// SPDX-License-Identifier: MPL-2.0

//! The page table cursor for mapping and querying over the page table.
//!
//! ## The page table lock protocol
//!
//! We provide a fine-grained lock protocol to allow concurrent accesses to
//! the page table. The protocol is originally proposed by Ruihan Li
//! <lrh2000@pku.edu.cn>.
//!
//! [`CursorMut::new`] accepts an address range, which indicates the page table
//! entries that may be visited by this cursor.
//!
//! Then, [`CursorMut::new`] finds an intermediate page table (not necessarily
//! the last-level or the top-level) which represents an address range that contains
//! the whole specified address range. It requires all locks from the root page
//! table to the intermediate page table, but then unlocks all locks excluding the
//! one for the intermediate page table. CursorMut then maintains the lock
//! guards from one for the intermediate page table to the leaf that the cursor is
//! currently manipulating.
//!
//! For example, if we're going to map the address range shown below:
//!
//! ```plain
//! Top-level page table node             A
//!                                      /
//!                                     B
//!                                    / \
//! Last-level page table nodes       C   D
//! Last-level PTEs               ---**...**---
//!                                  \__ __/
//!                                     V
//!                  Address range that we're going to map
//! ```
//!
//! When calling [`CursorMut::new`], it will:
//!  1. `lock(A)`, `lock(B)`, `unlock(A)`;
//!  2. `guards = [ locked(B) ]`.
//!
//! When calling [`CursorMut::map`], it will:
//!  1. `lock(C)`, `guards = [ locked(B), locked(C) ]`;
//!  2. Map some pages in `C`;
//!  3. `unlock(C)`, `lock_guard = [ locked(B) ]`;
//!  4. `lock(D)`, `lock_guard = [ locked(B), locked(D) ]`;
//!  5. Map some pages in D;
//!  6. `unlock(D)`, `lock_guard = [ locked(B) ]`;
//!
//!
//! ## Validity
//!
//! The page table cursor API will guarantee that the page table, as a data
//! structure, whose occupied memory will not suffer from data races. This is
//! ensured by the page table lock protocol. In other words, any operations
//! provided by the APIs (as long as safety requirements are met) will not
//! break the page table data structure (or other memory).
//!
//! However, the page table cursor creation APIs, [`CursorMut::new`] or
//! [`Cursor::new`], do not guarantee exclusive access to the virtual address
//! area you claim. From the lock protocol, you can see that there are chances
//! to create 2 cursors that claim the same virtual address range (one covers
//! another). In this case, the greater cursor may block if it wants to modify
//! the page table entries covered by the smaller cursor. Also, if the greater
//! cursor destructs the smaller cursor's parent page table node, it won't block
//! and the smaller cursor's change will not be visible. The user of the page
//! table cursor should add additional entry point checks to prevent these defined
//! behaviors if they are not wanted.

use core::any::TypeId;
use core::marker::PhantomData;
use core::ops::Range;

use align_ext::AlignExt;

use super::{
    Child, Entry, KernelMode, PageTable, PageTableEntryTrait, PageTableError, PageTableMode,
    PageTableNode, PagingConstsTrait, PagingLevel, UserMode, page_size, pte_index,
};
use crate::mm::kspace::should_map_as_tracked;
use crate::mm::page::meta::PageTablePageMeta;
use crate::mm::page::{DynPage, Page};
use crate::mm::{Paddr, PageProperty, Vaddr};

#[derive(Clone, Debug)]
pub enum PageTableItem {
    NotMapped {
        va: Vaddr,
        len: usize,
    },
    Mapped {
        va: Vaddr,
        page: DynPage,
        prop: PageProperty,
    },
    PageTableNode {
        page: DynPage,
    },
    #[allow(dead_code)]
    MappedUntracked {
        va: Vaddr,
        pa: Paddr,
        len: usize,
        prop: PageProperty,
    },
}

/// The cursor for traversal over the page table.
///
/// A slot is a PTE at any levels, which correspond to a certain virtual
/// memory range sized by the "page size" of the current level.
///
/// A cursor is able to move to the next slot, to read page properties,
/// and even to jump to a virtual address directly. We use a guard stack to
/// simulate the recursion, and adpot a page table locking protocol to
/// provide concurrency.
#[derive(Debug)]
pub struct Cursor<'a, M: PageTableMode, E: PageTableEntryTrait, C: PagingConstsTrait>
where
    [(); C::NR_LEVELS as usize]:,
{
    /// The lock guards of the cursor. The level 1 page table lock guard is at
    /// index 0, and the level N page table lock guard is at index N - 1.
    ///
    /// When destructing the cursor, the locks will be released in the order
    /// from low to high, exactly the reverse order of the acquisition.
    /// This behavior is ensured by the default drop implementation of Rust:
    /// <https://doc.rust-lang.org/reference/destructors.html>.
    guards: [Option<PageTableNode<E, C>>; C::NR_LEVELS as usize],
    /// The level of the page table that the cursor points to.
    level: PagingLevel,
    /// From `guard_level` to `level`, the locks are held in `guards`.
    guard_level: PagingLevel,
    /// The current virtual address that the cursor points to.
    va: Vaddr,
    /// The virtual address range that is locked.
    barrier_va: Range<Vaddr>,
    _phantom: PhantomData<&'a PageTable<M, E, C>>,
}

impl<'a, M: PageTableMode, E: PageTableEntryTrait, C: PagingConstsTrait> Cursor<'a, M, E, C>
where
    [(); C::NR_LEVELS as usize]:,
{
    /// Creates a cursor claiming the read access for the given range.
    ///
    /// The cursor created will only be able to query or jump within the given
    /// range. Out-of-bound accesses will result in panics or errors as return values,
    /// depending on the access method.
    ///
    /// Note that this function does not ensure exclusive access to the claimed
    /// virtual address range. The accesses using this cursor may block or fail.
    pub fn new(pt: &'a PageTable<M, E, C>, va: &Range<Vaddr>) -> Result<Self, PageTableError> {
        if !M::covers(va) || va.is_empty() {
            return Err(PageTableError::InvalidVaddrRange(va.start, va.end));
        }
        if va.start % C::BASE_PAGE_SIZE != 0 || va.end % C::BASE_PAGE_SIZE != 0 {
            return Err(PageTableError::UnalignedVaddr);
        }

        // Create a guard array that only hold the root node lock.
        let guards = core::array::from_fn(|i| {
            if i == (C::NR_LEVELS - 1) as usize {
                Some(pt.root.clone_shallow().lock())
            } else {
                None
            }
        });
        let mut cursor = Self {
            guards,
            level: C::NR_LEVELS,
            guard_level: C::NR_LEVELS,
            va: va.start,
            barrier_va: va.clone(),
            _phantom: PhantomData,
        };

        // Go down and get proper locks. The cursor should hold a lock of a
        // page table node containing the virtual address range.
        //
        // While going down, previous guards of too-high levels will be released.
        loop {
            let level_too_high = {
                let start_idx = pte_index::<C>(va.start, cursor.level);
                let end_idx = pte_index::<C>(va.end - 1, cursor.level);
                start_idx == end_idx
            };
            if !level_too_high {
                break;
            }

            let entry = cursor.cur_entry();
            if !entry.is_node() {
                break;
            }
            let Child::PageTable(child_pt) = entry.to_owned() else {
                unreachable!("Already checked");
            };

            cursor.push_level(child_pt.lock());

            // Release the guard of the previous (upper) level.
            cursor.guards[cursor.level as usize] = None;
            cursor.guard_level -= 1;
        }

        Ok(cursor)
    }

    /// Gets the information of the current slot.
    pub fn query(&mut self) -> Result<PageTableItem, PageTableError> {
        if self.va >= self.barrier_va.end {
            return Err(PageTableError::InvalidVaddr(self.va));
        }

        loop {
            let level = self.level;
            let va = self.va;

            match self.cur_entry().to_owned() {
                Child::PageTable(pt) => {
                    self.push_level(pt.lock());
                    continue;
                }
                Child::None => {
                    return Ok(PageTableItem::NotMapped { va, len: page_size::<C>(level) });
                }
                Child::Page(page, prop) => {
                    return Ok(PageTableItem::Mapped { va, page, prop });
                }
                Child::Untracked(pa, plevel, prop) => {
                    debug_assert_eq!(plevel, level);
                    return Ok(PageTableItem::MappedUntracked {
                        va,
                        pa,
                        len: page_size::<C>(level),
                        prop,
                    });
                }
            }
        }
    }

    /// Traverses forward in the current level to the next PTE.
    ///
    /// If reached the end of a page table node, it leads itself up to the next page of the parent
    /// page if possible.
    pub(in crate::mm) fn move_forward(&mut self) {
        let page_size = page_size::<C>(self.level);
        let next_va = self.va.align_down(page_size) + page_size;
        while self.level < self.guard_level && pte_index::<C>(next_va, self.level) == 0 {
            self.pop_level();
        }
        self.va = next_va;
    }

    /// Jumps to the given virtual address.
    /// If the target address is out of the range, this method will return `Err`.
    ///
    /// # Panics
    ///
    /// This method panics if the address has bad alignment.
    pub fn jump(&mut self, va: Vaddr) -> Result<(), PageTableError> {
        assert!(va % C::BASE_PAGE_SIZE == 0);
        if !self.barrier_va.contains(&va) {
            return Err(PageTableError::InvalidVaddr(va));
        }

        loop {
            let cur_node_start = self.va & !(page_size::<C>(self.level + 1) - 1);
            let cur_node_end = cur_node_start + page_size::<C>(self.level + 1);
            // If the address is within the current node, we can jump directly.
            if cur_node_start <= va && va < cur_node_end {
                self.va = va;
                return Ok(());
            }

            // There is a corner case that the cursor is depleted, sitting at the start of the
            // next node but the next node is not locked because the parent is not locked.
            if self.va >= self.barrier_va.end && self.level == self.guard_level {
                self.va = va;
                return Ok(());
            }

            debug_assert!(self.level < self.guard_level);
            self.pop_level();
        }
    }

    pub fn virt_addr(&self) -> Vaddr {
        self.va
    }

    /// Goes up a level.
    ///
    /// We release the current page if it has no mappings since the cursor
    /// only moves forward. And if needed we will do the final cleanup using
    /// this method after re-walk when the cursor is dropped.
    ///
    /// This method requires locks acquired before calling it. The discarded
    /// level will be unlocked.
    fn pop_level(&mut self) {
        self.guards[(self.level - 1) as usize] = None;
        self.level += 1;

        // TODO: Drop page tables if page tables become empty.
    }

    /// Goes down a level to a child page table.
    fn push_level(&mut self, child_pt: PageTableNode<E, C>) {
        self.level -= 1;
        debug_assert_eq!(self.level, child_pt.level());
        self.guards[(self.level - 1) as usize] = Some(child_pt);
    }

    fn should_map_as_tracked(&self) -> bool {
        (TypeId::of::<M>() == TypeId::of::<KernelMode>()
            || TypeId::of::<M>() == TypeId::of::<UserMode>())
            && should_map_as_tracked(self.va)
    }

    fn cur_entry(&mut self) -> Entry<'_, E, C> {
        let node = self.guards[(self.level - 1) as usize].as_mut().unwrap();
        node.entry(pte_index::<C>(self.va, self.level))
    }
}

impl<M: PageTableMode, E: PageTableEntryTrait, C: PagingConstsTrait> Iterator
    for Cursor<'_, M, E, C>
where
    [(); C::NR_LEVELS as usize]:,
{
    type Item = PageTableItem;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.query();
        if result.is_ok() {
            self.move_forward();
        }
        result.ok()
    }
}

/// The cursor of a page table that is capable of map, unmap or protect pages.
///
/// Also, it has all the capabilities of a [`Cursor`]. A virtual address range
/// in a page table can only be accessed by one cursor whether it is mutable or not.
#[derive(Debug)]
pub struct CursorMut<'a, M: PageTableMode, E: PageTableEntryTrait, C: PagingConstsTrait>(
    Cursor<'a, M, E, C>,
)
where
    [(); C::NR_LEVELS as usize]:;

impl<'a, M: PageTableMode, E: PageTableEntryTrait, C: PagingConstsTrait> CursorMut<'a, M, E, C>
where
    [(); C::NR_LEVELS as usize]:,
{
    /// Creates a cursor claiming the write access for the given range.
    ///
    /// The cursor created will only be able to map, query or jump within the given
    /// range. Out-of-bound accesses will result in panics or errors as return values,
    /// depending on the access method.
    ///
    /// Note that this function, the same as [`Cursor::new`], does not ensure exclusive
    /// access to the claimed virtual address range. The accesses using this cursor may
    /// block or fail.
    pub(super) fn new(
        pt: &'a PageTable<M, E, C>,
        va: &Range<Vaddr>,
    ) -> Result<Self, PageTableError> {
        Cursor::new(pt, va).map(|inner| Self(inner))
    }

    /// Jumps to the given virtual address.
    ///
    /// This is the same as [`Cursor::jump`].
    ///
    /// # Panics
    ///
    /// This method panics if the address is out of the range where the cursor is required to operate,
    /// or has bad alignment.
    pub fn jump(&mut self, va: Vaddr) -> Result<(), PageTableError> {
        self.0.jump(va)
    }

    /// Gets the current virtual address.
    pub fn virt_addr(&self) -> Vaddr {
        self.0.virt_addr()
    }

    /// Gets the information of the current slot.
    pub fn query(&mut self) -> Result<PageTableItem, PageTableError> {
        self.0.query()
    }

    // /// Copies the mapping from the given cursor to the current cursor.
    // ///
    // /// All the mappings in the current cursor's range must be empty. The
    // /// function allows the source cursor to operate on the mapping before
    // /// the copy happens. So it is equivalent to protect then duplicate.
    // /// Only the mapping is copied, the mapped pages are not copied.
    // ///
    // /// It can only copy tracked mappings since we consider the untracked
    // /// mappings not useful to be copied.
    // ///
    // /// After the operation, both cursors will advance by the specified length.
    // ///
    // /// # Safety
    // ///
    // /// The caller should ensure that
    // ///  - the range being copied with the operation does not affect kernel's
    // ///    memory safety.
    // ///  - both of the cursors are in tracked mappings.
    // ///
    // /// # Panics
    // ///
    // /// This function will panic if:
    // ///  - either one of the range to be copied is out of the range where any
    // ///    of the cursor is required to operate;
    // ///  - either one of the specified virtual address ranges only covers a
    // ///    part of a page.
    // ///  - the current cursor's range contains mapped pages.
    // pub unsafe fn copy_from(
    //     &mut self,
    //     src: &mut Self,
    //     len: usize,
    //     op: &mut impl FnMut(&mut PageProperty),
    // ) {
    //     assert!(len % page_size::<C>(1) == 0);
    //     let this_end = self.0.va + len;
    //     assert!(this_end <= self.0.barrier_va.end);
    //     let src_end = src.0.va + len;
    //     assert!(src_end <= src.0.barrier_va.end);

    //     while self.0.va < this_end && src.0.va < src_end {
    //         let src_va = src.0.va;
    //         let mut src_entry = src.0.cur_entry();

    //         match src_entry.to_owned() {
    //             Child::PageTable(pt) => {
    //                 let pt = pt.lock();
    //                 // If there's no mapped PTEs in the next level, we can
    //                 // skip to save time.
    //                 if pt.nr_children() != 0 {
    //                     src.0.push_level(pt);
    //                 } else {
    //                     src.0.move_forward();
    //                 }
    //                 continue;
    //             }
    //             Child::None => {
    //                 src.0.move_forward();
    //                 continue;
    //             }
    //             Child::Untracked(_, _, _) => {
    //                 panic!("Copying untracked mappings");
    //             }
    //             Child::Page(page, mut prop) => {
    //                 let mapped_page_size = page.size();

    //                 // Do protection.
    //                 src_entry.protect(op);

    //                 // Do copy.
    //                 op(&mut prop);
    //                 self.jump(src_va).unwrap();
    //                 let original = self.map(page, prop);
    //                 assert!(original.is_none());

    //                 // Only move the source cursor forward since `Self::map` will do it.
    //                 // This assertion is to ensure that they move by the same length.
    //                 debug_assert_eq!(mapped_page_size, page_size::<C>(src.0.level));
    //                 src.0.move_forward();
    //             }
    //         }
    //     }
    // }
}
