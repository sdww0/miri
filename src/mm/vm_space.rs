// SPDX-License-Identifier: MPL-2.0

//! Virtual memory space management.
//!
//! The [`VmSpace`] struct is provided to manage the virtual memory space of a
//! user. Cursors are used to traverse and modify over the virtual memory space
//! concurrently. The VM space cursor [`self::Cursor`] is just a wrapper over
//! the page table cursor [`super::page_table::Cursor`], providing efficient,
//! powerful concurrent accesses to the page table, and suffers from the same
//! validity concerns as described in [`super::page_table::cursor`].

use core::ops::Range;
use core::sync::atomic::{AtomicPtr, Ordering};

use super::Vaddr;
use crate::Error;
use crate::arch::mm::{PageTableEntry, PagingConsts, current_page_table_paddr};
use crate::mm::io::Fallible;
use crate::mm::page_table::{self, PageTable, PageTableItem, UserMode};
use crate::mm::{Frame, MAX_USERSPACE_VADDR, PageProperty, VmReader, VmWriter};

/// Virtual memory space.
///
/// A virtual memory space (`VmSpace`) can be created and assigned to a user
/// space so that the virtual memory of the user space can be manipulated
/// safely. For example,  given an arbitrary user-space pointer, one can read
/// and write the memory location referred to by the user-space pointer without
/// the risk of breaking the memory safety of the kernel space.
///
/// A newly-created `VmSpace` is not backed by any physical memory pages. To
/// provide memory pages for a `VmSpace`, one can allocate and map physical
/// memory ([`Frame`]s) to the `VmSpace` using the cursor.
///
/// A `VmSpace` can also attach a page fault handler, which will be invoked to
/// handle page faults generated from user space.
#[allow(clippy::type_complexity)]
#[derive(Debug)]
pub struct VmSpace {
    pt: PageTable<UserMode>,
}

impl VmSpace {
    /// Creates a new VM address space.
    // pub fn new() -> Self {
    //     Self {
    //         pt: KERNEL_PAGE_TABLE.get().unwrap().create_user_page_table(),
    //     }
    // }

    /// Gets an immutable cursor in the virtual address range.
    ///
    /// The cursor behaves like a lock guard, exclusively owning a sub-tree of
    /// the page table, preventing others from creating a cursor in it. So be
    /// sure to drop the cursor as soon as possible.
    ///
    /// The creation of the cursor may block if another cursor having an
    /// overlapping range is alive.
    pub fn cursor(&self, va: &Range<Vaddr>) -> Result<Cursor<'_>, crate::error::Error> {
        Ok(self.pt.cursor(va).map(Cursor)?)
    }

    /// Gets an mutable cursor in the virtual address range.
    ///
    /// The same as [`Self::cursor`], the cursor behaves like a lock guard,
    /// exclusively owning a sub-tree of the page table, preventing others
    /// from creating a cursor in it. So be sure to drop the cursor as soon as
    /// possible.
    ///
    /// The creation of the cursor may block if another cursor having an
    /// overlapping range is alive. The modification to the mapping by the
    /// cursor may also block or be overridden the mapping of another cursor.
    pub fn cursor_mut(&self, va: &Range<Vaddr>) -> Result<CursorMut<'_>, crate::error::Error> {
        Ok(self.pt.cursor_mut(va).map(|pt_cursor| {

            // The activation lock is held; other CPUs cannot activate this `VmSpace`.
            let ptr = ACTIVATED_VM_SPACE.load(Ordering::Relaxed) as *const VmSpace;

            CursorMut { pt_cursor }
        })?)
    }

    /// Creates a reader to read data from the user space of the current task.
    ///
    /// Returns `Err` if this `VmSpace` is not belonged to the user space of the current task
    /// or the `vaddr` and `len` do not represent a user space memory range.
    pub fn reader(
        &self,
        vaddr: Vaddr,
        len: usize,
    ) -> Result<VmReader<'_, Fallible>, crate::error::Error> {
        if current_page_table_paddr() != unsafe { self.pt.root_paddr() } {
            return Err(Error::AccessDenied);
        }

        if vaddr.checked_add(len).unwrap_or(usize::MAX) > MAX_USERSPACE_VADDR {
            return Err(Error::AccessDenied);
        }

        // `VmReader` is neither `Sync` nor `Send`, so it will not live longer than the current
        // task. This ensures that the correct page table is activated during the usage period of
        // the `VmReader`.
        //
        // SAFETY: The memory range is in user space, as checked above.
        Ok(unsafe { VmReader::<Fallible>::from_user_space(vaddr as *const u8, len) })
    }

    /// Creates a writer to write data into the user space.
    ///
    /// Returns `Err` if this `VmSpace` is not belonged to the user space of the current task
    /// or the `vaddr` and `len` do not represent a user space memory range.
    pub fn writer(
        &self,
        vaddr: Vaddr,
        len: usize,
    ) -> Result<VmWriter<'_, Fallible>, crate::error::Error> {
        if current_page_table_paddr() != unsafe { self.pt.root_paddr() } {
            return Err(Error::AccessDenied);
        }

        if vaddr.checked_add(len).unwrap_or(usize::MAX) > MAX_USERSPACE_VADDR {
            return Err(Error::AccessDenied);
        }

        // `VmWriter` is neither `Sync` nor `Send`, so it will not live longer than the current
        // task. This ensures that the correct page table is activated during the usage period of
        // the `VmWriter`.
        //
        // SAFETY: The memory range is in user space, as checked above.
        Ok(unsafe { VmWriter::<Fallible>::from_user_space(vaddr as *mut u8, len) })
    }
}

/// The cursor for querying over the VM space without modifying it.
///
/// It exclusively owns a sub-tree of the page table, preventing others from
/// reading or modifying the same sub-tree. Two read-only cursors can not be
/// created from the same virtual address range either.
pub struct Cursor<'a>(page_table::Cursor<'a, UserMode, PageTableEntry, PagingConsts>);

impl Iterator for Cursor<'_> {
    type Item = VmItem;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.query();
        if result.is_ok() {
            self.0.move_forward();
        }
        result.ok()
    }
}

impl Cursor<'_> {
    /// Query about the current slot.
    ///
    /// This function won't bring the cursor to the next slot.
    pub fn query(&mut self) -> Result<VmItem, crate::error::Error> {
        Ok(self.0.query().map(|item| item.try_into().unwrap())?)
    }

    /// Jump to the virtual address.
    pub fn jump(&mut self, va: Vaddr) -> Result<(), crate::error::Error> {
        self.0.jump(va)?;
        Ok(())
    }

    /// Get the virtual address of the current slot.
    pub fn virt_addr(&self) -> Vaddr {
        self.0.virt_addr()
    }
}

/// The cursor for modifying the mappings in VM space.
///
/// It exclusively owns a sub-tree of the page table, preventing others from
/// reading or modifying the same sub-tree.
pub struct CursorMut<'a> {
    pt_cursor: page_table::CursorMut<'a, UserMode, PageTableEntry, PagingConsts>,
}

impl CursorMut<'_> {
    /// Query about the current slot.
    ///
    /// This is the same as [`Cursor::query`].
    ///
    /// This function won't bring the cursor to the next slot.
    pub fn query(&mut self) -> Result<VmItem, crate::error::Error> {
        Ok(self.pt_cursor.query().map(|item| item.try_into().unwrap())?)
    }

    /// Jump to the virtual address.
    ///
    /// This is the same as [`Cursor::jump`].
    pub fn jump(&mut self, va: Vaddr) -> Result<(), crate::error::Error> {
        self.pt_cursor.jump(va)?;
        Ok(())
    }

    /// Get the virtual address of the current slot.
    pub fn virt_addr(&self) -> Vaddr {
        self.pt_cursor.virt_addr()
    }

    // /// Copies the mapping from the given cursor to the current cursor.
    // ///
    // /// All the mappings in the current cursor's range must be empty. The
    // /// function allows the source cursor to operate on the mapping before
    // /// the copy happens. So it is equivalent to protect then duplicate.
    // /// Only the mapping is copied, the mapped pages are not copied.
    // ///
    // /// After the operation, both cursors will advance by the specified length.
    // ///
    // /// Note that it will **NOT** flush the TLB after the operation. Please
    // /// make the decision yourself on when and how to flush the TLB using
    // /// the source's [`CursorMut::flusher`].
    // ///
    // /// # Panics
    // ///
    // /// This function will panic if:
    // ///  - either one of the range to be copied is out of the range where any
    // ///    of the cursor is required to operate;
    // ///  - either one of the specified virtual address ranges only covers a
    // ///    part of a page.
    // ///  - the current cursor's range contains mapped pages.
    // pub fn copy_from(
    //     &mut self,
    //     src: &mut Self,
    //     len: usize,
    //     op: &mut impl FnMut(&mut PageProperty),
    // ) {
    //     // SAFETY: Operations on user memory spaces are safe if it doesn't
    //     // involve dropping any pages.
    //     unsafe { self.pt_cursor.copy_from(&mut src.pt_cursor, len, op) }
    // }
}

/// The `Arc` pointer to the activated VM space on this CPU. If the pointer
/// is NULL, it means that the activated page table is merely the kernel
/// page table.
// TODO: If we are enabling ASID, we need to maintain the TLB state of each
// CPU, rather than merely the activated `VmSpace`. When ASID is enabled,
// the non-active `VmSpace`s can still have their TLB entries in the CPU!
static ACTIVATED_VM_SPACE: AtomicPtr<VmSpace> = AtomicPtr::new(core::ptr::null_mut());

/// The result of a query over the VM space.
#[derive(Debug)]
pub enum VmItem {
    /// The current slot is not mapped.
    NotMapped {
        /// The virtual address of the slot.
        va: Vaddr,
        /// The length of the slot.
        len: usize,
    },
    /// The current slot is mapped.
    Mapped {
        /// The virtual address of the slot.
        va: Vaddr,
        /// The mapped frame.
        frame: Frame,
        /// The property of the slot.
        prop: PageProperty,
    },
}

impl TryFrom<PageTableItem> for VmItem {
    type Error = &'static str;

    fn try_from(item: PageTableItem) -> core::result::Result<Self, Self::Error> {
        match item {
            PageTableItem::NotMapped { va, len } => Ok(VmItem::NotMapped { va, len }),
            PageTableItem::Mapped { va, page, prop } =>
                Ok(VmItem::Mapped {
                    va,
                    frame: page
                        .try_into()
                        .map_err(|_| "found typed memory mapped into `VmSpace`")?,
                    prop,
                }),
            PageTableItem::MappedUntracked { .. } =>
                Err("found untracked memory mapped into `VmSpace`"),
            PageTableItem::PageTableNode { .. } => {
                unreachable!()
            }
        }
    }
}
