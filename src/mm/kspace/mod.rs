// SPDX-License-Identifier: MPL-2.0

#![allow(dead_code)]

//! Kernel memory space management.
//!
//! The kernel memory space is currently managed as follows, if the
//! address width is 48 bits (with 47 bits kernel space).
//!
//! TODO: the cap of linear mapping (the start of vm alloc) are raised
//! to workaround for high IO in TDX. We need actual vm alloc API to have
//! a proper fix.
//!
//! ```text
//! +-+ <- the highest used address (0xffff_ffff_ffff_0000)
//! | |         For the kernel code, 1 GiB. Mapped frames are tracked.
//! +-+ <- 0xffff_ffff_8000_0000
//! | |
//! | |         Unused hole.
//! +-+ <- 0xffff_e100_0000_0000
//! | |         For frame metadata, 1 TiB. Mapped frames are untracked.
//! +-+ <- 0xffff_e000_0000_0000
//! | |         For [`KVirtArea<Tracked>`], 16 TiB. Mapped pages are tracked with handles.
//! +-+ <- 0xffff_d000_0000_0000
//! | |         For [`KVirtArea<Untracked>`], 16 TiB. Mapped pages are untracked.
//! +-+ <- the middle of the higher half (0xffff_c000_0000_0000)
//! | |
//! | |
//! | |
//! | |         For linear mappings, 64 TiB.
//! | |         Mapped physical addresses are untracked.
//! | |
//! | |
//! | |
//! +-+ <- the base of high canonical address (0xffff_8000_0000_0000)
//! ```
//!
//! If the address width is (according to [`crate::arch::mm::PagingConsts`])
//! 39 bits or 57 bits, the memory space just adjust proportionally.

// pub(crate) mod kvirt_area;

use core::ops::Range;

use align_ext::AlignExt;

use super::{Paddr, PagingConstsTrait, Vaddr};
use crate::arch::mm::{PageTableEntry, PagingConsts};

/// The shortest supported address width is 39 bits. And the literal
/// values are written for 48 bits address width. Adjust the values
/// by arithmetic left shift.
const ADDR_WIDTH_SHIFT: isize = PagingConsts::ADDRESS_WIDTH as isize - 48;

/// Start of the kernel address space.
/// This is the _lowest_ address of the x86-64's _high_ canonical addresses.
pub const KERNEL_BASE_VADDR: Vaddr = 0xffff_8000_0000_0000 << ADDR_WIDTH_SHIFT;
/// End of the kernel address space (non inclusive).
pub const KERNEL_END_VADDR: Vaddr = 0xffff_ffff_ffff_0000 << ADDR_WIDTH_SHIFT;

/// The kernel code is linear mapped to this address.
///
/// FIXME: This offset should be randomly chosen by the loader or the
/// boot compatibility layer. But we disabled it because OSTD
/// doesn't support relocatable kernel yet.
pub fn kernel_loaded_offset() -> usize {
    KERNEL_CODE_BASE_VADDR
}

#[cfg(target_arch = "x86_64")]
const KERNEL_CODE_BASE_VADDR: usize = 0xffff_ffff_8000_0000 << ADDR_WIDTH_SHIFT;
#[cfg(target_arch = "riscv64")]
const KERNEL_CODE_BASE_VADDR: usize = 0xffff_ffff_0000_0000 << ADDR_WIDTH_SHIFT;

const FRAME_METADATA_CAP_VADDR: Vaddr = 0xffff_e100_0000_0000 << ADDR_WIDTH_SHIFT;
const FRAME_METADATA_BASE_VADDR: Vaddr = 0xffff_e000_0000_0000 << ADDR_WIDTH_SHIFT;
pub(in crate::mm) const FRAME_METADATA_RANGE: Range<Vaddr> =
    FRAME_METADATA_BASE_VADDR..FRAME_METADATA_CAP_VADDR;

const TRACKED_MAPPED_PAGES_BASE_VADDR: Vaddr = 0xffff_d000_0000_0000 << ADDR_WIDTH_SHIFT;
pub const TRACKED_MAPPED_PAGES_RANGE: Range<Vaddr> =
    TRACKED_MAPPED_PAGES_BASE_VADDR..FRAME_METADATA_BASE_VADDR;

const VMALLOC_BASE_VADDR: Vaddr = 0xffff_c000_0000_0000 << ADDR_WIDTH_SHIFT;
pub const VMALLOC_VADDR_RANGE: Range<Vaddr> = VMALLOC_BASE_VADDR..TRACKED_MAPPED_PAGES_BASE_VADDR;

/// The base address of the linear mapping of all physical
/// memory in the kernel address space.
pub const LINEAR_MAPPING_BASE_VADDR: Vaddr = 0xffff_8000_0000_0000 << ADDR_WIDTH_SHIFT;
pub const LINEAR_MAPPING_VADDR_RANGE: Range<Vaddr> = LINEAR_MAPPING_BASE_VADDR..VMALLOC_BASE_VADDR;

/// Convert physical address to virtual address using offset, only available inside `ostd`
pub fn paddr_to_vaddr(pa: Paddr) -> usize {
    debug_assert!(pa < VMALLOC_BASE_VADDR - LINEAR_MAPPING_BASE_VADDR);
    pa + LINEAR_MAPPING_BASE_VADDR
}

/// Returns whether the given address should be mapped as tracked.
///
/// About what is tracked mapping, see [`crate::mm::page::meta::MapTrackingStatus`].
pub(crate) fn should_map_as_tracked(addr: Vaddr) -> bool {
    !(LINEAR_MAPPING_VADDR_RANGE.contains(&addr) || VMALLOC_VADDR_RANGE.contains(&addr))
}
