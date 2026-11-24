//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection.
//!
//! The guards are per address, so a write through another view of the same
//! pages is not seen at the guarded addresses. The runtime's own writers
//! open the guards of every address showing what they write
//! ([`writing`], as `kernel_will_write`; DirectStorage in `victory-hle`).
//!
//! shortcut: the title's own CPU writes through one view are not seen at
//! another view of the same pages a copy guards (no case is known to
//! matter). A system call writing into a guarded block fails (none is
//! known to).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// Page number → the asking that last found it written.
    written: BTreeMap<usize, u64>,
    /// Room for the kernel's answer.
    found: Vec<usize>,
    /// Per thread in a submission's preparation ([`begin_submission`]),
    /// the write-watched ranges asked in it, whose checks the journal
    /// answers.
    epochs: Vec<(std::thread::ThreadId, Spans)>,
}

static JOURNAL: Mutex<Journal> = Mutex::new(Journal {
    asked: 0,
    written: BTreeMap::new(),
    found: Vec::new(),
    epochs: Vec::new(),
});

fn journal() -> std::sync::MutexGuard<'static, Journal> {
    JOURNAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
///
/// The path is the allocation's remembered one ([`region`]): views go to
/// the guards and other memory answers `None` with no system call;
/// write-watched memory asked in this submission already
/// ([`begin_submission`]) is answered by the journal.
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    let (lo, hi) = (start / PAGE * PAGE, end.div_ceil(PAGE) * PAGE);
    match region(lo) {
        Some((_, top, Path::WriteWatched)) if hi <= top => watched(lo, hi, guarded),
        Some((_, _, Path::View)) => viewed(lo, hi, guarded),
        Some((_, _, Path::Neither)) => None,
        _ => writes_asking(start, end, guarded),
    }
}

/// [`writes`] by asking the kernel each time: `GetWriteWatch`, and if that
/// fails, the memory type.
fn writes_asking(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let mut j = journal();
    let j = &mut *j;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let asked = ask(j, first * PAGE, last * PAGE, true);
    let guarded = match asked {
        None => {
            if !guarded {
                return None;
            }
            guard(first * PAGE, last * PAGE)?
        }
        Some(_) => Vec::new(),
    };
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
    }
    let count = asked.unwrap_or(0);
    j.asked += 1;
    let now = j.asked;
    let pages = j.found[..count].iter().map(|&at| at / PAGE).chain(guarded);
    for page in pages.collect::<Vec<_>>() {
        j.written.insert(page, now);
    }
    Some((now, j.written.range(first..last).map(|(&p, &w)| (p, w)).collect()))
}

/// `GetWriteWatch` over `[lo, hi)` (with `reset`, clearing what it
/// reports): the number of written 4 KiB pages it put in `j.found`, `None`
/// if the range is not all one write-watched allocation's.
fn ask(j: &mut Journal, lo: usize, hi: usize, reset: bool) -> Option<usize> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let mut count = (hi - lo) / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(test)]
    tests::called(if reset { tests::RESET } else { tests::PROBE });
    // SAFETY: the answer buffer holds `count` entries.
    let failed = unsafe {
        GetWriteWatch(
            if reset { WRITE_WATCH_FLAG_RESET } else { 0 },
            lo as *const _,
            hi - lo,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    (!failed).then_some(count)
}

/// [`writes`] of write-watched `[lo, hi)` (whole pages, one allocation):
/// from the journal if this submission asked the range already, else one
/// `GetWriteWatch`.
fn watched(lo: usize, hi: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (lo / PAGE, hi / PAGE);
    let mut held = journal();
    let j = &mut *held;
    let me = std::thread::current().id();
    let epoch = j.epochs.iter().position(|(t, _)| *t == me);
    if epoch.is_some_and(|e| j.epochs[e].1.covers(lo, hi)) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Watches, 1);
        return Some((j.asked, j.written.range(first..last).map(|(&p, &w)| (p, w)).collect()));
    }
    let Some(count) = ask(j, lo, hi, true) else {
        // Not what was remembered: the memory changed unreported.
        drop(held);
        forget(lo, hi);
        return writes_asking(lo, hi, guarded);
    };
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Watches, 1);
    j.asked += 1;
    let now = j.asked;
    for i in 0..count {
        let page = j.found[i] / PAGE;
        j.written.insert(page, now);
    }
    if let Some(e) = epoch {
        j.epochs[e].1.insert(lo, hi);
    }
    Some((now, j.written.range(first..last).map(|(&p, &w)| (p, w)).collect()))
}

/// [`writes`] of `[lo, hi)` starting in a view: the guards, with no
/// `GetWriteWatch` and no type query.
fn viewed(lo: usize, hi: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    if !guarded {
        return None;
    }
    let (first, last) = (lo / PAGE, hi / PAGE);
    let mut j = journal();
    let j = &mut *j;
    let pages = guard_blocks(lo, hi);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Watches, 1);
    j.asked += 1;
    let now = j.asked;
    for page in pages {
        j.written.insert(page, now);
    }
    Some((now, j.written.range(first..last).map(|(&p, &w)| (p, w)).collect()))
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

struct Guards {
    /// Blocks we made read-only.
    protected: BTreeSet<usize>,
    /// Of those, the ones not written since a copy took them.
    clean: BTreeSet<usize>,
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: BTreeSet::new(),
    clean: BTreeSet::new(),
});

fn guards() -> std::sync::MutexGuard<'static, Guards> {
    GUARDS.lock().unwrap_or_else(|e| e.into_inner())
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    use windows_sys::Win32::System::Memory::{MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery};
    #[cfg(test)]
    tests::called(tests::TYPE);
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(start as *const _, &mut info, size_of_val(&info)) };
    if n == 0 || info.Type != MEM_MAPPED {
        return None;
    }
    Some(guard_blocks(start, end))
}

/// [`guard`] of `[start, end)`, known to start in a view.
fn guard_blocks(start: usize, end: usize) -> Vec<usize> {
    install();
    let mut g = guards();
    let mut written = Vec::new();
    let mut run: Option<usize> = None;
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    for block in first..=last {
        let open = block < last && !g.clean.contains(&block);
        if open {
            written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
            if g.protected.contains(&block) {
                // Written through the runtime, still read-only.
                g.clean.insert(block);
            } else {
                run.get_or_insert(block);
                continue;
            }
        }
        if let Some(from) = run.take() {
            protect(&mut g, from, block);
        }
    }
    written
}

/// `WIN32_MEMORY_REGION_INFORMATION` flag bits.
const REGION_PRIVATE: u32 = 1;
const REGION_MAPPED_DATA_FILE: u32 = 2;
const REGION_MAPPED_IMAGE: u32 = 4;
const REGION_MAPPED_PAGE_FILE: u32 = 8;

/// `QueryVirtualMemoryInformation(MemoryRegionInfo)` at `at`: the
/// allocation's base and size and the region flags; `None` if it fails
/// (free memory). It reads the allocation's descriptor alone (~0.2 us).
fn region_info(at: usize) -> Option<(usize, usize, u32)> {
    use windows_sys::Win32::System::Memory::{
        MemoryRegionInfo, QueryVirtualMemoryInformation, WIN32_MEMORY_REGION_INFORMATION,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    #[cfg(test)]
    tests::called(tests::REGION);
    let mut info = WIN32_MEMORY_REGION_INFORMATION::default();
    // SAFETY: our own process, a sized out structure.
    let ok = unsafe {
        QueryVirtualMemoryInformation(
            GetCurrentProcess(),
            at as *const _,
            MemoryRegionInfo,
            (&raw mut info).cast(),
            size_of_val(&info),
            std::ptr::null_mut(),
        )
    } != 0;
    // SAFETY: every bit pattern is a valid `u32`.
    let flags = unsafe { info.Anonymous.Flags };
    ok.then_some((info.AllocationBase as usize, info.RegionSize, flags))
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail (unmapped).
fn protect(g: &mut Guards, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let set = |from: usize, to: usize| {
        let mut old = 0;
        // SAFETY: title view memory; only its protection changes.
        unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        }
    };
    let note = |g: &mut Guards, from: usize, to: usize| {
        g.protected.extend(from..to);
        g.clean.extend(from..to);
    };
    if set(from, to) {
        note(g, from, to);
        return;
    }
    for block in from..to {
        if set(block, block + 1) {
            note(g, block, block + 1);
        }
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let blocks: Vec<usize> = g
        .protected
        .range(start / BLOCK..end.div_ceil(BLOCK))
        .copied()
        .collect();
    for &block in &blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
        let mut old = 0;
        // SAFETY: a block we made read-only.
        unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
    }
    !blocks.is_empty()
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    open(&mut guards(), at, at + len);
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded.
pub fn title_memory_remapped(at: usize, len: usize) {
    let mut g = guards();
    let blocks: Vec<usize> = g
        .protected
        .range(at / BLOCK..(at + len).div_ceil(BLOCK))
        .copied()
        .collect();
    for block in blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
    }
    drop(g);
    title_memory_changed(at, len);
}

// Fewer system calls per check (stop 4: their count, not the bytes they
// cover, set the time):
// - Each allocation's path (write-watched, view, neither) is remembered
//   ([`region`]) until the runtime reports a change of title memory there
//   ([`title_memory_changed`]): checks of views go straight to the guards
//   (no `GetWriteWatch`, no type query), others answer at once.
// - One `GetWriteWatch(RESET)` per write-watched allocation per submission
//   ([`begin_submission`]): the journal answers every check of the
//   submission; Copy and Run steps note what they write ([`note_written`]).
// - Unreadable 4 KiB pages of title reservations are remembered until a
//   change there ([`unmapped_in`]): their failing reads are made once.
//
// Correct with a stale path too: a write-watched one is checked by its
// `GetWriteWatch` (failing, the path is asked again); guards on memory that
// is no view still catch CPU writes; "neither" copies whole. Only
// remembered unmapped pages need every change reported: title reservations
// are made, committed and mapped only through `victory-hle` (XMem, and the
// title's own `VirtualAlloc`/`VirtualFree`), which report.

/// How an allocation's checks are answered.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Path {
    /// `GetWriteWatch` (private memory reserved with write watching).
    WriteWatched,
    /// The guards (section views).
    View,
    /// Not at all (other private memory, images): copied whole.
    Neither,
}

/// Disjoint byte ranges `[start, end)` by start; touching ones merge.
struct Spans(BTreeMap<usize, usize>);

impl Spans {
    const fn new() -> Spans {
        Spans(BTreeMap::new())
    }

    fn insert(&mut self, mut a: usize, mut b: usize) {
        if a >= b {
            return;
        }
        if let Some((&s, &e)) = self.0.range(..=a).next_back()
            && e >= a
        {
            a = s;
            b = b.max(e);
        }
        let inside: Vec<usize> = self.0.range(a..=b).map(|(&s, _)| s).collect();
        for s in inside {
            if let Some(e) = self.0.remove(&s) {
                b = b.max(e);
            }
        }
        self.0.insert(a, b);
    }

    fn remove(&mut self, a: usize, b: usize) {
        if a >= b {
            return;
        }
        let hit: Vec<(usize, usize)> = self
            .0
            .range(..b)
            .rev()
            .take_while(|&(_, &e)| e > a)
            .map(|(&s, &e)| (s, e))
            .collect();
        for (s, e) in hit {
            self.0.remove(&s);
            if s < a {
                self.0.insert(s, a);
            }
            if e > b {
                self.0.insert(b, e);
            }
        }
    }

    /// The parts of `[a, b)` inside spans, in order.
    fn within(&self, a: usize, b: usize) -> Vec<(usize, usize)> {
        let mut parts: Vec<(usize, usize)> = self
            .0
            .range(..b)
            .rev()
            .take_while(|&(_, &e)| e > a)
            .map(|(&s, &e)| (s.max(a), e.min(b)))
            .collect();
        parts.reverse();
        parts
    }

    fn covers(&self, a: usize, b: usize) -> bool {
        self.0.range(..=a).next_back().is_some_and(|(_, &e)| e >= b)
    }
}

/// What the checks remember of title memory.
struct Remembered {
    /// Allocations by base: end and path. Disjoint.
    regions: BTreeMap<usize, (usize, Path)>,
    /// Unreadable 4 KiB pages of title reservations.
    unmapped: Spans,
    /// Title reservations: XMem's and the title's own `VirtualAlloc`'s.
    reserved: Spans,
    /// The last changes reported (number, start, end), for answers asked
    /// before a change and remembered after it.
    recent: std::collections::VecDeque<(u64, usize, usize)>,
}

static REMEMBERED: Mutex<Remembered> = Mutex::new(Remembered {
    regions: BTreeMap::new(),
    unmapped: Spans::new(),
    reserved: Spans::new(),
    recent: std::collections::VecDeque::new(),
});

/// Changes of title memory reported; bumped with [`REMEMBERED`] held.
static CHANGES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn remembered() -> std::sync::MutexGuard<'static, Remembered> {
    REMEMBERED.lock().unwrap_or_else(|e| e.into_inner())
}

/// Reported changes kept for [`Remembered::changed_since`].
const RECENT: usize = 1024;

impl Remembered {
    /// Whether a change overlapping `[a, b)` was reported after change
    /// number `seen`.
    fn changed_since(&self, seen: u64, a: usize, b: usize) -> bool {
        if CHANGES.load(std::sync::atomic::Ordering::SeqCst) == seen {
            return false;
        }
        if self.recent.front().is_none_or(|&(n, ..)| n > seen + 1) {
            return true;
        }
        self.recent
            .iter()
            .rev()
            .take_while(|&&(n, ..)| n > seen)
            .any(|&(_, s, e)| s < b && a < e)
    }

    /// Forgets what is remembered of `[a, b)`.
    fn forget(&mut self, a: usize, b: usize) {
        let hit: Vec<usize> = self
            .regions
            .range(..b)
            .rev()
            .take_while(|&(_, &(e, _))| e > a)
            .map(|(&s, _)| s)
            .collect();
        for s in hit {
            self.regions.remove(&s);
        }
        self.unmapped
            .remove(a / KERNEL_PAGE * KERNEL_PAGE, b.div_ceil(KERNEL_PAGE) * KERNEL_PAGE);
    }
}

/// The runtime changed title memory `[at, at + len)` (allocated, freed,
/// committed, decommitted, mapped or unmapped it), after the system did:
/// what the checks remember there is forgotten.
pub fn title_memory_changed(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let mut r = remembered();
    let n = CHANGES.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
    if r.recent.len() == RECENT {
        r.recent.pop_front();
    }
    r.recent.push_back((n, at, at + len));
    r.forget(at, at + len);
}

/// The runtime reserved (`made`) or released title memory `[at, at +
/// len)`, after the system did: a [`title_memory_changed`] that also
/// keeps the title's reservations, where unreadable pages are remembered.
pub fn title_reservation(at: usize, len: usize, made: bool) {
    if len == 0 {
        return;
    }
    title_memory_changed(at, len);
    let mut r = remembered();
    if made {
        r.reserved.insert(at, at + len);
    } else {
        r.reserved.remove(at, at + len);
    }
}

/// The allocation holding `at` (base and end), for reports of releases.
pub fn allocation_of(at: usize) -> Option<(usize, usize)> {
    let (base, size, _) = region_info(at)?;
    Some((base, base + size))
}

/// The allocation holding `at` and its path: remembered, or asked (and
/// remembered unless title memory there changed meanwhile). `None` for
/// free memory or memory of no known path.
fn region(at: usize) -> Option<(usize, usize, Path)> {
    if let Some((&base, &(end, path))) = remembered().regions.range(..=at).next_back()
        && end > at
    {
        return Some((base, end, path));
    }
    let seen = CHANGES.load(std::sync::atomic::Ordering::SeqCst);
    let (base, size, flags) = region_info(at)?;
    let end = base + size;
    let path = if flags & REGION_PRIVATE != 0 {
        // Write watching is the allocation's: asked without a reset.
        let mut j = journal();
        let watched = ask(&mut j, base, base + KERNEL_PAGE, false).is_some();
        if watched { Path::WriteWatched } else { Path::Neither }
    } else if flags & (REGION_MAPPED_DATA_FILE | REGION_MAPPED_PAGE_FILE) != 0 {
        Path::View
    } else if flags & REGION_MAPPED_IMAGE != 0 {
        Path::Neither
    } else {
        return None;
    };
    let mut r = remembered();
    if !r.changed_since(seen, base, end) {
        r.forget(base, end);
        r.regions.insert(base, (end, path));
    }
    Some((base, end, path))
}

/// Forgets the path remembered for `[a, b)`, found wrong.
fn forget(a: usize, b: usize) {
    remembered().forget(a, b);
}

/// At the start of a submission's preparation, before any check: asks
/// the kernel once per write-watched allocation (over the bounding range
/// of `ranges`, the ranges the submission's checks will ask, or a few
/// parts of it if they are sparse), notes the answer in the journal, and
/// has the journal answer this thread's checks of those ranges until
/// [`end_submission`]. The title thread that submits is in the submission
/// until then; other threads' writes meanwhile race the GPU on the console
/// too, and are seen at the next submission.
pub(crate) fn begin_submission(ranges: impl IntoIterator<Item = (usize, usize)>) {
    let mut by_allocation: BTreeMap<usize, (usize, Vec<(usize, usize)>)> = BTreeMap::new();
    for (start, end) in ranges {
        if start >= end {
            continue;
        }
        let (lo, hi) = (start / PAGE * PAGE, end.div_ceil(PAGE) * PAGE);
        if let Some((base, top, Path::WriteWatched)) = region(lo)
            && hi <= top
        {
            by_allocation.entry(base).or_insert((top, Vec::new())).1.push((lo, hi));
        }
    }
    let mut held = journal();
    let j = &mut *held;
    let mut asked = Spans::new();
    let mut pages = Vec::new();
    for (base, (top, ranges)) in by_allocation {
        for (lo, hi) in askings(ranges) {
            let Some(count) = ask(j, lo, hi, true) else {
                forget(base, top);
                break;
            };
            pages.extend(j.found[..count].iter().map(|&at| at / PAGE));
            asked.insert(lo, hi);
        }
    }
    j.asked += 1;
    let now = j.asked;
    for page in pages {
        j.written.insert(page, now);
    }
    let me = std::thread::current().id();
    j.epochs.retain(|(t, _)| *t != me);
    j.epochs.push((me, asked));
}

/// The submission's preparation is over (the lists run next): checks ask
/// the kernel again.
pub(crate) fn end_submission() {
    let me = std::thread::current().id();
    journal().epochs.retain(|(t, _)| *t != me);
}

/// In a submission's preparation, a Copy or Run step wrote title memory
/// `[at, at + len)`: later checks of the submission see it written.
pub(crate) fn note_written(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let mut j = journal();
    let me = std::thread::current().id();
    if !j.epochs.iter().any(|(t, _)| *t == me) {
        return;
    }
    j.asked += 1;
    let now = j.asked;
    for page in at / PAGE..(at + len).div_ceil(PAGE) {
        j.written.insert(page, now);
    }
}

/// The ranges to ask of one allocation for `ranges` (whole pages): their
/// bounding range, or, if that is big and mostly not asked for, the
/// ranges merged across gaps up to 16 MiB.
fn askings(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    const BIG: usize = 64 << 20;
    const GAP: usize = 16 << 20;
    ranges.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (a, b) in ranges {
        match merged.last_mut() {
            Some((_, e)) if a <= *e => *e = (*e).max(b),
            _ => merged.push((a, b)),
        }
    }
    let (Some(&(lo, _)), Some(&(_, hi))) = (merged.first(), merged.last()) else {
        return merged;
    };
    let covered: usize = merged.iter().map(|(a, b)| b - a).sum();
    if hi - lo <= BIG || covered * 4 >= hi - lo {
        return vec![(lo, hi)];
    }
    let mut parts: Vec<(usize, usize)> = Vec::new();
    for (a, b) in merged {
        match parts.last_mut() {
            Some((_, e)) if a - *e <= GAP => *e = b,
            _ => parts.push((a, b)),
        }
    }
    parts
}

/// The change count, read before reads whose failures [`note_unmapped`]
/// then remembers.
pub(crate) fn changes() -> u64 {
    CHANGES.load(std::sync::atomic::Ordering::SeqCst)
}

/// The unreadable 4 KiB pages remembered in `[at, at + len)`, in order (cut
/// to it).
pub(crate) fn unmapped_in(at: usize, len: usize) -> Vec<(usize, usize)> {
    remembered().unmapped.within(at, at + len)
}

/// Reads of the 4 KiB pages `pages` (start, end) failed: remembered where
/// they lie in title reservations, unless title memory there changed after
/// change number `seen`.
pub(crate) fn note_unmapped(pages: &[(usize, usize)], seen: u64) {
    if pages.is_empty() {
        return;
    }
    let mut r = remembered();
    for &(a, b) in pages {
        if r.changed_since(seen, a, b) {
            continue;
        }
        for (a, b) in r.reserved.within(a, b) {
            r.unmapped.insert(a, b);
        }
    }
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason).
        let mut g = guards();
        if open(&mut g, at, at + 1) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    thread_local! {
        /// This thread's system calls: `GetWriteWatch` with reset, without
        /// (probes), `VirtualQuery` type queries, region queries.
        pub(super) static CALLS: std::cell::Cell<[u32; 4]> = const { std::cell::Cell::new([0; 4]) };
    }
    pub(super) const RESET: usize = 0;
    pub(super) const PROBE: usize = 1;
    pub(super) const TYPE: usize = 2;
    pub(super) const REGION: usize = 3;

    pub(super) fn called(i: usize) {
        let mut c = CALLS.get();
        c[i] += 1;
        CALLS.set(c);
    }

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        // The journal may hold older writes at these addresses (tests that
        // freed memory here); only this asking's count.
        let (first, before) = writes(start, end, false).expect("watched");
        assert!(before.iter().all(|&(_, w)| w < first));
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        let after: Vec<_> = after.into_iter().filter(|&(_, w)| w > first).collect();
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
        title_memory_changed(p, 4 * PAGE);
        title_memory_changed(plain.as_ptr() as usize, PAGE);
    }

    /// The 64 KiB pages a copy of pages `[o, o + n)` of `p` takes now, as
    /// `Shadow::copy_stale` decides (`copied`: its asking per page).
    fn take(copied: &mut [u64], p: usize, o: usize, n: usize) -> Vec<usize> {
        let (now, written) = writes(p + o * PAGE, p + (o + n) * PAGE, true).expect("watched");
        let mut taken = Vec::new();
        for (k, at) in copied.iter_mut().enumerate() {
            let page = p / PAGE + o + k;
            let w = written.iter().find(|&&(q, _)| q == page).map_or(0, |&(_, w)| w);
            if *at == 0 || w > *at {
                taken.push(o + k);
                *at = now;
            }
        }
        taken
    }

    /// One write-watched allocation, copies of it overlapping (a buffer, a
    /// 4 MiB window, 64 KiB view ranges), title writes before each
    /// submission and a Copy step's write in it. One `GetWriteWatch(RESET)`
    /// per submission answers every check; each copy takes exactly the
    /// pages written before the submission (all of them the first time),
    /// plus the Copy step's destination, which the next submission takes
    /// once more (its write is noted, not asked); after the Copy step only
    /// its destination is taken.
    #[test]
    fn one_asking_per_allocation_serves_every_copy() {
        const PAGES: usize = 128;
        const COPY_STEP: usize = 70;
        let copies: [(usize, usize); 6] = [(0, PAGES), (64, 64), (3, 1), (10, 2), (COPY_STEP, 1), (127, 1)];
        // SAFETY: a fresh write-watched allocation (as xmem's), freed at the end.
        let p = unsafe {
            VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
        } as usize;
        assert_ne!(p, 0);
        let write = |page: usize| {
            // SAFETY: committed memory of ours.
            unsafe { ((p + page * PAGE + 100) as *mut u8).write_volatile(page as u8) };
        };
        let mut copied: Vec<Vec<u64>> = copies.iter().map(|&(_, n)| vec![0; n]).collect();
        let mut x = 0x2545_f491_4f6c_dd1du64;
        for s in 0..6 {
            let mut written: Vec<usize> = (0..8)
                .map(|_| {
                    x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (x >> 33) as usize % PAGES
                })
                .chain([3])
                .collect();
            written.sort_unstable();
            written.dedup();
            for &page in &written {
                write(page);
            }
            CALLS.set([0; 4]);
            begin_submission(copies.iter().map(|&(o, n)| (p + o * PAGE, p + (o + n) * PAGE)));
            for ((&(o, n), c), i) in copies.iter().zip(&mut copied).zip(0..) {
                let taken = take(c, p, o, n);
                let expected: Vec<usize> = (o..o + n)
                    .filter(|q| s == 0 || written.contains(q) || *q == COPY_STEP)
                    .collect();
                assert_eq!(taken, expected, "submission {s}, copy {i}: the pages written before it");
            }
            // A Copy step writes title memory, then the copies flush again.
            write(COPY_STEP);
            note_written(p + COPY_STEP * PAGE + 100, 1);
            for (&(o, n), c) in copies.iter().zip(&mut copied) {
                let expected: &[usize] = if (o..o + n).contains(&COPY_STEP) { &[COPY_STEP] } else { &[] };
                assert_eq!(take(c, p, o, n), expected, "the Copy step's write, and only it, is taken");
            }
            end_submission();
            assert_eq!(CALLS.get()[RESET], 1, "submission {s}: one asking");
        }
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
        title_memory_changed(p, PAGES * PAGE);
    }

    /// A pagefile section's view of `blocks` blocks.
    fn view(blocks: usize) -> usize {
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile};
        // SAFETY: a pagefile section, mapped read-write (the view keeps it).
        unsafe {
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, (blocks * BLOCK) as u32, std::ptr::null());
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, blocks * BLOCK).Value as usize;
            windows_sys::Win32::Foundation::CloseHandle(section);
            assert_ne!(view, 0);
            view
        }
    }

    fn unmap(view: usize) {
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        // SAFETY: a view of ours.
        unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
    }

    /// The pages `writes` reports written by its own asking.
    fn fresh(start: usize, end: usize) -> Option<Vec<usize>> {
        let (now, written) = writes(start, end, true)?;
        Some(written.into_iter().filter(|&(_, w)| w == now).map(|(p, _)| p).collect())
    }

    /// Checks of a view make no `GetWriteWatch` and no `VirtualQuery` (one
    /// region query learns the path), and the guards still see the
    /// title's writes.
    #[test]
    fn view_checks_make_no_write_watch_calls() {
        let v = view(4);
        CALLS.set([0; 4]);
        let pages = |b: usize| (v / PAGE + b..v / PAGE + b + 1).collect::<Vec<_>>();
        assert_eq!(fresh(v, v + 4 * BLOCK), Some((v / PAGE..v / PAGE + 4).collect()), "new guards");
        assert_eq!(fresh(v, v + 4 * BLOCK), Some(Vec::new()));
        // SAFETY: our view; the write faults into the guard handler.
        unsafe { ((v + 2 * BLOCK + 8) as *mut u8).write_volatile(7) };
        assert_eq!(fresh(v, v + 4 * BLOCK), Some(pages(2)), "the write is seen");
        assert_eq!(fresh(v + 2 * BLOCK, v + 3 * BLOCK), Some(Vec::new()), "and guarded again");
        assert!(writes(v, v + BLOCK, false).is_none(), "views unguarded: not watched");
        let [reset, probe, query, region] = CALLS.get();
        assert_eq!((reset, probe, query), (0, 0, 0), "no write watch, no type query");
        assert!(region <= 1, "at most one region query learns the path");
        unmap(v);
        title_memory_remapped(v, 4 * BLOCK);
    }

    /// A remembered path is forgotten when the runtime reports a change: a
    /// placeholder (neither) that a view replaces, the view unmapped and
    /// another mapped, a write-watched block decommitted and committed
    /// again. The title's writes after each change are seen.
    #[test]
    fn changes_reported_forget_the_path() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_DECOMMIT, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3, PAGE_NOACCESS,
            UnmapViewOfFile2, VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a placeholder of the test's own; views of pagefile
        // sections replace part of it, as xmem maps physical pages.
        let ph = unsafe {
            VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                4 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            )
        } as usize;
        assert_ne!(ph, 0);
        assert!(writes(ph, ph + 2 * BLOCK, true).is_none(), "a placeholder: neither");
        let map = |fill: u8| {
            // SAFETY: a pagefile section shown over the placeholder's first
            // two blocks (split off first; failing harmlessly once split).
            unsafe {
                let section =
                    CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, (2 * BLOCK) as u32, std::ptr::null());
                assert!(!section.is_null());
                VirtualFree(ph as *mut _, 2 * BLOCK, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
                let shown = MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    ph as *const _,
                    0,
                    2 * BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                );
                assert_eq!(shown.Value as usize, ph);
                windows_sys::Win32::Foundation::CloseHandle(section);
                std::ptr::write_bytes(ph as *mut u8, fill, 2 * BLOCK);
            }
        };
        let unmap_view = || {
            // SAFETY: the view mapped above, back to a placeholder.
            unsafe {
                UnmapViewOfFile2(GetCurrentProcess(), MEMORY_MAPPED_VIEW_ADDRESS { Value: ph as *mut _ }, MEM_PRESERVE_PLACEHOLDER)
            };
        };
        map(1);
        assert!(writes(ph, ph + 2 * BLOCK, true).is_none(), "unreported: still remembered as neither");
        title_memory_remapped(ph, 2 * BLOCK);
        let both: Vec<usize> = (ph / PAGE..ph / PAGE + 2).collect();
        assert_eq!(fresh(ph, ph + 2 * BLOCK), Some(both.clone()), "reported: a view, guarded");
        // SAFETY: the view; the write faults into the guard handler.
        unsafe { ((ph + BLOCK + 8) as *mut u8).write_volatile(2) };
        assert_eq!(fresh(ph, ph + 2 * BLOCK), Some(vec![ph / PAGE + 1]), "a write in the view is seen");
        unmap_view();
        title_memory_remapped(ph, 2 * BLOCK);
        assert!(writes(ph, ph + 2 * BLOCK, true).is_none(), "unmapped: neither again");
        map(3);
        title_memory_remapped(ph, 2 * BLOCK);
        assert_eq!(fresh(ph, ph + 2 * BLOCK), Some(both), "another view: all new");
        // SAFETY: as above.
        unsafe { (ph as *mut u8).write_volatile(4) };
        assert_eq!(fresh(ph, ph + 2 * BLOCK), Some(vec![ph / PAGE]), "a write after the remap is seen");
        unmap_view();
        title_memory_remapped(ph, 2 * BLOCK);
        // SAFETY: the placeholder's parts.
        unsafe {
            VirtualFree(ph as *mut _, 0, MEM_RELEASE);
            VirtualFree((ph + 2 * BLOCK) as *mut _, 0, MEM_RELEASE);
        }
        title_memory_changed(ph, 4 * BLOCK);

        // Write-watched memory decommitted and committed again.
        // SAFETY: a write-watched allocation of the test's own.
        let p = unsafe {
            VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
        } as usize;
        assert_ne!(p, 0);
        assert!(fresh(p, p + 4 * PAGE).is_some());
        // SAFETY: part of our allocation.
        unsafe {
            assert_ne!(VirtualFree((p + PAGE) as *mut _, PAGE, MEM_DECOMMIT), 0);
        }
        title_memory_changed(p + PAGE, PAGE);
        // SAFETY: as above, then a write into it.
        unsafe {
            assert!(!VirtualAlloc((p + PAGE) as *const _, PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            ((p + PAGE + 8) as *mut u8).write_volatile(5);
        }
        title_memory_changed(p + PAGE, PAGE);
        assert_eq!(fresh(p, p + 4 * PAGE), Some(vec![p / PAGE + 1]), "a write after the commit is seen");
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
        title_memory_changed(p, 4 * PAGE);
    }

    /// Write watching is the allocation's: asked on a reserved (not
    /// committed) page of a write-watched reservation, it answers.
    #[test]
    fn write_watch_answers_on_reserved_pages() {
        // SAFETY: a write-watched reservation of the test's own.
        let p = unsafe { VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE | MEM_WRITE_WATCH, PAGE_READWRITE) } as usize;
        assert_ne!(p, 0);
        assert!(ask(&mut journal(), p, p + KERNEL_PAGE, false).is_some());
        // SAFETY: our reservation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(!read(&mut file), "the kernel does not fault into our handler");
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        assert_eq!(unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) }, b"from the disk");
        assert_eq!(guard(view, view + 2 * BLOCK), Some((view / PAGE..view / PAGE + 1).collect()));
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
        title_memory_remapped(view, 2 * BLOCK);
    }

    /// A write into a guarded block's pages through another view of the
    /// same section counts the block as written.
    #[test]
    #[ignore = "known gap (module docs, shortcut): guards are per address, so the title's CPU writes through another view are not seen"]
    fn writes_through_another_view_count_as_written() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write twice.
        let (a, b) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let map = || MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize;
            (map(), map())
        };
        assert!(a != 0 && b != 0 && a != b);
        // A copy takes both blocks of view A: guarded and clean.
        assert!(guard(a, a + 2 * BLOCK).is_some());
        assert_eq!(guard(a, a + 2 * BLOCK), Some(Vec::new()));
        // SAFETY: view B is ours and not guarded.
        unsafe { std::ptr::write_bytes((b + 100) as *mut u8, 2, 13) };
        // SAFETY: as above.
        assert_eq!(unsafe { ((a + 100) as *const u8).read() }, 2, "one memory");
        assert_eq!(
            guard(a, a + 2 * BLOCK),
            Some((a / PAGE..a / PAGE + 1).collect()),
            "the block written through view B is reported written at view A"
        );
        for v in [a, b] {
            // SAFETY: our views.
            unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: v as *mut _ }) };
        }
        title_memory_remapped(a, 2 * BLOCK);
    }
}
