//! The offline batch: every shader found goes through the translator, and
//! what it cannot handle yet is tallied (decision 0003).

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use victory_shader::emit::host::{self, Host};
use victory_shader::emit::{self, validator::Validator};
use victory_shader::structure::{self, Atom, LaneExit};
use victory_shader::{
    Container, FourCC, HwStage, NativeProgram, RootType, ShaderError, ir, native_programs,
};
use victory_shader::{io, resource};

use crate::shader::{MAX_FILE, Source, sources};

#[derive(Default)]
struct Tally {
    containers: usize,
    duplicates: usize,
    programs: usize,
    translated: usize,
    instructions: usize,
    structured: usize,
    vector: usize,
    guarded: usize,
    /// Guards that read a mask the pass cannot see into.
    opaque: usize,
    lane_branches: usize,
    uniform_branches: usize,
    /// Data joins with a lane mask on some edge: edges no lane reading
    /// the join comes over, and joins some such lane reads.
    dead_mask_edges: usize,
    live_mask_joins: usize,
    /// Those in compute programs, and the ones of them whose wave-wide
    /// value is known (`WaveJoin`), with uniform branches among them.
    cs_mask_joins: usize,
    cs_wave_joins: usize,
    cs_uniform_joins: usize,
    tracked: usize,
    loaded: usize,
    built: usize,
    constant: usize,
    /// Descriptors and scalar accesses with a computed address term.
    dynamic: usize,
    scalar: usize,
    used: usize,
    /// Used-resource entries whose root parameter no descriptor traces
    /// to, by resource and root type.
    untraced_used: BTreeMap<String, usize>,
    /// Root parameters descriptors trace to that no used entry names.
    unlisted: usize,
    /// Vertex and pixel programs, and those whose I/O was raised.
    io_programs: usize,
    io_raised: usize,
    interpolated: usize,
    vertex_values: usize,
    /// Exports by target kind.
    exports: BTreeMap<&'static str, usize>,
    /// Exports still packed as f16 pairs after the export patch.
    compressed: usize,
    /// Vertex and pixel programs every earlier stage handled, and how far
    /// emission got with them.
    emit_programs: usize,
    emitted: usize,
    validated: usize,
    pipelines: usize,
    /// The same for compute programs.
    cs_programs: usize,
    cs_emitted: usize,
    cs_validated: usize,
    cs_pipelines: usize,
    /// The same for mesh programs.
    ms_programs: usize,
    ms_emitted: usize,
    ms_validated: usize,
    ms_pipelines: usize,
    /// Reason → programs stopped by it, and the first one.
    failed: BTreeMap<String, (usize, String)>,
    /// With `--list`: one line per program and outcome.
    list: bool,
    lines: Vec<String>,
}

impl Tally {
    fn fail(&mut self, reason: String, example: impl FnOnce() -> String) {
        if self.list {
            let e = example();
            self.lines.push(format!("{e}  {reason}"));
            self.failed.entry(reason).or_insert((0, e)).0 += 1;
            return;
        }
        self.failed
            .entry(reason)
            .or_insert_with(|| (0, example()))
            .0 += 1;
    }

    fn merge(&mut self, other: Tally) {
        self.containers += other.containers;
        self.duplicates += other.duplicates;
        self.programs += other.programs;
        self.translated += other.translated;
        self.instructions += other.instructions;
        self.structured += other.structured;
        self.vector += other.vector;
        self.guarded += other.guarded;
        self.opaque += other.opaque;
        self.lane_branches += other.lane_branches;
        self.uniform_branches += other.uniform_branches;
        self.dead_mask_edges += other.dead_mask_edges;
        self.live_mask_joins += other.live_mask_joins;
        self.cs_mask_joins += other.cs_mask_joins;
        self.cs_wave_joins += other.cs_wave_joins;
        self.cs_uniform_joins += other.cs_uniform_joins;
        self.tracked += other.tracked;
        self.loaded += other.loaded;
        self.built += other.built;
        self.constant += other.constant;
        self.dynamic += other.dynamic;
        self.scalar += other.scalar;
        self.used += other.used;
        for (k, n) in other.untraced_used {
            *self.untraced_used.entry(k).or_default() += n;
        }
        self.unlisted += other.unlisted;
        self.io_programs += other.io_programs;
        self.io_raised += other.io_raised;
        self.interpolated += other.interpolated;
        self.vertex_values += other.vertex_values;
        self.compressed += other.compressed;
        self.emit_programs += other.emit_programs;
        self.emitted += other.emitted;
        self.validated += other.validated;
        self.pipelines += other.pipelines;
        self.cs_programs += other.cs_programs;
        self.cs_emitted += other.cs_emitted;
        self.cs_validated += other.cs_validated;
        self.cs_pipelines += other.cs_pipelines;
        self.ms_programs += other.ms_programs;
        self.ms_emitted += other.ms_emitted;
        self.ms_validated += other.ms_validated;
        self.ms_pipelines += other.ms_pipelines;
        for (k, n) in other.exports {
            *self.exports.entry(k).or_default() += n;
        }
        self.lines.extend(other.lines);
        for (reason, (n, example)) in other.failed {
            self.failed.entry(reason).or_insert((0, example)).0 += n;
        }
    }
}

/// An emitted vertex, mesh or pixel shader kept for `VICTORY_PAIRS`.
struct KeptShader {
    label: String,
    /// The first 16 hex digits of the container's BLAKE3, as run logs name
    /// shaders.
    hash: String,
    /// The BLAKE3 of its DXIL part (duplicates point at it).
    dxil: blake3::Hash,
    container: Vec<u8>,
    shader: emit::HostShader,
}

/// Shaders kept for `VICTORY_PAIRS`, shared between threads, and the
/// container hashes of duplicates with the DXIL hash of their original.
#[derive(Default)]
struct Kept {
    shaders: Vec<KeptShader>,
    aliases: Vec<(String, blake3::Hash)>,
}

/// The optional checks after emission, one set per thread.
struct Checks {
    /// Vertex, mesh and pixel shaders, kept to build the pipelines of
    /// `VICTORY_PAIRS` after the walk.
    keep: Option<std::sync::Arc<Mutex<Kept>>>,
    validator: Option<Validator>,
    host: Option<(Host, emit::HostShader)>,
}

pub fn run(
    paths: Vec<PathBuf>,
    validate: bool,
    pipelines: bool,
    list: bool,
    jobs: Option<usize>,
) -> Result<()> {
    let paths = crate::default_paths(paths)?;
    let dll = validate.then(Validator::default_path).flatten();
    if validate {
        Validator::load(dll.as_deref()).map_err(anyhow::Error::msg)?;
        println!(
            "validator    {}",
            dll.as_ref()
                .map_or("dxil.dll".into(), |p| p.display().to_string())
        );
    }
    if pipelines {
        Host::new(false).map_err(anyhow::Error::msg)?;
    }
    // `VICTORY_PAIRS`: comma-separated `first+pixel` pairs, each a label
    // substring or a container hash prefix as run logs give it; the entry
    // `oversized` pairs every mesh shader whose outputs exceed the
    // validator's limit with the pixel shaders of its effect.
    let pairs: Vec<(String, String)> = std::env::var("VICTORY_PAIRS")
        .unwrap_or_default()
        .split(',')
        .filter_map(|p| match p.trim() {
            "oversized" => Some(("oversized".to_owned(), String::new())),
            p => p
                .split_once('+')
                .map(|(v, p)| (v.trim().to_owned(), p.trim().to_owned())),
        })
        .collect();
    let kept = std::sync::Arc::new(Mutex::new(Kept::default()));
    let threads =
        jobs.unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()));
    let seen = Mutex::new(HashSet::new());
    let total = Mutex::new(Tally::default());
    for root in &paths {
        for entry in walkdir::WalkDir::new(root)
            .into_iter()
            .filter_map(Result::ok)
        {
            if !entry.file_type().is_file() || entry.metadata().map_or(0, |m| m.len()) > MAX_FILE {
                continue;
            }
            let Ok(data) = std::fs::read(entry.path()) else {
                continue;
            };
            let jobs = sources(&data);
            let next = AtomicUsize::new(0);
            std::thread::scope(|s| {
                for _ in 0..threads.min(jobs.len()) {
                    s.spawn(|| {
                        let mut t = Tally {
                            list,
                            ..Tally::default()
                        };
                        let mut checks = Checks {
                            keep: (!pairs.is_empty()).then(|| kept.clone()),
                            validator: validate
                                .then(|| Validator::load(dll.as_deref()).ok())
                                .flatten(),
                            host: pipelines
                                .then(|| Host::new(false).ok().map(|h| (h, host::stub_ps())))
                                .flatten(),
                        };
                        while let Some(job) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                            translate(job, entry.path(), &seen, &mut t, &mut checks);
                        }
                        lock(&total).merge(t);
                    });
                }
            });
        }
    }
    let t = total.into_inner().unwrap_or_else(|e| e.into_inner());
    println!(
        "containers   {} ({} duplicates skipped)",
        t.containers, t.duplicates
    );
    println!(
        "programs     {} translated of {} ({} IR instructions)",
        t.translated, t.programs, t.instructions
    );
    println!(
        "structured   {} of {} ({} vector instructions, {} guarded, {} on opaque masks; \
         branches {} per lane, {} uniform)",
        t.structured,
        t.translated,
        t.vector,
        t.guarded,
        t.opaque,
        t.lane_branches,
        t.uniform_branches
    );
    println!(
        "mask joins   {} read ({} edges proven dead); compute {}: {} exact ({} uniform), \
         {} refused",
        t.live_mask_joins,
        t.dead_mask_edges,
        t.cs_mask_joins,
        t.cs_wave_joins,
        t.cs_uniform_joins,
        t.cs_mask_joins - t.cs_wave_joins
    );
    println!(
        "resources    {} of {} tracked (descriptors {} loaded, {} built, {} constant; \
         {} scalar accesses; {} with a computed term)",
        t.tracked, t.translated, t.loaded, t.built, t.constant, t.scalar, t.dynamic
    );
    println!(
        "  used-resource entries {}: {} not traced {:?}; {} traced parameters not listed",
        t.used,
        t.untraced_used.values().sum::<usize>(),
        t.untraced_used,
        t.unlisted
    );
    println!(
        "I/O          {} of {} vertex and pixel programs raised ({} interpolations, \
         {} vertex values; exports {:?}, {} still f16 pairs)",
        t.io_raised, t.io_programs, t.interpolated, t.vertex_values, t.exports, t.compressed
    );
    println!(
        "DXIL         {} emitted of {} vertex and pixel programs; {} validated, {} pipelines",
        t.emitted, t.emit_programs, t.validated, t.pipelines
    );
    println!(
        "             {} emitted of {} compute programs; {} validated, {} pipelines",
        t.cs_emitted, t.cs_programs, t.cs_validated, t.cs_pipelines
    );
    println!(
        "             {} emitted of {} mesh programs; {} validated, {} pipelines",
        t.ms_emitted, t.ms_programs, t.ms_validated, t.ms_pipelines
    );
    if !pairs.is_empty() {
        link_pairs(&pairs, &lock(&kept), dll.as_deref(), pipelines)?;
    }
    let mut lines = t.lines;
    lines.sort();
    for l in lines {
        println!("{l}");
    }
    let mut failed: Vec<_> = t.failed.into_iter().collect();
    failed.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
    for (reason, (n, example)) in failed {
        println!("  {n:>7}  {reason}  (e.g. {example})");
    }
    Ok(())
}

fn translate(
    source: &Source,
    path: &Path,
    seen: &Mutex<HashSet<blake3::Hash>>,
    t: &mut Tally,
    checks: &mut Checks,
) {
    let example = || format!("{}{}", path.display(), source.label());
    let Some(bytes) = source.bytes() else {
        return t.fail("corrupt record".into(), example);
    };
    t.containers += 1;
    let Ok(container) = Container::parse(&bytes) else {
        return t.fail("no DXIL part (SM5 or older)".into(), example);
    };
    let Some(dxil) = container.part(FourCC::DXIL).map(|p| p.data) else {
        return t.fail("no DXIL part (SM5 or older)".into(), example);
    };
    let dxil_hash = blake3::hash(dxil);
    let hash = blake3::hash(&bytes).to_hex()[..16].to_owned();
    if !lock(seen).insert(dxil_hash) {
        t.duplicates += 1;
        if let Some(keep) = &checks.keep {
            lock(keep).aliases.push((hash, dxil_hash));
        }
        return;
    }
    let programs = match native_programs(dxil) {
        Ok(p) => p,
        Err(e) => return t.fail(e.to_string(), example),
    };
    for p in programs {
        t.programs += 1;
        match ir::build(&p.patched_code(), p.wave64, p.vgprs) {
            Ok(program) => {
                t.translated += 1;
                t.instructions += program.blocks.iter().map(|b| b.insts.len()).sum::<usize>();
                let res = match resource::track(&program, &p.layout, p.stage) {
                    Ok(r) => {
                        tally_resources(&p, &r, t);
                        Some(r)
                    }
                    Err(ShaderError::Unsupported { offset, what }) => {
                        t.fail(format!("resources: {what}"), || {
                            format!("{} {:?} at {offset:#x}", example(), p.stage)
                        });
                        None
                    }
                    Err(e) => {
                        t.fail(e.to_string(), || format!("{} {:?}", example(), p.stage));
                        None
                    }
                };
                let graphics = matches!(p.stage, HwStage::Vs | HwStage::Ps);
                let compute = p.stage == HwStage::Cs;
                let mut raised = None;
                if graphics || compute || p.stage == HwStage::Ms {
                    t.io_programs += usize::from(graphics);
                    match io::raise(&program, &p) {
                        Ok(r) => {
                            if graphics {
                                tally_io(&r, t);
                            }
                            raised = Some(r);
                        }
                        Err(ShaderError::Unsupported { offset, what }) => t
                            .fail(format!("I/O: {what}"), || {
                                format!("{} {:?} at {offset:#x}", example(), p.stage)
                            }),
                        Err(e) => t.fail(e.to_string(), || format!("{} {:?}", example(), p.stage)),
                    }
                }
                let structured = match structure::structure(&program) {
                    Ok(s) => {
                        tally_structured(&program, &s, compute, t);
                        Some(s)
                    }
                    Err(e) => {
                        let reason = match e {
                            ShaderError::Unsupported { what, .. } => format!("structure: {what}"),
                            ShaderError::Isa { reason, .. } => format!("structure: {reason}"),
                            e => e.to_string(),
                        };
                        t.fail(reason, || format!("{} {:?}", example(), p.stage));
                        None
                    }
                };
                if let (Some(r), Some(i), Some(s)) = (res, raised, structured) {
                    let label = || format!("{} {:?}", example(), p.stage);
                    *counter(t, p.stage, 0) += 1;
                    match emit::emit(&container, &p, &program, &s, &i, &r, None) {
                        Ok(shader) => {
                            *counter(t, p.stage, 1) += 1;
                            if let Some(keep) = &checks.keep
                                && matches!(shader.stage, HwStage::Vs | HwStage::Ms | HwStage::Ps)
                            {
                                lock(keep).shaders.push(KeptShader {
                                    label: label(),
                                    hash: hash.clone(),
                                    dxil: dxil_hash,
                                    container: bytes.to_vec(),
                                    shader: shader.clone(),
                                });
                            }
                            check(shader, t, checks, label);
                        }
                        Err(ShaderError::Unsupported { offset, what }) => t
                            .fail(format!("emit: {what}"), || {
                                format!("{} at {offset:#x}", label())
                            }),
                        Err(e) => t.fail(format!("emit: {e}"), label),
                    }
                }
            }
            Err(e) => {
                let reason = match e {
                    ShaderError::Unsupported { what, .. } => format!("unsupported: {what}"),
                    e => e.to_string(),
                };
                t.fail(reason, || format!("{} {:?}", example(), p.stage));
            }
        }
    }
}

/// Validates an emitted shader if asked to.
fn check(
    mut shader: emit::HostShader,
    t: &mut Tally,
    checks: &mut Checks,
    label: impl Fn() -> String,
) {
    // `VICTORY_EMIT_DUMP` with `--list`: every emitted shader as a file.
    if t.list
        && let Ok(dir) = std::env::var("VICTORY_EMIT_DUMP")
    {
        let name: String = label()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        let _ = std::fs::write(
            format!("{dir}/{}.dxil", &name[name.len().saturating_sub(100)..]),
            &shader.dxil,
        );
    }
    if let Some((h, stub_ps)) = &checks.host {
        let made = match shader.stage {
            HwStage::Vs => h.pipeline(&shader, stub_ps).map(drop),
            HwStage::Cs => h.compute_pipeline(&shader).map(drop),
            HwStage::Ms => h.mesh_pipeline(&shader, stub_ps).map(drop),
            _ => host::stub_vs(&shader).and_then(|vs| h.pipeline(&vs, &shader).map(drop)),
        };
        match made {
            Ok(()) => {
                *counter(t, shader.stage, 3) += 1;
                if t.list {
                    t.lines.push(format!("{}  pipeline created", label()));
                }
            }
            Err(e) => t.fail(format!("pipeline: {e}"), &label),
        }
    }
    let Some(v) = checks.validator.as_mut() else {
        return;
    };
    match v.validate(&mut shader.dxil) {
        Ok(()) => {
            *counter(t, shader.stage, 2) += 1;
            if t.list {
                t.lines.push(format!("{}  validated", label()));
            }
        }
        Err(text) => {
            let first = text
                .lines()
                .find(|l| l.contains("error"))
                .unwrap_or(text.lines().next().unwrap_or(""))
                .to_owned();
            if let Ok(dir) = std::env::var("VICTORY_EMIT_DUMP") {
                let n = t
                    .failed
                    .get(&format!("validator: {first}"))
                    .map_or(0, |f| f.0);
                if n < 3 {
                    let name = format!(
                        "{dir}/{}_{n}",
                        first
                            .chars()
                            .filter(char::is_ascii_alphanumeric)
                            .take(40)
                            .collect::<String>()
                    );
                    let _ = std::fs::write(format!("{name}.dxil"), &shader.dxil);
                    let _ = std::fs::write(format!("{name}.txt"), format!("{}\n{text}", label()));
                }
            }
            t.fail(format!("validator: {first}"), label);
        }
    }
}

/// The pipelines of `VICTORY_PAIRS`: a mesh shader whose outputs exceed
/// the validator's limit is translated again declaring only what the
/// pixel shader reads (`emit::translate_linked`), as the host pipeline
/// path does; each pair reports the output bytes, the validator's verdict
/// on the first stage, how the signatures link (`emit::link_mismatches`)
/// and, with `--pipelines`, whether the host creates the pipeline
/// (`VICTORY_D3D12_WARP` set: on WARP).
fn link_pairs(
    pairs: &[(String, String)],
    kept: &Kept,
    dll: Option<&Path>,
    pipelines: bool,
) -> Result<()> {
    let host = match pipelines {
        true => Some(
            Host::new(std::env::var_os("VICTORY_D3D12_WARP").is_some())
                .map_err(anyhow::Error::msg)?,
        ),
        false => None,
    };
    let mut validator = Validator::load(dll).ok();
    let find = |name: &str| {
        let dxil = kept
            .aliases
            .iter()
            .find(|(h, _)| h.starts_with(name))
            .map(|a| a.1);
        kept.shaders
            .iter()
            .find(|k| k.hash.starts_with(name) || dxil == Some(k.dxil) || k.label.contains(name))
    };
    let mut list: Vec<(&KeptShader, &KeptShader)> = Vec::new();
    let mut missing = 0;
    for (first, ps) in pairs {
        if first == "oversized" {
            list.extend(oversized_pairs(kept));
            continue;
        }
        match (find(first), find(ps)) {
            (Some(a), Some(b)) => list.push((a, b)),
            _ => {
                missing += 1;
                println!("pair {first} + {ps}: not emitted");
            }
        }
    }
    let (mut fit, mut validated, mut linked_ok, mut created) = (0, 0, 0, 0);
    let (mut oversized, mut unlinked) = (0, 0);
    for (first, ps) in &list {
        // As the host pipeline path does: the first stage translated again
        // in the pixel shader's input layout when it is too big or does
        // not link as it stands.
        let before = first.shader.mesh_output_bytes();
        let unlinking = emit::link_mismatches(&first.shader, &ps.shader).len();
        oversized += usize::from(before > emit::MESH_OUTPUT_LIMIT);
        unlinked += usize::from(unlinking > 0);
        let relink = before > emit::MESH_OUTPUT_LIMIT || unlinking > 0;
        let linked = if relink {
            emit::Link::pixel(&ps.shader)
                .and_then(|l| emit::translate_linked(&first.container, Some(&l)))
                .map_err(|e| format!("linked translation: {e}"))
        } else {
            Ok(first.shader.clone())
        };
        let result = linked.map(|s| {
            let after = s.mesh_output_bytes();
            fit += usize::from(after <= emit::MESH_OUTPUT_LIMIT);
            let verdict = match validator.as_mut().map(|v| v.validate(&mut s.dxil.clone())) {
                Some(Ok(())) => {
                    validated += 1;
                    "validated".to_owned()
                }
                Some(Err(text)) => text
                    .lines()
                    .find(|l| l.contains("error"))
                    .unwrap_or("validator error")
                    .to_owned(),
                None => "not validated".to_owned(),
            };
            let mismatches = emit::link_mismatches(&s, &ps.shader);
            let link = if mismatches.is_empty() {
                linked_ok += 1;
                "signatures link".to_owned()
            } else {
                format!("signatures: {}", mismatches.join(", "))
            };
            let made = host.as_ref().map(|host| {
                let made = if s.stage == HwStage::Ms {
                    host.mesh_pipeline(&s, &ps.shader)
                } else {
                    host.pipeline(&s, &ps.shader)
                };
                match made {
                    Ok(_) => {
                        created += 1;
                        "; pipeline created".to_owned()
                    }
                    Err(e) => format!("; pipeline: {e}"),
                }
            });
            let made = made.unwrap_or_default();
            let how = if relink { "relinked" } else { "as is" };
            format!(
                "outputs {before} -> {after} bytes, {unlinking} link mismatches; {how}; \
                 {verdict}; {link}{made}"
            )
        });
        let line = result.unwrap_or_else(|e| e);
        println!("pair {} + {}: {line}", first.label, ps.label);
    }
    println!(
        "pairs        {} found ({missing} not emitted), before: {oversized} past the output \
         limit, {unlinked} not linking; after: {fit} within the limit, {validated} validated, \
         {linked_ok} linking, {created} pipelines",
        list.len()
    );
    Ok(())
}

/// Every mesh shader past the output limit with the pixel shaders of its
/// effect: the same name suffix (`<name>_<8 hex>_Wrapped`), one per pixel
/// shader name, the record nearest the mesh shader's in the same file.
fn oversized_pairs(kept: &Kept) -> Vec<(&KeptShader, &KeptShader)> {
    // (file, record offset, effect suffix, shader name) of a label.
    let parse = |label: &str| {
        let (file, rest) = label.split_once('@')?;
        let (offset, rest) = rest.split_once(' ')?;
        let offset = u64::from_str_radix(offset.trim_start_matches("0x"), 16).ok()?;
        let name = rest.split(' ').next()?.strip_suffix("_Wrapped")?;
        let (_, effect) = name.rsplit_once('_')?;
        Some((file.to_owned(), offset, effect.to_owned(), name.to_owned()))
    };
    let pixels: Vec<_> = kept
        .shaders
        .iter()
        .filter(|k| k.shader.stage == HwStage::Ps)
        .filter_map(|k| Some((parse(&k.label)?, k)))
        .collect();
    let mut out = Vec::new();
    for ms in kept.shaders.iter().filter(|k| {
        k.shader.stage == HwStage::Ms && k.shader.mesh_output_bytes() > emit::MESH_OUTPUT_LIMIT
    }) {
        let Some((file, offset, effect, _)) = parse(&ms.label) else {
            continue;
        };
        let mut nearest: BTreeMap<&str, (u64, &KeptShader)> = BTreeMap::new();
        for ((f, o, e, name), ps) in &pixels {
            if *f != file || *e != effect {
                continue;
            }
            let d = o.abs_diff(offset);
            let best = nearest.entry(name).or_insert((d, ps));
            if d < best.0 {
                *best = (d, ps);
            }
        }
        out.extend(nearest.into_values().map(|(_, ps)| (ms, ps)));
    }
    out
}

/// Emission counts of `stage`'s kind: 0 programs, 1 emitted, 2 validated,
/// 3 pipelines.
fn counter(t: &mut Tally, stage: HwStage, k: usize) -> &mut usize {
    let [a, b, c, d] = match stage {
        HwStage::Cs => [
            &mut t.cs_programs,
            &mut t.cs_emitted,
            &mut t.cs_validated,
            &mut t.cs_pipelines,
        ],
        HwStage::Ms => [
            &mut t.ms_programs,
            &mut t.ms_emitted,
            &mut t.ms_validated,
            &mut t.ms_pipelines,
        ],
        _ => [
            &mut t.emit_programs,
            &mut t.emitted,
            &mut t.validated,
            &mut t.pipelines,
        ],
    };
    [a, b, c, d].into_iter().nth(k).expect("0 to 3")
}

fn tally_io(r: &io::Io, t: &mut Tally) {
    t.io_raised += 1;
    for (_, i) in &r.interps {
        match i {
            io::Interp::Interpolated { .. } | io::Interp::Raw { second: true, .. } => {
                t.interpolated += 1
            }
            io::Interp::Vertex { .. } => t.vertex_values += 1,
            io::Interp::Raw { .. } => {}
        }
    }
    for e in &r.exports {
        let kind = match e.target {
            io::Target::Color(_) => "color",
            io::Target::Depth => "depth",
            io::Target::Null => "null",
            io::Target::Position(_) => "position",
            io::Target::Param { .. } => "param",
            io::Target::Unread => "unread param",
            io::Target::Primitive => "primitive",
            io::Target::PrimitiveParam { .. } => "primitive param",
        };
        *t.exports.entry(kind).or_default() += 1;
        t.compressed += usize::from(e.compressed);
    }
}

fn tally_resources(p: &NativeProgram, r: &resource::Resources, t: &mut Tally) {
    t.tracked += 1;
    t.scalar += r.scalar.len();
    t.dynamic += r.scalar.iter().filter(|s| s.1.dynamic.is_some()).count();
    let address = |load| r.scalar.iter().find(|s| s.0 == load).map(|s| s.1);
    let mut traced = HashSet::new();
    for leaf in r.descriptors.iter().flat_map(|d| d.source.leaves()) {
        let a = match *leaf {
            resource::Source::Loaded { load, .. } => {
                t.loaded += 1;
                address(load)
            }
            resource::Source::Built { address, .. } => {
                t.built += 1;
                t.dynamic += usize::from(address.dynamic.is_some());
                Some(address)
            }
            resource::Source::Constant(_) => {
                t.constant += 1;
                None
            }
            resource::Source::Phi { .. } => unreachable!("leaves are no phis"),
        };
        // Heaps hold dynamic resources, which no used entry lists; code
        // tables are no root parameter.
        if let Some(a) = a
            && a.param != resource::CODE
            && !matches!(
                p.layout[a.param].root,
                RootType::ResourceHeap | RootType::SamplerHeap
            )
        {
            traced.insert(p.layout[a.param].user_data);
        }
    }
    // Root constants are listed as a CBV, read straight from the SGPRs.
    traced.extend(
        r.params
            .iter()
            .map(|&(_, param, _)| &p.layout[param])
            .filter(|l| l.root == RootType::Constants)
            .map(|l| l.user_data),
    );
    let listed: HashSet<u8> = p.used.iter().map(|u| u.user_data).collect();
    t.used += p.used.len();
    // Immediate samplers are literals in the code: the distinct sampler
    // values (decoded) are matched against the listed entries, which do not
    // say which value is whose.
    let mut values: Vec<emit::StaticSampler> = Vec::new();
    for d in r
        .descriptors
        .iter()
        .filter(|d| d.role == resource::Role::Sampler)
    {
        for leaf in d.source.leaves() {
            if let resource::Source::Constant(words) = leaf
                && let Ok(w) = <[u32; 4]>::try_from(&words[..words.len().min(4)])
            {
                let s = emit::StaticSampler::from_words(w);
                if !values.contains(&s) {
                    values.push(s);
                }
            }
        }
    }
    // Engines often declare equal samplers at several registers, so equal
    // values may stand for several entries, but never more values than
    // entries.
    let immediate = p
        .used
        .iter()
        .filter(|u| u.root == RootType::ImmediateSampler)
        .count();
    let samplers_match = !values.is_empty() && values.len() <= immediate;
    let matched = |u: &&victory_shader::UsedResource| {
        if u.root == RootType::ImmediateSampler {
            samplers_match
        } else {
            traced.contains(&u.user_data)
        }
    };
    for u in p.used.iter().filter(|u| !matched(u)) {
        *t.untraced_used
            .entry(format!("{:?} via {:?}", u.kind, u.root))
            .or_default() += 1;
    }
    t.unlisted += traced.difference(&listed).count();
}

fn tally_structured(
    program: &ir::Program,
    s: &structure::Structured,
    compute: bool,
    t: &mut Tally,
) {
    t.structured += 1;
    t.dead_mask_edges += s.dead_mask_edges.len();
    t.live_mask_joins += s.live_mask_joins;
    if compute {
        t.cs_mask_joins += s.live_mask_joins;
        t.cs_wave_joins += s.wave_joins.len();
        t.cs_uniform_joins += s
            .wave_joins
            .values()
            .filter(|&&w| w == structure::WaveJoin::Uniform)
            .count();
    }
    for (block, guards) in program.blocks.iter().zip(&s.guards) {
        t.vector += block
            .insts
            .iter()
            .filter(|i| i.read(ir::Reg::S(ir::EXEC)).is_some())
            .count();
        for &g in guards.iter().flatten() {
            t.guarded += 1;
            let support = s.bools.support(g);
            t.opaque += usize::from(support.iter().any(|a| matches!(a, Atom::Opaque(_))));
        }
    }
    for exit in &s.exits {
        if let LaneExit::Branch { when, .. } = *exit {
            if s.bools
                .support(when)
                .iter()
                .all(|a| matches!(a, Atom::Uniform(_)))
            {
                t.uniform_branches += 1;
            } else {
                t.lane_branches += 1;
            }
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}
