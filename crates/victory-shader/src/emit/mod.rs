//! DXIL emission (decision 0003, step 3): a console vertex, pixel, compute
//! or mesh shader as a PC DXIL container (SM 6.6) that host D3D12 accepts.
//! Compute shaders keep the console's thread-group size; one that depends
//! on the lane layout (cross-lane operations) declares the console's wave
//! size (`WaveSize`), so the host device must support it: an RX 9070 XT
//! reports 32 to 64 lanes, WARP 4 only (`WaveLaneCountMin`/`Max`, checked
//! 2026-11-24), so WARP refuses those pipelines. SGPR spills through VGPR
//! lanes need no lane layout ([`crate::ir::spill_traffic`]). Pixel and
//! vertex code keeps to wave operations whose result holds on any wave
//! size (see `lower::MaskData`), WARP's included.
//!
//! What the host binds for a translated shader ([`HostShader`]):
//!
//! * **User data** at `b0, space 0` (32-bit root constants or a constant
//!   buffer, per stage): dword `index` holds a root parameter's value, as
//!   [`UserData`] lists. A root-constant dword is the constant. A
//!   descriptor table is the byte offset of its first descriptor in its
//!   heap: the descriptor index times 32 for CBV/SRV/UAV tables, times 16
//!   for sampler tables (the console's descriptor sizes; the code adds
//!   offsets in those units). The heap bases of dynamic resources are 0
//!   and not listed.
//! * **Descriptors** through SM 6.6 `ResourceDescriptorHeap[]` and
//!   `SamplerDescriptorHeap[]`, at the index the console address gives, so
//!   the host heaps keep the title's descriptor order. The root signature
//!   needs `CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED` and
//!   `SAMPLER_HEAP_DIRECTLY_INDEXED`.
//! * **Immediate samplers** as static samplers `s<n>, space 0`, `n` the
//!   position in [`HostShader::samplers`] (the console sampler words) and
//!   [`HostShader::static_samplers`] (the same decoded into a D3D12 static
//!   sampler description). Comparison sampling (added 2026-11-23): an
//!   immediate sampler it uses has a comparison filter there (bit 0x80);
//!   a heap sampler it uses must be a comparison sampler on the host.
//! * **Root CBVs and root SRVs** (raw buffers) as root descriptors
//!   `b<i>` / `t<i>` in [`ROOT_CBV_SPACE`], `i` the title's root parameter
//!   index ([`HostShader::root_cbvs`], [`HostShader::root_srvs`]); the
//!   address the title sets is the buffer's.
//! * **Vertex fetch** (added 2026-11-23). Grand Theft Auto VI's mesh and
//!   vertex programs fetch vertices themselves with typed-format loads
//!   (`tbuffer_load_format_*`): the format (8_8_8_8 unorm, 16_16 snorm,
//!   2_10_10_10 unorm, 8_8_8_8 and 16_16_16_16 uint...) is in the
//!   instruction, so the translation reads raw bytes and converts them
//!   itself. Their descriptors are built around a root SRV (stride 1, no
//!   record limit): the host binds the root SRV `t<slot>` above with the
//!   address the title sets, nothing more. The vertex index, the stride
//!   and the attribute offset are the shader's own arithmetic on data in
//!   that buffer (per-stream headers), not descriptor fields.
//! * **Buffer views in the heap** are declared by how the shader reads
//!   them: raw (ByteAddressBuffer) for dword loads and stores, atomics and
//!   typed-format loads without an index; structured with the used-resource
//!   entry's stride when indexed; typed with four 32-bit float channels for
//!   `buffer_load_format_*` and `buffer_store_format_*`. The host's view at
//!   that index must be of that kind and start at the console descriptor's
//!   base address (raw: `R32_TYPELESS` with the raw flag), whatever kind the
//!   title's view description says. Typed-format loads from heap
//!   descriptors (1,024 world compute programs, ray-tracing material
//!   resolves) read raw views this way.
//!
//! * **Texture atomics** (added 2026-11-23): a texture UAV the shader uses
//!   atomically is declared a 32-bit unsigned single-channel view; the
//!   host's view at that index must be an R32 integer view (64-bit atomics
//!   stop the translation). Global loads from a root SRV read it raw.
//! * **Barycentrics read as data** (added 2026-11-24): the console's i and
//!   j of a pixel are the weights of vertices 1 and 2; read other than by
//!   interpolation, they become `SV_Barycentrics` channels y and z (one
//!   input per interpolation mode, two at most), which needs
//!   `BarycentricsSupported` (options 3). Interpolation alone declares none.
//!
//! Entry points: [`translate`] (container bytes as the title passes them
//! at pipeline creation → [`HostShader`]), [`emit`] (the same with the
//! earlier stages already run). The root constants of each stage must
//! cover [`HostShader::user_data_dwords`]. Stability: these types and the
//! binding scheme are what pipeline creation builds on; changes are
//! announced.
//! * **Signatures** as the console container has them (same semantics,
//!   registers and masks), so the input layout and stage linking are the
//!   title's; pixel outputs are plain 32-bit values.
//! * **Mesh shaders** ([`HwStage::Ms`], `ms_6_6`): user data and
//!   descriptors as above (the root constants visible to the mesh stage);
//!   the pipeline is a mesh pipeline (pipeline-state stream, no input
//!   layout) with the title's pixel shader. The thread group is the
//!   console's in lanes (`[numthreads]` in the DXIL), the vertex and
//!   primitive maxima its own, triangles only, no wave size required. The
//!   console numbers groups in x only: the draw path passes the title's
//!   count as `DispatchMesh(x, 1, 1)`. Per-primitive outputs (`PSG1`) are
//!   the console's, without `SV_PrimitiveID` and `SV_CullPrimitive` (the
//!   console never exports them).
//! * **Linked to a pixel shader** ([`translate_linked`], added
//!   2026-11-24): a vertex or mesh shader then declares its outputs in the
//!   translated pixel shader's input layout: each output the pixel shader
//!   reads (same semantic name and index) at the input's row, columns,
//!   type and interpolation mode; system values it does not read on rows
//!   past those; inputs nothing writes declared unwritten; other outputs
//!   left out, with the stores to them. The console links stages by
//!   semantic, and its vertex and pixel signatures place them at
//!   different registers, which PC linking refuses
//!   ([`link_mismatches`]). It also keeps mesh shaders under the
//!   validator's [`MESH_OUTPUT_LIMIT`] ([`HostShader::mesh_output_bytes`]):
//!   some console mesh programs declare more than any pixel shader reads.

mod bitcode;
mod container;
#[cfg(all(test, windows))]
mod diff_tests;
#[cfg(all(windows, any(test, feature = "host")))]
pub mod host;
mod lower;
mod sampler;
mod shader;
#[cfg(all(test, windows))]
mod synthetic_tests;
#[cfg(windows)]
pub mod validator;
#[cfg(all(test, windows))]
mod valu_tests;
mod wave;

use crate::dxbc::{Container, FourCC};
use crate::native::{HwStage, NativeProgram, RootType};
use crate::{ShaderError, io, ir, resource, structure};

pub use sampler::StaticSampler;
pub use shader::{ROOT_CBV_SPACE, SAMPLER_SPACE, USER_DATA_SPACE};

/// The loop limit without an override: 2^20.
pub const DEFAULT_LOOP_LIMIT: u32 = lower::LOOP_LIMIT;

/// The iteration count after which a translated loop is left, per lane and
/// per entry into the loop: [`DEFAULT_LOOP_LIMIT`].
///
/// With the `diagnostics` feature, `VICTORY_SHADER_LOOP_LIMIT` (decimal or
/// `0x` hex, at least 1) sets it instead. A loop that hits the limit in
/// every lane of a full-screen pass still runs for seconds (2^20 times a
/// 4K screen is about 10^13 iterations); a much lower limit shows whether
/// such loops are what makes a frame slow. The value is part of the
/// translation, so the shader cache must key on it (`victory-gpu`'s does).
pub fn loop_limit() -> u32 {
    static LIMIT: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        if cfg!(feature = "diagnostics") {
            parse_loop_limit(std::env::var("VICTORY_SHADER_LOOP_LIMIT").ok().as_deref())
        } else {
            lower::LOOP_LIMIT
        }
    })
}

/// `VICTORY_SHADER_LOOP_LIMIT`'s value; the default when it is unset,
/// unreadable or 0. Pure, for testing.
fn parse_loop_limit(value: Option<&str>) -> u32 {
    let parsed = value
        .map(str::trim)
        .and_then(|v| match v.strip_prefix("0x") {
            Some(hex) => u32::from_str_radix(hex, 16).ok(),
            None => v.parse().ok(),
        });
    match parsed {
        Some(n) if n > 0 => n,
        _ => lower::LOOP_LIMIT,
    }
}

/// A translated shader and what the host binds for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostShader {
    pub stage: HwStage,
    /// The DXIL container, hashed as D3D12 expects.
    pub dxil: Vec<u8>,
    /// The user-data dwords the shader reads.
    pub user_data: Vec<UserData>,
    /// Immediate samplers: the four sampler words of `s<n>`.
    pub samplers: Vec<[u32; 4]>,
    /// The same samplers decoded, `s<n>` at position n: the static sampler
    /// descriptions the host's root signature declares.
    pub static_samplers: Vec<StaticSampler>,
    /// Root CBVs the shader reads: root parameter indices, each bound as
    /// `b<index>` in [`ROOT_CBV_SPACE`].
    pub root_cbvs: Vec<u8>,
    /// Root SRVs (raw buffers) the shader reads: `t<index>` in
    /// [`ROOT_CBV_SPACE`].
    pub root_srvs: Vec<u8>,
}

impl HostShader {
    /// Size of the user-data buffer the shader declares, in dwords (a
    /// multiple of 4; 0 when it reads none). Root constants must cover it.
    pub fn user_data_dwords(&self) -> u32 {
        let n = self
            .user_data
            .iter()
            .map(|u| u.index + 1)
            .max()
            .unwrap_or(0);
        n.next_multiple_of(4)
    }

    /// A mesh shader's vertex plus primitive output bytes as the validator
    /// counts them: every declared component, 4 bytes, for each of the
    /// maximum vertices or primitives rounded up to 32. 0 for other stages.
    pub fn mesh_output_bytes(&self) -> u32 {
        if self.stage != HwStage::Ms {
            return 0;
        }
        let Ok(c) = Container::parse(&self.dxil) else {
            return 0;
        };
        let components = |fourcc| {
            c.part(fourcc)
                .and_then(|p| signature(p.data, false).ok())
                .map_or(0, |els| els.iter().map(|e| u32::from(e.cols)).sum::<u32>())
        };
        // PSV0: its runtime info's size, then the info (mesh maxima at 12
        // and 14).
        let max = |at: usize| {
            c.part(FourCC::PSV0)
                .and_then(|p| p.data.get(4 + at..6 + at))
                .map_or(0, |b| u32::from(u16::from_le_bytes([b[0], b[1]])))
        };
        4 * (components(FourCC::OSG1) * max(12).next_multiple_of(32)
            + components(FourCC::PSG1) * max(14).next_multiple_of(32))
    }
}

/// The most bytes of vertex plus primitive outputs the validator allows a
/// mesh shader.
pub const MESH_OUTPUT_LIMIT: u32 = 32768;

/// The inputs of the translated pixel shader a vertex or mesh shader
/// feeds, for [`translate_linked`]: the elements the previous stage
/// provides (arbitrary, position, clip and cull) as its `PSV0` lists them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Link {
    inputs: Vec<container::SigElement>,
}

impl Link {
    pub fn pixel(ps: &HostShader) -> Result<Self, ShaderError> {
        let [inputs, ..] =
            psv_elements(&ps.dxil).ok_or(ShaderError::Malformed("a pixel shader without PSV0"))?;
        let inputs = inputs
            .into_iter()
            .filter(|e| matches!(e.kind, 0 | 3 | 6 | 7))
            .map(|e| container::SigElement {
                name: e.name,
                index: e.index,
                kind: e.kind,
                comp: match e.comp {
                    1 => 5,
                    2 => 4,
                    _ => 9,
                },
                interp: e.interp,
                row: Some(u32::from(e.row)),
                col: e.col,
                cols: e.cols,
                usage: 0,
                half: false,
                console_read: true,
            })
            .collect();
        Ok(Self { inputs })
    }

    /// The pixel input that reads `e`: the same semantic.
    fn reads(&self, e: &container::SigElement) -> Option<&container::SigElement> {
        self.inputs.iter().find(|i| {
            i.kind == e.kind
                && i.index == e.index
                && (e.kind != 0 || i.name.eq_ignore_ascii_case(&e.name))
        })
    }

    /// What a shader writing `outputs` and `prims` (the console's
    /// elements) declares linked: each element the pixel shader reads in
    /// the input's place (row, columns, type and mode); a system value it
    /// does not read on a row of its own past the pixel shader's; other
    /// elements left out; and the inputs nothing writes, declared unwritten
    /// so the stages link.
    fn declare(
        &self,
        outputs: &[container::SigElement],
        prims: &[container::SigElement],
    ) -> lower::Linked {
        let mut free = self
            .inputs
            .iter()
            .filter_map(|i| i.row)
            .map(|r| r + 1)
            .max()
            .unwrap_or(0);
        let mut written = Vec::new();
        let mut place = |elements: &[container::SigElement],
                         declared: &mut Vec<container::SigElement>| {
            elements
                .iter()
                .map(|e| {
                    let at = match self.reads(e) {
                        Some(i) => {
                            written.push(i.clone());
                            // The name as the shader writes it (`PSV0`
                            // leaves system values unnamed).
                            container::SigElement {
                                name: e.name.clone(),
                                usage: 0,
                                half: e.half,
                                ..i.clone()
                            }
                        }
                        None if e.kind != 0 && e.row.is_some() => {
                            free += 1;
                            container::SigElement {
                                row: Some(free - 1),
                                col: 0,
                                usage: 0,
                                ..e.clone()
                            }
                        }
                        None => return None,
                    };
                    declared.push(at);
                    Some(declared.len() - 1)
                })
                .collect()
        };
        let mut linked = lower::Linked::default();
        linked.out_ids = place(outputs, &mut linked.outputs);
        linked.prim_ids = place(prims, &mut linked.prims);
        for i in &self.inputs {
            if i.kind == 0 && !written.contains(i) {
                linked.outputs.push(i.clone());
            }
        }
        linked
    }
}

/// How a pixel shader's inputs fail to link to a vertex or mesh shader's
/// outputs (both translated), from their `PSV0` parts: per arbitrary,
/// position, clip or cull input, an output (or per-primitive output) of
/// the same semantic must sit at the same row and first column, cover the
/// input's columns and use the same interpolation mode. Empty when they
/// link.
pub fn link_mismatches(first: &HostShader, ps: &HostShader) -> Vec<String> {
    let (Some([_, outputs, prims]), Some([inputs, ..])) =
        (psv_elements(&first.dxil), psv_elements(&ps.dxil))
    else {
        return vec!["no PSV0".to_owned()];
    };
    let mut out = Vec::new();
    for i in inputs.iter().filter(|i| matches!(i.kind, 0 | 3 | 6 | 7)) {
        let what = format!("{}{}", i.name, i.index);
        let same = |o: &&PsvElement| {
            o.kind == i.kind && o.index == i.index && o.name.eq_ignore_ascii_case(&i.name)
        };
        let Some(o) = outputs.iter().chain(&prims).find(same) else {
            out.push(format!("{what} not written"));
            continue;
        };
        if o.row != i.row || o.col != i.col || o.cols < i.cols {
            out.push(format!(
                "{what} at {}.{}+{} read at {}.{}+{}",
                o.row, o.col, o.cols, i.row, i.col, i.cols
            ));
        }
        if o.interp != i.interp {
            out.push(format!("{what} mode {} read as {}", o.interp, i.interp));
        }
    }
    out
}

/// A signature element as `PSV0` lists it.
struct PsvElement {
    name: String,
    index: u32,
    row: u8,
    col: u8,
    cols: u8,
    kind: u8,
    comp: u8,
    interp: u8,
}

/// The inputs, outputs and per-primitive outputs a translated shader's
/// `PSV0` lists (the layout [`container::psv`] writes).
fn psv_elements(dxil: &[u8]) -> Option<[Vec<PsvElement>; 3]> {
    let c = Container::parse(dxil).ok()?;
    let d = c.part(FourCC::PSV0)?.data;
    let word = |at: usize| crate::dxbc::u32_at(d, at).ok().map(|v| v as usize);
    let info = word(0)?;
    let counts = [*d.get(4 + 28)?, *d.get(4 + 29)?, *d.get(4 + 30)?];
    let mut at = 4 + info;
    let bindings = word(at)?;
    at += 4;
    if bindings > 0 {
        at += 4 + word(at)? * bindings;
    }
    let strings = d.get(at + 4..at + 4 + word(at)?)?;
    at += 4 + strings.len();
    let indices = word(at)?;
    let index_at = at + 4;
    at += 4 + 4 * indices;
    let size = match counts.iter().map(|&n| usize::from(n)).sum::<usize>() {
        0 => 0,
        _ => {
            at += 4;
            word(at - 4)?
        }
    };
    let mut lists: [Vec<PsvElement>; 3] = Default::default();
    let mut k = 0;
    for (list, n) in lists.iter_mut().zip(counts) {
        for _ in 0..n {
            let e = at + size * k;
            k += 1;
            let name = strings.get(word(e)?..)?.split(|&b| b == 0).next()?;
            let b = d.get(e + 8..e + 14)?;
            list.push(PsvElement {
                name: String::from_utf8_lossy(name).into_owned(),
                index: word(index_at + 4 * word(e + 4)?)? as u32,
                row: b[1],
                col: b[2] >> 4 & 3,
                cols: b[2] & 0xf,
                kind: b[3],
                comp: b[4],
                interp: b[5],
            });
        }
    }
    Some(lists)
}

/// One dword of user data and the root parameter that fills it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UserData {
    /// Dword in the user-data buffer (the console's user-data index).
    pub index: u32,
    /// Root parameter index in the title's root signature.
    pub slot: u8,
    /// `Constants` or `Table` (see the module notes for the value).
    pub root: RootType,
    /// Dword within the parameter.
    pub dword: u32,
}

/// Translates the vertex, pixel, compute or mesh program of a console
/// shader container.
pub fn translate(container: &[u8]) -> Result<HostShader, ShaderError> {
    translate_linked(container, None)
}

/// [`translate`], a vertex or mesh program declaring only the outputs
/// `link` reads.
pub fn translate_linked(container: &[u8], link: Option<&Link>) -> Result<HostShader, ShaderError> {
    let c = Container::parse(container)?;
    let dxil = c
        .part(FourCC::DXIL)
        .ok_or(ShaderError::Malformed("no DXIL part"))?;
    let native = crate::native_programs(dxil.data)?
        .into_iter()
        .find(|p| {
            matches!(
                p.stage,
                HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms
            )
        })
        .ok_or(ShaderError::Unsupported {
            offset: 0,
            what: "a container without a vertex, pixel, compute or mesh program",
        })?;
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
    let res = resource::track(&p, &native.layout, native.stage)?;
    let io = io::raise(&p, &native)?;
    let st = structure::structure(&p)?;
    emit(&c, &native, &p, &st, &io, &res, link)
}

/// Emits a program whose earlier stages already ran; a vertex or mesh
/// program declares only the outputs `link` reads.
pub fn emit(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
    link: Option<&Link>,
) -> Result<HostShader, ShaderError> {
    let sig = |fourcc, input| -> Result<Vec<container::SigElement>, ShaderError> {
        match c.part(fourcc) {
            Some(part) => signature(part.data, input),
            None => Ok(Vec::new()),
        }
    };
    let pixel = native.stage == HwStage::Ps;
    let mut inputs = sig(FourCC::ISG1, true)?;
    let mut outputs = sig(FourCC::OSG1, false)?;
    // Interpolation as DXC declares it by default; pixel inputs that are
    // read get the mode their barycentrics say.
    let interpolated = if pixel { &mut inputs } else { &mut outputs };
    for e in interpolated.iter_mut() {
        e.interp = match (e.kind, e.comp) {
            (3, _) => 4,
            // Constant, as DXC declares it (the AMD compiler crashed on a
            // front face without a mode).
            (13, _) => 1,
            (0 | 6 | 7, 9) => 2,
            (0, _) => 1,
            _ => 0,
        };
    }
    if pixel {
        // The console packs the front face into a row of attributes; PC
        // wants one mode per row, and the rasterizer makes the value, so it
        // gets a row of its own.
        let shared = |e: &container::SigElement, all: &[container::SigElement]| {
            all.iter()
                .any(|o| o.kind != 13 && o.row.is_some() && o.row == e.row)
        };
        let free = interpolated
            .iter()
            .filter_map(|e| e.row)
            .max()
            .map_or(0, |r| r + 1);
        if let Some(id) = interpolated
            .iter()
            .position(|e| e.kind == 13 && shared(e, interpolated))
        {
            interpolated[id].row = Some(free);
            interpolated[id].col = 0;
        }
    } else {
        // Vertex and mesh outputs: one mode per row (PC packing). The pixel
        // shader's inputs choose how values are interpolated.
        uniform_rows(interpolated);
    }
    // Conservative depth (SV_DepthLessEqual, SV_DepthGreaterEqual) needs a
    // centroid or sample position on PC, which moves SV_Position off the
    // pixel center the console reads; plain depth keeps the console's
    // values (only the early depth test is lost).
    if pixel {
        for e in outputs.iter_mut().filter(|e| matches!(e.kind, 18 | 19)) {
            e.kind = 17;
            e.name = "SV_Depth".into();
        }
    }
    // Per-primitive outputs are constant. The console never exports a
    // primitive id or cull flag a mesh shader writes (it culls with the
    // primitive export), so those rows are left out.
    let mut prims = sig(FourCC::PSG1, false)?;
    prims.retain(|e| !matches!(e.kind, 10 | 30));
    for e in &mut prims {
        e.interp = 1;
    }
    // Linked to a pixel shader: its inputs' layout.
    let linked = link.filter(|_| !pixel).map(|l| l.declare(&outputs, &prims));
    // A mesh group's id arrives multiplied by max(threads, vertices,
    // primitives) (the code divides by that); the thread count is in PSV0
    // from runtime info version 2 on.
    // shortcut: older containers fall back to the group's lane count, right
    // when the HLSL group fills its waves (Grand Theft Auto VI's do).
    let mesh_stride = match &native.io {
        crate::native::StageIo::Ms(ms) => {
            let psv = c.part(FourCC::PSV0).map(|p| p.data);
            let threads = psv
                .filter(|d| crate::dxbc::u32_at(d, 0).is_ok_and(|n| n >= 48))
                .and_then(|d| crate::dxbc::u32_at(d, 4 + 36).ok());
            threads.map_or(ms.threads, |t| {
                t.max(ms.max_vertices).max(ms.max_primitives)
            })
        }
        _ => 0,
    };
    let l = lower::lower(&lower::Parts {
        native,
        p,
        st,
        io,
        res,
        inputs,
        outputs,
        prims,
        linked,
        mesh_stride,
    })?;
    Ok(HostShader {
        stage: native.stage,
        dxil: l.dxil,
        user_data: l.user_data,
        static_samplers: l
            .samplers
            .iter()
            .enumerate()
            .map(|(n, &w)| {
                let mut s = StaticSampler::from_words(w);
                if l.cmp_samplers >> n & 1 != 0 {
                    s.filter |= 0x80;
                }
                s
            })
            .collect(),
        samplers: l.samplers,
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
    })
}

/// Elements sharing a row get one interpolation mode (constant where they
/// differ), as PC signatures pack them.
pub(crate) fn uniform_rows(elements: &mut [container::SigElement]) {
    for i in 0..elements.len() {
        let Some(row) = elements[i].row else {
            continue;
        };
        let in_row = |e: &container::SigElement| e.row == Some(row);
        let mode = elements[i].interp;
        if elements.iter().any(|e| in_row(e) && e.interp != mode) {
            for e in elements.iter_mut().filter(|e| in_row(e)) {
                e.interp = 1;
            }
        }
    }
}

/// The rows of an `ISG1`/`OSG1` part.
fn signature(part: &[u8], input: bool) -> Result<Vec<container::SigElement>, ShaderError> {
    use crate::dxbc::u32_at;
    let count = u32_at(part, 0)? as usize;
    let base = u32_at(part, 4)? as usize;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let at = base + 32 * i;
        let name_at = u32_at(part, at + 4)? as usize;
        let name = part
            .get(name_at..)
            .and_then(|s| s.split(|&b| b == 0).next())
            .ok_or(ShaderError::Malformed("signature name out of bounds"))?;
        let (mask, rw) = match part.get(at + 24..at + 26) {
            Some(&[mask, rw]) => (mask, rw),
            _ => return Err(ShaderError::Malformed("truncated")),
        };
        if u32_at(part, at + 28)? != 0 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a minimum-precision signature element",
            });
        }
        if u32_at(part, at)? != 0 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a signature element of another stream",
            });
        }
        let kind = match u32_at(part, at + 12)? {
            0 => 0,
            1 => 3,
            2 => 6,
            3 => 7,
            4 => 4,
            5 => 5,
            6 => 1,
            7 => 10,
            8 => 2,
            9 => 13,
            10 => 12,
            64 => 16,
            65 => 17,
            66 => 14,
            67 => 19,
            68 => 18,
            69 => 20,
            70 => 15,
            23 => 28, // barycentrics
            25 => 30, // cull primitive
            _ => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: "a system value in the signature",
                });
            }
        };
        // 16-bit elements are declared 32-bit; reading or writing one
        // converts (16-bit values travel packed on the console).
        let raw = u32_at(part, at + 16)?;
        let comp = match (kind, raw) {
            // DXIL declares the front face a uint (the validator insists).
            (13, _) => 5,
            (_, 1 | 4) => 5,
            (_, 2 | 5) => 4,
            (_, 3 | 6) => 9,
            (_, c) => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: match c {
                        0 => "a signature element of unknown type",
                        _ => "a 64-bit signature element",
                    },
                });
            }
        };
        let register = u32_at(part, at + 20)?;
        out.push(container::SigElement {
            name: String::from_utf8_lossy(name).into_owned(),
            index: u32_at(part, at + 8)?,
            kind,
            comp,
            interp: 0,
            row: (register != u32::MAX).then_some(register),
            col: mask.trailing_zeros().min(3) as u8,
            cols: mask.count_ones().max(1) as u8,
            usage: 0,
            half: (4..=6).contains(&raw),
            console_read: !input || rw != 0,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::bitcode::Attr;
    use super::lower::LOOP_LIMIT;
    use super::parse_loop_limit;

    #[test]
    fn the_loop_limit_switch_parses_or_falls_back() {
        assert_eq!(parse_loop_limit(None), LOOP_LIMIT);
        assert_eq!(parse_loop_limit(Some("4096")), 4096);
        assert_eq!(parse_loop_limit(Some(" 0x1000 ")), 4096);
        assert_eq!(parse_loop_limit(Some("1")), 1);
        for bad in ["", "0", "-5", "lots", "0x", "4294967296"] {
            assert_eq!(parse_loop_limit(Some(bad)), LOOP_LIMIT, "{bad:?}");
        }
    }
    use super::container::SigElement;
    use super::shader::Shader;
    use super::*;
    use crate::native::{ComputeIo, StageIo};

    /// A wave64 compute program that writes only EXEC's high dword
    /// (`s_mov_b32 exec_hi, 0`) before a vector instruction. The new mask is
    /// opaque and keyed by the unchanged low dword, so building its lane bit
    /// needed the mask's own value: lowering recursed until the stack
    /// overflowed, which takes the whole process down (found by
    /// random-input testing). Now the instruction fails like any other that
    /// cannot be expressed: its result is missing, and only a side effect
    /// that uses it would stop the translation.
    #[test]
    fn an_exec_high_write_does_not_overflow_the_stack() {
        // s_mov_b32 exec_hi, 0; v_mov_b32 v1, v0; s_endpgm
        let words = [0xBEFF_0380u32, 0x7E02_0300, 0xBF81_0000];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [64, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let res = resource::track(&p, &native.layout, native.stage).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        let container = crate::dxbc::build(&[]);
        let c = Container::parse(&container).unwrap();
        if let Err(e) = emit(&c, &native, &p, &st, &io, &res, None) {
            assert!(e.to_string().contains("its own value"), "{e}");
        }
    }

    /// A pixel shader writing a constant colour.
    fn constant_color() -> Vec<u8> {
        let mut s = Shader::new(true);
        s.outputs.push(SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
            console_read: true,
        });
        let b = s.entry;
        for (c, v) in [1.0f32, 0.5, 0.25, 1.0].into_iter().enumerate() {
            let args = [
                s.m.i32(0),
                s.m.i32(0),
                s.m.i8(c as u8),
                s.m.f32(v.to_bits()),
            ];
            s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
        }
        s.m.terminate(b, super::bitcode::Term::Ret);
        s.finish()
    }

    #[test]
    fn rows_mixing_modes_become_constant() {
        let e = |row, col, interp| SigElement {
            name: "TEXCOORD".into(),
            index: 0,
            kind: 0,
            comp: 9,
            interp,
            row: Some(row),
            col,
            cols: 1,
            usage: 0,
            half: false,
            console_read: true,
        };
        let mut sig = [e(0, 0, 2), e(0, 1, 1), e(1, 0, 2), e(1, 1, 2)];
        super::uniform_rows(&mut sig);
        let modes: Vec<u8> = sig.iter().map(|e| e.interp).collect();
        assert_eq!(modes, [1, 1, 2, 2]);
    }

    fn element(name: &str, kind: u8, row: u32, cols: u8, interp: u8) -> SigElement {
        SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp: 9,
            interp,
            row: Some(row),
            col: 0,
            cols,
            usage: (1 << cols) - 1,
            half: false,
            console_read: true,
        }
    }

    /// A translated shader with these signatures and no code.
    fn shell(
        stage: crate::HwStage,
        ins: Vec<SigElement>,
        outs: Vec<SigElement>,
    ) -> super::HostShader {
        let mut s = Shader::new(stage == crate::HwStage::Ps);
        s.inputs = ins;
        s.outputs = outs;
        if stage == crate::HwStage::Ms {
            s.prims = vec![element("PRIM", 0, 0, 1, 1)];
            s.threads = Some([128, 1, 1]);
            s.mesh = Some(super::shader::Mesh {
                max_vertices: 100,
                max_primitives: 64,
                topology: 2,
            });
        }
        let b = s.entry;
        s.m.terminate(b, super::bitcode::Term::Ret);
        super::HostShader {
            stage,
            dxil: s.finish(),
            user_data: Vec::new(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
        }
    }

    #[test]
    fn mesh_outputs_count_every_component_of_32_rounded_maxima() {
        let outs = vec![
            element("SV_Position", 3, 0, 4, 4),
            element("TEXCOORD", 0, 1, 3, 2),
        ];
        let ms = shell(crate::HwStage::Ms, Vec::new(), outs);
        assert_eq!(ms.mesh_output_bytes(), 4 * (7 * 128 + 64));
        let vs = shell(crate::HwStage::Vs, Vec::new(), Vec::new());
        assert_eq!(vs.mesh_output_bytes(), 0);
    }

    #[test]
    fn pixel_inputs_link_at_the_rows_and_modes_written() {
        let vs = shell(
            crate::HwStage::Vs,
            Vec::new(),
            vec![
                element("SV_Position", 3, 0, 4, 4),
                element("TEXCOORD", 0, 1, 4, 2),
            ],
        );
        let ps = |row, interp| {
            shell(
                crate::HwStage::Ps,
                vec![
                    element("SV_Position", 3, 0, 4, 4),
                    element("TEXCOORD", 0, row, 2, interp),
                ],
                vec![element("SV_Target", 16, 0, 4, 0)],
            )
        };
        assert!(super::link_mismatches(&vs, &ps(1, 2)).is_empty());
        assert_eq!(
            super::link_mismatches(&vs, &ps(2, 2)),
            ["TEXCOORD0 at 1.0+4 read at 2.0+2"]
        );
        assert_eq!(
            super::link_mismatches(&vs, &ps(1, 1)),
            ["TEXCOORD0 mode 2 read as 1"]
        );
    }

    #[test]
    fn links_place_outputs_where_the_pixel_shader_reads_them() {
        let mut texcoord = element("TEXCOORD", 0, 1, 2, 1);
        texcoord.col = 2;
        texcoord.comp = 5;
        let color = element("COLOR", 0, 2, 4, 2);
        // `PSV0` names no system value.
        let link = super::Link {
            inputs: vec![element("", 3, 0, 4, 4), texcoord.clone(), color.clone()],
        };
        let mut unread = element("TEXCOORD", 0, 6, 4, 2);
        unread.index = 1;
        let outputs = [
            element("TEXCOORD", 0, 5, 4, 2),
            unread,
            element("SV_Position", 3, 2, 4, 4),
            element("SV_ClipDistance", 6, 7, 1, 2),
        ];
        let linked = link.declare(&outputs, &[]);
        assert_eq!(linked.out_ids, [Some(0), None, Some(1), Some(2)]);
        let rows: Vec<_> = linked
            .outputs
            .iter()
            .map(|e| (e.row, e.col, e.cols))
            .collect();
        assert_eq!(
            rows,
            [
                (Some(1), 2, 2),
                (Some(0), 0, 4),
                (Some(3), 0, 1),
                (Some(2), 0, 4)
            ]
        );
        assert_eq!(linked.outputs[1].name, "SV_Position");
        assert_eq!((linked.outputs[0].comp, linked.outputs[0].interp), (5, 1));
        assert_eq!(linked.outputs[3], color);
    }

    #[test]
    #[cfg(windows)]
    fn stub_pipeline_is_created() {
        let Ok(host) = super::host::Host::new(false) else {
            eprintln!("no SM 6.6 device; skipped");
            return;
        };
        let ps = super::HostShader {
            stage: crate::HwStage::Ps,
            dxil: constant_color(),
            user_data: Vec::new(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
        };
        let file = |var: &str, stage| {
            std::env::var(var).ok().map(|p| super::HostShader {
                stage,
                dxil: std::fs::read(p).unwrap(),
                user_data: Vec::new(),
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
            })
        };
        let ps = file("VICTORY_REF_PS", crate::HwStage::Ps).unwrap_or(ps);
        let vs = file("VICTORY_REF_VS", crate::HwStage::Vs)
            .unwrap_or_else(|| super::host::stub_vs(&ps).unwrap());
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut vs.dxil.clone()).unwrap();
        }
        host.pipeline(&vs, &ps).unwrap();
        // The runtime checks the container hash: no hash or a wrong one is
        // refused. (Recent runtimes also take sixteen 0x01 bytes, a bypass
        // value; ours works with older ones too.)
        let mut wrong = ps.dxil[4..20].to_vec();
        wrong[0] ^= 1;
        for digest in [&[0u8; 16][..], &wrong] {
            let mut other = ps.clone();
            other.dxil[4..20].copy_from_slice(digest);
            assert!(
                host.pipeline(&vs, &other).is_err(),
                "digest {digest:02x?} accepted"
            );
        }
    }

    #[test]
    #[cfg(windows)]
    fn constant_color_validates() {
        let mut c = constant_color();
        let path = super::validator::Validator::default_path();
        let Ok(mut v) = super::validator::Validator::load(path.as_deref()) else {
            eprintln!("no dxil.dll; skipped");
            return;
        };
        let ours = c[4..20].to_vec();
        v.validate(&mut c).unwrap();
        assert_eq!(c[4..20], ours[..], "our hash differs from the validator's");
    }

    /// A pixel shader reading channel z of an input whose element starts at
    /// w. Looking the channel up subtracted the element's first channel
    /// before checking the channel lies in it, so in a build with overflow
    /// checks (tests, debug) any read below an element that starts at z or
    /// w panicked, whichever row it was in (found by random-input testing).
    #[test]
    fn reading_a_channel_below_an_input_element_does_not_underflow() {
        // v_interp_mov_f32 v3, p0, attr0.z; s_endpgm
        let words = [0xC80E_0202u32, 0xBF81_0000];
        let native = super::NativeProgram {
            stage: super::HwStage::Ps,
            wave64: false,
            vgprs: 4,
            layout: vec![],
            used: vec![],
            io: crate::native::StageIo::Ps {
                inputs: vec![crate::native::PsInput {
                    usage: crate::native::Usage::Generic,
                    row: 0,
                    mask: 0b1000,
                    attr: 0,
                    flat: false,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                }],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let w_only = SigElement {
            name: "TEXCOORD".into(),
            index: 0,
            kind: 0,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 3,
            cols: 1,
            usage: 0,
            half: false,
            console_read: true,
        };
        let isg1 = super::container::signature(&[w_only], true);
        let bytes = crate::dxbc::build(&[(super::FourCC::ISG1, &isg1)]);
        let c = super::Container::parse(&bytes).unwrap();
        let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        // Translated or refused with a reason; it must not panic.
        let _ = super::emit(&c, &native, &p, &st, &io, &res, None);
    }
}
