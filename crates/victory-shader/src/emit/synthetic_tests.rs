//! Console programs written here instruction by instruction, translated,
//! validated and run on the host device.

use windows::Win32::Graphics::Direct3D12::*;

use super::bitcode::{Attr, Term};
use super::container::{self, SigElement};
use super::host::{Dispatch, Draw, Host, Root, View};
use super::shader::Shader;
use super::validator::Validator;
use super::{HostShader, emit};
use crate::dxbc::Container;
use crate::native::{
    ComputeIo, HwStage, NativeProgram, PsInput, ResourceType, RootParam, RootType, StageIo, Usage,
    UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
/// The descriptor at the start of the table whose address is user-data
/// dword 0, in s[4:11].
const LOAD_UAV: [u32; 5] = [
    0xBE82_0300, // s_mov_b32 s2, s0
    0xBE83_0380, // s_mov_b32 s3, 0
    0xF40C_0101,
    0xFA00_0000, // s_load_dwordx8 s[4:11], s[2:3], null
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
];

/// One UAV table in user-data dword 0.
fn uav_table() -> (Vec<RootParam>, Vec<UsedResource>) {
    let layout = vec![RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data: 0,
        slot: 0,
    }];
    let used = vec![UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 1,
        dwords: 8,
        user_data: 0,
        slot: 0,
    }];
    (layout, used)
}

fn element(name: &str, kind: u8, row: u32) -> SigElement {
    SigElement {
        name: name.into(),
        index: 0,
        kind,
        comp: 9,
        interp: 0,
        row: Some(row),
        col: 0,
        cols: 4,
        usage: 0,
        half: false,
        console_read: true,
    }
}

/// Translates `words` as the program `native` describes, in a container
/// with these signatures.
fn translate(
    words: &[u32],
    mut native: NativeProgram,
    inputs: &[SigElement],
    outputs: &[SigElement],
) -> Result<HostShader, ShaderError> {
    native.code = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let res = resource::track(&p, &native.layout, native.stage)?;
    let io = io::raise(&p, &native)?;
    let st = structure::structure(&p)?;
    let isg1 = container::signature(inputs, true);
    let osg1 = container::signature(outputs, false);
    let bytes = container::build(&[(b"ISG1", &isg1[..]), (b"OSG1", &osg1[..])]);
    emit(
        &Container::parse(&bytes)?,
        &native,
        &p,
        &st,
        &io,
        &res,
        None,
    )
}

fn pixel(input_addr: u32, inputs: Vec<PsInput>) -> NativeProgram {
    NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Ps {
            inputs,
            input_addr,
            db_shader_control: 0,
            patches: vec![],
        },
        code: vec![],
    }
}

fn validate(s: &HostShader) {
    match Validator::load(Validator::default_path().as_deref()) {
        Ok(mut v) => v.validate(&mut s.dxil.clone()).unwrap(),
        Err(_) => eprintln!("no dxil.dll; not validated"),
    }
}

/// A vertex shader passing `POSITION` (row 0) through as `SV_Position`.
fn position_vs() -> HostShader {
    let mut s = Shader::new(false);
    let mut input = element("POSITION", 0, 0);
    input.usage = 0xF;
    s.inputs.push(input);
    let mut output = element("SV_Position", 3, 0);
    output.interp = 4;
    output.usage = 0xF;
    s.outputs.push(output);
    let b = s.entry;
    for col in 0..4 {
        let u = s.m.undef(s.t.i32);
        let args = [s.m.i32(0), s.m.i32(0), s.m.i8(col), u];
        let x = s.op(b, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, &args);
        let args = [s.m.i32(0), s.m.i32(0), s.m.i8(col), x];
        s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
    }
    s.m.terminate(b, Term::Ret);
    HostShader {
        stage: HwStage::Vs,
        dxil: s.finish(),
        user_data: Vec::new(),
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
        atomic64_views: Vec::new(),
        views: Vec::new(),
    }
}

fn typed_atomics_64(host: &Host) -> bool {
    let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS9::default();
    let mut h = D3D12_FEATURE_DATA_D3D12_OPTIONS11::default();
    // SAFETY: feature queries with matching structures.
    unsafe {
        host.device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS9,
                (&raw mut o).cast(),
                size_of_val(&o) as u32,
            )
            .is_ok()
            && host
                .device
                .CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS11,
                    (&raw mut h).cast(),
                    size_of_val(&h) as u32,
                )
                .is_ok()
            && o.AtomicInt64OnTypedResourceSupported.as_bool()
            && h.AtomicInt64OnDescriptorHeapResourceSupported.as_bool()
    }
}

#[test]
fn image_atomics_of_64_bits_run_on_the_host() {
    // 8 threads (x in v0) on an 8x4 R32G32 texture:
    // row 0: umax of (64 - x, x) at (x & 3, 0): the high dword decides;
    // row 1: add of 0xFFFFFFFF twice per texel at (x & 3, 1): a carry;
    // row 2: cmpswap 0 -> (x + 1, 7), then (x + 1, 7) -> (5, 6) with glc;
    // row 3: or of what that second cmpswap returned, at (x, 3).
    let mut words = LOAD_UAV.to_vec();
    words.extend([
        0x3602_0083, // v_and_b32 v1, 3, v0
        0x7E04_0280, // v_mov_b32 v2, 0
        0x4C06_00C0, // v_sub_nc_u32 v3, 64, v0
        0x7E08_0300, // v_mov_b32 v4, v0
        0xF05C_1308,
        0x0001_0301, // image_atomic_umax v[3:4], v[1:2], s[4:11] dmask:0x3 dim:2D unorm
        0x7E04_0281, // v_mov_b32 v2, 1
        0x7E06_02C1, // v_mov_b32 v3, -1
        0x7E08_0280, // v_mov_b32 v4, 0
        0xF044_1308,
        0x0001_0301, // image_atomic_add v[3:4], v[1:2], s[4:11] dmask:0x3 dim:2D unorm
        0x7E0E_0300, // v_mov_b32 v7, v0
        0x7E10_0282, // v_mov_b32 v8, 2
        0x4A06_0081, // v_add_nc_u32 v3, 1, v0
        0x7E08_0287, // v_mov_b32 v4, 7
        0x7E0A_0280, // v_mov_b32 v5, 0
        0x7E0C_0280, // v_mov_b32 v6, 0
        0xF040_1F08,
        0x0001_0307, // image_atomic_cmpswap v[3:6], v[7:8], s[4:11] dmask:0xf dim:2D unorm
        0x7E12_0285, // v_mov_b32 v9, 5
        0x7E14_0286, // v_mov_b32 v10, 6
        0x4A16_0081, // v_add_nc_u32 v11, 1, v0
        0x7E18_0287, // v_mov_b32 v12, 7
        0xF040_3F08,
        0x0001_0907, // image_atomic_cmpswap v[9:12], v[7:8], s[4:11] dmask:0xf dim:2D unorm glc
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x7E10_0283, // v_mov_b32 v8, 3
        0xF064_1308,
        0x0001_0907, // image_atomic_or v[9:10], v[7:8], s[4:11] dmask:0x3 dim:2D unorm
        S_ENDPGM,
    ]);
    let (layout, used) = uav_table();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 256,
        layout,
        used,
        // One user SGPR, thread id x in v0.
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2,
            threads: [8, 1, 1],
        }),
        code: vec![],
    };
    let cs = translate(&words, native, &[], &[]).unwrap();
    validate(&cs);
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 device; skipped");
        return;
    };
    if !typed_atomics_64(&host) {
        eprintln!("no 64-bit typed atomics on heap resources; skipped");
        return;
    }
    let (root, pso) = host.compute_pipeline(&cs).unwrap();
    // The table starts at heap index 0.
    let user = vec![0; cs.user_data_dwords().max(1) as usize];
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(user))],
                views: &[View::Uint2TextureUav(8, 4)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap();
    let texel = |x: u32, y: u32| {
        let at = 2 * (y * 8 + x) as usize;
        (out[0][at], out[0][at + 1])
    };
    for x in 0..8 {
        let k = x & 3;
        let (umax, add) = if x < 4 {
            ((64 - (k + 4), k + 4), (0xFFFF_FFFE, 1))
        } else {
            ((0, 0), (0, 0))
        };
        assert_eq!(texel(x, 0), umax, "umax at {x}");
        assert_eq!(texel(x, 1), add, "add at {x}");
        assert_eq!(texel(x, 2), (5, 6), "cmpswap at {x}");
        assert_eq!(texel(x, 3), (x + 1, 7), "cmpswap's old value at {x}");
    }
}

#[test]
fn pixel_image_atomics_of_64_bits_make_a_pipeline() {
    // Persp center i/j (v0, v1) and the position (v2, v3, ADDR 0x302): a
    // visibility-buffer write, umax of (i, j) at the pixel, color (i, j).
    let mut words = LOAD_UAV.to_vec();
    words.extend([
        0x7E08_0F02, // v_cvt_u32_f32 v4, v2
        0x7E0A_0F03, // v_cvt_u32_f32 v5, v3
        0x7E0C_0300, // v_mov_b32 v6, v0
        0x7E0E_0301, // v_mov_b32 v7, v1
        0xF05C_1308,
        0x0001_0604, // image_atomic_umax v[6:7], v[4:5], s[4:11] dmask:0x3 dim:2D unorm
        0xF800_180F,
        0x0100_0100, // exp mrt0 v0, v1, v0, v1 done vm
        S_ENDPGM,
    ]);
    let target = element("SV_Target", 16, 0);
    let position = element("SV_Position", 3, 0);
    let mut native = pixel(0x302, vec![]);
    (native.layout, native.used) = uav_table();
    let ps = translate(&words, native, &[position], &[target]).unwrap();
    validate(&ps);
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 device; skipped");
        return;
    };
    if !typed_atomics_64(&host) {
        eprintln!("no 64-bit typed atomics on heap resources; skipped");
        return;
    }
    host.pipeline(&super::host::stub_vs(&ps).unwrap(), &ps)
        .unwrap();
}

#[test]
fn barycentrics_read_as_data_are_the_weights_of_vertices_1_and_2() {
    // `exp mrt0 v0, v1, v0, v1`: the perspective-center i and j (ADDR 2).
    let words = [0xF800_180F, 0x0100_0100, S_ENDPGM];
    let target = element("SV_Target", 16, 0);
    let ps = translate(&words, pixel(2, vec![]), &[], &[target]).unwrap();
    validate(&ps);
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 device; skipped");
        return;
    };
    let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS3::default();
    // SAFETY: a feature query with its matching structure.
    let ok = unsafe {
        host.device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS3,
            (&raw mut o).cast(),
            size_of_val(&o) as u32,
        )
    };
    if ok.is_err() || !o.BarycentricsSupported.as_bool() {
        eprintln!("no barycentrics; skipped");
        return;
    }
    let vs = position_vs();
    let (root, pso) = host.pipeline(&vs, &ps).unwrap();
    // Vertex 0 at the bottom left, 1 two widths right, 2 two heights up:
    // at pixel (px, py) of a w x h target, b1 = px / 2w, b2 = (h - py) / 2h.
    let (w, h) = (8u32, 4u32);
    #[rustfmt::skip]
    let vertices = [
        -1.0f32, -1.0, 0.5, 1.0,
        3.0, -1.0, 0.5, 1.0,
        -1.0, 3.0, 0.5, 1.0,
    ];
    let out = host
        .draw(
            &root,
            &pso,
            &Draw {
                roots: vec![],
                textures: &[],
                samplers: &[],
                vertices: &vertices,
                stride: 16,
                size: (w, h),
                mesh_groups: 0,
            },
        )
        .unwrap();
    for y in 0..h {
        for x in 0..w {
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            let b1 = px / (2 * w) as f32;
            let b2 = (h as f32 - py) / (2 * h) as f32;
            let c = out[(y * w + x) as usize];
            let near = |a: f32, b: f32| (a - b).abs() < 1e-4;
            assert!(
                near(c[0], b1) && near(c[1], b2) && near(c[2], b1) && near(c[3], b2),
                "pixel ({x}, {y}): {c:?}, want ({b1}, {b2})"
            );
        }
    }
}

#[test]
fn interpolation_alone_declares_no_barycentrics() {
    // TEXCOORD.x from the perspective-center i/j, exported.
    let words = [
        0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
        0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
        0xF800_180F,
        0x0202_0202, // exp mrt0 v2, v2, v2, v2 done vm
        S_ENDPGM,
    ];
    let texcoord = PsInput {
        usage: Usage::Generic,
        row: 0,
        mask: 1,
        attr: 0,
        flat: false,
        fp16: false,
        pack2: false,
        row2: 0,
    };
    let mut input = element("TEXCOORD", 0, 0);
    input.cols = 1;
    let target = element("SV_Target", 16, 0);
    let ps = translate(&words, pixel(2, vec![texcoord]), &[input], &[target]).unwrap();
    validate(&ps);
    let found = ps.dxil.windows(15).any(|w| w == b"SV_Barycentrics");
    assert!(!found, "an SV_Barycentrics input");
}
