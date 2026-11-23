//! Console programs written here instruction by instruction, translated,
//! validated and run on the host device.

use windows::Win32::Graphics::Direct3D12::*;

use super::bitcode::{Attr, Term};
use super::container::{self, SigElement};
use super::host::{Draw, Host};
use super::shader::Shader;
use super::validator::Validator;
use super::{HostShader, emit};
use crate::dxbc::Container;
use crate::native::{HwStage, NativeProgram, PsInput, StageIo, Usage};
use crate::{ShaderError, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

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
    }
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
