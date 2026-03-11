#import bevy_sprite::{
    mesh2d_functions as mesh_functions,
    mesh2d_view_bindings::view,
}

#ifdef TONEMAP_IN_SHADER
#import bevy_core_pipeline::tonemapping
#endif

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(4) color: vec4<f32>,
    @location(10) dark_color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(4) color: vec4<f32>,
    @location(10) dark_color: vec4<f32>,
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    out.uv = vertex.uv;
    var model = mesh_functions::get_world_from_local(vertex.instance_index);
    out.world_position = mesh_functions::mesh2d_position_local_to_world(
        model,
        vec4<f32>(vertex.position, 1.0)
    );
    out.position = mesh_functions::mesh2d_position_world_to_clip(out.world_position);
    out.world_normal = mesh_functions::mesh2d_normal_local_to_world(vertex.normal, vertex.instance_index);
    out.color = vertex.color;
    out.dark_color = vertex.dark_color;
    return out;
}

@group(2) @binding(0)
var texture: texture_2d<f32>;
@group(2) @binding(1)
var texture_sampler: sampler;

@fragment
fn fragment(
    input: VertexOutput,
) -> @location(0) vec4<f32> {
    var tex_color = textureSample(texture, texture_sampler, input.uv);
    #ifdef WEB_UNPREMULTIPLY_TEXTURE
    // Browser WebGPU can behave as if straight-alpha Spine atlas texels were
    // uploaded premultiplied. Undo that before applying the normal non-PMA
    // Spine color math so translucent overlays do not wash out. The upload can
    // happen in nonlinear/sRGB space, so convert back to sRGB first, divide by
    // alpha there, then return to linear.
    if tex_color.a > 0.0 {
        let srgb = pow(max(tex_color.rgb, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
        let straight_srgb = srgb / tex_color.a;
        tex_color = vec4(pow(max(straight_srgb, vec3<f32>(0.0)), vec3<f32>(2.2)), tex_color.a);
    }
    #endif
    var color = vec4(
        ((tex_color.a - 1.0) * input.dark_color.a + 1.0 - tex_color.rgb) * input.dark_color.rgb + tex_color.rgb * input.color.rgb,
        tex_color.a * input.color.a,
    );
#ifdef TONEMAP_IN_SHADER
    color = tonemapping::tone_mapping(color, view.color_grading);
#endif
#ifdef SRGB_MESH2D_PASS
    let alpha = color.a;
    let rgb = max(color.rgb, vec3<f32>(0.0));
    let srgb = pow(rgb, vec3<f32>(1.0 / 2.2));
    color = vec4<f32>(srgb, alpha);
#endif
    return color;
}
