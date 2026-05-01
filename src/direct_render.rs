use std::{hash::Hash, marker::PhantomData, mem::offset_of, sync::Arc};

use bevy::{
    mesh::{MeshVertexBufferLayout, MeshVertexBufferLayoutRef, VertexBufferLayout},
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_asset::{RenderAssets, prepare_assets},
        render_phase::{
            AddRenderCommand, DrawFunctions, PhaseItemExtraIndex, RenderCommand,
            RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases,
        },
        render_resource::{
            Buffer, BufferDescriptor, BufferUsages, IndexFormat, PipelineCache, PrimitiveTopology,
            VertexAttribute, VertexFormat, VertexStepMode,
        },
        renderer::{RenderDevice, RenderQueue},
        sync_world::{MainEntity, MainEntityHashMap, MainEntityHashSet},
        view::{ExtractedView, RenderVisibleEntities},
    },
    sprite_render::{
        MATERIAL_2D_BIND_GROUP_INDEX, Material2d, Material2dKey, Material2dPipeline,
        Mesh2dPipelineKey, PreparedMaterial2d, RenderMaterial2dInstances, RenderMesh2dInstances,
        SetMaterial2dBindGroup, SetMesh2dBindGroup, SetMesh2dViewBindGroup, SrgbTransparent2d,
        ViewKeyCache,
    },
};

use crate::materials::{DARK_COLOR_ATTRIBUTE, DARK_COLOR_SHADER_POSITION};

/// CPU-side Spine geometry extracted to the render world for direct GPU upload.
#[derive(Component, Clone, Default)]
pub(crate) struct SpineDirectMesh {
    pub(crate) vertices: Vec<SpineDirectVertex>,
    pub(crate) indices: Vec<u16>,
}

impl ExtractComponent for SpineDirectMesh {
    type QueryData = &'static Self;
    type QueryFilter = ();
    type Out = Self;

    fn extract_component(
        item: bevy::ecs::query::QueryItem<'_, '_, Self::QueryData>,
    ) -> Option<Self::Out> {
        Some(item.clone())
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SpineDirectVertex {
    pub(crate) position: [f32; 3],
    pub(crate) normal: [f32; 3],
    pub(crate) uv: [f32; 2],
    pub(crate) color: [f32; 4],
    pub(crate) dark_color: [f32; 4],
}

#[derive(Resource)]
struct SpineDirectMeshLayout(MeshVertexBufferLayoutRef);

impl Default for SpineDirectMeshLayout {
    fn default() -> Self {
        let layout = MeshVertexBufferLayout::new(
            vec![
                Mesh::ATTRIBUTE_POSITION.id,
                Mesh::ATTRIBUTE_NORMAL.id,
                Mesh::ATTRIBUTE_UV_0.id,
                Mesh::ATTRIBUTE_COLOR.id,
                DARK_COLOR_ATTRIBUTE.id,
            ],
            VertexBufferLayout {
                array_stride: std::mem::size_of::<SpineDirectVertex>() as u64,
                step_mode: VertexStepMode::Vertex,
                attributes: vec![
                    VertexAttribute {
                        format: VertexFormat::Float32x3,
                        offset: offset_of!(SpineDirectVertex, position) as u64,
                        shader_location: 0,
                    },
                    VertexAttribute {
                        format: VertexFormat::Float32x3,
                        offset: offset_of!(SpineDirectVertex, normal) as u64,
                        shader_location: 1,
                    },
                    VertexAttribute {
                        format: VertexFormat::Float32x2,
                        offset: offset_of!(SpineDirectVertex, uv) as u64,
                        shader_location: 2,
                    },
                    VertexAttribute {
                        format: VertexFormat::Float32x4,
                        offset: offset_of!(SpineDirectVertex, color) as u64,
                        shader_location: 4,
                    },
                    VertexAttribute {
                        format: VertexFormat::Float32x4,
                        offset: offset_of!(SpineDirectVertex, dark_color) as u64,
                        shader_location: DARK_COLOR_SHADER_POSITION as u32,
                    },
                ],
            },
        );
        Self(MeshVertexBufferLayoutRef(Arc::new(layout)))
    }
}

#[derive(Resource, Default)]
struct SpineDirectMeshBuffers(MainEntityHashMap<SpineDirectPreparedMesh>);

struct SpineDirectPreparedMesh {
    vertex_buffer: Buffer,
    vertex_capacity: u64,
    index_buffer: Buffer,
    index_capacity: u64,
    index_count: u32,
}

/// Core direct-render resources. Material-specific queueing is installed by
/// [`SpineDirectMaterial2dPlugin`].
pub(crate) struct SpineDirectRenderPlugin;

impl Plugin for SpineDirectRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<SpineDirectMesh>::extract_visible());

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<SpineDirectMeshLayout>()
                .init_resource::<SpineDirectMeshBuffers>()
                .add_systems(
                    Render,
                    prepare_spine_direct_mesh_buffers.in_set(RenderSystems::PrepareResources),
                );
        }
    }
}

/// Queues direct-rendered Spine meshes that use material `M`.
pub struct SpineDirectMaterial2dPlugin<M: Material2d>(PhantomData<M>);

impl<M: Material2d> Default for SpineDirectMaterial2dPlugin<M> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<M> Plugin for SpineDirectMaterial2dPlugin<M>
where
    M: Material2d,
    M::Data: PartialEq + Eq + Hash + Clone,
{
    fn build(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_command::<SrgbTransparent2d, DrawSpineDirectMaterial2d<M>>()
                .add_systems(
                    Render,
                    queue_spine_direct_material2d_meshes::<M>
                        .in_set(RenderSystems::QueueMeshes)
                        .after(prepare_assets::<PreparedMaterial2d<M>>),
                );
        }
    }
}

type DrawSpineDirectMaterial2d<M> = (
    SetItemPipeline,
    SetMesh2dViewBindGroup<0>,
    SetMesh2dBindGroup<1>,
    SetMaterial2dBindGroup<M, MATERIAL_2D_BIND_GROUP_INDEX>,
    DrawSpineDirectMesh,
);

fn prepare_spine_direct_mesh_buffers(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut buffers: ResMut<SpineDirectMeshBuffers>,
    meshes: Query<(&MainEntity, &SpineDirectMesh)>,
) {
    let mut seen = MainEntityHashSet::default();

    for (main_entity, mesh) in &meshes {
        seen.insert(*main_entity);
        if mesh.vertices.is_empty() || mesh.indices.is_empty() {
            buffers.0.remove(main_entity);
            continue;
        }

        let vertex_bytes = bytemuck::cast_slice(mesh.vertices.as_slice());
        let index_bytes = bytemuck::cast_slice(mesh.indices.as_slice());
        let vertex_size = vertex_bytes.len() as u64;
        let index_size = aligned_copy_size(index_bytes.len()) as u64;

        let prepared = buffers.0.entry(*main_entity).or_insert_with(|| {
            SpineDirectPreparedMesh::new(&render_device, vertex_size, index_size)
        });
        prepared.ensure_capacity(&render_device, vertex_size, index_size);
        render_queue.write_buffer(&prepared.vertex_buffer, 0, vertex_bytes);
        write_aligned_index_buffer(&render_queue, &prepared.index_buffer, index_bytes);
        prepared.index_count = mesh.indices.len() as u32;
    }

    buffers
        .0
        .retain(|main_entity, _| seen.contains(main_entity));
}

fn aligned_copy_size(size: usize) -> usize {
    (size + 3) & !3
}

fn write_aligned_index_buffer(render_queue: &RenderQueue, buffer: &Buffer, bytes: &[u8]) {
    let aligned_len = bytes.len() & !3;
    if aligned_len > 0 {
        render_queue.write_buffer(buffer, 0, &bytes[..aligned_len]);
    }
    if aligned_len != bytes.len() {
        let mut tail = [0; 4];
        tail[..bytes.len() - aligned_len].copy_from_slice(&bytes[aligned_len..]);
        render_queue.write_buffer(buffer, aligned_len as u64, &tail);
    }
}

impl SpineDirectPreparedMesh {
    fn new(render_device: &RenderDevice, vertex_capacity: u64, index_capacity: u64) -> Self {
        Self {
            vertex_buffer: create_buffer(
                render_device,
                "spine_direct_vertex_buffer",
                vertex_capacity,
                BufferUsages::VERTEX,
            ),
            vertex_capacity,
            index_buffer: create_buffer(
                render_device,
                "spine_direct_index_buffer",
                index_capacity,
                BufferUsages::INDEX,
            ),
            index_capacity,
            index_count: 0,
        }
    }

    fn ensure_capacity(&mut self, render_device: &RenderDevice, vertex_size: u64, index_size: u64) {
        if vertex_size > self.vertex_capacity {
            self.vertex_buffer = create_buffer(
                render_device,
                "spine_direct_vertex_buffer",
                vertex_size,
                BufferUsages::VERTEX,
            );
            self.vertex_capacity = vertex_size;
        }
        if index_size > self.index_capacity {
            self.index_buffer = create_buffer(
                render_device,
                "spine_direct_index_buffer",
                index_size,
                BufferUsages::INDEX,
            );
            self.index_capacity = index_size;
        }
    }
}

fn create_buffer(
    render_device: &RenderDevice,
    label: &'static str,
    size: u64,
    usage: BufferUsages,
) -> Buffer {
    render_device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: size.max(1),
        usage: usage | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

#[allow(clippy::too_many_arguments)]
fn queue_spine_direct_material2d_meshes<M: Material2d>(
    material2d_pipeline: Res<Material2dPipeline<M>>,
    mut pipelines: ResMut<
        bevy::render::render_resource::SpecializedMeshPipelines<Material2dPipeline<M>>,
    >,
    pipeline_cache: Res<PipelineCache>,
    render_materials: Res<RenderAssets<PreparedMaterial2d<M>>>,
    mut render_mesh_instances: ResMut<RenderMesh2dInstances>,
    render_material_instances: Res<RenderMaterial2dInstances<M>>,
    mut srgb_render_phases: ResMut<ViewSortedRenderPhases<SrgbTransparent2d>>,
    views: Query<(&MainEntity, &ExtractedView, &RenderVisibleEntities)>,
    view_key_cache: Res<ViewKeyCache>,
    spine_meshes: Query<&SpineDirectMesh>,
    layout: Res<SpineDirectMeshLayout>,
    draw_functions: Res<DrawFunctions<SrgbTransparent2d>>,
) where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    if render_material_instances.is_empty() {
        return;
    }

    let draw_function = draw_functions.read().id::<DrawSpineDirectMaterial2d<M>>();

    for (view_entity, view, visible_entities) in &views {
        let Some(view_key) = view_key_cache.get(view_entity) else {
            continue;
        };
        let Some(srgb_phase) = srgb_render_phases.get_mut(&view.retained_view_entity) else {
            continue;
        };

        for (render_entity, visible_entity) in visible_entities.iter::<Mesh2d>() {
            let Ok(spine_mesh) = spine_meshes.get(*render_entity) else {
                continue;
            };
            if spine_mesh.vertices.is_empty() || spine_mesh.indices.is_empty() {
                continue;
            }
            let Some(material_asset_id) = render_material_instances.get(visible_entity) else {
                continue;
            };
            let Some(mesh_instance) = render_mesh_instances.get_mut(visible_entity) else {
                continue;
            };
            let Some(material_2d) = render_materials.get(*material_asset_id) else {
                continue;
            };

            mesh_instance.material_bind_group_id = material_2d.get_bind_group_id();
            let mesh_key = *view_key
                | Mesh2dPipelineKey::from_primitive_topology(PrimitiveTopology::TriangleList)
                | material_2d.properties.mesh_pipeline_key_bits;
            let pipeline = match pipelines.specialize(
                &pipeline_cache,
                &material2d_pipeline,
                Material2dKey {
                    mesh_key,
                    bind_group_data: material_2d.key.clone(),
                },
                &layout.0,
            ) {
                Ok(pipeline) => pipeline,
                Err(err) => {
                    error!("failed to specialize Spine direct 2D pipeline: {err}");
                    continue;
                }
            };

            srgb_phase.add(SrgbTransparent2d {
                entity: (*render_entity, *visible_entity),
                draw_function,
                pipeline,
                sort_key: bevy::math::FloatOrd(
                    mesh_instance.transforms.world_from_local.translation.z
                        + material_2d.properties.depth_bias,
                ),
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::None,
                extracted_index: usize::MAX,
                indexed: true,
            });
        }
    }
}

struct DrawSpineDirectMesh;

impl<P: bevy::render::render_phase::PhaseItem> RenderCommand<P> for DrawSpineDirectMesh {
    type Param = bevy::ecs::system::lifetimeless::SRes<SpineDirectMeshBuffers>;
    type ViewQuery = ();
    type ItemQuery = ();

    fn render<'w>(
        item: &P,
        _view: (),
        _entity: Option<()>,
        buffers: bevy::ecs::system::SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let buffers = buffers.into_inner();
        let Some(mesh) = buffers.0.get(&item.main_entity()) else {
            return RenderCommandResult::Skip;
        };

        pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        pass.set_index_buffer(mesh.index_buffer.slice(..), 0, IndexFormat::Uint16);
        pass.draw_indexed(0..mesh.index_count, 0, item.batch_range().clone());

        RenderCommandResult::Success
    }
}
