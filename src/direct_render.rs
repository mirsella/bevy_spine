use std::{hash::Hash, marker::PhantomData, mem::offset_of, sync::Arc};

use bevy::{
    mesh::{MeshVertexBufferLayout, MeshVertexBufferLayoutRef, VertexBufferLayout},
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        camera::ExtractedCamera,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_asset::{RenderAssets, prepare_assets},
        render_phase::{
            AddRenderCommand, DrawFunctions, PhaseItemExtraIndex, RenderCommand,
            RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases,
        },
        render_resource::{
            Buffer, BufferDescriptor, BufferUsages, IndexFormat, PipelineCache, PrimitiveTopology,
            SpecializedMeshPipelines, TextureFormat, VertexAttribute, VertexFormat, VertexStepMode,
        },
        renderer::{RenderDevice, RenderQueue},
        sync_component::SyncComponent,
        sync_world::{MainEntity, MainEntityHashMap},
        view::{ExtractedView, RenderVisibleEntities},
    },
};
use bevy_sprite_render::{
    AlphaMode2d, MATERIAL_2D_BIND_GROUP_INDEX, Material2d, Material2dKey, Material2dPipeline,
    Mesh2dPipelineKey, PreparedMaterial2d, RenderMaterial2dInstances, RenderMesh2dInstances,
    SetMaterial2dBindGroup, SetMesh2dBindGroup, SetMesh2dViewBindGroup, SrgbTransparent2d,
    ViewKeyCache, queue_material2d_meshes,
};

use crate::materials::{DARK_COLOR_ATTRIBUTE, DARK_COLOR_SHADER_POSITION};

const VERTEX_BUFFER_LABEL: &str = "spine_direct_packed_vertex_buffer";
const INDEX_BUFFER_LABEL: &str = "spine_direct_packed_index_buffer";
const VERTEX_SIZE: usize = std::mem::size_of::<SpineDirectVertex>();

/// CPU-side Spine geometry extracted to the render world for direct GPU upload.
#[derive(Component, Clone, Default)]
pub(crate) struct SpineDirectMesh {
    vertices: Vec<SpineDirectVertex>,
    indices: Vec<u16>,
}

impl SpineDirectMesh {
    pub(crate) fn write(
        &mut self,
        mesh_entity: Entity,
        vertices: &[[f32; 2]],
        indices: &[u16],
        uvs: &[[f32; 2]],
        colors: &[[f32; 4]],
        dark_colors: &[[f32; 4]],
    ) -> bool {
        let vertex_count = vertices.len();
        if uvs.len() != vertex_count {
            warn!(
                entity = ?mesh_entity,
                "Spine renderable has {vertex_count} vertices but {} uvs; skipping direct mesh update",
                uvs.len()
            );
            self.clear();
            return false;
        }
        if colors.len() != vertex_count || dark_colors.len() != vertex_count {
            warn!(
                entity = ?mesh_entity,
                "Spine renderable has {vertex_count} vertices but {} colors and {} dark colors; skipping direct mesh update",
                colors.len(),
                dark_colors.len()
            );
            self.clear();
            return false;
        }
        if indices
            .iter()
            .any(|index| usize::from(*index) >= vertex_count)
        {
            warn!(
                entity = ?mesh_entity,
                "Spine renderable has an index outside its {vertex_count} vertices; skipping direct mesh update"
            );
            self.clear();
            return false;
        }

        self.vertices.clear();
        self.vertices.reserve(vertex_count);
        for (((position, uv), color), dark_color) in
            vertices.iter().zip(uvs).zip(colors).zip(dark_colors)
        {
            self.vertices.push(SpineDirectVertex {
                position: [position[0], position[1], 0.0],
                normal: [0.0, 0.0, 0.0],
                uv: *uv,
                color: *color,
                dark_color: *dark_color,
            });
        }

        self.indices.clear();
        self.indices.extend_from_slice(indices);
        true
    }

    pub(crate) fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }
}

impl SyncComponent for SpineDirectMesh {
    type Target = Self;
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
struct SpineDirectVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    color: [f32; 4],
    dark_color: [f32; 4],
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
                array_stride: VERTEX_SIZE as u64,
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
struct SpineDirectMeshBuffers {
    vertex_buffer: Option<Buffer>,
    index_buffer: Option<Buffer>,
    vertex_capacity: u64,
    index_capacity: u64,
    meshes: MainEntityHashMap<std::ops::Range<u32>>,
    vertex_bytes: Vec<u8>,
    index_words: Vec<u32>,
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

/// Queues direct-rendered Spine meshes that use alpha-blended material `M`.
#[derive(Default)]
pub struct SpineDirectMaterial2dPlugin<M: Material2d>(PhantomData<M>);

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
                        .after(prepare_assets::<PreparedMaterial2d<M>>)
                        .after(queue_material2d_meshes::<M>),
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
    buffers.prepare(&render_device, &render_queue, meshes.iter());
}

impl SpineDirectMeshBuffers {
    fn prepare<'a>(
        &mut self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        meshes: impl IntoIterator<Item = (&'a MainEntity, &'a SpineDirectMesh)>,
    ) {
        self.meshes.clear();
        self.vertex_bytes.clear();
        self.index_words.clear();

        for (main_entity, mesh) in meshes {
            self.pack(*main_entity, mesh);
        }

        let vertex_size = self.vertex_bytes.len() as u64;
        let index_size = (self.index_words.len() * std::mem::size_of::<u32>()) as u64;
        self.ensure_capacity(render_device, vertex_size, index_size);
        self.upload(render_queue, vertex_size, index_size);
    }

    fn pack(&mut self, main_entity: MainEntity, mesh: &SpineDirectMesh) {
        if mesh.vertices.is_empty() || mesh.indices.is_empty() {
            return;
        }

        let Some(vertex_base) = u32::try_from(self.vertex_bytes.len() / VERTEX_SIZE).ok() else {
            error!(entity = ?main_entity, "too many packed Spine vertices; skipping mesh");
            return;
        };
        if vertex_base > u32::MAX - u32::from(u16::MAX) {
            error!(entity = ?main_entity, "packed Spine vertex base is too high for rewritten u32 indices; skipping mesh");
            return;
        }
        let Some(index_start) = u32::try_from(self.index_words.len()).ok() else {
            error!(entity = ?main_entity, "too many packed Spine indices; skipping mesh");
            return;
        };
        let Some(index_count) = u32::try_from(mesh.indices.len()).ok() else {
            error!(entity = ?main_entity, "Spine mesh has too many indices; skipping mesh");
            return;
        };
        let Some(index_end) = index_start.checked_add(index_count) else {
            error!(entity = ?main_entity, "too many packed Spine indices; skipping mesh");
            return;
        };

        self.vertex_bytes
            .extend_from_slice(bytemuck::cast_slice(mesh.vertices.as_slice()));
        self.index_words.extend(
            mesh.indices
                .iter()
                .map(|index| vertex_base + u32::from(*index)),
        );
        self.meshes.insert(main_entity, index_start..index_end);
    }

    fn ensure_capacity(&mut self, render_device: &RenderDevice, vertex_size: u64, index_size: u64) {
        if vertex_size > self.vertex_capacity {
            self.vertex_buffer = Some(render_device.create_buffer(&BufferDescriptor {
                label: Some(VERTEX_BUFFER_LABEL),
                size: vertex_size,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.vertex_capacity = vertex_size;
        }
        if index_size > self.index_capacity {
            self.index_buffer = Some(render_device.create_buffer(&BufferDescriptor {
                label: Some(INDEX_BUFFER_LABEL),
                size: index_size,
                usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.index_capacity = index_size;
        }
    }

    fn upload(&self, render_queue: &RenderQueue, vertex_size: u64, index_size: u64) {
        if vertex_size > 0 {
            let Some(buffer) = &self.vertex_buffer else {
                error!("missing Spine direct vertex buffer after capacity check; skipping upload");
                return;
            };
            render_queue.write_buffer(buffer, 0, &self.vertex_bytes);
        }

        if index_size > 0 {
            let Some(buffer) = &self.index_buffer else {
                error!("missing Spine direct index buffer after capacity check; skipping upload");
                return;
            };
            render_queue.write_buffer(buffer, 0, bytemuck::cast_slice(self.index_words.as_slice()));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn queue_spine_direct_material2d_meshes<M: Material2d>(
    material2d_pipeline: Res<Material2dPipeline<M>>,
    mut pipelines: ResMut<SpecializedMeshPipelines<Material2dPipeline<M>>>,
    pipeline_cache: Res<PipelineCache>,
    render_materials: Res<RenderAssets<PreparedMaterial2d<M>>>,
    mut render_mesh_instances: ResMut<RenderMesh2dInstances>,
    render_material_instances: Res<RenderMaterial2dInstances<M>>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<SrgbTransparent2d>>,
    views: Query<(
        &MainEntity,
        &ExtractedView,
        &ExtractedCamera,
        &RenderVisibleEntities,
    )>,
    view_key_cache: Res<ViewKeyCache>,
    spine_meshes: Query<&SpineDirectMesh>,
    layout: Res<SpineDirectMeshLayout>,
    draw_functions: Res<DrawFunctions<SrgbTransparent2d>>,
) where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    let draw_function = draw_functions.read().id::<DrawSpineDirectMaterial2d<M>>();

    for (view_entity, view, camera, view_visible_entities) in &views {
        let Some(view_key) = view_key_cache.get(view_entity) else {
            continue;
        };
        let Some(transparent_phase) = transparent_render_phases.get_mut(&view.retained_view_entity)
        else {
            continue;
        };

        let Some(visible_entities) = view_visible_entities.get::<Mesh2d>() else {
            continue;
        };

        for (_, main_entity) in &visible_entities.removed_entities {
            transparent_phase.remove(Entity::PLACEHOLDER, *main_entity);
        }

        for (render_entity, visible_entity) in visible_entities.iter_visible() {
            let Ok(spine_mesh) = spine_meshes.get(*render_entity) else {
                continue;
            };
            let Some(material_asset_id) = render_material_instances.get(visible_entity) else {
                // Another registered direct material queue owns this mesh.
                continue;
            };

            // This material type owns the entity, so replacing its phase key is safe.
            transparent_phase.remove(Entity::PLACEHOLDER, *visible_entity);

            if spine_mesh.vertices.is_empty() || spine_mesh.indices.is_empty() {
                continue;
            }
            let Some(mesh_instance) = render_mesh_instances.get_mut(visible_entity) else {
                warn!(
                    entity = ?visible_entity,
                    "visible Spine direct mesh has a material but no extracted Mesh2d instance; skipping"
                );
                continue;
            };
            let Some(material_2d) = render_materials.get(*material_asset_id) else {
                continue;
            };
            if material_2d.properties.alpha_mode != AlphaMode2d::Blend {
                warn!(
                    entity = ?visible_entity,
                    material = std::any::type_name::<M>(),
                    "Spine direct 2D rendering requires an alpha-blended Material2d; skipping mesh"
                );
                continue;
            }
            mesh_instance.material_bind_group_id = material_2d.get_bind_group_id();
            let mut mesh_key = *view_key
                | Mesh2dPipelineKey::from_primitive_topology_and_strip_index(
                    PrimitiveTopology::TriangleList,
                    None,
                )
                | material_2d.properties.mesh_pipeline_key_bits;
            mesh_key.remove(
                Mesh2dPipelineKey::COLOR_TARGET_FORMAT_RESERVED_BITS
                    | Mesh2dPipelineKey::OKLAB_COMPOSITING
                    | Mesh2dPipelineKey::TONEMAP_IN_SHADER
                    | Mesh2dPipelineKey::DEBAND_DITHER
                    | Mesh2dPipelineKey::TONEMAP_METHOD_RESERVED_BITS,
            );
            let target_format = if camera.hdr {
                TextureFormat::Rgba16Float
            } else {
                TextureFormat::Rgba8Unorm
            };
            mesh_key.insert(
                Mesh2dPipelineKey::from_target_format(target_format)
                    | Mesh2dPipelineKey::SRGB_COMPOSITING,
            );
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

            transparent_phase.add_retained(SrgbTransparent2d {
                entity: (Entity::PLACEHOLDER, *visible_entity),
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
        let Some(mesh) = buffers.meshes.get(&item.main_entity()) else {
            return RenderCommandResult::Skip;
        };
        let (Some(vertex_buffer), Some(index_buffer)) =
            (&buffers.vertex_buffer, &buffers.index_buffer)
        else {
            return RenderCommandResult::Skip;
        };

        pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        pass.set_index_buffer(index_buffer.slice(..), IndexFormat::Uint32);
        pass.draw_indexed(mesh.clone(), 0, item.batch_range().clone());

        RenderCommandResult::Success
    }
}
