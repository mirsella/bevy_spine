//! Materials for Spine meshes.
//!
//! To create a custom material for Spine, see [`SpineMaterial`].

use std::marker::PhantomData;

use bevy::asset::uuid_handle;
use bevy::mesh::{MeshVertexAttribute, MeshVertexBufferLayoutRef};
use bevy::sprite_render::{AlphaMode2d, Material2d, Material2dKey, MeshMaterial2d};
use bevy::{
    asset::Asset,
    ecs::system::{StaticSystemParam, SystemParam},
    prelude::*,
    reflect::TypePath,
    render::render_resource::{
        AsBindGroup, BlendComponent, BlendFactor, BlendOperation, BlendState,
        RenderPipelineDescriptor, SpecializedMeshPipelineError, VertexFormat,
    },
    shader::ShaderRef,
};
use rusty_spine::BlendMode;

use crate::{SpineMesh, SpineMeshState, SpineSettings, SpineSystem};

/// Trait for automatically applying materials to [`SpineMesh`] entities. Used by the built-in
/// materials but can also be used to create custom materials.
///
/// Implement the trait and add it with [`SpineMaterialPlugin`].
pub trait SpineMaterial: Material2d {
    /// System parameters to query when updating this material.
    type Params<'w, 's>: SystemParam;

    /// Runs every frame for every material and every [`SpineMesh`].
    ///
    /// `material` is [`None`] when the mesh has no material or its asset is missing. Return
    /// [`SpineMaterialUpdate::Keep`] when no asset change is needed. Default materials should be
    /// removed if a custom material is desired (see [`SpineSettings::default_materials`]).
    fn update(
        material: Option<&Self>,
        entity: Entity,
        renderable_data: &SpineMaterialInfo,
        params: &StaticSystemParam<Self::Params<'_, '_>>,
    ) -> SpineMaterialUpdate<Self>;
}

/// The change requested by [`SpineMaterial::update`].
pub enum SpineMaterialUpdate<T> {
    Keep,
    Set(T),
    Remove,
}

/// Add support for a new [`SpineMaterial`].
pub struct SpineMaterialPlugin<T: SpineMaterial> {
    _marker: PhantomData<T>,
}

impl<T: SpineMaterial> Default for SpineMaterialPlugin<T> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: SpineMaterial> Plugin for SpineMaterialPlugin<T> {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            update_materials::<T>.in_set(SpineSystem::UpdateMaterials),
        );
    }
}

/// Info necessary for a Spine material.
#[derive(Debug, Clone)]
pub struct SpineMaterialInfo {
    pub slot_index: Option<usize>,
    pub texture: Handle<Image>,
    pub blend_mode: BlendMode,
    pub premultiplied_alpha: bool,
}

fn update_materials<T: SpineMaterial>(
    mut commands: Commands,
    mut materials: ResMut<Assets<T>>,
    mesh_query: Query<(Entity, &SpineMesh, Option<&MeshMaterial2d<T>>)>,
    params: StaticSystemParam<T::Params<'_, '_>>,
) {
    for (mesh_entity, spine_mesh, material_handle) in mesh_query.iter() {
        let SpineMeshState::Renderable { info } = &spine_mesh.state else {
            continue;
        };
        let (material_id, material) = match material_handle {
            Some(handle) => {
                let id = handle.0.id();
                match materials.get(id) {
                    Some(material) => (Some(id), Some(material)),
                    None => {
                        error!(?mesh_entity, "Spine material asset is missing");
                        commands.entity(mesh_entity).remove::<MeshMaterial2d<T>>();
                        (None, None)
                    }
                }
            }
            None => (None, None),
        };

        match T::update(material, spine_mesh.spine_entity, info, &params) {
            SpineMaterialUpdate::Keep => {}
            SpineMaterialUpdate::Set(updated) => {
                if let Some(id) = material_id {
                    materials
                        .insert(id, updated)
                        .expect("existing Spine material ID must remain valid");
                } else {
                    let handle = materials.add(updated);
                    commands.entity(mesh_entity).insert(MeshMaterial2d(handle));
                }
            }
            SpineMaterialUpdate::Remove if material_id.is_some() => {
                commands.entity(mesh_entity).remove::<MeshMaterial2d<T>>();
            }
            SpineMaterialUpdate::Remove => {}
        }
    }
}

pub const DARK_COLOR_SHADER_POSITION: u64 = 10;
pub const DARK_COLOR_ATTRIBUTE: MeshVertexAttribute = MeshVertexAttribute::new(
    "Vertex_DarkColor",
    DARK_COLOR_SHADER_POSITION,
    VertexFormat::Float32x4,
);

pub const SHADER_HANDLE: Handle<Shader> = uuid_handle!("b5694dad-2246-5609-85e4-149838ce0219");

/// A [`SystemParam`] to query [`SpineSettings`].
///
/// Mostly used for the built-in materials but may be useful for implementing other materials.
#[derive(SystemParam)]
pub struct SpineSettingsQuery<'w, 's> {
    pub spine_settings_query: Query<'w, 's, &'static SpineSettings>,
}

macro_rules! material {
    ($(#[$($attrss:tt)*])* $name:ident, $blend_mode:expr, $premultiplied_alpha:expr, $blend_state:expr) => {
        $(#[$($attrss)*])*
        #[derive(Asset, Default, AsBindGroup, TypePath, Clone)]
        pub struct $name {
            #[texture(0)]
            #[sampler(1)]
            pub image: Handle<Image>,
        }

        impl $name {
            pub fn new(image: Handle<Image>) -> Self {
                Self { image }
            }
        }

        impl Material2d for $name {
            fn vertex_shader() -> ShaderRef {
                SHADER_HANDLE.into()
            }

            fn fragment_shader() -> ShaderRef {
                SHADER_HANDLE.into()
            }

            fn alpha_mode(&self) -> AlphaMode2d {
                AlphaMode2d::Blend
            }

            fn specialize(
                descriptor: &mut RenderPipelineDescriptor,
                layout: &MeshVertexBufferLayoutRef,
                _key: Material2dKey<Self>,
            ) -> Result<(), SpecializedMeshPipelineError> {
                let vertex_attributes = vec![
                    Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
                    Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
                    Mesh::ATTRIBUTE_UV_0.at_shader_location(2),
                    Mesh::ATTRIBUTE_COLOR.at_shader_location(4),
                    DARK_COLOR_ATTRIBUTE.at_shader_location(DARK_COLOR_SHADER_POSITION as u32),
                ];
                let vertex_buffer_layout = layout.0.get_layout(&vertex_attributes)?;
                descriptor.vertex.buffers = vec![vertex_buffer_layout];
                if let Some(fragment) = &mut descriptor.fragment {
                    if let Some(target_state) = &mut fragment.targets[0] {
                        target_state.blend = Some($blend_state);
                    }
                }
                descriptor.primitive.cull_mode = None;
                Ok(())
            }
        }

        impl SpineMaterial for $name {
            type Params<'w, 's> = SpineSettingsQuery<'w, 's>;

            fn update(
                material: Option<&Self>,
                entity: Entity,
                renderable_data: &SpineMaterialInfo,
                params: &StaticSystemParam<Self::Params<'_, '_>>,
            ) -> SpineMaterialUpdate<Self> {
                let spine_settings = params.spine_settings_query.get(entity).copied().unwrap_or(SpineSettings::default());
                if spine_settings.default_materials && renderable_data.blend_mode == $blend_mode && renderable_data.premultiplied_alpha == $premultiplied_alpha {
                    match material {
                        Some(material) if material.image == renderable_data.texture => SpineMaterialUpdate::Keep,
                        _ => SpineMaterialUpdate::Set(Self::new(renderable_data.texture.clone())),
                    }
                } else {
                    SpineMaterialUpdate::Remove
                }
            }
        }
    };
}

material!(
    /// Normal blend mode material, non-premultiplied-alpha
    SpineNormalMaterial,
    BlendMode::Normal,
    false,
    BlendState {
        color: BlendComponent {
            src_factor: BlendFactor::SrcAlpha,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        },
        alpha: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        },
    }
);

material!(
    /// Additive blend mode material, non-premultiplied-alpha
    SpineAdditiveMaterial,
    BlendMode::Additive,
    false,
    BlendState {
        color: BlendComponent {
            src_factor: BlendFactor::SrcAlpha,
            dst_factor: BlendFactor::One,
            operation: BlendOperation::Add,
        },
        alpha: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::One,
            operation: BlendOperation::Add,
        },
    }
);

material!(
    /// Multiply blend mode material, non-premultiplied-alpha
    SpineMultiplyMaterial,
    BlendMode::Multiply,
    false,
    BlendState {
        color: BlendComponent {
            src_factor: BlendFactor::Dst,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        },
        alpha: BlendComponent {
            src_factor: BlendFactor::OneMinusSrcAlpha,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        },
    }
);

material!(
    /// Screen blend mode material, non-premultiplied-alpha
    SpineScreenMaterial,
    BlendMode::Screen,
    false,
    BlendState {
        color: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        },
        alpha: BlendComponent {
            src_factor: BlendFactor::OneMinusSrc,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        },
    }
);

material!(
    /// Normal blend mode material, premultiplied-alpha
    SpineNormalPmaMaterial,
    BlendMode::Normal,
    true,
    BlendState {
        color: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        },
        alpha: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        },
    }
);

material!(
    /// Additive blend mode material, premultiplied-alpha
    SpineAdditivePmaMaterial,
    BlendMode::Additive,
    true,
    BlendState {
        color: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::One,
            operation: BlendOperation::Add,
        },
        alpha: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::One,
            operation: BlendOperation::Add,
        },
    }
);

material!(
    /// Multiply blend mode material, premultiplied-alpha
    SpineMultiplyPmaMaterial,
    BlendMode::Multiply,
    true,
    BlendState {
        color: BlendComponent {
            src_factor: BlendFactor::Dst,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        },
        alpha: BlendComponent {
            src_factor: BlendFactor::OneMinusSrcAlpha,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        },
    }
);

material!(
    /// Screen blend mode material, premultiplied-alpha
    SpineScreenPmaMaterial,
    BlendMode::Screen,
    true,
    BlendState {
        color: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        },
        alpha: BlendComponent {
            src_factor: BlendFactor::OneMinusSrc,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        },
    }
);
