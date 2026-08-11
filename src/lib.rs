//! A Bevy plugin for Spine 4.2
//!
//! Add [`SpinePlugin`] to your Bevy app and spawn a [`SkeletonDataHandle`] to get started!

use std::{
    collections::{HashMap, HashSet, VecDeque},
    mem::take,
    sync::{Arc, Mutex},
};

use crate::{
    assets::{AtlasLoader, SkeletonJsonLoader},
    materials::{DARK_COLOR_ATTRIBUTE, SHADER_HANDLE, SpineMaterialPlugin},
    rusty_spine::{
        AnimationStateData, BoneHandle, controller::SkeletonControllerSettings, draw::CullDirection,
    },
    textures::{SpineAtlasStatus, SpineTexture, SpineTextures},
};
use bevy::{
    asset::{AssetPath, RenderAssetUsages, load_internal_binary_asset},
    camera::{primitives::Aabb, visibility::RenderLayers},
    ecs::hierarchy::ChildSpawnerCommands,
    mesh::{Indices, MeshVertexAttribute},
    prelude::*,
    render::batching::NoAutomaticBatching,
    render::render_resource::{PrimitiveTopology, VertexFormat},
    sprite_render::Material2dPlugin,
};
use materials::{
    SpineAdditiveMaterial, SpineAdditivePmaMaterial, SpineMaterialInfo, SpineMultiplyMaterial,
    SpineMultiplyPmaMaterial, SpineNormalMaterial, SpineNormalPmaMaterial, SpineScreenMaterial,
    SpineScreenPmaMaterial,
};
use rusty_spine::{
    AnimationEvent, Physics, Skeleton,
    controller::{SkeletonCombinedRenderable, SkeletonRenderable},
};

pub use crate::{assets::*, crossfades::Crossfades, entity_sync::*, handle::*, rusty_spine::Color};
pub use direct_render::SpineDirectMaterial2dPlugin;
use direct_render::SpineDirectMesh;
pub use textures::{SpineAssetLoadFailedEvent, SpineTexturePathResolver};

/// See [`rusty_spine`] docs for more info.
pub use crate::rusty_spine::controller::SkeletonController;

pub use rusty_spine;

/// System sets for Spine systems.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, SystemSet)]
pub enum SpineSystem {
    /// Loads [`SkeletonData`] assets which must exist before a [`SkeletonDataHandle`] can fully
    /// load. A skeleton reaches [`SkeletonDataStatus::Loaded`] only after all atlas page images
    /// are ready.
    Load,
    /// Initializes loaded [`SkeletonDataHandle`] entities and their render helpers.
    Initialize,
    /// Advances all animations and processes Spine events (see [`SpineEvent`]).
    UpdateAnimation,
    /// Updates all Spine meshes.
    UpdateMeshes,
    /// Updates all Spine materials.
    UpdateMaterials,
}

/// Helper sets for interacting with Spine systems.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, SystemSet)]
pub enum SpineSet {
    /// Occurs after initialization commands are flushed but before Spine update systems.
    ///
    /// Readers of [`SpineReadyEvent`] in this set are guaranteed to observe the applied [`Spine`]
    /// component and helper entities, and can configure a newly initialized skeleton before its
    /// first update.
    OnReady,
    /// A helper Set occuring after [`SpineSystem::UpdateAnimation`] but before
    /// [`SpineSystem::UpdateMeshes`], so that systems can handle events immediately after the
    /// skeleton updates but before it renders.
    OnEvent,
    /// A helper set occuring simultaneously with [`SpineSystem::UpdateMeshes`], useful for custom
    /// mesh creation when using [`SpineDrawer::None`].
    OnUpdateMesh,
    /// The canonical phase for consumers to insert or replace [`SkeletonDataHandle`].
    ///
    /// This phase runs after [`SpineSystem::Load`] and before [`SpineSystem::Initialize`]. Deferred
    /// commands from systems in this set are flushed automatically before initialization. If the
    /// referenced [`SkeletonData`] is already [`SkeletonDataStatus::Loaded`], the new skeleton is
    /// initialized, animated, meshed, and made available to Bevy's transform/visibility propagation
    /// and render extraction in the same app frame. Asynchronously loading data remains explicitly
    /// loading and cannot render in that frame.
    Prepare,
}

/// Add Spine support to Bevy.
///
/// ```
/// # use bevy::prelude::*;
/// # use bevy_spine::SpinePlugin;
/// # fn doc() {
/// App::new()
///     .add_plugins(DefaultPlugins)
///     .add_plugins(SpinePlugin::default())
///     // ...
///     .run();
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SpinePlugin {
    built_in_materials: bool,
}

impl SpinePlugin {
    /// Avoids registering the built-in material pipelines when an application supplies every
    /// Spine material itself.
    pub fn without_built_in_materials() -> Self {
        Self {
            built_in_materials: false,
        }
    }
}

impl Default for SpinePlugin {
    fn default() -> Self {
        Self {
            built_in_materials: true,
        }
    }
}

impl Plugin for SpinePlugin {
    fn build(&self, app: &mut App) {
        if self.built_in_materials {
            app.add_plugins((
                Material2dPlugin::<SpineNormalMaterial>::default(),
                Material2dPlugin::<SpineAdditiveMaterial>::default(),
                Material2dPlugin::<SpineMultiplyMaterial>::default(),
                Material2dPlugin::<SpineScreenMaterial>::default(),
                Material2dPlugin::<SpineNormalPmaMaterial>::default(),
                Material2dPlugin::<SpineAdditivePmaMaterial>::default(),
                Material2dPlugin::<SpineMultiplyPmaMaterial>::default(),
                Material2dPlugin::<SpineScreenPmaMaterial>::default(),
            ))
            .add_plugins((
                SpineMaterialPlugin::<SpineNormalMaterial>::default(),
                SpineMaterialPlugin::<SpineAdditiveMaterial>::default(),
                SpineMaterialPlugin::<SpineMultiplyMaterial>::default(),
                SpineMaterialPlugin::<SpineScreenMaterial>::default(),
                SpineMaterialPlugin::<SpineNormalPmaMaterial>::default(),
                SpineMaterialPlugin::<SpineAdditivePmaMaterial>::default(),
                SpineMaterialPlugin::<SpineMultiplyPmaMaterial>::default(),
                SpineMaterialPlugin::<SpineScreenPmaMaterial>::default(),
            ))
            .add_plugins((
                direct_render::SpineDirectMaterial2dPlugin::<SpineNormalMaterial>::default(),
                direct_render::SpineDirectMaterial2dPlugin::<SpineAdditiveMaterial>::default(),
                direct_render::SpineDirectMaterial2dPlugin::<SpineMultiplyMaterial>::default(),
                direct_render::SpineDirectMaterial2dPlugin::<SpineScreenMaterial>::default(),
                direct_render::SpineDirectMaterial2dPlugin::<SpineNormalPmaMaterial>::default(),
                direct_render::SpineDirectMaterial2dPlugin::<SpineAdditivePmaMaterial>::default(),
                direct_render::SpineDirectMaterial2dPlugin::<SpineMultiplyPmaMaterial>::default(),
                direct_render::SpineDirectMaterial2dPlugin::<SpineScreenPmaMaterial>::default(),
            ));
        }

        app.add_plugins(direct_render::SpineDirectRenderPlugin)
            .add_plugins(SpineSyncPlugin::first())
            .register_type::<Crossfades>()
            .register_type::<SkeletonDataHandle>()
            .register_type::<SpineSync>()
            .register_type::<Spine>()
            .register_type::<SpineBone>()
            .register_type::<SpineMeshes>()
            .register_type::<SpineMesh>()
            .register_type::<SpineMeshState>()
            .register_type::<SpineSettings>()
            .register_type::<SpineDrawer>()
            .init_resource::<SpineEventQueue>()
            .init_resource::<SpineTexturePathResolver>()
            .insert_resource(SpineTextures::init())
            .add_message::<SpineAssetLoadFailedEvent>()
            .init_asset::<Atlas>()
            .init_asset::<SkeletonJson>()
            .init_asset::<SkeletonBinary>()
            .init_asset::<SkeletonData>()
            .register_asset_reflect::<Atlas>()
            .register_asset_reflect::<SkeletonJson>()
            .register_asset_reflect::<SkeletonBinary>()
            .register_asset_reflect::<SkeletonData>()
            .init_asset_loader::<AtlasLoader>()
            .init_asset_loader::<SkeletonJsonLoader>()
            .init_asset_loader::<SkeletonBinaryLoader>()
            .add_message::<SpineReadyEvent>()
            .add_message::<SpineEvent>()
            .configure_sets(
                Update,
                (
                    SpineSystem::Load,
                    SpineSet::Prepare,
                    SpineSystem::Initialize,
                    SpineSet::OnReady,
                    SpineSystem::UpdateAnimation,
                    SpineSet::OnEvent,
                )
                    .chain(),
            )
            .configure_sets(
                Update,
                (SpineSystem::UpdateMeshes, SpineSet::OnUpdateMesh)
                    .after(SpineSet::OnEvent)
                    .before(SpineSystem::UpdateMaterials),
            )
            .add_systems(
                Update,
                (
                    spine_load.in_set(SpineSystem::Load),
                    ApplyDeferred
                        .after(SpineSet::Prepare)
                        .before(SpineSystem::Initialize),
                    spine_cleanup_removed_handles.in_set(SpineSystem::Initialize),
                    spine_initialize.in_set(SpineSystem::Initialize),
                    ApplyDeferred
                        .after(SpineSystem::Initialize)
                        .before(SpineSet::OnReady),
                    spine_update_animation.in_set(SpineSystem::UpdateAnimation),
                    spine_update_meshes
                        .in_set(SpineSystem::UpdateMeshes)
                        .in_set(SpineSet::OnUpdateMesh),
                ),
            )
            .add_systems(PostUpdate, textures::adjust_spine_textures);

        load_internal_binary_asset!(
            app,
            SHADER_HANDLE,
            "spine.wgsl",
            |bytes: &[u8], path: String| Shader::from_wgsl(
                std::str::from_utf8(bytes).unwrap().to_owned(),
                path
            )
        );
    }
}

#[derive(Resource, Default)]
struct SpineEventQueue(Arc<Mutex<VecDeque<SpineEvent>>>);

/// A live Spine [`SkeletonController`] [`Component`], ready to be manipulated.
///
/// This component does not exist immediately when an entity is spawned with
/// [`SkeletonDataHandle`], since Spine assets may not yet be loaded. Querying for this component
/// type guarantees that all entities containing it have a Spine rig that is ready to use.
#[derive(Component, Debug, Reflect)]
#[reflect(Component, Debug, from_reflect = false)]
pub struct Spine(#[reflect(ignore)] pub SkeletonController);

/// When loaded, a [`Spine`] entity has children entities attached to it, each containing this
/// component.
///
/// To disable creation of these child entities, see [`SpineSettings::without_bone_entities`].
///
/// The bones are not automatically synchronized, but can be synchronized easily by adding a
/// [`SpineSync`] component.
#[derive(Component, Debug, Reflect)]
#[reflect(Component, Debug, from_reflect = false)]
pub struct SpineBone {
    pub spine_entity: Entity,
    #[reflect(ignore)]
    pub handle: BoneHandle,
    pub name: String,
    #[reflect(ignore)]
    pub parent: Option<SpineBoneParent>,
}

#[derive(Debug)]
pub struct SpineBoneParent {
    pub entity: Entity,
    pub handle: BoneHandle,
}

#[derive(Component, Clone, Reflect)]
#[reflect(Component, Clone)]
pub struct SpineMeshes;

#[derive(Component, Default, Clone, Copy)]
struct SpineMeshesUpdateState {
    has_renderable_geometry: bool,
    // `None` means no mesh update has been attempted yet.
    culled_frames: Option<u32>,
}

/// Marker component for child entities containing [`Mesh`] components for Spine rendering.
///
/// By default, the meshes may contain several meshes all combined into one to reduce draw calls
/// and improve performance. To interact with individual Spine meshes, see
/// [`SpineSettings::drawer`].
#[derive(Component, Debug, Clone, Reflect)]
#[reflect(opaque)]
#[reflect(Component, Debug, Clone)]
pub struct SpineMesh {
    pub spine_entity: Entity,
    pub handle: Handle<Mesh>,
    pub state: SpineMeshState,
}

/// The state of this [`SpineMesh`].
#[derive(Default, Component, Debug, Clone, Reflect)]
#[reflect(opaque)]
#[reflect(Component, Default, Debug, Clone)]
pub enum SpineMeshState {
    /// This Spine mesh contains no mesh data and should not render.
    #[default]
    Empty,
    /// This Spine mesh contains mesh data and should render.
    Renderable { info: SpineMaterialInfo },
}

impl core::ops::Deref for Spine {
    type Target = SkeletonController;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl core::ops::DerefMut for Spine {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Settings for how this Spine updates and renders.
///
/// Typically set alongside [`SkeletonDataHandle`] when spawning an entity.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Reflect)]
#[reflect(Component, Debug, PartialEq, Clone)]
pub struct SpineSettings {
    /// Create child entities for the skeleton's bones.
    pub spawn_bone_entities: bool,
    /// Indicates if default Spine materials should be used (default: `true`).
    ///
    /// If `false`, a custom [`SpineMaterial`](`materials::SpineMaterial`) should be configured for
    /// this Spine.
    pub default_materials: bool,
    /// The drawer this Spine should use to create its meshes.
    pub drawer: SpineDrawer,
    /// Keep rebuilding meshes even when all mesh children are currently out of view.
    ///
    /// Defaults to `false` to reduce CPU work for large numbers of off-screen skeletons.
    /// Hidden skeletons perform one initial mesh update. Set this to `true` if off-screen meshes
    /// must stay fully up to date after that initial update.
    pub update_meshes_when_invisible: bool,
    /// Upload 2D Spine geometry through a direct render path instead of mutating [`Mesh`] assets.
    ///
    /// This avoids per-frame mesh asset events and GPU mesh re-extraction for animated skeletons.
    /// Built-in Spine materials are registered automatically by [`SpinePlugin`]; custom
    /// [`Material2d`](bevy::sprite_render::Material2d) materials also need
    /// [`SpineDirectMaterial2dPlugin<M>`](SpineDirectMaterial2dPlugin).
    pub direct_2d_rendering: bool,
}

/// Drawer methods to use in [`SpineSettings`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Reflect)]
#[reflect(Debug, PartialEq, Clone)]
pub enum SpineDrawer {
    /// Draw each slot as a separate mesh, each represented by one [`SpineMesh`].
    ///
    /// Useful if individual meshes need separate materials, z-depth, or other rendering
    /// differences. Less performant, but more versatile than [`SpineDrawer::Combined`].
    Separated,
    /// Combine multiple slots into a single mesh.
    ///
    /// The default, and most performanent drawer method. Suitable for most use cases.
    Combined,
    /// Do not update meshes at all.
    None,
}

impl Default for SpineSettings {
    fn default() -> Self {
        Self {
            spawn_bone_entities: true,
            default_materials: true,
            drawer: SpineDrawer::Combined,
            update_meshes_when_invisible: false,
            direct_2d_rendering: false,
        }
    }
}

impl SpineSettings {
    /// Disables creation of [`SpineBone`] child entities. Renderable mesh children are unaffected.
    pub fn without_bone_entities(mut self) -> Self {
        self.spawn_bone_entities = false;
        self
    }
}

/// A [`Message`] sent when a skeleton has initialized.
///
/// Initialization uses deferred commands. Systems reading this message in [`SpineSet::OnReady`]
/// run after those commands are applied, so the entity's [`Spine`] component, mesh helpers, and
/// optional bone helpers are available before its first [`SkeletonController`] update.
#[derive(Debug, Clone, Message)]
pub struct SpineReadyEvent {
    /// The entity containing the [`Spine`] component.
    pub entity: Entity,
    /// A list of all bones (if spawned, see [`SpineBone`]).
    pub bones: HashMap<String, Entity>,
}

/// A Spine event fired from a playing animation.
///
/// Sent in [`SpineSystem::UpdateAnimation`].
///
/// ```
/// # use bevy::prelude::*;
/// # use bevy_spine::prelude::*;
/// // bevy system
/// fn on_spine_event(
///     mut spine_events: MessageReader<SpineEvent>,
///     mut commands: Commands,
///     asset_server: Res<AssetServer>,
/// ) {
///     for event in spine_events.read() {
///         if let SpineEvent::Event { name, entity, .. } = event {
///             println!("spine event fired: {}", name);
///             println!("from entity: {:?}", entity);
///         }
///     }
/// }
/// ```
#[derive(Debug, Clone, Message)]
pub enum SpineEvent {
    Start {
        entity: Entity,
        animation: String,
    },
    Interrupt {
        entity: Entity,
        animation: String,
    },
    End {
        entity: Entity,
        animation: String,
    },
    Complete {
        entity: Entity,
        animation: String,
    },
    Dispose {
        entity: Entity,
    },
    Event {
        entity: Entity,
        name: String,
        int: i32,
        float: f32,
        string: String,
        audio_path: String,
        volume: f32,
        balance: f32,
    },
}

#[allow(clippy::too_many_arguments)]
fn spine_load(
    mut skeleton_data_assets: ResMut<Assets<SkeletonData>>,
    mut skeleton_data_events: MessageReader<AssetEvent<SkeletonData>>,
    mut atlas_events: MessageReader<AssetEvent<Atlas>>,
    mut json_events: MessageReader<AssetEvent<SkeletonJson>>,
    mut binary_events: MessageReader<AssetEvent<SkeletonBinary>>,
    mut image_events: MessageReader<AssetEvent<Image>>,
    mut terminal_failures: MessageReader<SpineAssetLoadFailedEvent>,
    mut atlases: ResMut<Assets<Atlas>>,
    mut images: ResMut<Assets<Image>>,
    jsons: Res<Assets<SkeletonJson>>,
    binaries: Res<Assets<SkeletonBinary>>,
    mut spine_textures: ResMut<SpineTextures>,
    asset_server: Res<AssetServer>,
    path_resolver: Res<SpineTexturePathResolver>,
    mut previous_texture_revision: Local<Option<u64>>,
) {
    let skeleton_data_changed = skeleton_data_events.read().next().is_some();
    let json_changed = json_events.read().next().is_some();
    let binary_changed = binary_events.read().next().is_some();
    let image_changed = image_events.read().next().is_some();
    let mut changed_atlases = HashSet::new();
    let mut removed_atlases = HashSet::new();
    for event in atlas_events.read() {
        match event {
            AssetEvent::Added { id }
            | AssetEvent::Modified { id }
            | AssetEvent::LoadedWithDependencies { id } => {
                changed_atlases.insert(*id);
                removed_atlases.remove(id);
            }
            AssetEvent::Removed { id } => {
                changed_atlases.remove(id);
                removed_atlases.insert(*id);
            }
            AssetEvent::Unused { .. } => {}
        }
    }
    let dependency_assets_changed = skeleton_data_changed
        || !changed_atlases.is_empty()
        || !removed_atlases.is_empty()
        || json_changed
        || binary_changed
        || image_changed;

    let reset_paths = if dependency_assets_changed {
        spine_textures.clear_successfully_loaded(&asset_server)
    } else {
        Vec::new()
    };
    reset_skeleton_data_statuses(
        &mut skeleton_data_assets,
        &asset_server,
        &spine_textures,
        &reset_paths,
    );

    // Atlas page handles are resolved before skeleton data is parsed. The atlas readiness state is
    // advanced by image asset events in `adjust_spine_textures`, so this stays an O(1) state lookup
    // per skeleton instead of repeatedly scanning every atlas page.
    let texture_revision = spine_textures.update(
        asset_server.as_ref(),
        &mut atlases,
        &mut images,
        &path_resolver,
        &changed_atlases,
        &removed_atlases,
    );

    let mut terminal_failure_received = false;
    for failure in terminal_failures.read() {
        terminal_failure_received = true;
        error!(
            id = ?failure.id,
            path = %failure.path,
            error = %failure.error,
            "Spine asset failed terminally"
        );
        spine_textures.record_terminal_failure(&failure.path);
    }

    let textures_changed = previous_texture_revision
        .replace(texture_revision)
        .is_none_or(|previous| previous != texture_revision);
    if !dependency_assets_changed
        && !textures_changed
        && !terminal_failure_received
        && reset_paths.is_empty()
    {
        return;
    }

    let mut status_updates = Vec::new();
    for (id, skeleton_data_asset) in skeleton_data_assets.iter() {
        let SkeletonData {
            atlas_handle,
            kind,
            status,
            ..
        } = skeleton_data_asset;
        if skeleton_data_has_terminal_failure(skeleton_data_asset, &asset_server, &spine_textures) {
            status_updates.push((id, SkeletonDataStatus::Failed, None));
            continue;
        }
        if matches!(status, SkeletonDataStatus::Failed) {
            continue;
        }

        let Some(atlas) = atlases.get(atlas_handle) else {
            if matches!(status, SkeletonDataStatus::Loaded(_)) {
                status_updates.push((id, SkeletonDataStatus::Loading, None));
            }
            continue;
        };

        let atlas_ready = match spine_textures.atlas_status(atlas_handle) {
            Some(SpineAtlasStatus::Loaded) => true,
            Some(SpineAtlasStatus::Loading) => atlas.atlas.pages().next().is_none(),
            Some(SpineAtlasStatus::Failed) => {
                error!(
                    atlas = ?atlas_handle,
                    "Spine atlas page image failed to load or prepare"
                );
                status_updates.push((id, SkeletonDataStatus::Failed, None));
                continue;
            }
            None if atlas.atlas.pages().next().is_none() => true,
            None => {
                trace!(
                    atlas = ?atlas_handle,
                    "Spine atlas page readiness is not registered yet"
                );
                false
            }
        };
        if !atlas_ready {
            if matches!(status, SkeletonDataStatus::Loaded(_)) {
                status_updates.push((id, SkeletonDataStatus::Loading, None));
            }
            continue;
        }
        if matches!(status, SkeletonDataStatus::Loaded(_)) {
            continue;
        }

        let premultiplied_alpha = atlas.atlas.pages().next().map(|page| page.pma());
        let next_status = match kind {
            SkeletonDataKind::JsonFile(json_handle) => {
                let Some(json) = jsons.get(json_handle) else {
                    continue;
                };
                let skeleton_json = rusty_spine::SkeletonJson::new(atlas.atlas.clone());
                match skeleton_json.read_skeleton_data(&json.json) {
                    Ok(skeleton_data) => SkeletonDataStatus::Loaded(Arc::new(skeleton_data)),
                    Err(err) => {
                        error!("Failed to load Spine JSON skeleton data: {err}");
                        SkeletonDataStatus::Failed
                    }
                }
            }
            SkeletonDataKind::BinaryFile(binary_handle) => {
                let Some(binary) = binaries.get(binary_handle) else {
                    continue;
                };
                let skeleton_binary = rusty_spine::SkeletonBinary::new(atlas.atlas.clone());
                match skeleton_binary.read_skeleton_data(&binary.binary) {
                    Ok(skeleton_data) => SkeletonDataStatus::Loaded(Arc::new(skeleton_data)),
                    Err(err) => {
                        error!("Failed to load Spine binary skeleton data: {err}");
                        SkeletonDataStatus::Failed
                    }
                }
            }
        };
        status_updates.push((id, next_status, premultiplied_alpha));
    }

    for (id, status, premultiplied_alpha) in status_updates {
        let Some(skeleton_data_asset) = skeleton_data_assets.get_mut_untracked(id) else {
            warn!(id = ?id, "Spine skeleton disappeared before its load status could be updated");
            continue;
        };
        if let Some(premultiplied_alpha) = premultiplied_alpha
            && skeleton_data_asset.premultiplied_alpha != premultiplied_alpha
        {
            skeleton_data_asset.premultiplied_alpha = premultiplied_alpha;
        }
        if matches!(
            (&skeleton_data_asset.status, &status),
            (SkeletonDataStatus::Loading, SkeletonDataStatus::Loading)
                | (SkeletonDataStatus::Failed, SkeletonDataStatus::Failed)
        ) {
            continue;
        }
        skeleton_data_asset.status = status;
    }
}

fn reset_skeleton_data_statuses(
    skeleton_data_assets: &mut Assets<SkeletonData>,
    asset_server: &AssetServer,
    spine_textures: &SpineTextures,
    reset_paths: &[AssetPath<'static>],
) {
    if reset_paths.is_empty() {
        return;
    }
    let reset_ids: Vec<_> = skeleton_data_assets
        .iter()
        .filter_map(|(id, skeleton_data)| {
            (matches!(skeleton_data.status, SkeletonDataStatus::Failed)
                && reset_paths.iter().any(|path| {
                    skeleton_data_uses_path(skeleton_data, asset_server, path)
                        || spine_textures.atlas_uses_path(&skeleton_data.atlas_handle, path)
                }))
            .then_some(id)
        })
        .collect();
    for id in reset_ids {
        let Some(skeleton_data) = skeleton_data_assets.get_mut_untracked(id) else {
            continue;
        };
        skeleton_data.status = SkeletonDataStatus::Loading;
    }
}

fn skeleton_data_has_terminal_failure(
    skeleton_data: &SkeletonData,
    asset_server: &AssetServer,
    spine_textures: &SpineTextures,
) -> bool {
    [
        skeleton_data.atlas_handle.id().untyped(),
        match &skeleton_data.kind {
            SkeletonDataKind::JsonFile(handle) => handle.id().untyped(),
            SkeletonDataKind::BinaryFile(handle) => handle.id().untyped(),
        },
    ]
    .into_iter()
    .filter_map(|id| asset_server.get_path(id))
    .any(|path| spine_textures.has_terminal_failure(&path))
}

fn skeleton_data_uses_path(
    skeleton_data: &SkeletonData,
    asset_server: &AssetServer,
    path: &AssetPath<'_>,
) -> bool {
    [
        skeleton_data.atlas_handle.id().untyped(),
        match &skeleton_data.kind {
            SkeletonDataKind::JsonFile(handle) => handle.id().untyped(),
            SkeletonDataKind::BinaryFile(handle) => handle.id().untyped(),
        },
    ]
    .into_iter()
    .filter_map(|id| asset_server.get_path(id))
    .any(|dependency_path| dependency_path == *path)
}

#[derive(Component)]
#[relationship(relationship_target = SpineHelpers)]
struct SpineHelperOf(Entity);

#[derive(Component, Default)]
#[relationship_target(relationship = SpineHelperOf, linked_spawn)]
struct SpineHelpers(Vec<Entity>);

fn clear_spine_runtime(spine_entity: Entity, commands: &mut Commands) {
    if let Ok(mut root) = commands.get_entity(spine_entity) {
        root.despawn_related::<SpineHelpers>()
            .remove::<(Spine, SpineHelpers)>();
    }
}

fn spine_cleanup_removed_handles(
    mut removed_handles: RemovedComponents<SkeletonDataHandle>,
    runtime_roots: Query<(Has<Spine>, Has<SpineHelpers>), Without<SkeletonDataHandle>>,
    mut commands: Commands,
) {
    for entity in removed_handles.read() {
        let Ok((has_spine, has_helpers)) = runtime_roots.get(entity) else {
            continue;
        };
        if !has_spine && !has_helpers {
            continue;
        }
        clear_spine_runtime(entity, &mut commands);
    }
}

#[allow(clippy::type_complexity)]
fn spine_initialize(
    skeleton_query: Query<(
        Entity,
        Ref<SkeletonDataHandle>,
        &SpineSettings,
        Option<&Crossfades>,
        Option<&RenderLayers>,
        Has<Spine>,
        Option<&SpineHelpers>,
    )>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut ready_events: MessageWriter<SpineReadyEvent>,
    skeleton_data_assets: Res<Assets<SkeletonData>>,
    spine_event_queue: Res<SpineEventQueue>,
) {
    for (spine_entity, data_handle, settings, crossfades, render_layers, has_spine, helpers) in
        &skeleton_query
    {
        if !data_handle.is_changed() && has_spine {
            continue;
        }
        if data_handle.is_changed() || helpers.is_some_and(|helpers| !helpers.is_empty()) {
            clear_spine_runtime(spine_entity, &mut commands);
        }

        let Some(skeleton_data_asset) = skeleton_data_assets.get(&data_handle.0) else {
            continue;
        };
        match &skeleton_data_asset.status {
            SkeletonDataStatus::Loaded(skeleton_data) => {
                let mut animation_state_data = AnimationStateData::new(skeleton_data.clone());
                if let Some(crossfades) = crossfades {
                    crossfades.apply(&mut animation_state_data);
                }
                let mut controller =
                    SkeletonController::new(skeleton_data.clone(), Arc::new(animation_state_data))
                        .with_settings(
                            SkeletonControllerSettings::new()
                                .with_cull_direction(CullDirection::CounterClockwise)
                                .with_premultiplied_alpha(skeleton_data_asset.premultiplied_alpha),
                        );
                let events = spine_event_queue.0.clone();
                controller
                    .animation_state
                    .set_listener(move |_, animation_event| match animation_event {
                        AnimationEvent::Start { track_entry } => {
                            let mut events = events.lock().unwrap();
                            events.push_back(SpineEvent::Start {
                                entity: spine_entity,
                                animation: track_entry.animation().name().to_owned(),
                            });
                        }
                        AnimationEvent::Interrupt { track_entry } => {
                            let mut events = events.lock().unwrap();
                            events.push_back(SpineEvent::Interrupt {
                                entity: spine_entity,
                                animation: track_entry.animation().name().to_owned(),
                            });
                        }
                        AnimationEvent::End { track_entry } => {
                            let mut events = events.lock().unwrap();
                            events.push_back(SpineEvent::End {
                                entity: spine_entity,
                                animation: track_entry.animation().name().to_owned(),
                            });
                        }
                        AnimationEvent::Complete { track_entry } => {
                            let mut events = events.lock().unwrap();
                            events.push_back(SpineEvent::Complete {
                                entity: spine_entity,
                                animation: track_entry.animation().name().to_owned(),
                            });
                        }
                        AnimationEvent::Dispose { .. } => {
                            let mut events = events.lock().unwrap();
                            events.push_back(SpineEvent::Dispose {
                                entity: spine_entity,
                            });
                        }
                        AnimationEvent::Event {
                            name,
                            int,
                            float,
                            string,
                            audio_path,
                            volume,
                            balance,
                            ..
                        } => {
                            let mut events = events.lock().unwrap();
                            events.push_back(SpineEvent::Event {
                                entity: spine_entity,
                                name: name.to_owned(),
                                int,
                                float,
                                string: string.to_owned(),
                                audio_path: audio_path.to_owned(),
                                volume,
                                balance,
                            });
                        }
                    });
                controller.skeleton.set_to_setup_pose();
                let mesh_count = controller.skeleton.slots().count();
                let render_layers = render_layers.cloned();
                let mut bones = HashMap::new();
                let Ok(mut entity_commands) = commands.get_entity(spine_entity) else {
                    warn!(
                        entity = ?spine_entity,
                        "Spine finished loading, but the entity no longer exists; skipping spawn"
                    );
                    continue;
                };

                entity_commands
                    .with_children(|parent| {
                        let render_layers_for_children = render_layers.clone();
                        let mut spine_meshes_commands = parent.spawn((
                            Name::new("spine_meshes"),
                            SpineHelperOf(spine_entity),
                            SpineMeshes,
                            SpineMeshesUpdateState::default(),
                            Transform::default(),
                            GlobalTransform::default(),
                            Visibility::default(),
                            InheritedVisibility::default(),
                            ViewVisibility::default(),
                        ));

                        if let Some(render_layers) = &render_layers_for_children {
                            spine_meshes_commands.insert(render_layers.clone());
                        }
                        spine_meshes_commands.with_children(|parent| {
                            let render_layers_for_meshes = render_layers_for_children.clone();
                            let mut z = 0.;
                            for index in 0..mesh_count {
                                let mut mesh = Mesh::new(
                                    PrimitiveTopology::TriangleList,
                                    RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
                                );
                                empty_mesh(&mut mesh);
                                let mesh_handle = meshes.add(mesh);

                                let mut mesh_commands = parent.spawn((
                                    Name::new(format!("spine_mesh {index}")),
                                    SpineHelperOf(spine_entity),
                                    SpineMesh {
                                        spine_entity,
                                        handle: mesh_handle,
                                        state: SpineMeshState::Empty,
                                    },
                                    Transform::from_xyz(0., 0., z),
                                    GlobalTransform::default(),
                                    Visibility::default(),
                                    InheritedVisibility::default(),
                                    ViewVisibility::default(),
                                ));

                                if let Some(render_layers) = &render_layers_for_meshes {
                                    mesh_commands.insert(render_layers.clone());
                                }
                                z += 0.001;
                            }
                        });

                        if settings.spawn_bone_entities {
                            spawn_bones(
                                spine_entity,
                                None,
                                parent,
                                &controller.skeleton,
                                controller.skeleton.bone_root().handle(),
                                render_layers_for_children.as_ref(),
                                &mut bones,
                            );
                        }
                    })
                    .insert(Spine(controller));
                ready_events.write(SpineReadyEvent {
                    entity: spine_entity,
                    bones,
                });
            }
            SkeletonDataStatus::Loading | SkeletonDataStatus::Failed => {}
        }
    }
}

fn spawn_bones(
    spine_entity: Entity,
    bone_parent: Option<SpineBoneParent>,
    spawner: &mut ChildSpawnerCommands<'_>,
    skeleton: &Skeleton,
    bone: BoneHandle,
    render_layers: Option<&RenderLayers>,
    bones: &mut HashMap<String, Entity>,
) {
    if let Some(bone) = bone.get(skeleton) {
        let mut transform = Transform::default();
        transform.translation.x = bone.applied_x();
        transform.translation.y = bone.applied_y();
        transform.translation.z = 0.;
        transform.rotation = Quat::from_axis_angle(Vec3::Z, bone.applied_rotation().to_radians());
        transform.scale.x = bone.applied_scale_x();
        transform.scale.y = bone.applied_scale_y();
        let mut bone_entity_commands = spawner.spawn((
            Name::new(format!("spine_bone ({})", bone.data().name())),
            SpineHelperOf(spine_entity),
            transform,
            GlobalTransform::default(),
            Visibility::default(),
            InheritedVisibility::default(),
            ViewVisibility::default(),
        ));

        if let Some(render_layers) = render_layers {
            bone_entity_commands.insert(render_layers.clone());
        }
        let bone_entity = bone_entity_commands
            .insert(SpineBone {
                spine_entity,
                handle: bone.handle(),
                name: bone.data().name().to_owned(),
                parent: bone_parent,
            })
            .with_children(|parent| {
                for child in bone.children() {
                    spawn_bones(
                        spine_entity,
                        Some(SpineBoneParent {
                            entity: parent.target_entity(),
                            handle: bone.handle(),
                        }),
                        parent,
                        skeleton,
                        child.handle(),
                        render_layers,
                        bones,
                    );
                }
            })
            .id();
        bones.insert(bone.data().name().to_owned(), bone_entity);
    }
}

fn spine_update_animation(
    mut spine_query: Query<&mut Spine>,
    mut spine_events: MessageWriter<SpineEvent>,
    time: Res<Time>,
    spine_event_queue: Res<SpineEventQueue>,
) {
    for mut spine in &mut spine_query {
        spine.update(time.delta_secs(), Physics::Update);
    }
    {
        let mut events = spine_event_queue.0.lock().unwrap();
        while let Some(event) = events.pop_front() {
            spine_events.write(event);
        }
    }
}

enum SkeletonRenderableKind {
    Simple(Vec<SkeletonRenderable>),
    Combined(Vec<SkeletonCombinedRenderable>),
}

#[allow(clippy::type_complexity)]
fn spine_update_meshes(
    mut spine_query: Query<(&mut Spine, Option<&SpineSettings>, &InheritedVisibility)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mesh_query: Query<(
        Entity,
        &mut SpineMesh,
        &mut Transform,
        Option<&mut Aabb>,
        Option<&Mesh2d>,
        Option<&mut SpineDirectMesh>,
    )>,
    mesh_visibility_query: Query<&ViewVisibility, With<SpineMesh>>,
    mut commands: Commands,
    mut meshes_query: Query<(&ChildOf, &Children, &mut SpineMeshesUpdateState), With<SpineMeshes>>,
) {
    const CULLED_RECOVERY_INTERVAL_FRAMES: u32 = 60;

    fn write_spine_mesh_aabb(
        commands: &mut Commands,
        spine_mesh_entity: Entity,
        spine_mesh_aabb: &mut Option<Mut<Aabb>>,
        mesh_aabb: Aabb,
    ) {
        if let Some(current_aabb) = spine_mesh_aabb {
            **current_aabb = mesh_aabb;
        } else if let Ok(mut entity) = commands.get_entity(spine_mesh_entity) {
            entity.insert(mesh_aabb);
        }
    }

    for (meshes_parent, meshes_children, mut update_state) in meshes_query.iter_mut() {
        let Ok((mut spine, spine_settings, inherited_visibility)) =
            spine_query.get_mut(meshes_parent.parent())
        else {
            warn!(
                entity = ?meshes_parent.parent(),
                "SpineMeshes parent has no ready Spine component; skipping mesh update"
            );
            continue;
        };

        let SpineSettings {
            drawer,
            update_meshes_when_invisible,
            direct_2d_rendering,
            ..
        } = spine_settings.copied().unwrap_or_default();

        // Newly initialized skeletons must produce their first geometry even though Bevy has not
        // propagated their inherited visibility yet. Established hidden skeletons retain the
        // normal off-screen optimization.
        if update_state.culled_frames.is_some()
            && !inherited_visibility.get()
            && !update_meshes_when_invisible
        {
            continue;
        }

        if !update_meshes_when_invisible
            && inherited_visibility.get()
            && update_state.has_renderable_geometry
        {
            let any_visible = meshes_children.iter().any(|child| {
                mesh_visibility_query
                    .get(child)
                    .is_ok_and(|visibility| visibility.get())
            });
            if !any_visible {
                let culled_frames = update_state.culled_frames.get_or_insert(0);
                *culled_frames = culled_frames.saturating_add(1);

                if *culled_frames < CULLED_RECOVERY_INTERVAL_FRAMES {
                    continue;
                }
            } else {
                update_state.culled_frames = Some(0);
            }
        }

        let mut renderables = match drawer {
            SpineDrawer::Combined => {
                let mut renderables = spine.0.combined_renderables();
                // rusty_spine can emit empty leading combined renderables when early slots are hidden.
                renderables.retain(|renderable| {
                    renderable.attachment_renderer_object.is_some()
                        && !renderable.vertices.is_empty()
                        && !renderable.indices.is_empty()
                });
                SkeletonRenderableKind::Combined(renderables)
            }
            SpineDrawer::Separated => SkeletonRenderableKind::Simple(spine.0.renderables()),
            SpineDrawer::None => continue,
        };
        let mut z = 0.;
        let mut renderable_index = 0;
        let mut has_renderable_geometry = false;
        for child in meshes_children.iter() {
            if let Ok((
                spine_mesh_entity,
                mut spine_mesh,
                mut spine_mesh_transform,
                mut spine_mesh_aabb,
                spine_2d_mesh,
                mut direct_mesh,
            )) = mesh_query.get_mut(child)
            {
                if direct_2d_rendering {
                    let direct_mesh_2d = Mesh2d(Handle::<Mesh>::default());
                    if spine_2d_mesh != Some(&direct_mesh_2d)
                        && let Ok(mut entity) = commands.get_entity(spine_mesh_entity)
                    {
                        entity.insert((direct_mesh_2d, NoAutomaticBatching));
                    }
                } else {
                    let mesh_2d = Mesh2d(spine_mesh.handle.clone());
                    if spine_2d_mesh != Some(&mesh_2d)
                        && let Ok(mut entity) = commands.get_entity(spine_mesh_entity)
                    {
                        entity.insert(mesh_2d);
                    }

                    if direct_mesh.is_some()
                        && let Ok(mut entity) = commands.get_entity(spine_mesh_entity)
                    {
                        entity.remove::<(SpineDirectMesh, NoAutomaticBatching)>();
                    }
                }

                let mut mesh = if direct_2d_rendering {
                    None
                } else {
                    let Some(mesh) = meshes.get_mut(&spine_mesh.handle) else {
                        warn!(
                            "Spine mesh asset {:?} is missing; skipping mesh update",
                            spine_mesh.handle
                        );
                        continue;
                    };
                    Some(mesh)
                };
                let mut empty = true;
                'render: {
                    let (
                        slot_index,
                        attachment_renderer_object,
                        vertices,
                        indices,
                        uvs,
                        colors,
                        dark_colors,
                        blend_mode,
                        premultiplied_alpha,
                    ) = match &mut renderables {
                        SkeletonRenderableKind::Simple(vec) => {
                            let Some(renderable) = vec.get_mut(renderable_index) else {
                                break 'render;
                            };
                            let colors = vec![
                                [
                                    renderable.color.r,
                                    renderable.color.g,
                                    renderable.color.b,
                                    renderable.color.a
                                ];
                                renderable.vertices.len()
                            ];
                            let dark_colors = vec![
                                [
                                    renderable.dark_color.r,
                                    renderable.dark_color.g,
                                    renderable.dark_color.b,
                                    renderable.dark_color.a
                                ];
                                renderable.vertices.len()
                            ];
                            (
                                Some(renderable.slot_index),
                                renderable.attachment_renderer_object,
                                take(&mut renderable.vertices),
                                take(&mut renderable.indices),
                                take(&mut renderable.uvs),
                                colors,
                                dark_colors,
                                renderable.blend_mode,
                                renderable.premultiplied_alpha,
                            )
                        }
                        SkeletonRenderableKind::Combined(vec) => {
                            let Some(renderable) = vec.get_mut(renderable_index) else {
                                break 'render;
                            };
                            (
                                None,
                                renderable.attachment_renderer_object,
                                take(&mut renderable.vertices),
                                take(&mut renderable.indices),
                                take(&mut renderable.uvs),
                                take(&mut renderable.colors),
                                take(&mut renderable.dark_colors),
                                renderable.blend_mode,
                                renderable.premultiplied_alpha,
                            )
                        }
                    };
                    let Some(attachment_render_object) = attachment_renderer_object else {
                        break 'render;
                    };
                    if vertices.is_empty() || indices.is_empty() {
                        break 'render;
                    }

                    let mut min_x = f32::INFINITY;
                    let mut min_y = f32::INFINITY;
                    let mut max_x = f32::NEG_INFINITY;
                    let mut max_y = f32::NEG_INFINITY;
                    for [x, y] in &vertices {
                        min_x = min_x.min(*x);
                        min_y = min_y.min(*y);
                        max_x = max_x.max(*x);
                        max_y = max_y.max(*y);
                    }
                    let mesh_aabb = Aabb::from_min_max(
                        Vec3::new(min_x, min_y, 0.),
                        Vec3::new(max_x, max_y, 0.),
                    );

                    let spine_texture =
                        unsafe { &mut *(attachment_render_object as *mut SpineTexture) };
                    let Some(texture_handle) = spine_texture.resolved_handle.clone() else {
                        warn_once!(
                            path = %spine_texture.path,
                            "Spine renderable has no resolved atlas texture handle; skipping mesh update"
                        );
                        break 'render;
                    };
                    let mesh_updated = if direct_2d_rendering {
                        let mut next_direct_mesh = None;
                        let updated = {
                            let direct_mesh = if let Some(direct_mesh) = direct_mesh.as_deref_mut()
                            {
                                direct_mesh
                            } else {
                                next_direct_mesh.insert(SpineDirectMesh::default())
                            };
                            direct_mesh.write(
                                spine_mesh_entity,
                                &vertices,
                                &indices,
                                &uvs,
                                &colors,
                                &dark_colors,
                            )
                        };
                        if updated
                            && let Some(next_direct_mesh) = next_direct_mesh
                            && let Ok(mut entity) = commands.get_entity(spine_mesh_entity)
                        {
                            entity.insert(next_direct_mesh);
                        }
                        updated
                    } else if let Some(mesh) = mesh.as_deref_mut() {
                        let normals = vec![[0., 0., 0.]; vertices.len()];
                        mesh.insert_indices(Indices::U16(indices));
                        mesh.insert_attribute(
                            MeshVertexAttribute::new("Vertex_Position", 0, VertexFormat::Float32x2),
                            vertices,
                        );
                        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
                        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
                        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
                        mesh.insert_attribute(DARK_COLOR_ATTRIBUTE, dark_colors);
                        true
                    } else {
                        false
                    };

                    if !mesh_updated {
                        break 'render;
                    }

                    spine_mesh.state = SpineMeshState::Renderable {
                        info: SpineMaterialInfo {
                            slot_index,
                            texture: texture_handle,
                            blend_mode,
                            premultiplied_alpha,
                        },
                    };
                    spine_mesh_transform.translation.z = z;
                    write_spine_mesh_aabb(
                        &mut commands,
                        spine_mesh_entity,
                        &mut spine_mesh_aabb,
                        mesh_aabb,
                    );
                    z += 0.001;
                    empty = false;
                    has_renderable_geometry = true;
                }
                if empty {
                    spine_mesh.state = SpineMeshState::Empty;
                    if direct_2d_rendering {
                        if let Some(direct_mesh) = direct_mesh.as_deref_mut() {
                            direct_mesh.clear();
                        }
                    } else if let Some(mesh) = mesh.as_deref_mut() {
                        empty_mesh(mesh);
                    }
                    write_spine_mesh_aabb(
                        &mut commands,
                        spine_mesh_entity,
                        &mut spine_mesh_aabb,
                        Aabb::from_min_max(Vec3::ZERO, Vec3::ZERO),
                    );
                }
                renderable_index += 1;
            }
        }

        update_state.has_renderable_geometry |= has_renderable_geometry;
        update_state.culled_frames = Some(0);
    }
}

fn empty_mesh(mesh: &mut Mesh) {
    let positions: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
    let normals: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
    let uvs: Vec<[f32; 2]> = vec![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];
    let colors: Vec<[f32; 4]> = vec![
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ];
    let dark_colors: Vec<[f32; 4]> = vec![
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ];

    mesh.remove_indices();
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_attribute(DARK_COLOR_ATTRIBUTE, dark_colors);
}

mod assets;
mod crossfades;
mod direct_render;
mod entity_sync;
mod handle;

pub mod materials;
pub mod textures;

#[doc(hidden)]
pub mod prelude {
    pub use crate::{
        Crossfades, SkeletonController, SkeletonData, SkeletonDataHandle, Spine,
        SpineAssetLoadFailedEvent, SpineBone, SpineDirectMaterial2dPlugin, SpineEvent, SpineMesh,
        SpineMeshState, SpinePlugin, SpineReadyEvent, SpineSet, SpineSettings, SpineSync,
        SpineSyncSet, SpineSyncSystem, SpineSystem, SpineTexturePathResolver,
    };
    pub use rusty_spine::{BoneHandle, SlotHandle};
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use bevy::asset::AssetPlugin;
    use bevy::sprite_render::MeshMaterial2d;

    use super::*;

    #[derive(Resource)]
    struct SpawnedSkeletons {
        loaded: Entity,
        loading: Entity,
    }

    #[derive(Resource)]
    struct ReplaceSkeleton(Entity);

    #[derive(Resource)]
    struct FinishLoading(Entity);

    #[derive(Default, Resource)]
    struct ObservedReady(Vec<Entity>);

    fn loaded_spineboy_skeleton() -> Arc<rusty_spine::SkeletonData> {
        let atlas = Arc::new(
            rusty_spine::Atlas::new(
                include_bytes!("../assets/spineboy/export/spineboy-pma.atlas"),
                Path::new(""),
            )
            .expect("test atlas should parse"),
        );
        for page in atlas.pages() {
            let mut renderer_object = page.renderer_object();
            let texture = unsafe { renderer_object.get::<SpineTexture>() }
                .expect("Spine texture callbacks should attach page metadata");
            texture.resolved_handle = Some(Handle::default());
        }
        Arc::new(
            rusty_spine::SkeletonJson::new(atlas)
                .read_skeleton_data(include_bytes!(
                    "../assets/spineboy/export/spineboy-pro.json"
                ))
                .expect("test skeleton should parse"),
        )
    }

    fn loaded_skeleton_asset() -> SkeletonData {
        SkeletonData {
            atlas_handle: default(),
            kind: SkeletonDataKind::JsonFile(default()),
            status: SkeletonDataStatus::Loaded(loaded_spineboy_skeleton()),
            premultiplied_alpha: true,
        }
    }

    fn test_app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .init_asset::<Image>()
            .init_asset::<Mesh>()
            .init_asset::<Shader>()
            .init_asset::<SpineNormalPmaMaterial>()
            .add_plugins(SpinePlugin::without_built_in_materials())
            // The CPU material updater only: no `Material2dPlugin` or GPU render resources.
            .add_plugins(SpineMaterialPlugin::<SpineNormalPmaMaterial>::default())
            .init_resource::<ObservedReady>()
            .add_systems(
                Update,
                (
                    spawn_preloaded_skeletons.run_if(not(resource_exists::<SpawnedSkeletons>)),
                    replace_with_preloaded_skeleton.run_if(resource_exists::<ReplaceSkeleton>),
                    finish_loading.run_if(resource_exists::<FinishLoading>),
                )
                    .in_set(SpineSet::Prepare),
            )
            .add_systems(Update, observe_ready.in_set(SpineSet::OnReady));
        app
    }

    fn observe_ready(
        mut events: MessageReader<SpineReadyEvent>,
        ready: Query<&SpineHelpers, With<Spine>>,
        mut observed: ResMut<ObservedReady>,
    ) {
        for event in events.read() {
            let helpers = ready
                .get(event.entity)
                .expect("OnReady must observe applied Spine runtime components");
            assert!(!helpers.is_empty());
            observed.0.push(event.entity);
        }
    }

    fn spawn_preloaded_skeletons(
        mut commands: Commands,
        mut skeletons: ResMut<Assets<SkeletonData>>,
    ) {
        let loaded = skeletons.add(loaded_skeleton_asset());
        let loading = skeletons.add(SkeletonData::new_from_json(default(), default()));
        let loaded = commands
            .spawn((
                SkeletonDataHandle(loaded),
                SpineSettings::default(),
                Visibility::Visible,
            ))
            .id();
        let loading = commands
            .spawn((
                SkeletonDataHandle(loading),
                SpineSettings::default().without_bone_entities(),
            ))
            .id();
        commands.insert_resource(SpawnedSkeletons { loaded, loading });
    }

    fn replace_with_preloaded_skeleton(
        mut commands: Commands,
        mut skeletons: ResMut<Assets<SkeletonData>>,
        replacement: Res<ReplaceSkeleton>,
    ) {
        commands
            .entity(replacement.0)
            .insert(SkeletonDataHandle(skeletons.add(loaded_skeleton_asset())));
        commands.remove_resource::<ReplaceSkeleton>();
    }

    fn finish_loading(
        mut commands: Commands,
        handles: Query<&SkeletonDataHandle>,
        mut skeletons: ResMut<Assets<SkeletonData>>,
        request: Res<FinishLoading>,
    ) {
        let handle = &handles.get(request.0).unwrap().0;
        let asset = skeletons.get_mut_untracked(handle).unwrap();
        asset.status = SkeletonDataStatus::Loaded(loaded_spineboy_skeleton());
        asset.premultiplied_alpha = true;
        commands.remove_resource::<FinishLoading>();
    }

    fn renderable_meshes_for(world: &mut World, spine_entity: Entity) -> Vec<Entity> {
        let mut query = world.query::<(Entity, &SpineMesh)>();
        query
            .iter(world)
            .filter_map(|(entity, spine_mesh)| {
                (spine_mesh.spine_entity == spine_entity
                    && matches!(spine_mesh.state, SpineMeshState::Renderable { .. }))
                .then_some(entity)
            })
            .collect()
    }

    fn spine_helpers(world: &World, spine_entity: Entity) -> Vec<Entity> {
        world
            .get::<SpineHelpers>(spine_entity)
            .map(|helpers| helpers.iter().collect())
            .unwrap_or_default()
    }

    fn reparent(world: &mut World, entities: &[Entity], parent: Entity) {
        for &entity in entities {
            world.entity_mut(entity).insert(ChildOf(parent));
        }
    }

    fn assert_ready_and_renderable(app: &mut App, spine_entity: Entity) {
        let renderable_meshes = renderable_meshes_for(app.world_mut(), spine_entity);
        let world = app.world();
        assert!(world.entity(spine_entity).contains::<Spine>());
        let mesh_entity = renderable_meshes
            .into_iter()
            .find(|entity| {
                world
                    .get::<MeshMaterial2d<SpineNormalPmaMaterial>>(*entity)
                    .is_some()
            })
            .expect("ready Spine should have renderable geometry and its built-in material");
        let spine_mesh = world.get::<SpineMesh>(mesh_entity).unwrap();
        assert_eq!(
            world.get::<Mesh2d>(mesh_entity).unwrap().0,
            spine_mesh.handle
        );
        assert!(world.entity(mesh_entity).contains::<Aabb>());
        let mesh = world
            .resource::<Assets<Mesh>>()
            .get(&spine_mesh.handle)
            .unwrap();
        assert!(mesh.count_vertices() > 0);
        assert!(mesh.indices().is_some_and(|indices| !indices.is_empty()));
        assert_eq!(
            world.resource::<ObservedReady>().0.last(),
            Some(&spine_entity)
        );
    }

    fn assert_despawned(world: &World, entities: &[Entity]) {
        assert!(
            entities
                .iter()
                .all(|entity| !world.entities().contains(*entity))
        );
    }

    #[test]
    fn preloaded_prepare_and_runtime_rebuilds_finish_in_the_same_update() {
        let mut app = test_app();
        app.update();
        let (loaded, loading) = {
            let spawned = app.world().resource::<SpawnedSkeletons>();
            (spawned.loaded, spawned.loading)
        };
        assert_ready_and_renderable(&mut app, loaded);
        assert!(!app.world().entity(loading).contains::<Spine>());

        let initial_helpers = spine_helpers(app.world(), loaded);
        let unrelated_parent = app.world_mut().spawn_empty().id();
        let authored_child = app.world_mut().spawn(ChildOf(loaded)).id();
        reparent(app.world_mut(), &initial_helpers, unrelated_parent);
        app.world_mut().insert_resource(ReplaceSkeleton(loaded));
        app.update();
        assert_ready_and_renderable(&mut app, loaded);
        assert_despawned(app.world(), &initial_helpers);
        assert!(app.world().entities().contains(authored_child));

        let replacement_helpers = spine_helpers(app.world(), loaded);
        reparent(app.world_mut(), &replacement_helpers, unrelated_parent);
        app.world_mut().entity_mut(loaded).remove::<Spine>();
        app.world_mut().insert_resource(FinishLoading(loaded));
        app.update();
        assert_ready_and_renderable(&mut app, loaded);
        assert_despawned(app.world(), &replacement_helpers);
        assert!(app.world().entities().contains(authored_child));

        let rebuilt_helpers = spine_helpers(app.world(), loaded);
        reparent(app.world_mut(), &rebuilt_helpers, unrelated_parent);
        app.world_mut()
            .entity_mut(loaded)
            .remove::<SkeletonDataHandle>();
        app.update();
        assert!(!app.world().entity(loaded).contains::<Spine>());
        assert!(!app.world().entity(loaded).contains::<SpineHelpers>());
        assert_despawned(app.world(), &rebuilt_helpers);
        assert!(app.world().entities().contains(authored_child));
        assert!(app.world().entities().contains(unrelated_parent));
    }

    #[test]
    fn loading_request_initializes_after_the_asset_becomes_loaded() {
        let mut app = test_app();
        app.update();
        app.update();
        let loading = app.world().resource::<SpawnedSkeletons>().loading;
        assert!(!app.world().entity(loading).contains::<Spine>());
        assert!(!app.world().entity(loading).contains::<SpineHelpers>());

        app.world_mut().insert_resource(FinishLoading(loading));
        app.update();
        assert_ready_and_renderable(&mut app, loading);
        assert!(
            spine_helpers(app.world(), loading)
                .iter()
                .all(|entity| !app.world().entity(*entity).contains::<SpineBone>())
        );
    }
}
