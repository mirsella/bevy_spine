//! A Bevy plugin for Spine 4.2
//!
//! Add [`SpineCorePlugin`] to your Bevy app and spawn a [`SkeletonDataHandle`] to get started.
//! Add [`SpineDefaultMaterialPlugin`] too when using the built-in 2D materials.

use std::{
    collections::{HashMap, VecDeque},
    mem::take,
    sync::{Arc, Mutex},
};

use bevy::{
    asset::{RenderAssetUsages, load_internal_binary_asset},
    camera::visibility::RenderLayers,
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    mesh::{Indices, MeshVertexAttribute, VertexAttributeValues},
    prelude::*,
    render::batching::NoAutomaticBatching,
    render::render_resource::{PrimitiveTopology, VertexFormat},
    sprite_render::Material2dPlugin,
};
use direct_render::{SpineDirectMesh, SpineDirectVertex};
use materials::{
    SpineAdditiveMaterial, SpineAdditivePmaMaterial, SpineMaterialInfo, SpineMultiplyMaterial,
    SpineMultiplyPmaMaterial, SpineNormalMaterial, SpineNormalPmaMaterial, SpineScreenMaterial,
    SpineScreenPmaMaterial,
};
use rusty_spine::{
    AnimationEvent, BlendMode, Physics, Skeleton,
    atlas::{AtlasFilter, AtlasWrap},
    controller::{SkeletonCombinedRenderable, SkeletonRenderable},
};
use textures::SpineTextureConfig;

use crate::{
    assets::{AtlasLoader, SkeletonJsonLoader},
    materials::{DARK_COLOR_ATTRIBUTE, SHADER_HANDLE, SpineMaterialPlugin},
    rusty_spine::{
        AnimationStateData, BoneHandle, controller::SkeletonControllerSettings, draw::CullDirection,
    },
    textures::{SpineTextureCreateEvent, SpineTextures},
};

const SPINE_POSITION_ATTRIBUTE: MeshVertexAttribute =
    MeshVertexAttribute::new("Vertex_Position", 0, VertexFormat::Float32x2);

pub use crate::{assets::*, crossfades::Crossfades, entity_sync::*, handle::*, rusty_spine::Color};

/// See [`rusty_spine`] docs for more info.
pub use crate::rusty_spine::controller::SkeletonController;

#[cfg(feature = "ui")]
pub use crate::ui::*;

pub use rusty_spine;

fn required_mesh_count(drawer: SpineDrawer, slot_count: usize, renderable_count: usize) -> usize {
    match drawer {
        SpineDrawer::None => 0,
        SpineDrawer::Combined => renderable_count.max(1),
        SpineDrawer::Separated => slot_count.max(1),
    }
}

/// System sets for Spine systems.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, SystemSet)]
pub enum SpineSystem {
    /// Loads [`SkeletonData`] assets which must exist before a [`SkeletonDataHandle`] can fully
    /// load.
    Load,
    /// Spawns helper entities associated with entities containing [`SkeletonDataHandle`] for
    /// drawing meshes and (optionally) adding bone entities (see [`SpineLoader`]).
    Spawn,
    /// An [`bevy::ecs::schedule::ApplyDeferred`] to load the spine helper entities this frame.
    SpawnFlush,
    /// Sends [`SpineReadyEvent`] after [`SpineSystem::SpawnFlush`], indicating [`Spine`]
    /// components on newly spawned entities can now be interacted with.
    Ready,
    /// Advances all animations and processes Spine events (see [`SpineEvent`]).
    UpdateAnimation,
    /// Updates all Spine meshes.
    UpdateMeshes,
    /// Updates all Spine materials.
    UpdateMaterials,
    /// Adjusts Spine textures to render properly.
    AdjustSpineTextures,
}

/// Helper sets for interacting with Spine systems.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, SystemSet)]
pub enum SpineSet {
    /// A helper Set occuring after [`SpineSystem::Ready`] but before Spine update systems, so that
    /// systems can configure a newly spawned skeleton before they are updated for the first time.
    OnReady,
    /// A helper Set occuring after [`SpineSystem::UpdateAnimation`] but before
    /// [`SpineSystem::UpdateMeshes`], so that systems can handle events immediately after the
    /// skeleton updates but before it renders.
    OnEvent,
    /// A helper set occuring simultaneously with [`SpineSystem::UpdateMeshes`], useful for custom
    /// mesh creation when using [`SpineDrawer::None`].
    OnUpdateMesh,
}

/// Core Spine loading, animation, mesh, texture, and optional UI support.
///
/// ```
/// # use bevy::prelude::*;
/// # use bevy_spine::{SpineCorePlugin, SpineDefaultMaterialPlugin};
/// # fn doc() {
/// App::new()
///     .add_plugins(DefaultPlugins)
///     .add_plugins((SpineCorePlugin, SpineDefaultMaterialPlugin))
///     // ...
///     .run();
/// # }
/// ```
/// Use this with a custom [`SpineMaterial`](`materials::SpineMaterial`) to avoid running the
/// default material update systems every frame.
pub struct SpineCorePlugin;

impl Plugin for SpineCorePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(SpineSyncPlugin::first())
            .add_plugins(direct_render::SpineDirectRenderPlugin)
            .register_type::<Crossfades>()
            .register_type::<SkeletonDataHandle>()
            .register_type::<SpineSync>()
            .register_type::<Spine>()
            .register_type::<SpineBone>()
            .register_type::<SpineMeshes>()
            .register_type::<SpineMesh>()
            .register_type::<SpineMeshState>()
            .register_type::<SpineLoader>()
            .register_type::<SpineSettings>()
            .register_type::<SpineMeshType>()
            .register_type::<SpineDrawer>()
            .init_resource::<SpineEventQueue>()
            .init_resource::<SpineTextureHandleCache>()
            .insert_resource(SpineTextures::init())
            .insert_resource(SpineReadyEvents::default())
            .add_message::<SpineTextureCreateEvent>()
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
            .add_systems(
                Update,
                (
                    spine_load.in_set(SpineSystem::Load),
                    spine_spawn
                        .in_set(SpineSystem::Spawn)
                        .after(SpineSystem::Load),
                    spine_ready
                        .in_set(SpineSystem::Ready)
                        .after(SpineSystem::Spawn)
                        .before(SpineSet::OnReady),
                    spine_update_animation
                        .in_set(SpineSystem::UpdateAnimation)
                        .after(SpineSet::OnReady)
                        .before(SpineSet::OnEvent),
                    spine_update_meshes
                        .in_set(SpineSystem::UpdateMeshes)
                        .in_set(SpineSet::OnUpdateMesh)
                        .after(SpineSystem::UpdateAnimation)
                        .after(SpineSet::OnEvent),
                    ApplyDeferred
                        .in_set(SpineSystem::SpawnFlush)
                        .after(SpineSystem::Spawn)
                        .before(SpineSystem::Ready),
                ),
            )
            .add_systems(
                PostUpdate,
                adjust_spine_textures.in_set(SpineSystem::AdjustSpineTextures),
            );

        #[cfg(feature = "ui")]
        app.add_plugins(ui::SpineUiPlugin);

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

/// Built-in Spine 2D materials.
pub struct SpineDefaultMaterialPlugin;

impl Plugin for SpineDefaultMaterialPlugin {
    fn build(&self, app: &mut App) {
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
        ));
    }
}

#[derive(Resource, Default)]
struct SpineEventQueue(Arc<Mutex<VecDeque<SpineEvent>>>);

#[derive(Resource, Default)]
struct SpineTextureHandleCache(HashMap<String, Handle<Image>>);

impl SpineTextureHandleCache {
    fn load(&mut self, asset_server: &AssetServer, path: &str) -> Handle<Image> {
        self.0
            .entry(path.to_owned())
            .or_insert_with(|| asset_server.load(path.to_owned()))
            .clone()
    }
}

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
/// To disable creation of these child entities, see [`SpineLoader::without_children`].
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
    initialized: bool,
    culled_frames: u32,
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

/// The async loader for Spine assets. Waits for Spine assets to be ready in the [`AssetServer`],
/// then initializes child entities, and finally attaches the live [`Spine`] component.
///
/// When spawning a [`SkeletonDataHandle`], a [`SpineLoader`] is added automatically. It will create
/// child entities representing the bones of a skeleton (see [`SpineBone`]). These bones are not
/// synchronized (see [`SpineSync`]), and can be disabled entirely using
/// [`SpineLoader::without_children`].
#[derive(Component, Debug, Reflect)]
#[reflect(Component, Debug)]
pub enum SpineLoader {
    /// The spine rig is still loading.
    Loading {
        /// If true, will spawn child entities for each bone in the skeleton (see [`SpineBone`]).
        with_children: bool,
    },
    /// The spine rig is ready.
    Ready,
    /// The spine rig failed to load.
    Failed,
}

impl Default for SpineLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl SpineLoader {
    pub fn new() -> Self {
        Self::with_children()
    }

    pub fn with_children() -> Self {
        Self::Loading {
            with_children: true,
        }
    }

    /// Load a [`Spine`] entity without child entities containing [`SpineBone`] components.
    ///
    /// Renderable mesh child entities are still created.
    ///
    /// ```
    /// # use bevy::prelude::*;
    /// # use bevy_spine::{SkeletonDataHandle, SpineLoader};
    /// # fn doc(mut commands: Commands) {
    /// commands.spawn((
    ///     SkeletonDataHandle::default(),
    ///     SpineLoader::without_children(),
    /// ));
    /// # }
    /// ```
    pub fn without_children() -> Self {
        Self::Loading {
            with_children: false,
        }
    }
}

/// Settings for how this Spine updates and renders.
///
/// Typically set alongside [`SkeletonDataHandle`] when spawning an entity.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Reflect)]
#[reflect(Component, Debug, PartialEq, Clone)]
pub struct SpineSettings {
    /// Indicates if default Spine materials should be used (default: `true`).
    ///
    /// If `false`, a custom [`SpineMaterial`](`materials::SpineMaterial`) should be configured for
    /// this Spine.
    pub default_materials: bool,
    /// Indicates how the meshes should be drawn.
    pub mesh_type: SpineMeshType,
    /// The drawer this Spine should use to create its meshes.
    pub drawer: SpineDrawer,
    /// Keep rebuilding meshes even when all mesh children are currently out of view.
    ///
    /// Defaults to `false` to reduce CPU work for large numbers of off-screen skeletons.
    /// Set this to `true` if off-screen meshes must stay fully up to date.
    pub update_meshes_when_invisible: bool,
    /// Upload 2D Spine geometry through a direct render path instead of mutating [`Mesh`] assets.
    ///
    /// This avoids per-frame mesh asset events and GPU mesh re-extraction for animated skeletons.
    /// It is only used with [`SpineMeshType::Mesh2D`]; 3D meshes continue through Bevy meshes.
    pub direct_2d_rendering: bool,
}

#[derive(Component, Clone, Copy, Debug)]
pub(crate) struct SpineRenderOwner;

/// Mesh types to use in [`SpineSettings`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Reflect)]
#[reflect(Debug, PartialEq, Clone)]
pub enum SpineMeshType {
    /// Render meshes in 2D.
    Mesh2D,
    /// Render meshes in 3D. Requires a custom [`SpineMaterial`](`materials::SpineMaterial`) since
    /// the default materials do not support 3D meshes.
    Mesh3D,
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
            default_materials: true,
            mesh_type: SpineMeshType::Mesh2D,
            drawer: SpineDrawer::Combined,
            update_meshes_when_invisible: false,
            direct_2d_rendering: false,
        }
    }
}

/// A [`Message`] which is sent once a [`SpineLoader`] has fully loaded a skeleton and attached the
/// [`Spine`] component.
///
/// For convenience, systems receiving this event can be added to the [`SpineSet::OnReady`] set to
/// receive this after events are sent, but before the first [`SkeletonController`] update.
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
///     mut spine_events: EventReader<SpineEvent>,
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

/// Queued ready events, to be sent after [`SpineSystem::SpawnFlush`].
#[derive(Default, Resource)]
struct SpineReadyEvents(Vec<SpineReadyEvent>);

#[allow(clippy::too_many_arguments)]
fn spine_load(
    mut skeleton_data_assets: ResMut<Assets<SkeletonData>>,
    mut texture_create_events: MessageWriter<SpineTextureCreateEvent>,
    mut atlases: ResMut<Assets<Atlas>>,
    jsons: Res<Assets<SkeletonJson>>,
    binaries: Res<Assets<SkeletonBinary>>,
    mut spine_textures: ResMut<SpineTextures>,
    asset_server: Res<AssetServer>,
) {
    // check if any assets are loading, else, early out to avoid triggering change detection
    let mut loading = false;
    for (_, skeleton_data_asset) in skeleton_data_assets.iter() {
        if matches!(skeleton_data_asset.status, SkeletonDataStatus::Loading) {
            loading = true;
            break;
        }
    }
    if loading {
        for (_, skeleton_data_asset) in skeleton_data_assets.iter_mut() {
            let SkeletonData {
                atlas_handle,
                kind,
                status,
                premultiplied_alpha,
            } = skeleton_data_asset;
            if matches!(status, SkeletonDataStatus::Loading) {
                let atlas = if let Some(atlas) = atlases.get(atlas_handle) {
                    atlas
                } else {
                    continue;
                };
                if let Some(page) = atlas.atlas.pages().next() {
                    *premultiplied_alpha = page.pma();
                }
                match kind {
                    SkeletonDataKind::JsonFile(json_handle) => {
                        let json = if let Some(json) = jsons.get(json_handle) {
                            json
                        } else {
                            continue;
                        };
                        let skeleton_json = rusty_spine::SkeletonJson::new(atlas.atlas.clone());
                        match skeleton_json.read_skeleton_data(&json.json) {
                            Ok(skeleton_data) => {
                                *status = SkeletonDataStatus::Loaded(Arc::new(skeleton_data));
                            }
                            Err(_err) => {
                                *status = SkeletonDataStatus::Failed;
                                continue;
                            }
                        }
                    }
                    SkeletonDataKind::BinaryFile(binary_handle) => {
                        let binary = if let Some(binary) = binaries.get(binary_handle) {
                            binary
                        } else {
                            continue;
                        };
                        let skeleton_binary = rusty_spine::SkeletonBinary::new(atlas.atlas.clone());
                        match skeleton_binary.read_skeleton_data(&binary.binary) {
                            Ok(skeleton_data) => {
                                *status = SkeletonDataStatus::Loaded(Arc::new(skeleton_data));
                            }
                            Err(_err) => {
                                // TODO: print error?
                                *status = SkeletonDataStatus::Failed;
                                continue;
                            }
                        }
                    }
                }
            }
        }
    }

    spine_textures.update(
        asset_server.as_ref(),
        &mut atlases,
        &mut texture_create_events,
    );
}

#[allow(clippy::too_many_arguments)]
fn spine_spawn(
    mut skeleton_query: Query<(
        &mut SpineLoader,
        Entity,
        &SkeletonDataHandle,
        Option<&SpineSettings>,
        Option<&Crossfades>,
        Option<&RenderLayers>,
        Option<&SpineRenderOwner>,
    )>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut ready_events: ResMut<SpineReadyEvents>,
    mut skeleton_data_assets: ResMut<Assets<SkeletonData>>,
    spine_event_queue: Res<SpineEventQueue>,
) {
    for (
        mut spine_loader,
        spine_entity,
        data_handle,
        spine_settings,
        crossfades,
        render_layers,
        render_owner,
    ) in skeleton_query.iter_mut()
    {
        if let SpineLoader::Loading { with_children } = spine_loader.as_ref() {
            let skeleton_data_asset =
                if let Some(skeleton_data_asset) = skeleton_data_assets.get_mut(&data_handle.0) {
                    skeleton_data_asset
                } else {
                    continue;
                };
            match &skeleton_data_asset.status {
                SkeletonDataStatus::Loaded(skeleton_data) => {
                    let mut animation_state_data = AnimationStateData::new(skeleton_data.clone());
                    if let Some(crossfades) = crossfades {
                        crossfades.apply(&mut animation_state_data);
                    }
                    let mut controller = SkeletonController::new(
                        skeleton_data.clone(),
                        Arc::new(animation_state_data),
                    )
                    .with_settings(
                        SkeletonControllerSettings::new()
                            .with_cull_direction(CullDirection::CounterClockwise)
                            .with_premultiplied_alpha(skeleton_data_asset.premultiplied_alpha),
                    );
                    let settings = spine_settings.copied().unwrap_or_default();
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
                    let mut bones = HashMap::new();
                    let render_layers = render_layers.cloned();
                    let render_owner = render_owner.copied();
                    if let Ok(mut entity_commands) = commands.get_entity(spine_entity) {
                        entity_commands
                            .with_children(|parent| {
                                // TODO: currently, a mesh is created for each slot, however when we use the
                                // combined drawer, this many meshes is usually not necessary. instead, we
                                // may want to dynamically create meshes as needed in the render system
                                let render_layers_for_children = render_layers.clone();
                                let mut spine_meshes_commands = parent.spawn((
                                    Name::new("spine_meshes"),
                                    SpineMeshes,
                                    SpineMeshesUpdateState::default(),
                                    Transform::from_xyz(0., 0., 0.),
                                    GlobalTransform::default(),
                                    Visibility::default(),
                                    InheritedVisibility::default(),
                                    ViewVisibility::default(),
                                ));

                                if let Some(render_layers) = &render_layers_for_children {
                                    spine_meshes_commands.insert(render_layers.clone());
                                }
                                if let Some(render_owner) = render_owner {
                                    spine_meshes_commands.insert(render_owner);
                                }

                                spine_meshes_commands.with_children(|parent| {
                                    let render_layers_for_meshes =
                                        render_layers_for_children.clone();
                                    let initial_mesh_count = required_mesh_count(
                                        settings.drawer,
                                        controller.skeleton.slots().count(),
                                        controller.combined_renderables().len(),
                                    );
                                    spawn_spine_mesh_children(
                                        parent,
                                        &mut meshes,
                                        spine_entity,
                                        initial_mesh_count,
                                        &render_layers_for_meshes,
                                        render_owner,
                                        0,
                                    );
                                });
                                if *with_children {
                                    spawn_bones(
                                        spine_entity,
                                        None,
                                        parent,
                                        &controller.skeleton,
                                        controller.skeleton.bone_root().handle(),
                                        render_layers_for_children.as_ref(),
                                        render_owner.as_ref(),
                                        &mut bones,
                                    );
                                }
                            })
                            .insert(Spine(controller));
                    }
                    *spine_loader = SpineLoader::Ready;
                    ready_events.0.push(SpineReadyEvent {
                        entity: spine_entity,
                        bones,
                    });
                }
                SkeletonDataStatus::Loading => {}
                SkeletonDataStatus::Failed => {
                    *spine_loader = SpineLoader::Failed;
                }
            }
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
    render_owner: Option<&SpineRenderOwner>,
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
            transform,
            GlobalTransform::default(),
            Visibility::default(),
            InheritedVisibility::default(),
            ViewVisibility::default(),
        ));

        if let Some(render_layers) = render_layers {
            bone_entity_commands.insert(render_layers.clone());
        }
        if let Some(render_owner) = render_owner {
            bone_entity_commands.insert(*render_owner);
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
                        render_owner,
                        bones,
                    );
                }
            })
            .id();
        bones.insert(bone.data().name().to_owned(), bone_entity);
    }
}

fn spawn_spine_mesh_children(
    spawner: &mut ChildSpawnerCommands<'_>,
    meshes: &mut Assets<Mesh>,
    spine_entity: Entity,
    mesh_count: usize,
    render_layers: &Option<RenderLayers>,
    render_owner: Option<SpineRenderOwner>,
    start_index: usize,
) {
    let mut z = start_index as f32 * 0.001;
    for index in start_index..start_index + mesh_count {
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
        );
        empty_mesh(&mut mesh);
        let mesh_handle = meshes.add(mesh);
        let mut mesh_commands = spawner.spawn((
            Name::new(format!("spine_mesh {index}")),
            SpineMesh {
                spine_entity,
                handle: mesh_handle.clone(),
                state: SpineMeshState::Empty,
            },
            Transform::from_xyz(0., 0., z),
            GlobalTransform::default(),
            Visibility::default(),
            InheritedVisibility::default(),
            ViewVisibility::default(),
        ));

        if let Some(render_layers) = render_layers {
            mesh_commands.insert(render_layers.clone());
        }
        if let Some(render_owner) = render_owner {
            mesh_commands.insert(render_owner);
        }

        z += 0.001;
    }
}

fn spine_ready(
    mut ready_events: ResMut<SpineReadyEvents>,
    mut ready_writer: MessageWriter<SpineReadyEvent>,
) {
    for event in take(&mut ready_events.0).into_iter() {
        ready_writer.write(event);
    }
}

fn spine_update_animation(
    mut spine_query: Query<(Entity, &mut Spine)>,
    mut spine_events: MessageWriter<SpineEvent>,
    time: Res<Time>,
    spine_event_queue: Res<SpineEventQueue>,
) {
    for (_, mut spine) in spine_query.iter_mut() {
        spine.update(time.delta_secs(), Physics::Update);
    }
    {
        let mut events = spine_event_queue.0.lock().unwrap();
        while let Some(event) = events.pop_front() {
            spine_events.write(event);
        }
    }
}

enum SpineRenderables {
    Simple(Vec<SkeletonRenderable>),
    Combined(Vec<SkeletonCombinedRenderable>),
}

#[derive(Clone, Copy)]
enum SpineVertexColors<'a> {
    Fill([f32; 4], usize),
    Slice(&'a [[f32; 4]]),
}

impl SpineVertexColors<'_> {
    fn len(self) -> usize {
        match self {
            Self::Fill(_, len) => len,
            Self::Slice(values) => values.len(),
        }
    }

    fn get(self, index: usize) -> [f32; 4] {
        match self {
            Self::Fill(color, _) => color,
            Self::Slice(values) => values[index],
        }
    }
}

struct SpineRenderableRef<'a> {
    slot_index: Option<usize>,
    attachment_renderer_object: Option<*const rusty_spine::c::c_void>,
    vertices: &'a [[f32; 2]],
    indices: &'a [u16],
    uvs: &'a [[f32; 2]],
    colors: SpineVertexColors<'a>,
    dark_colors: SpineVertexColors<'a>,
    blend_mode: BlendMode,
    premultiplied_alpha: bool,
}

impl SpineRenderables {
    fn len(&self) -> usize {
        match self {
            Self::Simple(renderables) => renderables.len(),
            Self::Combined(renderables) => renderables.len(),
        }
    }

    fn get(&self, index: usize) -> Option<SpineRenderableRef<'_>> {
        match self {
            Self::Simple(renderables) => {
                let renderable = renderables.get(index)?;
                let color = [
                    renderable.color.r,
                    renderable.color.g,
                    renderable.color.b,
                    renderable.color.a,
                ];
                let dark_color = [
                    renderable.dark_color.r,
                    renderable.dark_color.g,
                    renderable.dark_color.b,
                    renderable.dark_color.a,
                ];

                Some(SpineRenderableRef {
                    slot_index: Some(renderable.slot_index),
                    attachment_renderer_object: renderable.attachment_renderer_object,
                    vertices: renderable.vertices.as_slice(),
                    indices: renderable.indices.as_slice(),
                    uvs: renderable.uvs.as_slice(),
                    colors: SpineVertexColors::Fill(color, renderable.vertices.len()),
                    dark_colors: SpineVertexColors::Fill(dark_color, renderable.vertices.len()),
                    blend_mode: renderable.blend_mode,
                    premultiplied_alpha: renderable.premultiplied_alpha,
                })
            }
            Self::Combined(renderables) => {
                let renderable = renderables.get(index)?;

                Some(SpineRenderableRef {
                    slot_index: None,
                    attachment_renderer_object: renderable.attachment_renderer_object,
                    vertices: renderable.vertices.as_slice(),
                    indices: renderable.indices.as_slice(),
                    uvs: renderable.uvs.as_slice(),
                    colors: SpineVertexColors::Slice(renderable.colors.as_slice()),
                    dark_colors: SpineVertexColors::Slice(renderable.dark_colors.as_slice()),
                    blend_mode: renderable.blend_mode,
                    premultiplied_alpha: renderable.premultiplied_alpha,
                })
            }
        }
    }
}

fn combined_renderable_has_mesh(renderable: &SkeletonCombinedRenderable) -> bool {
    // rusty_spine's combined drawer may emit an empty leading renderable when
    // early draw-order slots are hidden. It should not allocate or shift meshes.
    renderable.attachment_renderer_object.is_some()
        && !renderable.vertices.is_empty()
        && !renderable.indices.is_empty()
}

#[allow(clippy::type_complexity)]
fn spine_update_meshes(
    mut spine_query: Query<(&mut Spine, Option<&SpineSettings>, &InheritedVisibility)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mesh_query: Query<(
        Entity,
        &mut SpineMesh,
        &mut Transform,
        Option<&Mesh2d>,
        Option<&Mesh3d>,
        Option<&mut SpineDirectMesh>,
    )>,
    mesh_visibility_query: Query<&ViewVisibility, With<SpineMesh>>,
    mut commands: Commands,
    mut meshes_query: Query<
        (
            Entity,
            &ChildOf,
            &Children,
            &mut SpineMeshesUpdateState,
            Option<&RenderLayers>,
            Option<&SpineRenderOwner>,
        ),
        With<SpineMeshes>,
    >,
    asset_server: Res<AssetServer>,
    mut texture_handle_cache: ResMut<SpineTextureHandleCache>,
) {
    const CULLED_RECOVERY_INTERVAL_FRAMES: u32 = 60;

    for (
        meshes_entity,
        meshes_parent,
        meshes_children,
        mut update_state,
        render_layers,
        render_owner,
    ) in meshes_query.iter_mut()
    {
        let Ok((mut spine, spine_mesh_type, inherited_visibility)) =
            spine_query.get_mut(meshes_parent.parent())
        else {
            continue;
        };

        if !inherited_visibility.get() {
            continue;
        }

        let SpineSettings {
            mesh_type,
            drawer,
            update_meshes_when_invisible,
            direct_2d_rendering,
            ..
        } = spine_mesh_type.cloned().unwrap_or(SpineSettings::default());
        let use_direct_2d_rendering = direct_2d_rendering && mesh_type == SpineMeshType::Mesh2D;

        if !update_meshes_when_invisible && update_state.initialized {
            let any_visible = meshes_children.iter().any(|child| {
                mesh_visibility_query
                    .get(child)
                    .is_ok_and(|visibility| visibility.get())
            });
            if !any_visible {
                update_state.culled_frames = update_state.culled_frames.saturating_add(1);

                if update_state.culled_frames < CULLED_RECOVERY_INTERVAL_FRAMES {
                    continue;
                }
            } else {
                update_state.culled_frames = 0;
            }
        }

        let renderables = match drawer {
            SpineDrawer::Combined => {
                let mut renderables = spine.0.combined_renderables();
                renderables.retain(combined_renderable_has_mesh);
                SpineRenderables::Combined(renderables)
            }
            SpineDrawer::Separated => SpineRenderables::Simple(spine.0.renderables()),
            SpineDrawer::None => continue,
        };
        let required_mesh_count =
            required_mesh_count(drawer, spine.skeleton.slots().count(), renderables.len());
        let dynamic_mesh_children = drawer == SpineDrawer::Combined;
        let existing_mesh_count = meshes_children.len();
        if dynamic_mesh_children && existing_mesh_count < required_mesh_count {
            if let Ok(mut meshes_entity_commands) = commands.get_entity(meshes_entity) {
                meshes_entity_commands.with_children(|parent| {
                    spawn_spine_mesh_children(
                        parent,
                        &mut meshes,
                        meshes_parent.parent(),
                        required_mesh_count - existing_mesh_count,
                        &render_layers.cloned(),
                        render_owner.copied(),
                        existing_mesh_count,
                    );
                });
            }
        }
        if dynamic_mesh_children {
            for extra_child in meshes_children.iter().skip(required_mesh_count) {
                if let Ok(mut entity) = commands.get_entity(extra_child) {
                    entity.try_despawn();
                }
            }
        }
        let mut z = 0.;
        let mut renderable_index = 0;
        for child in meshes_children.iter() {
            if dynamic_mesh_children && renderable_index >= required_mesh_count {
                break;
            }
            if let Ok((
                spine_mesh_entity,
                mut spine_mesh,
                mut spine_mesh_transform,
                spine_2d_mesh,
                spine_3d_mesh,
                mut direct_mesh,
            )) = mesh_query.get_mut(child)
            {
                macro_rules! apply_mesh {
                    ($mesh:ident, $condition:expr, $attach:expr, $deattach:ty) => {
                        if $condition {
                            if !$mesh.is_some() {
                                if let Ok(mut entity) = commands.get_entity(spine_mesh_entity) {
                                    entity.insert($attach);
                                }
                            }
                        } else {
                            if $mesh.is_some() {
                                if let Ok(mut entity) = commands.get_entity(spine_mesh_entity) {
                                    entity.remove::<$deattach>();
                                }
                            }
                        }
                    };
                }
                if use_direct_2d_rendering {
                    if spine_2d_mesh.is_none_or(|mesh| mesh.0 != Handle::<Mesh>::default()) {
                        if let Ok(mut entity) = commands.get_entity(spine_mesh_entity) {
                            entity.insert((Mesh2d(Handle::<Mesh>::default()), NoAutomaticBatching));
                        }
                    }
                } else {
                    apply_mesh!(
                        spine_2d_mesh,
                        mesh_type == SpineMeshType::Mesh2D,
                        Mesh2d(spine_mesh.handle.clone()),
                        Mesh2d
                    );
                    if direct_mesh.is_some()
                        && let Ok(mut entity) = commands.get_entity(spine_mesh_entity)
                    {
                        entity.remove::<(SpineDirectMesh, NoAutomaticBatching)>();
                    }
                }
                apply_mesh!(
                    spine_3d_mesh,
                    mesh_type == SpineMeshType::Mesh3D && !use_direct_2d_rendering,
                    Mesh3d(spine_mesh.handle.clone()),
                    Mesh3d
                );
                let rendered = if let Some(renderable) = renderables.get(renderable_index) {
                    if let Some(attachment_render_object) = renderable.attachment_renderer_object {
                        let texture_path = unsafe { &*(attachment_render_object as *const String) };
                        let texture_handle = texture_handle_cache.load(&asset_server, texture_path);
                        let mesh_updated = if use_direct_2d_rendering {
                            if let Some(direct_mesh) = direct_mesh.as_deref_mut() {
                                set_direct_mesh_data(direct_mesh, &renderable)
                            } else {
                                let mut next_direct_mesh = SpineDirectMesh::default();
                                let updated =
                                    set_direct_mesh_data(&mut next_direct_mesh, &renderable);
                                if updated
                                    && let Ok(mut entity) = commands.get_entity(spine_mesh_entity)
                                {
                                    entity.insert(next_direct_mesh);
                                }
                                updated
                            }
                        } else if let Some(mesh) = meshes.get_mut(&spine_mesh.handle) {
                            set_u16_indices(mesh, renderable.indices);
                            set_float32x2_attribute(
                                mesh,
                                SPINE_POSITION_ATTRIBUTE,
                                renderable.vertices,
                            );
                            set_zero_normals(mesh, renderable.vertices.len());
                            set_float32x2_attribute(mesh, Mesh::ATTRIBUTE_UV_0, renderable.uvs);
                            set_float32x4_attribute(mesh, Mesh::ATTRIBUTE_COLOR, renderable.colors);
                            set_float32x4_attribute(
                                mesh,
                                DARK_COLOR_ATTRIBUTE,
                                renderable.dark_colors,
                            );
                            true
                        } else {
                            warn!(
                                "Spine mesh asset {:?} is missing; skipping mesh update",
                                spine_mesh.handle
                            );
                            false
                        };
                        if mesh_updated {
                            spine_mesh.state = SpineMeshState::Renderable {
                                info: SpineMaterialInfo {
                                    slot_index: renderable.slot_index,
                                    texture: texture_handle,
                                    blend_mode: renderable.blend_mode,
                                    premultiplied_alpha: renderable.premultiplied_alpha,
                                },
                            };
                            spine_mesh_transform.translation.z = z;
                            z += 0.001;
                        }
                        mesh_updated
                    } else {
                        false
                    }
                } else {
                    false
                };
                if !rendered {
                    if !matches!(spine_mesh.state, SpineMeshState::Empty) {
                        spine_mesh.state = SpineMeshState::Empty;
                        if use_direct_2d_rendering {
                            if let Some(direct_mesh) = direct_mesh.as_deref_mut() {
                                direct_mesh.vertices.clear();
                                direct_mesh.indices.clear();
                            }
                        } else if let Some(mesh) = meshes.get_mut(&spine_mesh.handle) {
                            empty_mesh(mesh);
                        }
                    }
                }
                renderable_index += 1;
            }
        }

        update_state.initialized = true;
        update_state.culled_frames = 0;
    }
}

fn set_direct_mesh_data(mesh: &mut SpineDirectMesh, renderable: &SpineRenderableRef<'_>) -> bool {
    let vertex_count = renderable.vertices.len();
    if renderable.uvs.len() != vertex_count {
        warn!(
            "Spine renderable has {} vertices but {} uvs; skipping direct mesh update",
            vertex_count,
            renderable.uvs.len()
        );
        mesh.vertices.clear();
        mesh.indices.clear();
        return false;
    }
    if renderable.colors.len() != vertex_count || renderable.dark_colors.len() != vertex_count {
        warn!(
            "Spine renderable has mismatched color data for {} vertices; skipping direct mesh update",
            vertex_count
        );
        mesh.vertices.clear();
        mesh.indices.clear();
        return false;
    }

    mesh.vertices.clear();
    mesh.vertices.reserve(vertex_count);
    for index in 0..vertex_count {
        let position = renderable.vertices[index];
        mesh.vertices.push(SpineDirectVertex {
            position: [position[0], position[1], 0.0],
            normal: [0.0, 0.0, 0.0],
            uv: renderable.uvs[index],
            color: renderable.colors.get(index),
            dark_color: renderable.dark_colors.get(index),
        });
    }

    mesh.indices.clear();
    mesh.indices.extend_from_slice(renderable.indices);
    true
}

fn set_u16_indices(mesh: &mut Mesh, indices: &[u16]) {
    if let Some(Indices::U16(current)) = mesh.indices_mut() {
        current.clear();
        current.extend_from_slice(indices);
    } else {
        mesh.insert_indices(Indices::U16(indices.to_vec()));
    }
}

fn set_float32x2_attribute(mesh: &mut Mesh, attribute: MeshVertexAttribute, values: &[[f32; 2]]) {
    if let Some(VertexAttributeValues::Float32x2(current)) = mesh.attribute_mut(attribute.id) {
        current.clear();
        current.extend_from_slice(values);
    } else {
        mesh.insert_attribute(attribute, values.to_vec());
    }
}

fn set_float32x3_attribute(mesh: &mut Mesh, attribute: MeshVertexAttribute, values: &[[f32; 3]]) {
    if let Some(VertexAttributeValues::Float32x3(current)) = mesh.attribute_mut(attribute.id) {
        current.clear();
        current.extend_from_slice(values);
    } else {
        mesh.insert_attribute(attribute, values.to_vec());
    }
}

fn set_float32x4_attribute(
    mesh: &mut Mesh,
    attribute: MeshVertexAttribute,
    values: SpineVertexColors<'_>,
) {
    if let Some(VertexAttributeValues::Float32x4(current)) = mesh.attribute_mut(attribute.id) {
        match values {
            SpineVertexColors::Fill(value, len) => {
                current.clear();
                current.resize(len, value);
            }
            SpineVertexColors::Slice(values) => {
                current.clear();
                current.extend_from_slice(values);
            }
        }
    } else {
        let values = match values {
            SpineVertexColors::Fill(value, len) => vec![value; len],
            SpineVertexColors::Slice(values) => values.to_vec(),
        };
        mesh.insert_attribute(attribute, values);
    }
}

fn set_zero_normals(mesh: &mut Mesh, len: usize) {
    if let Some(VertexAttributeValues::Float32x3(current)) =
        mesh.attribute_mut(Mesh::ATTRIBUTE_NORMAL.id)
    {
        current.clear();
        current.resize(len, [0.0, 0.0, 0.0]);
    } else {
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 0.0, 0.0]; len]);
    }
}

fn empty_mesh(mesh: &mut Mesh) {
    const EMPTY_VERTEX_COUNT: usize = 3;
    const EMPTY_VEC3: [[f32; 3]; EMPTY_VERTEX_COUNT] = [[0.0, 0.0, 0.0]; EMPTY_VERTEX_COUNT];
    const EMPTY_UVS: [[f32; 2]; EMPTY_VERTEX_COUNT] = [[0.0, 0.0]; EMPTY_VERTEX_COUNT];
    const TRANSPARENT: [f32; 4] = [0.0, 0.0, 0.0, 0.0];

    mesh.remove_indices();
    set_float32x3_attribute(mesh, Mesh::ATTRIBUTE_POSITION, &EMPTY_VEC3);
    set_zero_normals(mesh, EMPTY_VERTEX_COUNT);
    set_float32x2_attribute(mesh, Mesh::ATTRIBUTE_UV_0, &EMPTY_UVS);
    set_float32x4_attribute(
        mesh,
        Mesh::ATTRIBUTE_COLOR,
        SpineVertexColors::Fill(TRANSPARENT, EMPTY_VERTEX_COUNT),
    );
    set_float32x4_attribute(
        mesh,
        DARK_COLOR_ATTRIBUTE,
        SpineVertexColors::Fill(TRANSPARENT, EMPTY_VERTEX_COUNT),
    );
}

#[derive(Default)]
struct FixSpineTextures {
    handles: Vec<(Handle<Image>, SpineTextureConfig)>,
}

/// Adjusts Spine textures to render properly.
fn adjust_spine_textures(
    mut local: Local<FixSpineTextures>,
    mut spine_texture_create_events: MessageReader<SpineTextureCreateEvent>,
    mut images: ResMut<Assets<Image>>,
) {
    for spine_texture_create_event in spine_texture_create_events.read() {
        local.handles.push((
            spine_texture_create_event.handle.clone(),
            spine_texture_create_event.config,
        ));
    }
    local.handles.retain(|(handle, handle_config)| {
        if let Some(image) = images.get_mut(handle) {
            fn convert_filter(filter: AtlasFilter) -> ImageFilterMode {
                match filter {
                    AtlasFilter::Nearest => ImageFilterMode::Nearest,
                    AtlasFilter::Linear => ImageFilterMode::Linear,
                    _ => {
                        warn!("Unsupported Spine filter: {:?}", filter);
                        ImageFilterMode::Nearest
                    }
                }
            }
            fn convert_wrap(wrap: AtlasWrap) -> ImageAddressMode {
                match wrap {
                    AtlasWrap::ClampToEdge => ImageAddressMode::ClampToEdge,
                    AtlasWrap::MirroredRepeat => ImageAddressMode::MirrorRepeat,
                    AtlasWrap::Repeat => ImageAddressMode::Repeat,
                    _ => {
                        warn!("Unsupported Spine wrap mode: {:?}", wrap);
                        ImageAddressMode::ClampToEdge
                    }
                }
            }
            image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
                min_filter: convert_filter(handle_config.min_filter),
                mag_filter: convert_filter(handle_config.mag_filter),
                address_mode_u: convert_wrap(handle_config.u_wrap),
                address_mode_v: convert_wrap(handle_config.v_wrap),
                ..Default::default()
            });
            // The RGB components exported from Spine were premultiplied in nonlinear space, but need to be
            // multiplied in linear space to render properly in Bevy.
            if handle_config.premultiplied_alpha {
                if let Some(data) = &mut image.data {
                    for i in 0..(data.len() / 4) {
                        let mut rgba = Srgba::rgba_u8(
                            data[i * 4],
                            data[i * 4 + 1],
                            data[i * 4 + 2],
                            data[i * 4 + 3],
                        );
                        if rgba.alpha != 0. {
                            rgba = Srgba::new(
                                rgba.red / rgba.alpha,
                                rgba.green / rgba.alpha,
                                rgba.blue / rgba.alpha,
                                rgba.alpha,
                            );
                        } else {
                            rgba = Srgba::new(0., 0., 0., 0.);
                        }
                        let mut linear_rgba = LinearRgba::from(rgba);
                        linear_rgba.red *= linear_rgba.alpha;
                        linear_rgba.green *= linear_rgba.alpha;
                        linear_rgba.blue *= linear_rgba.alpha;
                        rgba = Srgba::from(linear_rgba);
                        data[i * 4] = (rgba.red * 255.) as u8;
                        data[i * 4 + 1] = (rgba.green * 255.) as u8;
                        data[i * 4 + 2] = (rgba.blue * 255.) as u8;
                        data[i * 4 + 3] = (rgba.alpha * 255.) as u8;
                    }
                }
            }
            false
        } else {
            true
        }
    });
}

mod assets;
mod crossfades;
mod direct_render;
mod entity_sync;
mod handle;
#[cfg(feature = "ui")]
mod ui;

pub mod materials;
pub mod textures;

pub use direct_render::SpineDirectMaterial2dPlugin;

#[doc(hidden)]
pub mod prelude {
    pub use crate::{
        Crossfades, SkeletonController, SkeletonData, SkeletonDataHandle, Spine, SpineBone,
        SpineCorePlugin, SpineDefaultMaterialPlugin, SpineDirectMaterial2dPlugin, SpineEvent,
        SpineLoader, SpineMesh, SpineMeshState, SpineReadyEvent, SpineSet, SpineSettings,
        SpineSync, SpineSyncSet, SpineSyncSystem, SpineSystem,
    };
    #[cfg(feature = "ui")]
    pub use crate::{SpineUiFit, SpineUiNode, SpineUiProxy, SpineUiReadyEvent, SpineUiSkeleton};
    pub use rusty_spine::{BoneHandle, SlotHandle};
}
