//! Events and lifecycle state for textures referenced by Spine atlases.

use std::{
    collections::{HashMap, HashSet, hash_map::Entry},
    fmt,
    path::Path,
    sync::{Arc, Once},
};

use bevy::{
    asset::{AssetEvent, AssetLoadError, AssetPath, UntypedAssetId},
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::render_resource::{TextureDimension, TextureFormat},
};
use rusty_spine::atlas::{AtlasFilter, AtlasWrap};

use crate::Atlas;

/// The renderer object attached to one live Spine atlas page.
///
/// The page owns the only persistent strong image handle. Events and short-lived local variables
/// may clone it while they are being processed, but no cache keeps a second handle alive.
#[derive(Debug)]
pub(crate) struct SpineTexture {
    pub(crate) path: String,
    pub(crate) resolved_handle: Option<Handle<Image>>,
    pub(crate) config: SpineTextureConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpineTexturePageStatus {
    Loading,
    Loaded,
    Failed,
}

/// Readiness of every image referenced by a Spine atlas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpineAtlasStatus {
    Loading,
    Loaded,
    /// A page failed an invariant check or the application reported a terminal load failure.
    Failed,
}

type SpineTexturePathResolverFn = dyn Fn(&str, bool) -> String + Send + Sync;

/// Resolves the logical page path written in an atlas to the path that should be loaded.
///
/// The callback receives the page's premultiplied-alpha flag so applications can select a
/// compressed variant for ordinary pages and the canonical PNG for PMA pages.
#[derive(Resource, Clone)]
pub struct SpineTexturePathResolver {
    resolver: Arc<SpineTexturePathResolverFn>,
}

impl fmt::Debug for SpineTexturePathResolver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SpineTexturePathResolver")
            .finish_non_exhaustive()
    }
}

impl SpineTexturePathResolver {
    pub fn new(resolver: impl Fn(&str, bool) -> String + Send + Sync + 'static) -> Self {
        Self {
            resolver: Arc::new(resolver),
        }
    }

    pub fn resolve(&self, logical_path: &str, premultiplied_alpha: bool) -> String {
        (self.resolver)(logical_path, premultiplied_alpha)
    }
}

impl Default for SpineTexturePathResolver {
    fn default() -> Self {
        Self::new(|path, _| path.to_owned())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SpineTextureConfig {
    premultiplied_alpha: bool,
    min_filter: AtlasFilter,
    mag_filter: AtlasFilter,
    u_wrap: AtlasWrap,
    v_wrap: AtlasWrap,
}

#[derive(Resource, Default)]
pub(crate) struct SpineTextures {
    data: SpineTexturesData,
}

/// A terminal asset load failure selected by the application's retry policy.
///
/// Ordinary [`bevy::asset::AssetLoadFailedEvent`] messages are intentionally not terminal: the application may
/// retry the same path and a later asset event will make the page ready.
#[derive(Debug, Clone, Message)]
pub struct SpineAssetLoadFailedEvent {
    pub id: UntypedAssetId,
    pub path: AssetPath<'static>,
    pub error: AssetLoadError,
}

#[derive(Default)]
pub(crate) struct SpineTexturesData {
    pages: HashMap<SpinePageKey, SpinePageState>,
    pages_by_atlas: HashMap<AssetId<Atlas>, HashSet<SpinePageKey>>,
    pages_by_image: HashMap<AssetId<Image>, HashSet<SpinePageKey>>,
    atlases: HashMap<AssetId<Atlas>, SpineAtlasState>,
    image_states: HashMap<AssetId<Image>, SpineImageState>,
    // Path-keyed so a later SkeletonData/page generation observes an earlier terminal event.
    terminal_failures: HashSet<AssetPath<'static>>,
    revision: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SpinePageKey {
    atlas: AssetId<Atlas>,
    logical_path: String,
    index: usize,
}

#[derive(Debug, Default)]
struct SpineAtlasState {
    live_pages: usize,
    pending_pages: usize,
    failed_pages: usize,
}

#[derive(Debug)]
struct SpinePageState {
    image: Option<AssetId<Image>>,
    path: Option<AssetPath<'static>>,
    config: SpineTextureConfig,
    status: SpineTexturePageStatus,
    terminal_failure: bool,
}

#[derive(Debug, Default)]
struct SpineImageState {
    /// The image content generation that received PMA correction.
    ///
    /// `AssetEvent::Modified` can describe a sampler-only mutation after preparation. The data
    /// identity lets that event be ignored without hashing the image or applying PMA twice.
    pma_prepared: Option<ImageContentIdentity>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ImageContentIdentity {
    data_pointer: usize,
    data_length: usize,
    data_capacity: usize,
    format: TextureFormat,
    dimension: TextureDimension,
    width: u32,
    height: u32,
    depth_or_array_layers: u32,
    mip_level_count: u32,
    sample_count: u32,
}

impl SpineAtlasState {
    fn status(&self) -> SpineAtlasStatus {
        if self.failed_pages != 0 {
            SpineAtlasStatus::Failed
        } else if self.live_pages != 0 && self.pending_pages == 0 {
            SpineAtlasStatus::Loaded
        } else {
            SpineAtlasStatus::Loading
        }
    }
}

impl SpineTextures {
    pub(crate) fn init() -> Self {
        install_texture_callbacks();
        Self::default()
    }

    pub(crate) fn update(
        &mut self,
        asset_server: &AssetServer,
        atlases: &mut Assets<Atlas>,
        images: &mut Assets<Image>,
        path_resolver: &SpineTexturePathResolver,
        changed_atlases: &HashSet<AssetId<Atlas>>,
        removed_atlases: &HashSet<AssetId<Atlas>>,
    ) -> u64 {
        let mut created_images = HashSet::new();
        let data = &mut self.data;

        for atlas_id in removed_atlases {
            unregister_atlas(data, *atlas_id);
        }
        for atlas_id in changed_atlases {
            sync_atlas(
                data,
                *atlas_id,
                atlases,
                asset_server,
                path_resolver,
                &mut created_images,
            );
        }
        prune_terminal_failures(data, asset_server);

        for image_id in created_images {
            prepare_image(data, image_id, images, false);
        }
        data.revision
    }

    pub(crate) fn atlas_status(&self, atlas: &Handle<Atlas>) -> Option<SpineAtlasStatus> {
        self.data
            .atlases
            .get(&atlas.id())
            .map(SpineAtlasState::status)
    }

    /// Retains a terminal failure even when no page currently uses the path.
    pub(crate) fn record_terminal_failure(&mut self, path: &AssetPath<'_>) {
        let path = path.clone_owned();
        let data = &mut self.data;
        data.terminal_failures.insert(path.clone());
        let page_ids: Vec<_> = data
            .pages
            .iter()
            .filter_map(|(page_id, page)| {
                (page.path.as_ref() == Some(&path)).then_some(page_id.clone())
            })
            .collect();
        for page_id in &page_ids {
            mark_page_terminal_failure(data, page_id);
        }
    }

    pub(crate) fn has_terminal_failure(&self, path: &AssetPath<'_>) -> bool {
        self.data.terminal_failures.contains(path)
    }

    pub(crate) fn atlas_uses_path(&self, atlas: &Handle<Atlas>, path: &AssetPath<'_>) -> bool {
        self.data.pages.iter().any(|(page_id, page)| {
            page_id.atlas == atlas.id()
                && page
                    .path
                    .as_ref()
                    .is_some_and(|page_path| page_path == path)
        })
    }

    /// Clears a retained failure after a successful asset generation.
    pub(crate) fn clear_terminal_failure(&mut self, path: &AssetPath<'_>) {
        let path = path.clone_owned();
        let data = &mut self.data;
        if !data.terminal_failures.remove(&path) {
            return;
        }
        let page_ids: Vec<_> = data
            .pages
            .iter()
            .filter_map(|(page_id, page)| {
                (page.path.as_ref() == Some(&path)).then_some(page_id.clone())
            })
            .collect();
        for page_id in page_ids {
            if let Some(page) = data.pages.get_mut(&page_id) {
                page.terminal_failure = false;
            }
            set_page_status(data, &page_id, SpineTexturePageStatus::Loading);
        }
    }

    pub(crate) fn clear_successfully_loaded(
        &mut self,
        asset_server: &AssetServer,
    ) -> Vec<AssetPath<'static>> {
        let paths: Vec<_> = {
            let data = &self.data;
            data.terminal_failures
                .iter()
                .filter(|path| {
                    asset_server
                        .get_path_ids((*path).clone())
                        .into_iter()
                        .any(|id| asset_server.is_loaded_with_dependencies(id))
                })
                .cloned()
                .collect()
        };
        for path in &paths {
            self.clear_terminal_failure(path);
        }
        paths
    }

    #[cfg(test)]
    fn set_page_status(
        &mut self,
        atlas: AssetId<Atlas>,
        page_id: SpinePageKey,
        status: SpineTexturePageStatus,
    ) -> bool {
        if page_id.atlas != atlas {
            return false;
        }
        set_page_status(&mut self.data, &page_id, status)
    }
}

#[derive(Debug)]
struct SpineAtlasPage {
    key: SpinePageKey,
    index: usize,
    config: SpineTextureConfig,
    metadata_attached: bool,
}

fn install_texture_callbacks() {
    static INSTALL: Once = Once::new();

    INSTALL.call_once(|| {
        rusty_spine::extension::set_create_texture_cb(|page, path| {
            page.renderer_object().set(SpineTexture {
                path: path.to_owned(),
                resolved_handle: None,
                config: SpineTextureConfig {
                    premultiplied_alpha: page.pma(),
                    min_filter: page.min_filter(),
                    mag_filter: page.mag_filter(),
                    u_wrap: page.u_wrap(),
                    v_wrap: page.v_wrap(),
                },
            });
        });
        rusty_spine::extension::set_dispose_texture_cb(|page| {
            let mut renderer_object = page.renderer_object();
            if unsafe { renderer_object.get::<SpineTexture>() }.is_none() {
                error!("Spine atlas page was disposed without its texture metadata");
                return;
            }
            unsafe { renderer_object.dispose::<SpineTexture>() };
        });
    });
}

fn atlas_pages(atlas_id: AssetId<Atlas>, atlas: &Atlas) -> Vec<SpineAtlasPage> {
    atlas
        .atlas
        .pages()
        .enumerate()
        .map(|(index, page)| {
            let mut renderer_object = page.renderer_object();
            let (path, config, metadata_attached) =
                if let Some(texture) = unsafe { renderer_object.get::<SpineTexture>() } {
                    (texture.path.clone(), texture.config, true)
                } else {
                    error!(
                        atlas = ?atlas_id,
                        index,
                        "Spine atlas page is missing renderer texture metadata"
                    );
                    (
                        page.name().to_owned(),
                        SpineTextureConfig {
                            premultiplied_alpha: page.pma(),
                            min_filter: page.min_filter(),
                            mag_filter: page.mag_filter(),
                            u_wrap: page.u_wrap(),
                            v_wrap: page.v_wrap(),
                        },
                        false,
                    )
                };
            SpineAtlasPage {
                key: SpinePageKey {
                    atlas: atlas_id,
                    logical_path: path,
                    index,
                },
                index,
                config,
                metadata_attached,
            }
        })
        .collect()
}

fn sync_atlas(
    data: &mut SpineTexturesData,
    atlas_id: AssetId<Atlas>,
    atlases: &mut Assets<Atlas>,
    asset_server: &AssetServer,
    path_resolver: &SpineTexturePathResolver,
    created_images: &mut HashSet<AssetId<Image>>,
) {
    let Some(atlas) = atlases.get(atlas_id) else {
        error!(atlas = ?atlas_id, "Spine atlas event referenced a missing asset");
        unregister_atlas(data, atlas_id);
        return;
    };
    let pages = atlas_pages(atlas_id, atlas);
    let page_keys = pages
        .iter()
        .map(|page| page.key.clone())
        .collect::<HashSet<_>>();
    let previous_pages = data
        .pages_by_atlas
        .insert(atlas_id, page_keys.clone())
        .unwrap_or_default();
    for page in previous_pages.difference(&page_keys) {
        unregister_page(data, page);
    }
    if page_keys.is_empty() {
        data.pages_by_atlas.remove(&atlas_id);
    }

    for page in pages {
        register_page(data, page.key.clone(), page.config);
        if !page.metadata_attached {
            set_page_status(data, &page.key, SpineTexturePageStatus::Failed);
            continue;
        }

        let logical_path = page.key.logical_path.as_str();
        let resolved_path = path_resolver.resolve(logical_path, page.config.premultiplied_alpha);
        if page.config.premultiplied_alpha
            && Path::new(&resolved_path)
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("ktx2"))
        {
            error!(path = %resolved_path, "PMA Spine atlas page resolved to compressed data");
            set_page_status(data, &page.key, SpineTexturePageStatus::Failed);
            continue;
        }

        let resolved_path = AssetPath::parse(&resolved_path).into_owned();
        if !set_page_path(data, &page.key, resolved_path.clone()) {
            error!(
                atlas = ?atlas_id,
                index = page.index,
                "Spine page state disappeared before its path was stored"
            );
            continue;
        }
        let handle: Handle<Image> = asset_server.load(resolved_path);
        if !set_page_image(data, &page.key, handle.id(), page.config) {
            error!(
                atlas = ?atlas_id,
                index = page.index,
                "Spine page state disappeared before its image handle was stored"
            );
            continue;
        }
        let Some(atlas) = atlases.get_mut_untracked(atlas_id) else {
            error!(atlas = ?atlas_id, "Spine atlas disappeared while storing a page handle");
            set_page_status(data, &page.key, SpineTexturePageStatus::Failed);
            continue;
        };
        if with_page_texture(atlas, page.index, |texture| {
            texture.resolved_handle = Some(handle.clone());
        })
        .is_none()
        {
            error!(
                atlas = ?atlas_id,
                index = page.index,
                "Spine page metadata disappeared while storing its image handle"
            );
            set_page_status(data, &page.key, SpineTexturePageStatus::Failed);
            continue;
        }
        created_images.insert(handle.id());
    }
}

fn register_page(data: &mut SpineTexturesData, page_id: SpinePageKey, config: SpineTextureConfig) {
    let (new_page, config_changed, terminal_failure) = match data.pages.entry(page_id.clone()) {
        Entry::Vacant(entry) => {
            entry.insert(SpinePageState {
                image: None,
                path: None,
                config,
                status: SpineTexturePageStatus::Loading,
                terminal_failure: false,
            });
            (true, false, false)
        }
        Entry::Occupied(mut entry) => {
            let page = entry.get_mut();
            let config_changed = page.config != config;
            page.config = config;
            (false, config_changed, page.terminal_failure)
        }
    };
    if new_page {
        data.pages_by_atlas
            .entry(page_id.atlas)
            .or_default()
            .insert(page_id.clone());
        let atlas = data.atlases.entry(page_id.atlas).or_default();
        atlas.live_pages += 1;
        atlas.pending_pages += 1;
        data.revision = data.revision.wrapping_add(1);
    } else if config_changed && !terminal_failure {
        set_page_status(data, &page_id, SpineTexturePageStatus::Loading);
    }
}

fn set_page_status(
    data: &mut SpineTexturesData,
    page_id: &SpinePageKey,
    status: SpineTexturePageStatus,
) -> bool {
    let Some(previous) = data.pages.get(page_id).map(|page| page.status) else {
        return false;
    };
    if previous == status {
        return true;
    }
    let Some(atlas) = data.atlases.get_mut(&page_id.atlas) else {
        error!(atlas = ?page_id.atlas, "Spine page has no atlas readiness state");
        return false;
    };
    match previous {
        SpineTexturePageStatus::Loading => atlas.pending_pages -= 1,
        SpineTexturePageStatus::Failed => atlas.failed_pages -= 1,
        SpineTexturePageStatus::Loaded => {}
    }
    match status {
        SpineTexturePageStatus::Loading => atlas.pending_pages += 1,
        SpineTexturePageStatus::Failed => atlas.failed_pages += 1,
        SpineTexturePageStatus::Loaded => {}
    }
    let Some(page) = data.pages.get_mut(page_id) else {
        return false;
    };
    page.status = status;
    data.revision = data.revision.wrapping_add(1);
    true
}

fn unregister_atlas(data: &mut SpineTexturesData, atlas_id: AssetId<Atlas>) {
    let page_ids = data.pages_by_atlas.remove(&atlas_id).unwrap_or_default();
    for page_id in page_ids {
        unregister_page(data, &page_id);
    }
}

fn unregister_page(data: &mut SpineTexturesData, page_id: &SpinePageKey) {
    let Some(page) = data.pages.remove(page_id) else {
        return;
    };
    if let Some(pages) = data.pages_by_atlas.get_mut(&page_id.atlas) {
        pages.remove(page_id);
        if pages.is_empty() {
            data.pages_by_atlas.remove(&page_id.atlas);
        }
    }
    if let Some(atlas) = data.atlases.get_mut(&page_id.atlas) {
        match page.status {
            SpineTexturePageStatus::Loading => atlas.pending_pages -= 1,
            SpineTexturePageStatus::Failed => atlas.failed_pages -= 1,
            SpineTexturePageStatus::Loaded => {}
        }
        atlas.live_pages -= 1;
        if atlas.live_pages == 0 {
            data.atlases.remove(&page_id.atlas);
        }
    }
    if let Some(image_id) = page.image {
        remove_image_page(data, image_id, page_id);
    }
    data.revision = data.revision.wrapping_add(1);
}

fn set_page_image(
    data: &mut SpineTexturesData,
    page_id: &SpinePageKey,
    image: AssetId<Image>,
    config: SpineTextureConfig,
) -> bool {
    let Some((previous_image, previous_config, terminal_failure)) = data
        .pages
        .get(page_id)
        .map(|page| (page.image, page.config, page.terminal_failure))
    else {
        return false;
    };
    if previous_image != Some(image) {
        if let Some(previous_image) = previous_image {
            remove_image_page(data, previous_image, page_id);
        }
        data.pages_by_image
            .entry(image)
            .or_default()
            .insert(page_id.clone());
    }
    let Some(page) = data.pages.get_mut(page_id) else {
        return false;
    };
    page.image = Some(image);
    page.config = config;
    if (previous_image != Some(image) || previous_config != config) && !terminal_failure {
        set_page_status(data, page_id, SpineTexturePageStatus::Loading);
    }
    true
}

fn set_page_path(
    data: &mut SpineTexturesData,
    page_id: &SpinePageKey,
    path: AssetPath<'static>,
) -> bool {
    let terminal_failure = data.terminal_failures.contains(&path);
    let Some(page) = data.pages.get_mut(page_id) else {
        return false;
    };
    page.path = Some(path);
    if terminal_failure {
        mark_page_terminal_failure(data, page_id);
    }
    true
}

fn pages_for_image(data: &SpineTexturesData, image: AssetId<Image>) -> Vec<SpinePageKey> {
    data.pages_by_image
        .get(&image)
        .into_iter()
        .flatten()
        .cloned()
        .collect()
}

fn remove_image_page(data: &mut SpineTexturesData, image: AssetId<Image>, page_id: &SpinePageKey) {
    let Some(pages) = data.pages_by_image.get_mut(&image) else {
        error!(?image, ?page_id, "Spine image index is missing a live page");
        return;
    };
    pages.remove(page_id);
    if pages.is_empty() {
        data.pages_by_image.remove(&image);
        data.image_states.remove(&image);
    }
}

fn fail_pages(data: &mut SpineTexturesData, page_ids: &[SpinePageKey], reason: &str) {
    error!(%reason, "Spine atlas page preparation failed terminally");
    for page_id in page_ids {
        mark_page_terminal_failure(data, page_id);
    }
}

fn mark_page_terminal_failure(data: &mut SpineTexturesData, page_id: &SpinePageKey) {
    if !data.pages.contains_key(page_id) {
        return;
    }
    set_page_status(data, page_id, SpineTexturePageStatus::Failed);
    if let Some(page) = data.pages.get_mut(page_id) {
        page.terminal_failure = true;
    }
}

fn with_page_texture<R>(
    atlas: &mut Atlas,
    index: usize,
    callback: impl FnOnce(&mut SpineTexture) -> R,
) -> Option<R> {
    let page = atlas.atlas.pages().nth(index)?;
    let mut renderer_object = page.renderer_object();
    let texture = unsafe { renderer_object.get::<SpineTexture>() }?;
    Some(callback(texture))
}

fn prune_terminal_failures(data: &mut SpineTexturesData, asset_server: &AssetServer) {
    let stale_paths: Vec<_> = data
        .terminal_failures
        .iter()
        .filter(|path| {
            !data
                .pages
                .values()
                .any(|page| page.path.as_ref() == Some(*path))
                && asset_server.get_path_ids((*path).clone()).is_empty()
        })
        .cloned()
        .collect();
    for path in stale_paths {
        trace!(%path, "discarding terminal Spine asset failure after its assets left scope");
        data.terminal_failures.remove(&path);
    }
}

/// Adjusts Spine samplers and PMA pixels after the corresponding image asset is available.
///
/// This is event-driven by image asset changes. It never scans the world or reprocesses every live
/// image every frame.
pub(crate) fn adjust_spine_textures(
    mut image_events: MessageReader<AssetEvent<Image>>,
    mut images: ResMut<Assets<Image>>,
    mut spine_textures: ResMut<SpineTextures>,
) {
    let mut new_generations = HashSet::new();
    let mut loaded_events = HashSet::new();
    for event in image_events.read() {
        match event {
            AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                new_generations.insert(*id);
                loaded_events.remove(id);
            }
            AssetEvent::LoadedWithDependencies { id } => {
                loaded_events.insert(*id);
            }
            AssetEvent::Removed { id } => {
                spine_textures.remove_image(*id);
                new_generations.remove(id);
                loaded_events.remove(id);
            }
            AssetEvent::Unused { id } => {
                if spine_textures.has_image(*id) {
                    error!(id = ?id, "Spine page image became unused while its page was live");
                }
            }
        }
    }

    for image_id in new_generations {
        spine_textures.prepare_image(image_id, &mut images, true);
    }
    for image_id in loaded_events {
        spine_textures.prepare_image(image_id, &mut images, false);
    }
}

impl SpineTextures {
    fn prepare_image(
        &mut self,
        image_id: AssetId<Image>,
        images: &mut Assets<Image>,
        new_generation: bool,
    ) {
        prepare_image(&mut self.data, image_id, images, new_generation);
    }

    fn remove_image(&mut self, image_id: AssetId<Image>) {
        let data = &mut self.data;
        data.image_states.remove(&image_id);
        for page_id in pages_for_image(data, image_id) {
            let Some(terminal_failure) = data.pages.get(&page_id).map(|page| page.terminal_failure)
            else {
                continue;
            };
            if !terminal_failure {
                set_page_status(data, &page_id, SpineTexturePageStatus::Loading);
            }
        }
    }

    fn has_image(&self, image_id: AssetId<Image>) -> bool {
        self.data.pages_by_image.contains_key(&image_id)
    }
}

fn prepare_image(
    data: &mut SpineTexturesData,
    image_id: AssetId<Image>,
    images: &mut Assets<Image>,
    new_generation: bool,
) {
    let page_ids = pages_for_image(data, image_id);
    if page_ids.is_empty() {
        return;
    }

    let content_identity = images.get(image_id).map(image_content_identity);
    let same_prepared_content = new_generation
        && data
            .image_states
            .get(&image_id)
            .and_then(|state| state.pma_prepared)
            .is_some_and(|prepared| Some(prepared) == content_identity);
    if new_generation && !same_prepared_content {
        data.image_states.entry(image_id).or_default().pma_prepared = None;
    }

    if new_generation && !same_prepared_content {
        for page_id in &page_ids {
            let Some(terminal_failure) = data.pages.get(page_id).map(|page| page.terminal_failure)
            else {
                continue;
            };
            if !terminal_failure {
                set_page_status(data, page_id, SpineTexturePageStatus::Loading);
            }
        }
    }

    let mut config = None;
    for page_id in &page_ids {
        let Some(page) = data.pages.get(page_id) else {
            continue;
        };
        if page.terminal_failure {
            return;
        }
        if let Some(previous) = config {
            if !same_sampling(previous, page.config) {
                fail_pages(
                    data,
                    &page_ids,
                    "Spine atlas pages sharing one image use different sampler settings",
                );
                return;
            }
            if previous.premultiplied_alpha != page.config.premultiplied_alpha {
                fail_pages(
                    data,
                    &page_ids,
                    "Spine atlas pages sharing one image disagree on premultiplied alpha",
                );
                return;
            }
            config = Some(previous);
        } else {
            config = Some(page.config);
        }
    }
    let Some(config) = config else {
        return;
    };
    if page_ids.iter().all(|id| {
        data.pages
            .get(id)
            .is_some_and(|page| page.status == SpineTexturePageStatus::Loaded)
    }) {
        return;
    }

    let Some(image) = images.get_mut_untracked(image_id) else {
        // A failed load is deliberately left Loading. The application's retry policy owns the
        // next attempt and its Added/Modified event will call this function again.
        return;
    };
    let Some(sampler) = sampler_for(config) else {
        fail_pages(data, &page_ids, "invalid Spine atlas sampler settings");
        return;
    };
    image.sampler = sampler;
    let pma_already_prepared = data
        .image_states
        .get(&image_id)
        .and_then(|state| state.pma_prepared)
        .is_some_and(|prepared| Some(prepared) == content_identity);
    if config.premultiplied_alpha
        && !pma_already_prepared
        && let Err(error) = prepare_pma(image)
    {
        fail_pages(
            data,
            &page_ids,
            &format!("invalid premultiplied Spine atlas image: {error:?}"),
        );
        return;
    }
    if config.premultiplied_alpha {
        let Some(content_identity) = content_identity else {
            fail_pages(
                data,
                &page_ids,
                "Spine atlas image content disappeared during PMA preparation",
            );
            return;
        };
        data.image_states.entry(image_id).or_default().pma_prepared = Some(content_identity);
    }

    for page_id in page_ids {
        set_page_status(data, &page_id, SpineTexturePageStatus::Loaded);
    }
}

fn image_content_identity(image: &Image) -> ImageContentIdentity {
    let (data_pointer, data_length, data_capacity) = image
        .data
        .as_ref()
        .map(|data| (data.as_ptr() as usize, data.len(), data.capacity()))
        .unwrap_or_default();
    let descriptor = &image.texture_descriptor;
    ImageContentIdentity {
        data_pointer,
        data_length,
        data_capacity,
        format: descriptor.format,
        dimension: descriptor.dimension,
        width: descriptor.size.width,
        height: descriptor.size.height,
        depth_or_array_layers: descriptor.size.depth_or_array_layers,
        mip_level_count: descriptor.mip_level_count,
        sample_count: descriptor.sample_count,
    }
}

fn same_sampling(left: SpineTextureConfig, right: SpineTextureConfig) -> bool {
    left.min_filter == right.min_filter
        && left.mag_filter == right.mag_filter
        && left.u_wrap == right.u_wrap
        && left.v_wrap == right.v_wrap
}

fn sampler_for(config: SpineTextureConfig) -> Option<ImageSampler> {
    Some(ImageSampler::Descriptor(ImageSamplerDescriptor {
        min_filter: filter_for(config.min_filter)?,
        mag_filter: filter_for(config.mag_filter)?,
        address_mode_u: wrap_for(config.u_wrap)?,
        address_mode_v: wrap_for(config.v_wrap)?,
        ..Default::default()
    }))
}

fn filter_for(filter: AtlasFilter) -> Option<ImageFilterMode> {
    match filter {
        AtlasFilter::UnknownFilter => None,
        AtlasFilter::Nearest => Some(ImageFilterMode::Nearest),
        AtlasFilter::Linear
        | AtlasFilter::Mipmap
        | AtlasFilter::MipmapNearestNearest
        | AtlasFilter::MipmapLinearNearest
        | AtlasFilter::MipmapNearestLinear
        | AtlasFilter::MipmapLinearLinear => Some(ImageFilterMode::Linear),
    }
}

fn wrap_for(wrap: AtlasWrap) -> Option<ImageAddressMode> {
    match wrap {
        AtlasWrap::MirroredRepeat => Some(ImageAddressMode::MirrorRepeat),
        AtlasWrap::ClampToEdge => Some(ImageAddressMode::ClampToEdge),
        AtlasWrap::Repeat => Some(ImageAddressMode::Repeat),
        AtlasWrap::Unknown => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PmaError {
    Compressed,
    WrongFormat,
    WrongDimensions,
    MissingCpuData,
    WrongDataLength,
}

fn prepare_pma(image: &mut Image) -> Result<(), PmaError> {
    if image.is_compressed() {
        return Err(PmaError::Compressed);
    }
    if image.texture_descriptor.format != TextureFormat::Rgba8UnormSrgb {
        return Err(PmaError::WrongFormat);
    }
    let size = image.texture_descriptor.size;
    if image.texture_descriptor.dimension != TextureDimension::D2
        || size.width == 0
        || size.height == 0
        || size.depth_or_array_layers != 1
        || image.texture_descriptor.mip_level_count != 1
        || image.texture_descriptor.sample_count != 1
    {
        return Err(PmaError::WrongDimensions);
    }
    let Some(expected_len) = (size.width as usize)
        .checked_mul(size.height as usize)
        .and_then(|pixels| pixels.checked_mul(4))
    else {
        return Err(PmaError::WrongDataLength);
    };
    let Some(data) = image.data.as_mut() else {
        return Err(PmaError::MissingCpuData);
    };
    if data.len() != expected_len {
        return Err(PmaError::WrongDataLength);
    }

    for [red, green, blue, alpha] in data.as_chunks_mut::<4>().0 {
        let mut rgba = Srgba::rgba_u8(*red, *green, *blue, *alpha);
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
        let mut linear = LinearRgba::from(rgba);
        linear.red *= linear.alpha;
        linear.green *= linear.alpha;
        linear.blue *= linear.alpha;
        rgba = Srgba::from(linear);
        *red = (rgba.red * 255.) as u8;
        *green = (rgba.green * 255.) as u8;
        *blue = (rgba.blue * 255.) as u8;
        *alpha = (rgba.alpha * 255.) as u8;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_page(atlas: AssetId<Atlas>, index: usize) -> SpinePageKey {
        SpinePageKey {
            atlas,
            logical_path: format!("page-{index}.png"),
            index,
        }
    }

    fn test_config() -> SpineTextureConfig {
        SpineTextureConfig {
            premultiplied_alpha: false,
            min_filter: AtlasFilter::Linear,
            mag_filter: AtlasFilter::Linear,
            u_wrap: AtlasWrap::ClampToEdge,
            v_wrap: AtlasWrap::ClampToEdge,
        }
    }

    fn pma_config() -> SpineTextureConfig {
        SpineTextureConfig {
            premultiplied_alpha: true,
            ..test_config()
        }
    }

    #[test]
    fn resolver_is_pma_aware() {
        let resolver = SpineTexturePathResolver::new(|path, pma| {
            if pma {
                path.to_owned()
            } else {
                format!("bc7/{path}").replace(".png", ".ktx2")
            }
        });
        for (pma, expected) in [
            (true, "cards/castle/castle.png"),
            (false, "bc7/cards/castle/castle.ktx2"),
        ] {
            assert_eq!(resolver.resolve("cards/castle/castle.png", pma), expected);
        }
    }

    #[test]
    fn pma_validation_rejects_invalid_inputs_without_mutation() {
        let cases = [
            (TextureFormat::Bc7RgbaUnormSrgb, None, PmaError::Compressed),
            (TextureFormat::Rgba8Unorm, None, PmaError::WrongFormat),
        ];
        for (format, data, expected) in cases {
            let mut image = Image::default();
            image.texture_descriptor.format = format;
            image.data = data;
            assert_eq!(prepare_pma(&mut image), Err(expected));
        }

        let mut image = Image {
            data: Some(vec![1, 2, 3]),
            ..default()
        };
        let original = image.data.clone();
        assert_eq!(prepare_pma(&mut image), Err(PmaError::WrongDataLength));
        assert_eq!(image.data, original);
    }

    #[test]
    fn atlas_status_tracks_all_live_pages() {
        let mut textures = SpineTextures::init();
        let atlas: Handle<Atlas> =
            bevy::asset::uuid_handle!("f3bb4f7f-a4df-4ed8-9c8e-64f5cce0c0b1");
        let page1 = test_page(atlas.id(), 1);
        let page2 = test_page(atlas.id(), 2);
        register_page(&mut textures.data, page1.clone(), test_config());
        register_page(&mut textures.data, page2.clone(), test_config());
        for (page_id, page_status, expected) in [
            (
                page1.clone(),
                SpineTexturePageStatus::Loading,
                SpineAtlasStatus::Loading,
            ),
            (
                page2.clone(),
                SpineTexturePageStatus::Loading,
                SpineAtlasStatus::Loading,
            ),
            (
                page1.clone(),
                SpineTexturePageStatus::Loaded,
                SpineAtlasStatus::Loading,
            ),
            (
                page2.clone(),
                SpineTexturePageStatus::Loaded,
                SpineAtlasStatus::Loaded,
            ),
            (
                page2.clone(),
                SpineTexturePageStatus::Loading,
                SpineAtlasStatus::Loading,
            ),
            (
                page2.clone(),
                SpineTexturePageStatus::Loaded,
                SpineAtlasStatus::Loaded,
            ),
            (
                page2.clone(),
                SpineTexturePageStatus::Failed,
                SpineAtlasStatus::Failed,
            ),
        ] {
            assert!(textures.set_page_status(atlas.id(), page_id, page_status));
            assert_eq!(textures.atlas_status(&atlas), Some(expected));
        }
        unregister_page(&mut textures.data, &page1);
        assert_eq!(
            textures.atlas_status(&atlas),
            Some(SpineAtlasStatus::Failed)
        );
    }

    #[test]
    fn missing_image_stays_loading_until_a_later_asset_event() {
        let atlas: Handle<Atlas> =
            bevy::asset::uuid_handle!("f3bb4f7f-a4df-4ed8-9c8e-64f5cce0c0b2");
        let image: Handle<Image> =
            bevy::asset::uuid_handle!("f3bb4f7f-a4df-4ed8-9c8e-64f5cce0c0b3");
        let mut textures = SpineTextures::default();
        let page = test_page(atlas.id(), 7);
        register_page(&mut textures.data, page.clone(), test_config());
        assert!(set_page_image(
            &mut textures.data,
            &page,
            image.id(),
            test_config()
        ));
        let mut images = Assets::default();
        textures.prepare_image(image.id(), &mut images, false);
        assert_eq!(
            textures.atlas_status(&atlas),
            Some(SpineAtlasStatus::Loading)
        );

        images.insert(image.id(), Image::default()).unwrap();
        textures.prepare_image(image.id(), &mut images, false);
        assert_eq!(
            textures.atlas_status(&atlas),
            Some(SpineAtlasStatus::Loaded)
        );
    }

    #[test]
    fn terminal_failure_before_page_registration_survives_page_recreation() {
        let atlas: Handle<Atlas> =
            bevy::asset::uuid_handle!("f3bb4f7f-a4df-4ed8-9c8e-64f5cce0c0b8");
        let path = AssetPath::parse("chests/chest.png");
        let mut textures = SpineTextures::default();

        textures.record_terminal_failure(&path);
        for index in [1, 2] {
            let page = test_page(atlas.id(), index);
            register_page(&mut textures.data, page.clone(), test_config());
            assert!(set_page_path(&mut textures.data, &page, path.clone_owned()));
            assert_eq!(
                textures.atlas_status(&atlas),
                Some(SpineAtlasStatus::Failed)
            );
            unregister_page(&mut textures.data, &page);
        }

        textures.clear_terminal_failure(&path);
        assert!(!textures.has_terminal_failure(&path));
    }

    #[test]
    fn shared_image_conflicts_fail_every_page() {
        for conflicting_config in [
            pma_config(),
            SpineTextureConfig {
                min_filter: AtlasFilter::Nearest,
                ..test_config()
            },
        ] {
            let atlas: Handle<Atlas> =
                bevy::asset::uuid_handle!("f3bb4f7f-a4df-4ed8-9c8e-64f5cce0c0b4");
            let image: Handle<Image> =
                bevy::asset::uuid_handle!("f3bb4f7f-a4df-4ed8-9c8e-64f5cce0c0b5");
            let mut textures = SpineTextures::default();
            let page1 = test_page(atlas.id(), 1);
            let page2 = test_page(atlas.id(), 2);
            register_page(&mut textures.data, page1.clone(), test_config());
            register_page(&mut textures.data, page2.clone(), conflicting_config);
            assert!(set_page_image(
                &mut textures.data,
                &page1,
                image.id(),
                test_config()
            ));
            assert!(set_page_image(
                &mut textures.data,
                &page2,
                image.id(),
                conflicting_config
            ));

            let mut images = Assets::default();
            images.insert(image.id(), Image::default()).unwrap();
            textures.prepare_image(image.id(), &mut images, false);

            assert_eq!(
                textures.atlas_status(&atlas),
                Some(SpineAtlasStatus::Failed)
            );
        }
    }

    #[test]
    fn pma_is_applied_once_per_image_generation() {
        let atlas: Handle<Atlas> =
            bevy::asset::uuid_handle!("f3bb4f7f-a4df-4ed8-9c8e-64f5cce0c0b6");
        let image: Handle<Image> =
            bevy::asset::uuid_handle!("f3bb4f7f-a4df-4ed8-9c8e-64f5cce0c0b7");
        let mut textures = SpineTextures::default();
        let page1 = test_page(atlas.id(), 1);
        let page2 = test_page(atlas.id(), 2);
        register_page(&mut textures.data, page1.clone(), pma_config());
        assert!(set_page_image(
            &mut textures.data,
            &page1,
            image.id(),
            pma_config()
        ));

        let raw = vec![128, 64, 32, 128];
        let mut images = Assets::default();
        images
            .insert(
                image.id(),
                Image {
                    data: Some(raw.clone()),
                    ..default()
                },
            )
            .unwrap();
        textures.prepare_image(image.id(), &mut images, false);
        let prepared = images.get(image.id()).unwrap().data.clone();

        // Simulate an external sampler-only mutation emitting Modified without replacing pixels.
        textures.prepare_image(image.id(), &mut images, true);
        assert_eq!(images.get(image.id()).unwrap().data, prepared);

        register_page(&mut textures.data, page2.clone(), pma_config());
        assert!(set_page_image(
            &mut textures.data,
            &page2,
            image.id(),
            pma_config()
        ));
        textures.prepare_image(image.id(), &mut images, false);
        assert_eq!(images.get(image.id()).unwrap().data, prepared);

        images.get_mut_untracked(image.id()).unwrap().data = Some(raw);
        textures.prepare_image(image.id(), &mut images, true);
        assert_eq!(images.get(image.id()).unwrap().data, prepared);

        unregister_page(&mut textures.data, &page1);
        assert!(textures.data.image_states.contains_key(&image.id()));
        unregister_page(&mut textures.data, &page2);
        assert!(!textures.data.image_states.contains_key(&image.id()));
        assert!(!textures.data.pages_by_image.contains_key(&image.id()));
    }
}
