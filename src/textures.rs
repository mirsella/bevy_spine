//! Events related to textures loaded by Spine.

use std::{
    mem::take,
    sync::{Arc, Mutex},
};

use bevy::prelude::*;
use rusty_spine::atlas::{AtlasFilter, AtlasWrap};

use crate::Atlas;

#[derive(Debug)]
struct PendingSpineTexture {
    path: String,
    atlas_address: usize,
    config: SpineTextureConfig,
}

#[derive(Debug, Clone, Copy)]
pub struct SpineTextureConfig {
    pub premultiplied_alpha: bool,
    pub min_filter: AtlasFilter,
    pub mag_filter: AtlasFilter,
    pub u_wrap: AtlasWrap,
    pub v_wrap: AtlasWrap,
}

#[derive(Resource)]
pub(crate) struct SpineTextures {
    pending: Arc<Mutex<Vec<PendingSpineTexture>>>,
}

/// An [`Event`] fired for each texture loaded by Spine.
///
/// Sent in [`SpineSystem::Load`](`crate::SpineSystem::Load`).
#[derive(Debug, Clone, Message)]
pub struct SpineTextureCreateEvent {
    pub path: String,
    pub handle: Handle<Image>,
    pub atlas: Handle<Atlas>,
    pub config: SpineTextureConfig,
}

impl SpineTextures {
    pub(crate) fn init() -> Self {
        let pending = Arc::new(Mutex::new(Vec::new()));

        let pending_create = pending.clone();
        rusty_spine::extension::set_create_texture_cb(move |page, path| {
            let path = path.to_owned();
            pending_create.lock().unwrap().push(PendingSpineTexture {
                path: path.clone(),
                atlas_address: page.atlas().c_ptr() as usize,
                config: SpineTextureConfig {
                    premultiplied_alpha: page.pma(),
                    min_filter: page.min_filter(),
                    mag_filter: page.mag_filter(),
                    u_wrap: page.u_wrap(),
                    v_wrap: page.v_wrap(),
                },
            });
            page.renderer_object().set(path);
        });

        rusty_spine::extension::set_dispose_texture_cb(move |page| unsafe {
            page.renderer_object().dispose::<String>();
        });

        Self { pending }
    }

    pub fn update(
        &mut self,
        asset_server: &AssetServer,
        atlases: &mut Assets<Atlas>,
        create_events: &mut MessageWriter<SpineTextureCreateEvent>,
    ) {
        for texture in take(&mut *self.pending.lock().unwrap()) {
            let Some(atlas) = find_matching_atlas(atlases, texture.atlas_address) else {
                continue;
            };
            create_events.write(SpineTextureCreateEvent {
                path: texture.path.clone(),
                handle: asset_server.load(texture.path),
                atlas,
                config: texture.config,
            });
        }
    }
}

fn find_matching_atlas(atlases: &mut Assets<Atlas>, atlas_address: usize) -> Option<Handle<Atlas>> {
    let id = atlases
        .iter()
        .find_map(|(id, atlas)| (atlas.atlas.c_ptr() as usize == atlas_address).then_some(id))?;
    atlases.get_strong_handle(id)
}
