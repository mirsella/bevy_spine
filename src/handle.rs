use crate::SkeletonData;
use bevy::prelude::*;

#[derive(Default, Component, Clone)]
pub struct SkeletonDataHandle(pub Handle<SkeletonData>);

impl From<Handle<SkeletonData>> for SkeletonDataHandle {
    fn from(handle: Handle<SkeletonData>) -> Self {
        Self(handle)
    }
}
