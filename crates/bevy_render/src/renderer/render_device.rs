use super::RenderQueue;
use crate::render_resource::{
    BindGroup, BindGroupLayout, Buffer, ComputePipeline, RawRenderPipelineDescriptor,
    RenderPipeline, Sampler, Texture,
};
use crate::renderer::WgpuWrapper;
use bevy_ecs::resource::Resource;
use wgpu::{
    util::DeviceExt, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BufferAsyncError, BufferBindingType, PollError, PollStatus,
};

#[cfg(target_arch = "wasm32")]
use alloc::sync::Arc;
#[cfg(target_arch = "wasm32")]
use std::collections::HashMap;
#[cfg(target_arch = "wasm32")]
use std::sync::Mutex;

/// This GPU device is responsible for the creation of most rendering and compute resources.
#[derive(Resource, Clone)]
pub struct RenderDevice {
    device: WgpuWrapper<wgpu::Device>,
    #[cfg(target_arch = "wasm32")]
    bind_group_cache: Arc<Mutex<BindGroupCache>>,
}

#[cfg(target_arch = "wasm32")]
const BIND_GROUP_CACHE_MAX_ENTRIES: usize = 256;

/// Two-generation bind group cache. Entries survive up to 2 "epochs"
/// (each epoch ≈ one render frame). Entries not reused across epochs
/// are dropped, releasing their references to GPU buffers/textures.
/// A hard cap prevents unbounded growth even in pathological cases.
#[cfg(target_arch = "wasm32")]
struct BindGroupCache {
    map: HashMap<u64, CachedBindGroup>,
    epoch: u64,
    access_count: u64,
    epoch_start: u64,
    accesses_first_epoch: u64,
}

#[cfg(target_arch = "wasm32")]
struct CachedBindGroup {
    bind_group: BindGroup,
    epoch: u64,
}

#[cfg(target_arch = "wasm32")]
impl BindGroupCache {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
            epoch: 0,
            access_count: 0,
            epoch_start: 0,
            accesses_first_epoch: 0,
        }
    }

    fn get_or_insert(
        &mut self,
        key: u64,
        create: impl FnOnce() -> BindGroup,
    ) -> BindGroup {
        self.access_count += 1;
        self.maybe_advance_epoch();

        if let Some(entry) = self.map.get_mut(&key) {
            entry.epoch = self.epoch;
            return entry.bind_group.clone();
        }

        let bind_group = create();
        self.map.insert(key, CachedBindGroup {
            bind_group: bind_group.clone(),
            epoch: self.epoch,
        });

        // Hard cap: if we somehow exceed the limit, drop everything.
        if self.map.len() > BIND_GROUP_CACHE_MAX_ENTRIES {
            self.map.clear();
        }

        bind_group
    }

    fn maybe_advance_epoch(&mut self) {
        let accesses_this_epoch = self.access_count - self.epoch_start;
        // Calibrate epoch length from the first epoch (= one frame's worth).
        // Use a minimum of 32 to avoid thrashing on tiny scenes.
        let epoch_len = self.accesses_first_epoch.max(32);

        if accesses_this_epoch >= epoch_len {
            if self.epoch == 0 {
                self.accesses_first_epoch = accesses_this_epoch;
            }
            self.epoch += 1;
            self.epoch_start = self.access_count;
            // Evict entries not used in the previous epoch.
            let min_epoch = self.epoch.saturating_sub(1);
            self.map.retain(|_, v| v.epoch >= min_epoch);
        }
    }
}

impl From<wgpu::Device> for RenderDevice {
    fn from(device: wgpu::Device) -> Self {
        Self::new(WgpuWrapper::new(device))
    }
}

impl RenderDevice {
    pub fn new(device: WgpuWrapper<wgpu::Device>) -> Self {
        Self {
            device,
            #[cfg(target_arch = "wasm32")]
            bind_group_cache: Arc::new(Mutex::new(BindGroupCache::new())),
        }
    }

    /// List all [`Features`](wgpu::Features) that may be used with this device.
    ///
    /// Functions may panic if you use unsupported features.
    #[inline]
    pub fn features(&self) -> wgpu::Features {
        self.device.features()
    }

    /// List all [`Limits`](wgpu::Limits) that were requested of this device.
    ///
    /// If any of these limits are exceeded, functions may panic.
    #[inline]
    pub fn limits(&self) -> wgpu::Limits {
        self.device.limits()
    }

    /// Creates a [`ShaderModule`](wgpu::ShaderModule) from either SPIR-V or WGSL source code.
    ///
    /// # Safety
    ///
    /// Creates a shader module with user-customizable runtime checks which allows shaders to
    /// perform operations which can lead to undefined behavior like indexing out of bounds,
    /// To avoid UB, ensure any unchecked shaders are sound!
    /// This method should never be called for user-supplied shaders.
    #[inline]
    pub unsafe fn create_shader_module(
        &self,
        desc: wgpu::ShaderModuleDescriptor,
    ) -> wgpu::ShaderModule {
        #[cfg(feature = "spirv_shader_passthrough")]
        match &desc.source {
            wgpu::ShaderSource::SpirV(source)
                if self
                    .features()
                    .contains(wgpu::Features::EXPERIMENTAL_PASSTHROUGH_SHADERS) =>
            {
                // SAFETY:
                // This call passes binary data to the backend as-is and can potentially result in a driver crash or bogus behavior.
                // No attempt is made to ensure that data is valid SPIR-V.
                unsafe {
                    self.device.create_shader_module_passthrough(
                        wgpu::ShaderModuleDescriptorPassthrough {
                            label: desc.label,
                            spirv: Some(source.clone()),
                            ..Default::default()
                        },
                    )
                }
            }
            // SAFETY:
            //
            // This call passes binary data to the backend as-is and can potentially result in a driver crash or bogus behavior.
            // No attempt is made to ensure that data is valid SPIR-V.
            _ => unsafe {
                self.device
                    .create_shader_module_trusted(desc, wgpu::ShaderRuntimeChecks::unchecked())
            },
        }
        #[cfg(not(feature = "spirv_shader_passthrough"))]
        // SAFETY: the caller is responsible for upholding the safety requirements
        unsafe {
            self.device
                .create_shader_module_trusted(desc, wgpu::ShaderRuntimeChecks::unchecked())
        }
    }

    /// Creates and validates a [`ShaderModule`](wgpu::ShaderModule) from either SPIR-V or WGSL source code.
    ///
    /// See [`ValidateShader`](bevy_shader::ValidateShader) for more information on the tradeoffs involved with shader validation.
    #[inline]
    pub fn create_and_validate_shader_module(
        &self,
        desc: wgpu::ShaderModuleDescriptor,
    ) -> wgpu::ShaderModule {
        #[cfg(feature = "spirv_shader_passthrough")]
        match &desc.source {
            wgpu::ShaderSource::SpirV(_source) => panic!("no safety checks are performed for spirv shaders. use `create_shader_module` instead"),
            _ => self.device.create_shader_module(desc),
        }
        #[cfg(not(feature = "spirv_shader_passthrough"))]
        self.device.create_shader_module(desc)
    }

    /// Check for resource cleanups and mapping callbacks.
    ///
    /// Return `true` if the queue is empty, or `false` if there are more queue
    /// submissions still in flight. (Note that, unless access to the [`wgpu::Queue`] is
    /// coordinated somehow, this information could be out of date by the time
    /// the caller receives it. `Queue`s can be shared between threads, so
    /// other threads could submit new work at any time.)
    ///
    /// no-op on the web, device is automatically polled.
    #[inline]
    pub fn poll(&self, maintain: wgpu::PollType) -> Result<PollStatus, PollError> {
        self.device.poll(maintain)
    }

    /// Creates an empty [`CommandEncoder`](wgpu::CommandEncoder).
    #[inline]
    pub fn create_command_encoder(
        &self,
        desc: &wgpu::CommandEncoderDescriptor,
    ) -> wgpu::CommandEncoder {
        self.device.create_command_encoder(desc)
    }

    /// Creates an empty [`RenderBundleEncoder`](wgpu::RenderBundleEncoder).
    #[inline]
    pub fn create_render_bundle_encoder(
        &self,
        desc: &wgpu::RenderBundleEncoderDescriptor,
    ) -> wgpu::RenderBundleEncoder<'_> {
        self.device.create_render_bundle_encoder(desc)
    }

    /// Creates a new [`BindGroup`](wgpu::BindGroup).
    ///
    /// On WASM targets, bind groups are cached by a hash of their layout and
    /// entries to avoid recreating identical groups every frame. Without this
    /// cache, the browser's garbage collector must periodically sweep thousands
    /// of short-lived GPU object wrappers, causing visible frame stutters.
    /// See <https://github.com/bevyengine/bevy/issues/22545>.
    pub fn create_bind_group<'a>(
        &self,
        label: impl Into<wgpu::Label<'a>>,
        layout: &'a BindGroupLayout,
        entries: &'a [BindGroupEntry<'a>],
    ) -> BindGroup {
        #[cfg(target_arch = "wasm32")]
        {
            let key = compute_bind_group_hash(layout, entries);
            let label = label.into();
            let device = &self.device;
            self.bind_group_cache.lock().unwrap().get_or_insert(key, || {
                BindGroup::from(device.create_bind_group(&BindGroupDescriptor {
                    label,
                    layout,
                    entries,
                }))
            })
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let wgpu_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                label: label.into(),
                layout,
                entries,
            });
            BindGroup::from(wgpu_bind_group)
        }
    }

    /// Creates a [`BindGroupLayout`](wgpu::BindGroupLayout).
    #[inline]
    pub fn create_bind_group_layout<'a>(
        &self,
        label: impl Into<wgpu::Label<'a>>,
        entries: &'a [BindGroupLayoutEntry],
    ) -> BindGroupLayout {
        BindGroupLayout::from(
            self.device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: label.into(),
                    entries,
                }),
        )
    }

    /// Creates a [`PipelineLayout`](wgpu::PipelineLayout).
    #[inline]
    pub fn create_pipeline_layout(
        &self,
        desc: &wgpu::PipelineLayoutDescriptor,
    ) -> wgpu::PipelineLayout {
        self.device.create_pipeline_layout(desc)
    }

    /// Creates a [`RenderPipeline`].
    #[inline]
    pub fn create_render_pipeline(&self, desc: &RawRenderPipelineDescriptor) -> RenderPipeline {
        let wgpu_render_pipeline = self.device.create_render_pipeline(desc);
        RenderPipeline::from(wgpu_render_pipeline)
    }

    /// Creates a [`ComputePipeline`].
    #[inline]
    pub fn create_compute_pipeline(
        &self,
        desc: &wgpu::ComputePipelineDescriptor,
    ) -> ComputePipeline {
        let wgpu_compute_pipeline = self.device.create_compute_pipeline(desc);
        ComputePipeline::from(wgpu_compute_pipeline)
    }

    /// Creates a [`Buffer`].
    pub fn create_buffer(&self, desc: &wgpu::BufferDescriptor) -> Buffer {
        let wgpu_buffer = self.device.create_buffer(desc);
        Buffer::from(wgpu_buffer)
    }

    /// Creates a [`Buffer`] and initializes it with the specified data.
    pub fn create_buffer_with_data(&self, desc: &wgpu::util::BufferInitDescriptor) -> Buffer {
        let wgpu_buffer = self.device.create_buffer_init(desc);
        Buffer::from(wgpu_buffer)
    }

    /// Creates a new [`Texture`] and initializes it with the specified data.
    ///
    /// `desc` specifies the general format of the texture.
    /// `data` is the raw data.
    pub fn create_texture_with_data(
        &self,
        render_queue: &RenderQueue,
        desc: &wgpu::TextureDescriptor,
        order: wgpu::util::TextureDataOrder,
        data: &[u8],
    ) -> Texture {
        let wgpu_texture =
            self.device
                .create_texture_with_data(render_queue.as_ref(), desc, order, data);
        Texture::from(wgpu_texture)
    }

    /// Creates a new [`Texture`].
    ///
    /// `desc` specifies the general format of the texture.
    pub fn create_texture(&self, desc: &wgpu::TextureDescriptor) -> Texture {
        let wgpu_texture = self.device.create_texture(desc);
        Texture::from(wgpu_texture)
    }

    /// Creates a new [`Sampler`].
    ///
    /// `desc` specifies the behavior of the sampler.
    pub fn create_sampler(&self, desc: &wgpu::SamplerDescriptor) -> Sampler {
        let wgpu_sampler = self.device.create_sampler(desc);
        Sampler::from(wgpu_sampler)
    }

    /// Initializes [`Surface`](wgpu::Surface) for presentation.
    ///
    /// # Panics
    ///
    /// - A old [`SurfaceTexture`](wgpu::SurfaceTexture) is still alive referencing an old surface.
    /// - Texture format requested is unsupported on the surface.
    pub fn configure_surface(&self, surface: &wgpu::Surface, config: &wgpu::SurfaceConfiguration) {
        surface.configure(&self.device, config);
    }

    /// Returns the wgpu [`Device`](wgpu::Device).
    pub fn wgpu_device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn map_buffer(
        &self,
        buffer: &wgpu::BufferSlice,
        map_mode: wgpu::MapMode,
        callback: impl FnOnce(Result<(), BufferAsyncError>) + Send + 'static,
    ) {
        buffer.map_async(map_mode, callback);
    }

    // Rounds up `row_bytes` to be a multiple of [`wgpu::COPY_BYTES_PER_ROW_ALIGNMENT`].
    pub const fn align_copy_bytes_per_row(row_bytes: usize) -> usize {
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;

        // If row_bytes is aligned calculate a value just under the next aligned value.
        // Otherwise calculate a value greater than the next aligned value.
        let over_aligned = row_bytes + align - 1;

        // Round the number *down* to the nearest aligned value.
        (over_aligned / align) * align
    }

    pub fn get_supported_read_only_binding_type(
        &self,
        buffers_per_shader_stage: u32,
    ) -> BufferBindingType {
        if self.limits().max_storage_buffers_per_shader_stage >= buffers_per_shader_stage {
            BufferBindingType::Storage { read_only: true }
        } else {
            BufferBindingType::Uniform
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn compute_bind_group_hash(layout: &BindGroupLayout, entries: &[BindGroupEntry]) -> u64 {
    use core::hash::{Hash, Hasher};
    use std::hash::DefaultHasher;

    let mut hasher = DefaultHasher::new();
    layout.id().hash(&mut hasher);
    entries.len().hash(&mut hasher);
    for entry in entries {
        entry.binding.hash(&mut hasher);
        hash_binding_resource(&entry.resource, &mut hasher);
    }
    hasher.finish()
}

#[cfg(target_arch = "wasm32")]
fn hash_binding_resource<H: core::hash::Hasher>(resource: &wgpu::BindingResource, hasher: &mut H) {
    use core::hash::Hash;

    core::mem::discriminant(resource).hash(hasher);
    match resource {
        wgpu::BindingResource::Buffer(binding) => {
            binding.buffer.hash(hasher);
            binding.offset.hash(hasher);
            binding.size.map(|s| s.get()).hash(hasher);
        }
        wgpu::BindingResource::BufferArray(arr) => {
            arr.len().hash(hasher);
            for b in *arr {
                b.buffer.hash(hasher);
                b.offset.hash(hasher);
                b.size.map(|s| s.get()).hash(hasher);
            }
        }
        wgpu::BindingResource::Sampler(sampler) => {
            (*sampler).hash(hasher);
        }
        wgpu::BindingResource::SamplerArray(arr) => {
            arr.len().hash(hasher);
            for s in *arr {
                (*s).hash(hasher);
            }
        }
        wgpu::BindingResource::TextureView(view) => {
            (*view).hash(hasher);
        }
        wgpu::BindingResource::TextureViewArray(arr) => {
            arr.len().hash(hasher);
            for v in *arr {
                (*v).hash(hasher);
            }
        }
        _ => {
            255u8.hash(hasher);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_copy_bytes_per_row() {
        // Test for https://github.com/bevyengine/bevy/issues/16992
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;

        assert_eq!(RenderDevice::align_copy_bytes_per_row(0), 0);
        assert_eq!(RenderDevice::align_copy_bytes_per_row(1), align);
        assert_eq!(RenderDevice::align_copy_bytes_per_row(align + 1), align * 2);
        assert_eq!(RenderDevice::align_copy_bytes_per_row(align), align);
    }
}
