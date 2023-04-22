package render

import "core:sync"
import "core:strings"
import "core:slice"
import "core:fmt"
import "core:runtime"
import "core:mem"
import "core:container/queue"

import vk "vendor:vulkan"
import "vendor:glfw"

hello :: proc() {
    fmt.println("hello from render!")
}

WIDTH :: 400
HEIGHT :: 600

ENABLE_VALIDATION_LAYERS :: ODIN_DEBUG || #config(ENABLE_VALIDATION_LAYERS, false)

vk_assert :: proc(result: vk.Result, loc := #caller_location) {
    if result != .SUCCESS {
        fmt.panicf("Failure performing operation at %v", loc)
    }
}

validation_layers := []cstring{"VK_LAYER_KHRONOS_validation"}


Renderer :: struct {
    window: glfw.WindowHandle,
    surface: vk.SurfaceKHR,
    instance: vk.Instance,
    debug_messenger: vk.DebugUtilsMessengerEXT,
    physical_device: vk.PhysicalDevice,
    device: vk.Device,
    main_queue: vk.Queue,
    swapchain: Swapchain,
    descriptors: Descriptors,
    pipeline: Pipeline,
    syncs: Syncs,
}

Image :: struct {
    width, height: int,
    data: [][4]u8,
}

Buffer :: struct {
    buffer: vk.Buffer,
    memory: vk.DeviceMemory,
}

Texture :: struct {
    image: vk.Image,
    image_memory: vk.DeviceMemory,
    image_view: vk.ImageView,
}

Swapchain :: struct {
    swapchain: vk.SwapchainKHR,
    extent: vk.Extent2D,
    format: vk.Format,
    images: []vk.Image,
    image_views: []vk.ImageView,
}

Descriptors :: struct {
    layout: vk.DescriptorSetLayout,
    pool: vk.DescriptorPool,
    sets: [MAX_FRAMES_IN_FLIGHT]vk.DescriptorSet,
}

Pipeline :: struct {
    layout: vk.PipelineLayout,
    pipeline: vk.Pipeline,
}

Syncs :: struct {
    image_avails: [MAX_FRAMES_IN_FLIGHT]vk.Semaphore,
    render_finishes: [MAX_FRAMES_IN_FLIGHT]vk.Semaphore,
    inflight_fences: [MAX_FRAMES_IN_FLIGHT]vk.Fence,
}

WriterHandle :: distinct u64
@thread_local _thread_global_handle: WriterHandle
_global_atomic_counter: u64

global_renderer: Renderer

mesh_shader :: #load("../shaders/mesh.spv")
fragment_shader :: #load("../shaders/frag.spv")

MAX_FRAMES_IN_FLIGHT :: 2
DEFAULT_THREAD_CAPACITY :: 4
global_command_pools: [dynamic]vk.CommandPool
global_command_buffers: [dynamic][MAX_FRAMES_IN_FLIGHT]vk.CommandBuffer

// Gets a WriterHandle which is necessary to write commands
// into a command buffer. Each writer needs a handle (they may be the same)
// and should just treat it as an opaque type
register_writer :: proc() -> WriterHandle {
    if _thread_global_handle == {} {
        current_value := sync.atomic_load(&_global_atomic_counter)
        value, swapped := sync.atomic_compare_exchange_strong(&_global_atomic_counter, current_value, current_value + 1)
        for !swapped {
            current_value = sync.atomic_load(&_global_atomic_counter)
            value, swapped = sync.atomic_compare_exchange_strong(&_global_atomic_counter, current_value, current_value + 1)
        }
        _thread_global_handle = WriterHandle(value + 1) // cas returns previous value
        assert(value + 1 <= DEFAULT_THREAD_CAPACITY) // TODO add more pools
    }
    
    return _thread_global_handle
}

get_command_buffer :: proc(writer: WriterHandle, current_frame: int) -> vk.CommandBuffer {
    assert(writer != WriterHandle{})
    buffer := global_command_buffers[int(writer - 1)][current_frame]
    vk.ResetCommandBuffer(buffer, {})
    return buffer;
}

global_render_create_texture :: proc(image: Image) -> Texture {
    return renderer_create_texture(global_renderer, image)
}

global_render_destroy_texture :: proc(texture: Texture) {
    renderer_destroy_texture(global_renderer, texture)
}

renderer_create_texture :: proc(renderer: Renderer, image: Image) -> Texture {
    image_data_as_bytes := slice.to_bytes(image.data)
    image_bytes_len := vk.DeviceSize(len(image_data_as_bytes))
    staging_buffer, staging_memory := create_buffer(renderer.physical_device, renderer.device, image_bytes_len, {.TRANSFER_SRC}, {.HOST_VISIBLE, .HOST_COHERENT})
    defer destroy_buffer(renderer.device, staging_buffer, staging_memory)

    data: rawptr
    vk.MapMemory(renderer.device, staging_memory, 0, image_bytes_len, nil, &data)
    mem.copy(data, raw_data(image_data_as_bytes), int(image_bytes_len))
    vk.UnmapMemory(renderer.device, staging_memory)

    texture, texture_memory := create_image(renderer.physical_device, renderer.device, u32(image.width), u32(image.height), 1, .R8G8B8A8_UNORM, .OPTIMAL, {.TRANSFER_DST, .SAMPLED, .STORAGE}, {.DEVICE_LOCAL})
    
    command_buffer := scoped_single_time_commands(renderer.device, global_command_pools[int(_thread_global_handle)], renderer.main_queue)
	// transition to transfer_dst_optimal
    vk.CmdPipelineBarrier(command_buffer, {.TOP_OF_PIPE}, {.TRANSFER}, {}, 0, nil, 0, nil, 1, &vk.ImageMemoryBarrier{
		sType = .IMAGE_MEMORY_BARRIER,
		oldLayout = .UNDEFINED,
		newLayout = .TRANSFER_DST_OPTIMAL,
		srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		image = texture,
		subresourceRange = {
			aspectMask = {.COLOR},
			baseMipLevel = 0,
			levelCount = 1,
			baseArrayLayer = 0,
			layerCount = 1,
		},
		srcAccessMask = nil,
		dstAccessMask = nil,
	})

    // copy staging buffer to texture
	vk.CmdCopyBufferToImage(command_buffer, staging_buffer, texture, .TRANSFER_DST_OPTIMAL, 1, &vk.BufferImageCopy{
		bufferOffset = 0,
		bufferRowLength = 0,
		bufferImageHeight = 0,

		imageSubresource = {
			aspectMask = {.COLOR},
			mipLevel = 0,
			baseArrayLayer = 0,
			layerCount = 1,
		},

		imageOffset = {0, 0, 0},
		imageExtent = {u32(image.width), u32(image.height), 1},

	})

    vk.CmdPipelineBarrier(command_buffer, {.TRANSFER}, {.FRAGMENT_SHADER}, {}, 0, nil, 0, nil, 1, &vk.ImageMemoryBarrier{
		sType = .IMAGE_MEMORY_BARRIER,
		oldLayout = .TRANSFER_DST_OPTIMAL,
		newLayout = .SHADER_READ_ONLY_OPTIMAL,
		srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		image = texture,
		subresourceRange = {
			aspectMask = {.COLOR},
			baseMipLevel = 0,
			levelCount = 1,
			baseArrayLayer = 0,
			layerCount = 1,
		},
		srcAccessMask = {.TRANSFER_WRITE},
		dstAccessMask = {.SHADER_READ},
	})

    texture_image_view := create_image_view(renderer.device, texture, .R8G8B8A8_UNORM, {.COLOR}, 1)

    return {image = texture, image_view = texture_image_view, image_memory = texture_memory}
}

renderer_destroy_texture :: proc(renderer: Renderer, texture: Texture) {
    defer vk.FreeMemory(renderer.device, texture.image_memory, nil)
    defer vk.DestroyImage(renderer.device, texture.image, nil)
    defer vk.DestroyImageView(renderer.device, texture.image_view, nil)
}

global_render_running :: proc() -> bool {
    return !bool(glfw.WindowShouldClose(global_renderer.window))
}

global_render_handle_events :: proc() {
    glfw.WaitEventsTimeout(0.001)
}

global_render_init :: proc() {
    global_command_pools = make([dynamic]vk.CommandPool, DEFAULT_THREAD_CAPACITY)
    global_command_buffers = make([dynamic][MAX_FRAMES_IN_FLIGHT]vk.CommandBuffer, DEFAULT_THREAD_CAPACITY)

    glfw.Init()
    vk.load_proc_addresses_global(rawptr(glfw.GetInstanceProcAddress))
    glfw.WindowHint(glfw.CLIENT_API, glfw.NO_API)

    global_renderer.window = glfw.CreateWindow(WIDTH, HEIGHT, "Hello Dynamic Rendering Vulkan", nil, nil)
	assert(global_renderer.window != nil, "Window could not be crated")

    glfw.SetWindowUserPointer(global_renderer.window, &global_renderer)
	glfw.SetFramebufferSizeCallback(global_renderer.window, framebuffer_resize_callback)

    glfw.SetScrollCallback(global_renderer.window, scroll_callback)
    glfw.SetCursorPosCallback(global_renderer.window, cursor_pos_callback)
    glfw.SetMouseButtonCallback(global_renderer.window, mouse_button_callback)

    instance_extension := get_required_instance_extensions()
    defer delete(instance_extension)

    enabled := []vk.ValidationFeatureEnableEXT{.DEBUG_PRINTF}
    features := vk.ValidationFeaturesEXT{
        sType = .VALIDATION_FEATURES_EXT,
        enabledValidationFeatureCount = 1,
        pEnabledValidationFeatures = raw_data(enabled),
    }
    if vk.CreateInstance(&vk.InstanceCreateInfo{
        sType = .INSTANCE_CREATE_INFO,
        enabledExtensionCount = u32(len(instance_extension)),
        ppEnabledExtensionNames = raw_data(instance_extension),
        pApplicationInfo = &vk.ApplicationInfo{
            sType = .APPLICATION_INFO,
            pApplicationName = "GOL in Vulkan",
            applicationVersion = vk.MAKE_VERSION(0, 1, 1),
            pEngineName = "No Engine",
            engineVersion = vk.MAKE_VERSION(0, 1, 0),
            apiVersion = vk.API_VERSION_1_3,
        },
        enabledLayerCount = u32(len(validation_layers)),
        ppEnabledLayerNames = raw_data(validation_layers),
        pNext = &features,
    }, nil, &global_renderer.instance) != .SUCCESS {
        panic("Couldn not create instance")
    }

    get_instance_proc_addr :: proc "system" (
		instance: vk.Instance,
		name: cstring,
	) -> vk.ProcVoidFunction {
		f := glfw.GetInstanceProcAddress(instance, name)
		return (vk.ProcVoidFunction)(f)
	}
	vk.GetInstanceProcAddr = get_instance_proc_addr
	vk.load_proc_addresses(global_renderer.instance)

    vk.CreateDebugUtilsMessengerEXT(global_renderer.instance, &vk.DebugUtilsMessengerCreateInfoEXT{
        sType = .DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        messageSeverity = {.VERBOSE, .INFO, .WARNING, .ERROR},
        messageType = {.GENERAL, .VALIDATION, .PERFORMANCE},
        pfnUserCallback = debug_callback,
    }, nil, &global_renderer.debug_messenger)

    if glfw.CreateWindowSurface(global_renderer.instance, global_renderer.window, nil, &global_renderer.surface) != .SUCCESS {
        panic("Could not create surface")
    }

    graphics_family: u32 = 1000
    present_family: u32 = 1000

    { // CREATE LOGICAL DEVICE AND GET QUEUE HANDLE
        device_count: u32
        vk.EnumeratePhysicalDevices(global_renderer.instance, &device_count, nil)
        devices := make([]vk.PhysicalDevice, device_count, context.temp_allocator)
        vk.EnumeratePhysicalDevices(global_renderer.instance, &device_count, raw_data(devices))
        physical_device := devices[0]
        global_renderer.physical_device = physical_device

        queue_family_count: u32
        vk.GetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nil)
        queue_families := make([]vk.QueueFamilyProperties, queue_family_count, context.temp_allocator)
        vk.GetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, raw_data(queue_families))

        for queue_family, i in queue_families {
            if .GRAPHICS in queue_family.queueFlags && .COMPUTE in queue_family.queueFlags {
                graphics_family = u32(i)
                break
            }
        }
        for queue_family, i in queue_families {
            present_support: b32
            vk.GetPhysicalDeviceSurfaceSupportKHR(physical_device, u32(i), global_renderer.surface, &present_support)
            if present_support {
                present_family = u32(i)
                break
            }
        }

        queue_set := make(map[u32]u32, 100, context.temp_allocator)
        queue_set[graphics_family] = 1
        queue_set[present_family] = 1
        assert(len(queue_set) == 1)

        queue_create_infos := make([dynamic]vk.DeviceQueueCreateInfo, 0, len(queue_set), context.temp_allocator)
        priority: f32 = 1.0

        for queue_family, _ in queue_set {
            append(&queue_create_infos, vk.DeviceQueueCreateInfo{
                sType = .DEVICE_QUEUE_CREATE_INFO,
                queueFamilyIndex = queue_family,
                queueCount = 1,
                pQueuePriorities = &priority,
            })
        }

        device_features := vk.PhysicalDeviceFeatures{
            samplerAnisotropy = true,
        }

        device_extensions := make([]cstring, len(DEVICE_EXTENSION_LIST), context.temp_allocator)
        for ext, i in DEVICE_EXTENSION_LIST {
            device_extensions[i] = strings.clone_to_cstring(ext, context.temp_allocator)
        }

        if vk.CreateDevice(physical_device, &vk.DeviceCreateInfo{
            sType = .DEVICE_CREATE_INFO,
            pQueueCreateInfos = raw_data(queue_create_infos[:]),
            queueCreateInfoCount = u32(len(queue_create_infos)),
            pEnabledFeatures = &device_features,
            enabledExtensionCount = u32(len(device_extensions)),
            ppEnabledExtensionNames = raw_data(device_extensions),
            pNext = &vk.PhysicalDeviceDynamicRenderingFeaturesKHR{
                sType = .PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR,
                dynamicRendering = true,
                pNext = &vk.PhysicalDeviceSynchronization2Features{
                    sType = .PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
                    synchronization2 = true,
                    pNext = &vk.PhysicalDeviceMeshShaderFeaturesEXT{
                        sType = .PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
                        meshShader = true,
                        pNext = &vk.PhysicalDeviceMaintenance4Features{
                            sType = .PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES,
                            maintenance4 = true,
                        },
                    },
                },
            },
        }, nil, &global_renderer.device) != .SUCCESS {
            panic("Could not create logical device!")
        }
        vk.GetDeviceQueue(global_renderer.device, graphics_family, 0, &global_renderer.main_queue)

        // CREATE SWAP CHAIN AND SWAP CHAIN IMAGES/VIEWS
        capabilities: vk.SurfaceCapabilitiesKHR
        vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, global_renderer.surface, &capabilities)

        image_count := capabilities.minImageCount + 1
        if capabilities.maxImageCount > 0 {
            image_count = clamp(image_count, capabilities.minImageCount, capabilities.maxImageCount)
            image_count = clamp(image_count, capabilities.minImageCount, MAX_FRAMES_IN_FLIGHT)
        }

        width, height := glfw.GetFramebufferSize(global_renderer.window)
        extent := vk.Extent2D{
            width = clamp(u32(width), capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
            height = clamp(u32(height), capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
        }

        if vk.CreateSwapchainKHR(global_renderer.device, &vk.SwapchainCreateInfoKHR{
            sType = .SWAPCHAIN_CREATE_INFO_KHR,
            surface = global_renderer.surface,
            minImageCount = image_count,
            imageFormat = .B8G8R8A8_SRGB,
            imageColorSpace = .SRGB_NONLINEAR,
            imageExtent = extent,
            imageArrayLayers = 1,
            imageUsage = {.COLOR_ATTACHMENT},
            preTransform = capabilities.currentTransform,
            compositeAlpha = {.OPAQUE},
            presentMode = .FIFO,
            clipped = true,
            oldSwapchain = {},
        }, nil, &global_renderer.swapchain.swapchain) != .SUCCESS {
            panic("Couldn't create swap chain!")
        }

        sc_image_count: u32
        vk.GetSwapchainImagesKHR(global_renderer.device, global_renderer.swapchain.swapchain, &sc_image_count, nil)
        global_renderer.swapchain.images = make([]vk.Image, sc_image_count)
        vk.GetSwapchainImagesKHR(global_renderer.device, global_renderer.swapchain.swapchain, &sc_image_count, raw_data(global_renderer.swapchain.images))
        global_renderer.swapchain.extent = extent
        global_renderer.swapchain.format = .B8G8R8A8_SRGB
        global_renderer.swapchain.image_views = make([]vk.ImageView, sc_image_count)
        for image, i in global_renderer.swapchain.images {
            global_renderer.swapchain.image_views[i] = create_image_view(global_renderer.device, image, .B8G8R8A8_SRGB, {.COLOR}, 1)
        }


    }

    { // COMMAND POOLS
        using global_renderer
        for pool, i in &global_command_pools {
            if vk.CreateCommandPool(device, &vk.CommandPoolCreateInfo{
                sType = .COMMAND_POOL_CREATE_INFO,
                flags = {.RESET_COMMAND_BUFFER},
                queueFamilyIndex = graphics_family,
            }, nil, &pool) != .SUCCESS {
                panic("Failed to create command pool!")
            }

            vk_assert(vk.AllocateCommandBuffers(device, &vk.CommandBufferAllocateInfo{
                sType = .COMMAND_BUFFER_ALLOCATE_INFO,
                commandPool = pool,
                level = .PRIMARY,
                commandBufferCount = MAX_FRAMES_IN_FLIGHT,
            }, raw_data(global_command_buffers[i][:])))
        }
    }


    // THESE THINGS BELOW ARE ACTUALLY PROGRAM SPECIFIC WHILE THE STUFF ABOVE
    // CAN BE A BIT MORE GENERAL I THINK

    { // DESCRIPTOR SET LAYOUT
        using global_renderer
        graphics_layout := [?]vk.DescriptorSetLayoutBinding{
            {
                binding = 0,
                descriptorCount = 1,
                descriptorType = .UNIFORM_BUFFER,
                stageFlags = {.MESH_SHADER_EXT},
            },
            {
                binding = 1,
                descriptorCount = 1,
                descriptorType = .SAMPLER,
                stageFlags = {.FRAGMENT},
            },
            {
                binding = 2,
                descriptorCount = 4,
                descriptorType = .SAMPLED_IMAGE,
                stageFlags = {.FRAGMENT},
            }
        }

        if vk.CreateDescriptorSetLayout(device, &vk.DescriptorSetLayoutCreateInfo{
            sType = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount = u32(len(graphics_layout[:])),
            pBindings = raw_data(graphics_layout[:]),
        }, nil, &descriptors.layout) != .SUCCESS {
            panic("Failed to create descriptor set layout!")
        }
    }

    { // PIPELINE
        using global_renderer
        mesh_shader_module := create_shader_module(device, mesh_shader)
        frag_shader_module := create_shader_module(device, fragment_shader)
        defer vk.DestroyShaderModule(device, mesh_shader_module, nil)
        defer vk.DestroyShaderModule(device, frag_shader_module, nil)

        shader_stages := [?]vk.PipelineShaderStageCreateInfo{
            {
                sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage = {.MESH_SHADER_EXT},
                module = mesh_shader_module,
                pName = "main",
            },
            {
                sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage = {.FRAGMENT},
                module = frag_shader_module,
                pName = "main",
            },
        }

        dynamic_states := [?]vk.DynamicState{.VIEWPORT, .SCISSOR}
        dynamic_state := vk.PipelineDynamicStateCreateInfo{
            sType = .PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            dynamicStateCount = u32(len(dynamic_states)),
            pDynamicStates = raw_data(dynamic_states[:]),
        }

        viewport_state := vk.PipelineViewportStateCreateInfo{
            sType = .PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewportCount = 1,
            scissorCount = 1,
        }

        rasterizer := vk.PipelineRasterizationStateCreateInfo{
            sType = .PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            polygonMode = .FILL,
            cullMode = {.FRONT},
            frontFace = .COUNTER_CLOCKWISE,
            lineWidth = 1.0,
        }
        multisampling := vk.PipelineMultisampleStateCreateInfo{
            sType = .PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            rasterizationSamples = {._1},
            minSampleShading = 1.0,
        }
        color_blend_attachment := vk.PipelineColorBlendAttachmentState {
            colorWriteMask = {.R, .G, .B, .A},
            srcColorBlendFactor = .ONE,
            dstColorBlendFactor = .ZERO,
            colorBlendOp = .ADD,
            srcAlphaBlendFactor = .ONE,
            dstAlphaBlendFactor = .ZERO,
            alphaBlendOp = .ADD,
        }
    
        color_blending := vk.PipelineColorBlendStateCreateInfo {
            sType           = .PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            logicOp         = .COPY,
            attachmentCount = 1,
            pAttachments    = &color_blend_attachment,
        }

        if vk.CreatePipelineLayout(device, &vk.PipelineLayoutCreateInfo{
            sType = .PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount = 1,
            pSetLayouts = &descriptors.layout,
        }, nil, &pipeline.layout) != .SUCCESS {
            panic("Error creating pipeline layout")
        }

        if vk.CreateGraphicsPipelines(device, {}, 1, &vk.GraphicsPipelineCreateInfo{
            sType = .GRAPHICS_PIPELINE_CREATE_INFO,
            stageCount = u32(len(shader_stages)),
            pStages = raw_data(shader_stages[:]),
            pViewportState = &viewport_state,
            pRasterizationState = &rasterizer,
            pMultisampleState = &multisampling,
            pColorBlendState = &color_blending,
            pDynamicState = &dynamic_state,
            layout = pipeline.layout,
            basePipelineIndex = -1,
            pNext = &vk.PipelineRenderingCreateInfoKHR{
                sType = .PIPELINE_RENDERING_CREATE_INFO_KHR,
                colorAttachmentCount = 1,
                pColorAttachmentFormats = &swapchain.format,
            },
        }, nil, &pipeline.pipeline) != .SUCCESS {
            panic("failed to create pipeline")
        }
    }

    { // Descriptor pool/sets
        using global_renderer
        pool_sizes := []vk.DescriptorPoolSize{
            {
                type = .COMBINED_IMAGE_SAMPLER,
                descriptorCount = u32(MAX_FRAMES_IN_FLIGHT),
            },
            {
                type = .UNIFORM_BUFFER,
                descriptorCount = u32(MAX_FRAMES_IN_FLIGHT),
            },
            {
                type = .SAMPLED_IMAGE,
                descriptorCount = 4 * u32(MAX_FRAMES_IN_FLIGHT),
            },
        }

        if vk.CreateDescriptorPool(device, &vk.DescriptorPoolCreateInfo{
            sType = .DESCRIPTOR_POOL_CREATE_INFO,
            poolSizeCount = u32(len(pool_sizes)),
            pPoolSizes = raw_data(pool_sizes),
            maxSets = u32(MAX_FRAMES_IN_FLIGHT),
        }, nil, &descriptors.pool) != .SUCCESS {
            panic("Failed to create descriptor pool!")
        }

        layouts := [MAX_FRAMES_IN_FLIGHT]vk.DescriptorSetLayout{
            descriptors.layout,
            descriptors.layout,
        }

        vk_assert(vk.AllocateDescriptorSets(device, &vk.DescriptorSetAllocateInfo{
            sType = .DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool = descriptors.pool,
            descriptorSetCount = u32(MAX_FRAMES_IN_FLIGHT),
            pSetLayouts = raw_data(layouts[:]),
        }, raw_data(descriptors.sets[:])))
    }

    { // SYNC OBJECTS
        using global_renderer
        for i in 0..<MAX_FRAMES_IN_FLIGHT {
            s1 := vk.CreateSemaphore(device, &vk.SemaphoreCreateInfo{
                sType = .SEMAPHORE_CREATE_INFO,
            }, nil, &syncs.image_avails[i])
            s2 := vk.CreateSemaphore(device, &vk.SemaphoreCreateInfo{
                sType = .SEMAPHORE_CREATE_INFO,
            }, nil, &syncs.render_finishes[i])
            fen := vk.CreateFence(device, &vk.FenceCreateInfo{
                sType = .FENCE_CREATE_INFO,
                flags = {.SIGNALED},
            }, nil, &syncs.inflight_fences[i])
        
            if s1 != .SUCCESS || s2 != .SUCCESS || fen != .SUCCESS {
                panic("failed to create sync objects")
            }
        }
    }
}

global_render_destroy :: proc() {
    defer glfw.Terminate()
    defer glfw.DestroyWindow(global_renderer.window)
    defer vk.DestroyInstance(global_renderer.instance, nil)
    defer vk.DestroySurfaceKHR(global_renderer.instance, global_renderer.surface, nil)
    defer vk.DestroyDebugUtilsMessengerEXT(global_renderer.instance, global_renderer.debug_messenger, nil)
    defer vk.DestroyDevice(global_renderer.device, nil)
    defer vk.DestroySwapchainKHR(global_renderer.device, global_renderer.swapchain.swapchain, nil)
    defer {
        defer delete(global_renderer.swapchain.images)
        for image_view in global_renderer.swapchain.image_views {
            vk.DestroyImageView(global_renderer.device, image_view, nil)
        }
    }
    defer delete(global_renderer.swapchain.image_views)
    defer vk.DestroyDescriptorSetLayout(global_renderer.device, global_renderer.descriptors.layout, nil)
    defer vk.DestroyPipelineLayout(global_renderer.device, global_renderer.pipeline.layout, nil)
    defer vk.DestroyPipeline(global_renderer.device, global_renderer.pipeline.pipeline, nil)

    defer vk.DestroyDescriptorPool(global_renderer.device, global_renderer.descriptors.pool, nil)

    defer {
        for i in 0..<MAX_FRAMES_IN_FLIGHT {
            vk.DestroySemaphore(global_renderer.device, global_renderer.syncs.image_avails[i], nil)
            vk.DestroySemaphore(global_renderer.device, global_renderer.syncs.render_finishes[i], nil)
            vk.DestroyFence(global_renderer.device, global_renderer.syncs.inflight_fences[i], nil)
        }
    }

    defer {
        for pool, i in global_command_pools {
            vk.FreeCommandBuffers(global_renderer.device, pool, len(global_command_buffers[i]), raw_data(global_command_buffers[i][:]))
            vk.DestroyCommandPool(global_renderer.device, pool, nil)
        }
    }

    defer vk.DeviceWaitIdle(global_renderer.device)
}

get_required_instance_extensions :: proc() -> (result: [dynamic]cstring) {

	extensions := glfw.GetRequiredInstanceExtensions()
	append(&result, ..extensions)

	if ENABLE_VALIDATION_LAYERS {
		append(&result, vk.EXT_DEBUG_UTILS_EXTENSION_NAME)
	}
	return
}

@(deferred_in_out = end_single_time_commands)
scoped_single_time_commands :: proc(device: vk.Device, command_pool: vk.CommandPool, submit_queue: vk.Queue) -> vk.CommandBuffer {
	return begin_single_time_commands(device, command_pool, submit_queue)
}


// Creates a temporary command buffer for one time submit / oneshot commands
// to be written to GPU
begin_single_time_commands :: proc(device: vk.Device, command_pool: vk.CommandPool, submit_queue: vk.Queue) -> (buffer: vk.CommandBuffer) {
	vk.AllocateCommandBuffers(device, &vk.CommandBufferAllocateInfo{
		sType = .COMMAND_BUFFER_ALLOCATE_INFO,
		level = .PRIMARY,
		commandPool = command_pool,
		commandBufferCount = 1,
	}, &buffer)

	vk.BeginCommandBuffer(buffer, &vk.CommandBufferBeginInfo{
		sType = .COMMAND_BUFFER_BEGIN_INFO,
		flags = {.ONE_TIME_SUBMIT},
	})
	
	return
}

// Ends the temporary command buffer and submits the commands
end_single_time_commands :: proc(device: vk.Device, command_pool: vk.CommandPool, submit_queue: vk.Queue, buffer: vk.CommandBuffer) {
	buffer := buffer

	vk.EndCommandBuffer(buffer)

	vk.QueueSubmit(submit_queue, 1, &vk.SubmitInfo{
		sType = .SUBMIT_INFO,
		commandBufferCount = 1,
		pCommandBuffers = &buffer,
	}, {})
	vk.QueueWaitIdle(submit_queue)

	vk.FreeCommandBuffers(device, command_pool, 1, &buffer)
}

Zoom_Event :: f64
Drag_Start :: distinct struct {}
Drag_End :: distinct struct {}
Mouse_Pos :: [2]int

Simulator_Event :: union {
    Zoom_Event,
    Drag_Start,
    Drag_End,
    Mouse_Pos,
}