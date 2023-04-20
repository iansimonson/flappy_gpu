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

    { // DESCRIPTOR SET LAYOUT
        using global_renderer
        // graphics_layout := [?]vk.DescriptorSetLayoutBinding{
        //     {
        //         binding = 0,
        //         descriptorCount = 1,
        //         descriptorType = .COMBINED_IMAGE_SAMPLER,
        //         stageFlags = {.FRAGMENT},
        //     },
        // }
        graphics_layout := [?]vk.DescriptorSetLayoutBinding{
            {
                binding = 0,
                descriptorCount = 1,
                descriptorType = .UNIFORM_BUFFER,
                stageFlags = {.MESH_SHADER_EXT},
            },
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

    { // Descriptor pool/sets
        using global_renderer
        pool_sizes := []vk.DescriptorPoolSize{
            {
                type = .COMBINED_IMAGE_SAMPLER,
                descriptorCount = u32(MAX_FRAMES_IN_FLIGHT),
            },
            {
                type = .STORAGE_IMAGE,
                descriptorCount = 2 * u32(MAX_FRAMES_IN_FLIGHT),
            },
        }

        if vk.CreateDescriptorPool(device, &vk.DescriptorPoolCreateInfo{
            sType = .DESCRIPTOR_POOL_CREATE_INFO,
            poolSizeCount = u32(len(pool_sizes)),
            pPoolSizes = raw_data(pool_sizes),
            maxSets = 2 * u32(MAX_FRAMES_IN_FLIGHT),
        }, nil, &descriptors.pool) != .SUCCESS {
            panic("Failed to create descriptor pool!")
        }

        layouts := [MAX_FRAMES_IN_FLIGHT]vk.DescriptorSetLayout{
            descriptors.layout,
            descriptors.layout,
        }
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

get_device :: proc(instance: vk.Instance) -> vk.Device {
    return {}
}

framebuffer_resize_callback :: proc "c" (window: glfw.WindowHandle, width, height: i32) {
}

DEVICE_EXTENSION_LIST := [?]string{
    vk.KHR_SWAPCHAIN_EXTENSION_NAME,
    vk.KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
    vk.KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
    vk.EXT_MESH_SHADER_EXTENSION_NAME,
    vk.KHR_SPIRV_1_4_EXTENSION_NAME,
    // vk.KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME,
}

INSTANCE_EXTENSION_LIST := [?]string{
    // vk.KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
}

create_image :: proc(physical_device: vk.PhysicalDevice, device: vk.Device, width, height, mip_levels: u32, format: vk.Format, tiling: vk.ImageTiling, usage: vk.ImageUsageFlags, properties: vk.MemoryPropertyFlags) -> (image: vk.Image, memory: vk.DeviceMemory) {
	if vk.CreateImage(device, &vk.ImageCreateInfo{
		sType = .IMAGE_CREATE_INFO,
		imageType = .D2,
		extent = vk.Extent3D{width = width, height = height, depth = 1},
		mipLevels = mip_levels,
		arrayLayers = 1,
		format = format,
		tiling = tiling,
		initialLayout = .UNDEFINED,
		usage = usage,
		sharingMode = .EXCLUSIVE,
		samples = {._1},
		flags = nil,
	}, nil, &image) != .SUCCESS {
		panic("Failed to create image!")
	}

	mem_requirements: vk.MemoryRequirements
	vk.GetImageMemoryRequirements(device, image, &mem_requirements)

	if vk.AllocateMemory(device, &vk.MemoryAllocateInfo{
		sType = .MEMORY_ALLOCATE_INFO,
		allocationSize = mem_requirements.size,
		memoryTypeIndex = find_memory_type(physical_device, mem_requirements.memoryTypeBits, properties),
	}, nil, &memory) != .SUCCESS {
		panic("failed to allocate image memory!")
	}

	vk.BindImageMemory(device, image, memory, 0)
	return
}

create_image_view :: proc(device: vk.Device, image: vk.Image, format: vk.Format, aspect_flags: vk.ImageAspectFlags, mip_levels: u32) -> (view: vk.ImageView) {
	if vk.CreateImageView(device, &vk.ImageViewCreateInfo{
		sType = .IMAGE_VIEW_CREATE_INFO,
		image = image,
		viewType = .D2,
		format = format,
		subresourceRange = {
			aspectMask = aspect_flags,
			baseMipLevel = 0,
			levelCount = mip_levels,
			baseArrayLayer = 0,
			layerCount = 1,
		},
	}, nil, &view) != .SUCCESS {
		panic("Failed to create texture image")
	}
	return
}

create_shader_module :: proc(device: vk.Device, code: []byte) -> (sm: vk.ShaderModule) {
	if result := vk.CreateShaderModule(
		   device,
		   &vk.ShaderModuleCreateInfo{
			   sType = .SHADER_MODULE_CREATE_INFO,
			   codeSize = len(code),
			   pCode = (^u32)(raw_data(code)),
		   },
		   nil,
		   &sm,
	   ); result != .SUCCESS {
		panic("Failed to create shader module")
	}
	return
}

create_buffer :: proc(physical_device: vk.PhysicalDevice, device: vk.Device, size: vk.DeviceSize, usage: vk.BufferUsageFlags, properties: vk.MemoryPropertyFlags) -> (buffer: vk.Buffer, memory: vk.DeviceMemory) {
	buffer_info := vk.BufferCreateInfo{
		sType = .BUFFER_CREATE_INFO,
		size = size,
		usage = usage,
		sharingMode = .EXCLUSIVE,
	}

	if vk.CreateBuffer(device, &buffer_info, nil, &buffer) != .SUCCESS {
		fmt.panicf("Failed to create buffer: {%v, %v, %v}\n", size, usage, properties)
	}

	mem_requirements: vk.MemoryRequirements
	vk.GetBufferMemoryRequirements(device, buffer, &mem_requirements)

	alloc_info := vk.MemoryAllocateInfo{
		sType = .MEMORY_ALLOCATE_INFO,
		allocationSize = mem_requirements.size,
		memoryTypeIndex = find_memory_type(physical_device, mem_requirements.memoryTypeBits, properties),
	}

	if vk.AllocateMemory(device, &alloc_info, nil, &memory) != .SUCCESS {
		fmt.panicf("failed to allocate memory for the buffer: {%v, %v, %v}\n", size, usage, properties)
	}

	vk.BindBufferMemory(device, buffer, memory, 0)

	return
}

destroy_buffer :: proc(device: vk.Device, buffer: vk.Buffer, memory: vk.DeviceMemory) {
	defer vk.DestroyBuffer(device, buffer, nil)
	defer  vk.FreeMemory(device, memory, nil)
}

copy_buffer :: proc(device: vk.Device, src, dst: vk.Buffer, copy_infos: []vk.BufferCopy) {
	temp_command_buffer := scoped_single_time_commands(device, global_command_pools[int(_thread_global_handle)], global_renderer.main_queue)

	vk.CmdCopyBuffer(temp_command_buffer, src, dst, u32(len(copy_infos)), raw_data(copy_infos))
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

find_memory_type :: proc(physical_device: vk.PhysicalDevice, type_filter: u32, properties: vk.MemoryPropertyFlags) -> u32 {
	mem_properties: vk.PhysicalDeviceMemoryProperties
	vk.GetPhysicalDeviceMemoryProperties(physical_device, &mem_properties)

	for i in 0..<mem_properties.memoryTypeCount {
		if type_filter & (1 << i) != 0 && (mem_properties.memoryTypes[i].propertyFlags & properties == properties) {
			return i
		}
	}

	panic("Failed to find suitable memory type!")

}

transition_image_layout :: proc(device: vk.Device, queue: vk.Queue, image: vk.Image, format: vk.Format, old_layout, new_layout: vk.ImageLayout, mip_levels: u32) {
	command_buffer := scoped_single_time_commands(device, global_command_pools[int(_thread_global_handle)], queue)
	barrier := vk.ImageMemoryBarrier{
		sType = .IMAGE_MEMORY_BARRIER,
		oldLayout = old_layout,
		newLayout = new_layout,
		srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		image = image,
		subresourceRange = {
			aspectMask = {.COLOR},
			baseMipLevel = 0,
			levelCount = mip_levels,
			baseArrayLayer = 0,
			layerCount = 1,
		},
		srcAccessMask = nil,
		dstAccessMask = nil,
	}

	source_stage, destination_stage: vk.PipelineStageFlags

	if old_layout == .UNDEFINED && new_layout == .TRANSFER_DST_OPTIMAL {
		barrier.srcAccessMask = nil
		barrier.dstAccessMask = nil
		
		source_stage = {.TOP_OF_PIPE}
		destination_stage = {.TRANSFER}
	} else if old_layout == .TRANSFER_DST_OPTIMAL && new_layout == .SHADER_READ_ONLY_OPTIMAL {
		barrier.srcAccessMask = {.TRANSFER_WRITE}
		barrier.dstAccessMask = {.SHADER_READ}

		source_stage = {.TRANSFER}
		destination_stage = {.FRAGMENT_SHADER}
	} else {
		panic("unsupported layout transition!")
	}
	vk.CmdPipelineBarrier(command_buffer, source_stage, destination_stage, {}, 0, nil, 0, nil, 1, &barrier)
}

debug_callback :: proc "system" (
	message_severity: vk.DebugUtilsMessageSeverityFlagsEXT,
	message_type: vk.DebugUtilsMessageTypeFlagsEXT,
	p_callback_data: ^vk.DebugUtilsMessengerCallbackDataEXT,
	p_user_data: rawptr,
) -> b32 {
	context = runtime.default_context()
    fmt.println()
    fmt.printf("MESSAGE: (")
    for ms in vk.DebugUtilsMessageSeverityFlagEXT {
        if ms in message_severity {
            fmt.printf("%v, ", ms)
        }
    }
    for t in vk.DebugUtilsMessageTypeFlagEXT {
        if t in message_type {
            fmt.printf("%v", t)
        }
    }
    fmt.printf(")\n")
    fmt.println("---------------")
    fmt.printf("%#v\n", p_callback_data.pMessage)
    fmt.println()

	return false
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

scroll_callback :: proc "c" (window: glfw.WindowHandle, xoffset, yoffset: f64) {
    context = runtime.default_context()
    fmt.println("scroll(", xoffset, yoffset, ")")
    // sync.mutex_guard(&queue_lock)
    // queue.push(&global_queue, yoffset)
}

cursor_pos_callback :: proc "c" (window: glfw.WindowHandle, xpos, ypos: f64) {
    context = runtime.default_context()
    fmt.println("cursor(", xpos, ypos, ")")
    // sync.mutex_guard(&queue_lock)
    // queue.push(&global_queue, [2]int{int(xpos), int(ypos)})
}

mouse_button_callback :: proc "c" (window: glfw.WindowHandle, button, action, mods: i32) {
    context = runtime.default_context()
    fmt.println("button(", button, action, mods, ")")
    
    if button == glfw.MOUSE_BUTTON_LEFT && action == glfw.PRESS {
        fmt.println("ADDED")
        // sync.mutex_guard(&queue_lock)
        // queue.push(&global_queue, Drag_Start{})
    } else if button == glfw.MOUSE_BUTTON_LEFT && action == glfw.RELEASE {
        fmt.println("RELEASED!")
        // sync.mutex_guard(&queue_lock)
        // queue.push(&global_queue, Drag_End{})
    }
}