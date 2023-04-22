package render

import vk "vendor:vulkan"

import "core:fmt"

/*
    These are generally useful across all instances of render
    and will probably need to be copy/pasted each time
*/

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