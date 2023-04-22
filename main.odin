package main

import "render"

import "vendor:glfw"
import vk "vendor:vulkan"
import stbi "vendor:stb/image"
import "core:fmt"
import "core:mem"

UniformData :: struct {
    bird: [4]f32,
    column0: [4]f32,
    column1: [4]f32,
    padding: [4]f32,
}

FLAPPY_OFFSETS := [2]int{
    140, 100,
}

FLAPPY_ENDS := [2]int{
    666, 226,
}

Frames := [][2]int{
    {140, 100},
    {315, 100},
    {490, 100},
}

SIZES := [2]int{
    175, 126
}

load_image :: proc(filename: cstring) -> (image: render.Image) {
    width, height, channels: i32
    desired_channels: i32 = 4
    pixels := stbi.load(filename, &width, &height, &channels, desired_channels)
    pixels_as_u32 := ([^][4]u8)(pixels)
    image.width, image.height = int(width), int(height)
    image.data = pixels_as_u32[:width*height]
    return
}

free_image :: proc(image: render.Image) {
    stbi.image_free(raw_data(image.data))
}

main :: proc() {
    render.global_render_init()
    defer render.global_render_destroy()


    flappy_image := load_image("./assets/flappy_bird_spritesheet.jpg")
    defer free_image(flappy_image)
    pipe_image := load_image("./assets/mario_pipe.png")
    defer free_image(pipe_image)

    flappy_texture := render.global_render_create_texture(flappy_image)
    defer render.global_render_destroy_texture(flappy_texture)
    pipe_texture := render.global_render_create_texture(pipe_image)
    defer render.global_render_destroy_texture(pipe_texture)

    write_handle := render.register_writer()

    uniform_buffer_size := render.MAX_FRAMES_IN_FLIGHT * size_of(UniformData)
    uniform_buffer, uniform_memory := render.create_buffer(render.global_renderer.physical_device, render.global_renderer.device, vk.DeviceSize(uniform_buffer_size), {.UNIFORM_BUFFER}, {.HOST_VISIBLE, .HOST_COHERENT})
    defer render.destroy_buffer(render.global_renderer.device, uniform_buffer, uniform_memory)
    raw_uniform: rawptr
	vk.MapMemory(render.global_renderer.device, uniform_memory, 0, vk.DeviceSize(uniform_buffer_size), nil, &raw_uniform)

    sampler: vk.Sampler
    {
        using render.global_renderer
        props: vk.PhysicalDeviceProperties
        vk.GetPhysicalDeviceProperties(physical_device, &props)

        if vk.CreateSampler(device, &vk.SamplerCreateInfo{
            sType = .SAMPLER_CREATE_INFO,
            magFilter = .LINEAR,
            minFilter = .LINEAR,
            mipmapMode = .LINEAR,
            addressModeU = .REPEAT,
            addressModeV = .REPEAT,
            addressModeW = .REPEAT,
            anisotropyEnable = false,
            maxAnisotropy = props.limits.maxSamplerAnisotropy,
            borderColor = .INT_OPAQUE_BLACK,
            unnormalizedCoordinates = false,
            compareEnable = false,
            compareOp = .ALWAYS,
        }, nil, &sampler) != .SUCCESS {
            panic("Could not create sampler")
        }
    }

    game_data := UniformData{
        bird = {-0.5, 0, 0.1, 0.1},
        column0 = {0, 0, 0.3, 1.0},
        column1 = {0.7, 0, 0.3, 1.0},
    }

    mem.copy(raw_uniform, &game_data, size_of(UniformData))
    mem.copy(rawptr(uintptr(raw_uniform) + uintptr(size_of(UniformData))), &game_data, size_of(UniformData))


    image_infos := []vk.DescriptorImageInfo{
        {
            imageLayout = .SHADER_READ_ONLY_OPTIMAL,
            imageView = flappy_texture.image_view,
        },
        {
            imageLayout = .SHADER_READ_ONLY_OPTIMAL,
            imageView = pipe_texture.image_view,
        },
        {
            imageLayout = .SHADER_READ_ONLY_OPTIMAL,
            imageView = pipe_texture.image_view,
        },
        {
            imageLayout = .SHADER_READ_ONLY_OPTIMAL,
            imageView = pipe_texture.image_view,
        },
    }

    for i in 0..<render.MAX_FRAMES_IN_FLIGHT {
        descriptor_sets := []vk.WriteDescriptorSet{
            {
                sType = .WRITE_DESCRIPTOR_SET,
                dstSet = render.global_renderer.descriptors.sets[i],
                dstBinding = 0,
                dstArrayElement = 0,
                descriptorType = .UNIFORM_BUFFER,
                descriptorCount = 1,
                pBufferInfo = &vk.DescriptorBufferInfo{
                    buffer = uniform_buffer,
                    offset = vk.DeviceSize(i * size_of(UniformData)),
                    range = size_of(UniformData),
                },
            },
            {
                sType = .WRITE_DESCRIPTOR_SET,
                dstSet = render.global_renderer.descriptors.sets[i],
                dstBinding = 1,
                dstArrayElement = 0,
                descriptorType = .SAMPLER,
                descriptorCount = 1,
                pImageInfo = &vk.DescriptorImageInfo{
                    sampler = sampler,
                },
            },
            {
                sType = .WRITE_DESCRIPTOR_SET,
                dstSet = render.global_renderer.descriptors.sets[i],
                dstBinding = 2,
                dstArrayElement = 0,
                descriptorType = .SAMPLED_IMAGE,
                descriptorCount = u32(len(image_infos)),
                pImageInfo = raw_data(image_infos),
            },
        }
        vk.UpdateDescriptorSets(render.global_renderer.device, u32(len(descriptor_sets)), raw_data(descriptor_sets), 0, nil)
    }

    step: int
    for {
        using render.global_renderer

        render.global_render_handle_events()
        if !render.global_render_running() do break

        current_frame := step % render.MAX_FRAMES_IN_FLIGHT
        vk.WaitForFences(device, 1, &syncs.inflight_fences[current_frame], true, max(u64))

        image_index: u32 = max(u32)
        result := vk.AcquireNextImageKHR(device, swapchain.swapchain, max(u64), syncs.image_avails[current_frame], {}, &image_index)
        if result == .ERROR_OUT_OF_DATE_KHR {
            panic("Need to recreate swapchain")
        }
        else if result != .SUCCESS && result != .SUBOPTIMAL_KHR {
            panic("Couldn't get swapchain")
        }

        assert(image_index == u32(current_frame))
        vk.ResetFences(device, 1, &syncs.inflight_fences[current_frame])

        command_buffer := render.get_command_buffer(write_handle, current_frame)
        render.vk_assert(vk.BeginCommandBuffer(command_buffer, &vk.CommandBufferBeginInfo{
            sType = .COMMAND_BUFFER_BEGIN_INFO,
        }))

        frag_shader_image_barriers := [?]vk.ImageMemoryBarrier2{
            { // swapchain image
                sType = .IMAGE_MEMORY_BARRIER_2,
                dstAccessMask = {.COLOR_ATTACHMENT_WRITE},
                dstStageMask = {.COLOR_ATTACHMENT_OUTPUT},
                oldLayout = .UNDEFINED,
                newLayout = .COLOR_ATTACHMENT_OPTIMAL,
                image = swapchain.images[current_frame],
                subresourceRange = {
                    aspectMask = {.COLOR},
                    baseMipLevel = 0,
                    levelCount = 1,
                    baseArrayLayer = 0,
                    layerCount = 1,
                },
            },
        }

        dependency_info := vk.DependencyInfoKHR{
            sType = .DEPENDENCY_INFO_KHR,
            imageMemoryBarrierCount = len(frag_shader_image_barriers),
            pImageMemoryBarriers = raw_data(frag_shader_image_barriers[:]),
        }
        vk.CmdPipelineBarrier2(command_buffer, &dependency_info)

        clear_value := vk.ClearValue{color = vk.ClearColorValue{float32 = {0, 0, 0, 1}}}
        vk.CmdBeginRenderingKHR(command_buffer, &vk.RenderingInfoKHR{
            sType = .RENDERING_INFO_KHR,
            renderArea = {
                extent = swapchain.extent,
            },
            layerCount = 1,
            colorAttachmentCount = 1,
            pColorAttachments = &vk.RenderingAttachmentInfoKHR{
                sType = .RENDERING_ATTACHMENT_INFO_KHR,
                imageView = swapchain.image_views[current_frame],
                imageLayout = .ATTACHMENT_OPTIMAL_KHR,
                loadOp = .CLEAR,
                storeOp = .STORE,
                clearValue = clear_value,
            },
        })
        vk.CmdSetViewport(command_buffer, 0, 1, &vk.Viewport{
            width = f32(swapchain.extent.width),
            height = f32(swapchain.extent.height),
            maxDepth = 1,
        })
        vk.CmdSetScissor(command_buffer, 0, 1, &vk.Rect2D{
            extent = swapchain.extent,
        })

        vk.CmdBindPipeline(command_buffer, .GRAPHICS, pipeline.pipeline)
        vk.CmdBindDescriptorSets(command_buffer, .GRAPHICS, pipeline.layout, 0, 1, &descriptors.sets[current_frame], 0, nil)
        vk.CmdDrawMeshTasksEXT(command_buffer, 1, 1, 1)

        vk.CmdEndRenderingKHR(command_buffer)

        vk.CmdPipelineBarrier(command_buffer, {.COLOR_ATTACHMENT_OUTPUT}, {.BOTTOM_OF_PIPE}, nil, 0, nil, 0, nil, 1, &vk.ImageMemoryBarrier{
            sType = .IMAGE_MEMORY_BARRIER,
            srcAccessMask = {.COLOR_ATTACHMENT_WRITE},
            oldLayout = .COLOR_ATTACHMENT_OPTIMAL,
            newLayout = .PRESENT_SRC_KHR,
            image = swapchain.images[current_frame],
            subresourceRange = {
                aspectMask = {.COLOR},
                baseMipLevel = 0,
                levelCount = 1,
                baseArrayLayer = 0,
                layerCount = 1,
            },
        })

        render.vk_assert(vk.EndCommandBuffer(command_buffer))

        dst_stage_mask := vk.PipelineStageFlags{.COLOR_ATTACHMENT_OUTPUT}

        render.vk_assert(vk.QueueSubmit(main_queue, 1, &vk.SubmitInfo{
            sType = .SUBMIT_INFO,
            commandBufferCount = 1,
            pCommandBuffers = &command_buffer,
            waitSemaphoreCount = 1,
            pWaitSemaphores = &syncs.image_avails[current_frame],
            pWaitDstStageMask = &dst_stage_mask,
            signalSemaphoreCount = 1,
            pSignalSemaphores = &syncs.render_finishes[current_frame],
        }, syncs.inflight_fences[current_frame]))

        render.vk_assert(vk.QueuePresentKHR(main_queue, &vk.PresentInfoKHR{
            sType = .PRESENT_INFO_KHR,
            waitSemaphoreCount = 1,
            pWaitSemaphores = &syncs.render_finishes[current_frame],
            swapchainCount = 1,
            pSwapchains = &swapchain.swapchain,
            pImageIndices = &image_index,
            pResults = nil,
        }))

        step += 1
        fmt.println("step!")
    }

}

general := `
Single point for flappy bird location
Flappy Bird width/height (constants, radius?)

Two points for the current column location
Column width (radius) / inter-column height (also radius) (constants)

constant vertices for the background/squares/little bar that moves

compute shader for determining if game over?

then need to figure out the truetype thing
`

GOAL := `

The purpose of this is to make a flappy bird clone. So the window should be phone vertical size / aspect ratio
Flappy bird only has 2 sets of columns, the ones you're going through and the other ones.
Once the closest columns go off screen to the left the new ones appear on the right (when right side of column crosses edge of screen reset to edge of other screen)

Maybe we can learn mmesh/task shaders? I'd like to put almost everything on compute shaders each frame

Couple plans could be:
1. Mesh shader that converts a single point to the 2 rectangles of the column?
2. 

1. Make a box
2. Make 4 boxes for the columns
3. Columns randomize center height and then they just go until offscreen
4. It'll essentially be a UBO for each point/column, yeah that makes sense



background -> fixed
2 points for the columns
1 point for the bird
foreground rectangles (ground) -> fixed at first

Counter (stb_truetype? look this up later)
`