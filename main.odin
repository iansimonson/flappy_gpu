package main

import "render"

import "vendor:glfw"
import vk "vendor:vulkan"
import stbi "vendor:stb/image"
import "core:fmt"
import "core:mem"
import "core:time"
import "core:math/rand"

UniformData :: struct {
    bird: [4]f32,
    bird_texture_coords: [4]f32,
    column0: [4]f32,
    column1: [4]f32,
    stripes_texture_coords: [4]f32,
    render_points: b32,
    padding0: [3]f32,
    padding: [2][4]f32,
}

FLAPPY_FULL_SIZE := [2]f32{
    820.0, 328.0,
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

load_texture :: proc(file_name: cstring) -> render.Texture {
    img := load_image(file_name)
    defer free_image(img)
    txt := render.global_render_create_texture(img)
    return txt
}

unload_texture :: proc(texture: render.Texture) {
    render.global_render_destroy_texture(texture)
}

@(deferred_out = unload_texture)
scope_texture :: proc(file_name: cstring) -> render.Texture {
    return load_texture(file_name)
}

jump: bool
render_points: bool

State :: enum {
    Idle,
    Started,
    Dead,
}

global_state: State

main :: proc() {
    render.global_render_init()
    defer render.global_render_destroy()

    glfw.SetMouseButtonCallback(render.global_renderer.window, mouse_callback)
    glfw.SetKeyCallback(render.global_renderer.window, key_callback)

    flappy_texture := scope_texture("./assets/flappy_bird_spritesheet.jpg")
    pipe_texture := scope_texture("./assets/mario_pipe.png")
    ground_texture := scope_texture("./assets/Ground cover thing.png")
    stripe_texture := scope_texture("./assets/shitty_stripes.png")
    background_texture := scope_texture("./assets/background.png")
    red_texture := scope_texture("./assets/red.png")


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
    defer vk.DestroySampler(render.global_renderer.device, sampler, nil)

    starting_column_locations := [2][4]f32{
        {3.15, rand.float32_range(-0.7, 0.5), 0.15, 0.75},
        {4.3, rand.float32_range(-0.7, 0.5), 0.15, 0.75},
    }

    bird_animation := [4][4]f32{
        {140.0 / FLAPPY_FULL_SIZE.x, 100.0 / FLAPPY_FULL_SIZE.y, 175.0/FLAPPY_FULL_SIZE.x, 126.0/FLAPPY_FULL_SIZE.y},
        {(140.0 + 175.0) / FLAPPY_FULL_SIZE.x, 100.0 / FLAPPY_FULL_SIZE.y, 175.0/FLAPPY_FULL_SIZE.x, 126.0/FLAPPY_FULL_SIZE.y},
        {(140.0 + 2 * 175.0) / FLAPPY_FULL_SIZE.x, 100.0 / FLAPPY_FULL_SIZE.y, 175.0/FLAPPY_FULL_SIZE.x, 126.0/FLAPPY_FULL_SIZE.y},
        {(140.0 + 175.0) / FLAPPY_FULL_SIZE.x, 100.0 / FLAPPY_FULL_SIZE.y, 175.0/FLAPPY_FULL_SIZE.x, 126.0/FLAPPY_FULL_SIZE.y}, // goes back down
    }

    // columns should always be a distance of 1 apart
    game_data := UniformData{
        bird = {-0.1, 0, 0.12, 0.06},
        bird_texture_coords = bird_animation[0],
        column0 = starting_column_locations[0],
        column1 = starting_column_locations[1],
        stripes_texture_coords = {0, 0, 0.5, 1},
        render_points = b32(render_points),
    }

    // fmt.println(game_data.bird_texture_coords)

    mapped_game_data: [2]rawptr
    mapped_game_data[0] = raw_uniform
    mapped_game_data[1] = rawptr(uintptr(raw_uniform) + uintptr(size_of(UniformData)))
    mem.copy(mapped_game_data[0], &game_data, size_of(UniformData))
    mem.copy(mapped_game_data[1], &game_data, size_of(UniformData))


    // I did this for 4 textures b/c I'll need more in a little bit
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
            imageView = ground_texture.image_view,
        },
        {
            imageLayout = .SHADER_READ_ONLY_OPTIMAL,
            imageView = stripe_texture.image_view,
        },
        {
            imageLayout = .SHADER_READ_ONLY_OPTIMAL,
            imageView = background_texture.image_view,
        },
        {
            imageLayout = .SHADER_READ_ONLY_OPTIMAL,
            imageView = red_texture.image_view,
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

    previous_time := time.now()

    step: int
    gravity := f32(3.0)
    velocity := f32(0)
    score: int
    scored: [2]bool
    animation_time: f64 = 0
    for {
        using render.global_renderer
        // fmt.printf("STEP: %v\nGame State:\n%#v", step, game_data)
        current_frame := step % render.MAX_FRAMES_IN_FLIGHT
        now := time.now()
        diff := time.diff(previous_time, now)
        seconds := time.duration_seconds(diff)
        previous_time = now
        column_change := [4]f32{-f32(seconds / 4.5) * 2.0, 0, 0, 0}

        animation_time += seconds * 10
        bird_frame := int(animation_time) % 4
        game_data.bird_texture_coords = bird_animation[bird_frame]

        game_data.stripes_texture_coords -= column_change / 4.3
        if game_data.stripes_texture_coords.x > 0.5 {
            game_data.stripes_texture_coords.x = 0
        }

        game_data.render_points = b32(render_points)

        switch global_state {
        case .Started:
            hit_column :: proc(bird, column: [4]f32) -> bool {
                within_box_x := bird.x + bird.z >= column.x - column.z && bird.x - bird.z <= column.x + column.z
                within_box_y := bird.y - bird.w >= column.y - 0.25 && bird.y + bird.w <= column.y + 0.25

                if within_box_x do return !within_box_y
                else do return false
            }

            hit_ground :: proc(bird: [4]f32) -> bool {
                return bird.y + bird.w >= 0.8
            }

            using game_data
            if jump {
                velocity = clamp(velocity + -1.4, -2.0, -1)
                jump = false
            }

            velocity += f32(seconds) * gravity
            bird.y += velocity*f32(seconds)
    
    
            column0 += column_change
            column1 += column_change
            if column0.x < (-1 - column0.z) {
                column0.x = 1.0 + column0.z
                column0.y = rand.float32_range(-0.7, 0.5)
                scored[0] = false
            }
            if column1.x < (-1 - column1.z) {
                column1.x = 1.0 + column1.z
                column1.y = rand.float32_range(-0.7, 0.5)
                scored[0] = false
            }

            if column0.x < bird.x {
                score += int(!scored[0])
                scored[0] = true
            }
            if column1.x < bird.x {
                score += int(!scored[1])
                scored[1] = true
            }

            if hit_column(bird, column0) || hit_column(bird, column1) || hit_ground(bird) {
                fmt.printf("bird:\n%#v\ncolum0:\n%#v\ncolumn1:\n%#v", bird, column0, column1)
                fmt.println("GAME OVER")
                fmt.println("SCORE:", score)
                column0 = starting_column_locations[0]
                column1 = starting_column_locations[1]
                scored = false
                bird.y = 0
                score = 0
                global_state = .Dead
            }
        case .Dead:
            // TODO: print "best"
        case .Idle:
        }

        render.global_render_handle_events()
        if !render.global_render_running() do break

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

        mem.copy(mapped_game_data[current_frame], &game_data, size_of(UniformData))

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
    }
    vk.DeviceWaitIdle(render.global_renderer.device)

}

mouse_callback :: proc "c" (window: glfw.WindowHandle, button, action, mods: i32) {
    if button == glfw.MOUSE_BUTTON_LEFT && action == glfw.PRESS {
        jump = true
        if global_state == .Idle || global_state == .Dead {
            global_state = .Started
        }
    }
}

key_callback :: proc "c" (window: glfw.WindowHandle, key, scancode, action, mods: i32) {
    if key == glfw.KEY_S && action == glfw.PRESS {
        render_points = !render_points
    }
}