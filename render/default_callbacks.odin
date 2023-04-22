package render

import "vendor:glfw"

import "core:fmt"
import "core:runtime"

framebuffer_resize_callback :: proc "c" (window: glfw.WindowHandle, width, height: i32) {
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