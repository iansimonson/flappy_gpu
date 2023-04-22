package render

import vk "vendor:vulkan"

import "core:fmt"
import "core:runtime"

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