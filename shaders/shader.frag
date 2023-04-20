#version 450

layout (location = 0) in VertexInput {
    vec4 color;
} vertexColorInput;

layout (location = 0) out vec4 outFragColor;

// layout (binding = 1) uniform sampler2D texSampler;

void main() {
    outFragColor = vertexColorInput.color;
}