#version 450

layout (location = 0) in VertexTexture {
    vec2 textureCoord;
    ivec2 imageIdx;
} vertexTexture;

layout (location = 0) out vec4 outFragColor;

layout(push_constant) uniform PER_OBJECT {
    int imgIdx;
};

layout (binding = 1) uniform sampler samp;
layout (binding = 2) uniform texture2D textures[4];

void main() {
    outFragColor = vec4(texture(sampler2D(textures[vertexTexture.imageIdx.x], samp), vertexTexture.textureCoord).xyz, 1);
}