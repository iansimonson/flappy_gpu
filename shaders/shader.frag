#version 450

#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_debug_printf : enable


layout (location = 0) in VertexTexture {
    vec2 textureCoord;
} vertexTexture;
layout (location = 1) in VertexImgIndex {
    flat ivec2 imageIdx;
} vertexImg;

layout (location = 0) out vec4 outFragColor;

layout (binding = 1) uniform sampler samp;
layout (binding = 2) uniform texture2D textures[4];

void main() {
    //debugPrintfEXT("textureCoord: %d, %d", vertexImg.imageIdx.x, vertexImg.imageIdx.y);
    vec4 color = texture(sampler2D(textures[vertexImg.imageIdx.x], samp), vertexTexture.textureCoord);
    if (color.b > 0.9 && color.r > 0.9 && color.g < 0.3) {
        discard;
    } else {
        outFragColor = color;
    }
}