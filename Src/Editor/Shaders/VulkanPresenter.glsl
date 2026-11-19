#version 450
// Vulkan editor presentation: pixel coordinates become clip coordinates through
// the viewport scale/translation, then a sampled UI texture modulates tint.
#ifdef EVANS_VERTEX_STAGE
layout(location = 0) in vec2 InPosition;
layout(location = 1) in vec2 InTexCoord;
layout(location = 2) in vec4 InColor;
layout(push_constant) uniform ViewportTransform { vec2 Scale; vec2 Translate; } Viewport;
layout(location = 0) out vec2 TexCoord;
layout(location = 1) out vec4 Color;
void main()
{
    gl_Position = vec4(InPosition * Viewport.Scale + Viewport.Translate, 0.0, 1.0);
    TexCoord = InTexCoord; Color = InColor;
}
#else
layout(set = 0, binding = 0) uniform sampler2D Image;
layout(location = 0) in vec2 TexCoord;
layout(location = 1) in vec4 Color;
layout(location = 0) out vec4 OutColor;
void main() { OutColor = Color * texture(Image, TexCoord); }
#endif
