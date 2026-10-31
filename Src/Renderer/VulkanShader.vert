#version 450

layout(location = 0) in vec3 InPosition;
layout(location = 1) in vec3 InNormal;
layout(location = 2) in vec4 InColor;

layout(set = 0, binding = 0, std140) uniform DrawConstants
{
    mat4 World;
    mat4 ViewProjection;
    mat4 LightViewProjection;
    vec4 PipelineColor;
    vec4 Parameters;
} Constants;

layout(location = 0) out vec3 WorldNormal;
layout(location = 1) out vec4 VertexColor;
layout(location = 2) out vec4 LightPosition;

void main()
{
    vec4 world = Constants.World * vec4(InPosition, 1.0);
    vec3 normal = normalize(mat3(Constants.World) * InNormal);
    if (Constants.Parameters.z > 0.5)
    {
        world.xyz += normal * Constants.Parameters.y;
    }

    vec4 clip = Constants.ViewProjection * world;
    clip.y = -clip.y;
    gl_Position = clip;
    WorldNormal = normal;
    VertexColor = InColor;
    LightPosition = Constants.LightViewProjection * world;
}
