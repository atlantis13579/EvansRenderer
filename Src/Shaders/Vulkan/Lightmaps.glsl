#version 450
// Vulkan GLSL is compiled directly at runtime. Surface draws use the same local
// positions and world/projection order as opaque rendering. Only clip-space Y
// is flipped; CPU atlas coordinates retain their logical top-left row order.
layout(set=0,binding=0,std140,row_major) uniform LightmapUniforms
{
    mat4 World, ViewProjection;
    vec4 DiffuseColor, Appearance;
    uvec4 Options;
} Lightmap;
const uint CompositePass = 1u;
#ifdef EVANS_VERTEX_STAGE
// Equal-depth receivers must use the opaque pass's invariant projection math.
invariant gl_Position;
precise gl_Position;
layout(location=0) in vec3 Position;
layout(location=1) in vec2 MaterialCoordinates;
layout(location=2) in vec2 LightmapCoordinates;
layout(location=3) in vec4 Color;
layout(location=0) out vec2 MaterialUV;
layout(location=1) out vec2 AtlasUV;
layout(location=2) out vec4 VertexColor;
const float ColorBreakpoint = 0.04045, LinearScale = 12.92;
const float TransferOffset = 0.055, TransferScale = 1.055, TransferPower = 2.4;
vec3 DecodeColor(vec3 encoded)
{
    encoded = clamp(encoded,0.0,1.0);
    vec3 low = encoded/LinearScale;
    vec3 high = pow((encoded+TransferOffset)/TransferScale,vec3(TransferPower));
    return mix(high,low,lessThanEqual(encoded,vec3(ColorBreakpoint)));
}
void main()
{
    MaterialUV = vec2(0.0); AtlasUV = vec2(0.0); VertexColor = vec4(0.0);
    if (Lightmap.Options.w == CompositePass)
    {
        vec2 triangle = vec2((gl_VertexIndex<<1)&2,gl_VertexIndex&2);
        gl_Position = vec4(triangle*2.0-1.0,0.0,1.0); return;
    }
    vec4 worldPosition = Lightmap.World*vec4(Position,1.0);
    gl_Position = Lightmap.ViewProjection*worldPosition;
    gl_Position.y = -gl_Position.y;
    MaterialUV = vec2(MaterialCoordinates.x,1.0-MaterialCoordinates.y);
    AtlasUV = LightmapCoordinates; VertexColor = vec4(DecodeColor(Color.rgb),Color.a);
}
#else
layout(location=0) in vec2 MaterialUV;
layout(location=1) in vec2 AtlasUV;
layout(location=2) in vec4 VertexColor;
layout(set=0,binding=1) uniform sampler MaterialSampler;
layout(set=0,binding=2) uniform texture2D FrontAtlas;
layout(set=0,binding=3) uniform texture2D BackAtlas;
layout(set=0,binding=4) uniform texture2D DiffuseMap;
layout(set=0,binding=5) uniform texture2D OpacityMap;
layout(set=0,binding=6) uniform texture2D SceneColor;
layout(set=0,binding=7) uniform texture2D Contribution;
layout(location=0) out vec4 OutColor;
const float HalfFloatMaximum = 65504.0, TexelCenter = 0.5;
vec3 IrradianceOverPi(bool front)
{
    if (!front && Lightmap.Options.z == 0u) return vec3(0.0);
    vec2 texel = AtlasUV*Lightmap.Appearance.w-TexelCenter;
    ivec2 first = ivec2(floor(texel)); vec2 blend = fract(texel);
    vec3 filtered = vec3(0.0);
    for (int row=0;row<2;++row) for (int column=0;column<2;++column)
    {
        ivec2 position = clamp(first+ivec2(column,row),ivec2(0),ivec2(int(Lightmap.Appearance.w)-1));
        vec3 sampleValue = front ? texelFetch(sampler2D(FrontAtlas,MaterialSampler),position,0).rgb :
            texelFetch(sampler2D(BackAtlas,MaterialSampler),position,0).rgb;
        float weight = (column == 0 ? 1.0-blend.x : blend.x)*(row == 0 ? 1.0-blend.y : blend.y);
        filtered += sampleValue*weight;
    }
    return max(filtered,vec3(0.0));
}
void main()
{
    if (Lightmap.Options.w == CompositePass)
    {
        ivec2 pixel = ivec2(gl_FragCoord.xy);
        vec4 scene = texelFetch(sampler2D(SceneColor,MaterialSampler),pixel,0);
        vec3 indirect = texelFetch(sampler2D(Contribution,MaterialSampler),pixel,0).rgb;
        float visibility = Lightmap.Options.x != 0u ? clamp(texelFetch(sampler2D(FrontAtlas,MaterialSampler),pixel,0).r,0.0,1.0) : 1.0;
        OutColor = vec4(clamp(scene.rgb+indirect*visibility,0.0,HalfFloatMaximum),scene.a); return;
    }
    vec4 material = VertexColor*Lightmap.DiffuseColor;
    if (Lightmap.Options.x != 0u) material *= texture(sampler2D(DiffuseMap,MaterialSampler),MaterialUV);
    float opacity = material.a;
    vec3 surfaceMap = Lightmap.Options.y != 0 ? texture(sampler2D(OpacityMap,MaterialSampler),MaterialUV).rgb : vec3(1, 1, 1);
    float diffuseWeight = Lightmap.Options.y > 1 ? 0.96 * (1 - Lightmap.Appearance.z * surfaceMap.b) : Lightmap.Appearance.z;
    opacity *= surfaceMap.r;
    if (clamp(opacity,0.0,1.0) < Lightmap.Appearance.y) discard;
    vec3 indirect = IrradianceOverPi(gl_FrontFacing)*clamp(material.rgb,0.0,1.0)*Lightmap.Appearance.x*diffuseWeight;
    OutColor = vec4(clamp(indirect,0.0,HalfFloatMaximum),1.0);
}
#endif
