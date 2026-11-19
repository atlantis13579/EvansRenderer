#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
// Native ES lightmaps: data rows are uploaded in logical atlas order. Scene and
// contribution images use GL raster coordinates, so their texelFetch operations
// need no Y inversion. Receiver clip depth alone converts [0,1] to GL [-1,1].
uniform mat4 World, ViewProjection;
uniform vec4 DiffuseColor, Appearance;
uniform uvec4 Options;
const uint CompositePass = 1u;
#ifdef EVANS_VERTEX_STAGE
// Match camera-depth producers and consumers across separately linked programs.
// GLSL ES 3.00, section 4.6: invariant raster positions prevent replay holes.
invariant gl_Position;
layout(location=0) in vec3 InPosition;
layout(location=1) in vec2 InMaterialUV;
layout(location=2) in vec2 InAtlasUV;
layout(location=3) in vec4 InColor;
out vec2 MaterialUV, AtlasUV;
out vec4 VertexColor;
const float SRGBThreshold = 0.04045, SRGBLinearScale = 12.92;
const float SRGBOffset = 0.055, SRGBScale = 1.055, SRGBExponent = 2.4;
vec3 LinearVertex(vec3 encoded)
{
    encoded = clamp(encoded,0.0,1.0);
    vec3 linearSegment = encoded/SRGBLinearScale;
    vec3 powerSegment = pow((encoded+SRGBOffset)/SRGBScale,vec3(SRGBExponent));
    return mix(powerSegment,linearSegment,lessThanEqual(encoded,vec3(SRGBThreshold)));
}
void main()
{
    MaterialUV = vec2(0.0); AtlasUV = vec2(0.0); VertexColor = vec4(0.0);
    if (Options.w == CompositePass)
    {
        vec2 corner = vec2((gl_VertexID<<1)&2,gl_VertexID&2);
        gl_Position = vec4(corner*2.0-1.0,0.0,1.0); return;
    }
    vec4 worldPosition = World*vec4(InPosition,1.0);
    vec4 clip = ViewProjection*worldPosition;
    clip.z = clip.z*2.0-clip.w;
    gl_Position = clip;
    MaterialUV = vec2(InMaterialUV.x,1.0-InMaterialUV.y);
    AtlasUV = InAtlasUV; VertexColor = vec4(LinearVertex(InColor.rgb),InColor.a);
}
#else
in vec2 MaterialUV, AtlasUV;
in vec4 VertexColor;
uniform sampler2D FrontAtlas, BackAtlas, DiffuseMap, OpacityMap, SceneColor, Contribution;
layout(location=0) out vec4 OutColor;
const float MaximumRadiance = 65504.0, PixelCenter = 0.5;
vec3 SampleAtlas(bool front)
{
    if (!front && Options.z == 0u) return vec3(0.0);
    vec2 location = AtlasUV*Appearance.w-PixelCenter;
    ivec2 origin = ivec2(floor(location)); vec2 fraction = fract(location); vec3 result = vec3(0.0);
    // Explicit bilinear interpolation also works without float-texture linear
    // filtering extensions. Chart gutters isolate all four base-level taps.
    for (int y=0;y<2;++y) for (int x=0;x<2;++x)
    {
        ivec2 cell = clamp(origin+ivec2(x,y),ivec2(0),ivec2(int(Appearance.w)-1));
        vec3 value = front ? texelFetch(FrontAtlas,cell,0).rgb : texelFetch(BackAtlas,cell,0).rgb;
        result += value*(x == 0 ? 1.0-fraction.x : fraction.x)*(y == 0 ? 1.0-fraction.y : fraction.y);
    }
    return max(result,vec3(0.0));
}
void main()
{
    if (Options.w == CompositePass)
    {
        ivec2 pixel = ivec2(gl_FragCoord.xy); vec4 scene = texelFetch(SceneColor,pixel,0);
        float visibility = Options.x != 0u ? clamp(texelFetch(FrontAtlas,pixel,0).r,0.0,1.0) : 1.0;
        OutColor = vec4(clamp(scene.rgb+texelFetch(Contribution,pixel,0).rgb*visibility,0.0,MaximumRadiance),scene.a); return;
    }
    vec4 material = VertexColor*DiffuseColor;
    if (Options.x != 0u) material *= texture(DiffuseMap,MaterialUV);
    float opacity = material.a;
    vec3 surfaceMap = Options.y != 0u ? texture(OpacityMap,MaterialUV).rgb : vec3(1, 1, 1);
    float diffuseWeight = Options.y > 1u ? 0.96 * (1.0 - Appearance.z * surfaceMap.b) : Appearance.z;
    opacity *= surfaceMap.r;
    if (clamp(opacity,0.0,1.0) < Appearance.y) discard;
    vec3 indirect = SampleAtlas(gl_FrontFacing)*clamp(material.rgb,0.0,1.0)*Appearance.x*diffuseWeight;
    OutColor = vec4(clamp(indirect,0.0,MaximumRadiance),1.0);
}
#endif
