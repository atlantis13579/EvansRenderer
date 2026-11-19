#include <metal_stdlib>
using namespace metal;
// Native Metal surface lightmaps. Packed receiver vertices preserve the shared
// 44-byte stream ABI. Atlas filtering uses explicit integer reads so neither
// float linear-filter support nor generated mipmaps are prerequisites.
struct LightmapUniforms
{
    float4 World[4], ViewProjection[4];
    float4 DiffuseColor, Appearance;
    uint4 Options;
};
struct ReceiverData
{
    packed_float3 Position;
    packed_float2 MaterialUV, AtlasUV;
    packed_float4 Color;
};
struct LightmapVertex
{
    float4 Position [[position]];
    float2 MaterialUV, AtlasUV;
    float4 Color;
};
constant uint CompositePass = 1;
constant float MaximumRadiance = 65504, PixelCenter = 0.5f;
constant float SRGBThreshold = 0.04045f, SRGBLinearScale = 12.92f;
constant float SRGBOffset = 0.055f, SRGBScale = 1.055f, SRGBExponent = 2.4f;
float4 TransformPosition(constant float4* rows,float4 position)
{
    return float4(dot(rows[0],position),dot(rows[1],position),dot(rows[2],position),dot(rows[3],position));
}
float3 LinearColor(float3 value)
{
    value = saturate(value);
    return select(pow((value+SRGBOffset)/SRGBScale,float3(SRGBExponent)),value/SRGBLinearScale,value <= SRGBThreshold);
}
vertex LightmapVertex VSMain(uint id [[vertex_id]],device const ReceiverData* vertices [[buffer(0)]],
    constant LightmapUniforms& uniforms [[buffer(1)]])
{
    LightmapVertex result; result.MaterialUV = result.AtlasUV = float2(0); result.Color = float4(0);
    if (uniforms.Options.w == CompositePass)
    {
        float2 corner = float2((id<<1)&2,id&2); result.Position = float4(corner*2-1,0,1); return result;
    }
    ReceiverData vertex = vertices[id];
    float4 worldPosition = TransformPosition(uniforms.World,float4(float3(vertex.Position),1));
    result.Position = TransformPosition(uniforms.ViewProjection,worldPosition);
    result.MaterialUV = float2(vertex.MaterialUV.x,1-vertex.MaterialUV.y);
    result.AtlasUV = float2(vertex.AtlasUV); result.Color = float4(LinearColor(float4(vertex.Color).xyz),vertex.Color.w);
    return result;
}
float3 FilterAtlas(texture2d<float> frontAtlas,texture2d<float> backAtlas,float2 uv,float side,bool front,bool hasBack)
{
    if (!front && !hasBack) return float3(0);
    float2 location = uv*side-PixelCenter;
    int2 origin = int2(floor(location)); float2 fraction = fract(location); float3 value = float3(0);
    for (int row=0;row<2;++row) for (int column=0;column<2;++column)
    {
        uint2 cell = uint2(clamp(origin+int2(column,row),int2(0),int2(int(side)-1)));
        float3 sampleValue = front ? frontAtlas.read(cell).rgb : backAtlas.read(cell).rgb;
        value += sampleValue*(column == 0 ? 1-fraction.x : fraction.x)*(row == 0 ? 1-fraction.y : fraction.y);
    }
    return max(value,float3(0));
}
fragment float4 PSMain(LightmapVertex input [[stage_in]],bool front [[front_facing]],
    constant LightmapUniforms& uniforms [[buffer(0)]],texture2d<float> frontAtlas [[texture(0)]],
    texture2d<float> backAtlas [[texture(1)]],texture2d<float> diffuseMap [[texture(2)]],
    texture2d<float> opacityMap [[texture(3)]],texture2d<float> sceneColor [[texture(4)]],texture2d<float> contribution [[texture(5)]],
    sampler materialSampler [[sampler(0)]])
{
    if (uniforms.Options.w == CompositePass)
    {
        uint2 pixel = uint2(input.Position.xy); float4 scene = sceneColor.read(pixel);
        float visibility = uniforms.Options.x != 0 ? saturate(frontAtlas.read(pixel).r) : 1.0f;
        return float4(clamp(scene.rgb+contribution.read(pixel).rgb*visibility,0.0f,MaximumRadiance),scene.a);
    }
    float4 material = input.Color*uniforms.DiffuseColor;
    if (uniforms.Options.x != 0) material *= diffuseMap.sample(materialSampler,input.MaterialUV);
    float opacity = material.a;
    float3 surfaceMap = uniforms.Options.y != 0 ? opacityMap.sample(materialSampler,input.MaterialUV).rgb : float3(1, 1, 1);
    float diffuseWeight = uniforms.Options.y > 1 ? 0.96 * (1 - uniforms.Appearance.z * surfaceMap.b) : uniforms.Appearance.z;
    opacity *= surfaceMap.r;
    if (saturate(opacity) < uniforms.Appearance.y) discard_fragment();
    float3 irradiance = FilterAtlas(frontAtlas,backAtlas,input.AtlasUV,uniforms.Appearance.w,front,uniforms.Options.z != 0);
    float3 indirect = irradiance*saturate(material.rgb)*uniforms.Appearance.x*diffuseWeight;
    return float4(clamp(indirect,0.0f,MaximumRadiance),1);
}
