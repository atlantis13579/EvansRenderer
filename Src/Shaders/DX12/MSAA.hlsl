// DirectX 12 MSAA resolve stages. Raster attachments retain separate coverage
// and depth per sample; hardware ResolveSubresource averages linear radiance.
// Screen effects need one surface, selected by the closest stored depth. Fetch
// its attributes and velocity without blending across geometric discontinuities.
// https://learn.microsoft.com/windows/win32/api/d3d12/nf-d3d12-id3d12graphicscommandlist-resolvesubresource
Texture2DMS<float> OpaqueSamples : register(t0);
Texture2DMS<float4> DiffuseSamples : register(t1);
Texture2DMS<float4> NormalSamples : register(t2);
Texture2DMS<float4> SpecularSamples : register(t3);
Texture2DMS<float4> AmbientSamples : register(t4);
Texture2DMS<float4> EmissionSamples : register(t5);
Texture2DMS<float4> VelocitySamples : register(t6);
Texture2D<float4> FilteredOpaqueColor : register(t7);

float4 VSMain(uint corner : SV_VertexID) : SV_POSITION
{
    float2 coordinate = float2((corner << 1u) & 2u, corner & 2u);
    return float4(coordinate * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
}

struct RepresentativeSurface
{
    uint Sample;
    float Depth;
};
RepresentativeSurface SelectSurface(uint2 location)
{
    uint width, height, samples;
    OpaqueSamples.GetDimensions(width, height, samples);
    RepresentativeSurface surface;
    surface.Sample = 0u;
    surface.Depth = OpaqueSamples.Load(location, surface.Sample);
    [loop] for (uint candidate = 1u; candidate < samples; ++candidate)
    {
        float candidateDepth = OpaqueSamples.Load(location, candidate);
        if (candidateDepth < surface.Depth)
        {
            surface.Sample = candidate;
            surface.Depth = candidateDepth;
        }
    }
    return surface;
}

float PSDepth(float4 fragment : SV_POSITION) : SV_Depth
{
    return SelectSurface(uint2(fragment.xy)).Depth;
}

struct ResolvedMaterial
{
    float4 DiffuseOpacity : SV_TARGET0;
    float4 NormalRoughness : SV_TARGET1;
    float4 SpecularShininess : SV_TARGET2;
    float4 AmbientMetallic : SV_TARGET3;
    float4 Emission : SV_TARGET4;
    float Depth : SV_Depth;
};
ResolvedMaterial PSMaterial(float4 fragment : SV_POSITION)
{
    uint2 location = uint2(fragment.xy);
    RepresentativeSurface surface = SelectSurface(location);
    ResolvedMaterial material;
    material.Depth = surface.Depth;
    material.DiffuseOpacity = DiffuseSamples.Load(location, surface.Sample);
    material.NormalRoughness = NormalSamples.Load(location, surface.Sample);
    material.SpecularShininess = SpecularSamples.Load(location, surface.Sample);
    material.AmbientMetallic = AmbientSamples.Load(location, surface.Sample);
    material.Emission = EmissionSamples.Load(location, surface.Sample);
    return material;
}

float4 PSVelocity(float4 fragment : SV_POSITION) : SV_TARGET
{
    uint2 location = uint2(fragment.xy);
    RepresentativeSurface surface = SelectSurface(location);
    return VelocitySamples.Load(location, surface.Sample);
}

float4 PSRestoreColor(float4 fragment : SV_POSITION) : SV_TARGET
{
    // The output merger broadcasts this pixel color into every color sample.
    // Depth stays untouched for transparent occlusion. This approximates the
    // processed background at silhouettes containing multiple opaque surfaces.
    return FilteredOpaqueColor.Load(int3(int2(fragment.xy), 0));
}
