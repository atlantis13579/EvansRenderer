#version 450
// Vulkan GLSL is compiled directly at runtime. No SPIR-V array or foreign shader
// is involved. Spherical harmonics store cosine-convolved radiance/pi; a separate
// angular kernel projects normalized distance moments into octahedral tiles.
// https://graphics.stanford.edu/papers/envmap/ ; https://jcgt.org/published/0008/02/01/
#ifdef EVANS_VERTEX_STAGE
void main()
{
    vec2 corner = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
#else
layout(set=0, binding=0, std140, row_major) uniform ProbeParameters
{
    mat4 InverseViewProjection;
    vec4 GridMinimum, GridMaximum, Viewport, Appearance, Visibility;
    uvec4 Dimensions, Options;
} Probe;
layout(set=0, binding=1) uniform texture2D SampleTable;
layout(set=0, binding=9) uniform sampler NearestSampler;
layout(location=0) out vec4 OutputValue;
const float Pi = 3.14159265358979323846, SHStorageScale = 0.25, HalfFloatLimit = 65504.0;
const float WeightEpsilon = 1e-12, TexelCenter = 0.5;
const uint ProjectSH = 0u;
const int SampleTableBorder = 1;
float BasisValue(int coefficient, vec3 w)
{
    switch (coefficient)
    {
    case 0: return sqrt(1.0 / (4.0 * Pi));
    case 1: return w.y * sqrt(3.0 / (4.0 * Pi));
    case 2: return w.z * sqrt(3.0 / (4.0 * Pi));
    case 3: return w.x * sqrt(3.0 / (4.0 * Pi));
    case 4: return w.x*w.y * sqrt(15.0 / (4.0 * Pi));
    case 5: return w.y*w.z * sqrt(15.0 / (4.0 * Pi));
    case 6: return (3.0*w.z*w.z - 1.0) * sqrt(5.0 / (16.0 * Pi));
    case 7: return w.x*w.z * sqrt(15.0 / (4.0 * Pi));
    default: return (w.x*w.x - w.y*w.y) * sqrt(15.0 / (16.0 * Pi));
    }
}
vec4 ReadSample(ivec2 location) { return texelFetch(sampler2D(SampleTable, NearestSampler), location, 0); }
vec3 OctahedralDirection(ivec2 cell)
{
    vec2 p = (vec2(cell) + TexelCenter) / float(Probe.Options.z) * 2.0 - 1.0;
    vec3 n = vec3(p, 1.0-abs(p.x)-abs(p.y));
    float folding = max(-n.z, 0.0);
    n.xy += vec2(n.x >= 0.0 ? -folding : folding, n.y >= 0.0 ? -folding : folding);
    return normalize(n);
}
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    if (Probe.Options.x == ProjectSH)
    {
        vec3 value = vec3(0.0);
        for (uint ray = 0u; ray < Probe.Dimensions.w; ++ray)
        {
            int x = int(ray) + SampleTableBorder;
            vec4 direction = ReadSample(ivec2(x,0));
            value += ReadSample(ivec2(x,pixel.y+SampleTableBorder)).rgb * (direction.w * BasisValue(pixel.x,direction.xyz));
        }
        float cosineIntegralOverPi = pixel.x == 0 ? 1.0 : pixel.x <= 3 ? 2.0/3.0 : 0.25;
        value *= cosineIntegralOverPi * SHStorageScale;
        OutputValue = vec4(clamp(value,-HalfFloatLimit,HalfFloatLimit),ReadSample(ivec2(0,pixel.y+SampleTableBorder)).w);
        return;
    }
    int size = int(Probe.Options.z), columns = int(Probe.Visibility.w);
    ivec2 tile = pixel / size;
    int probeIndex = tile.x + tile.y * columns;
    if (probeIndex >= int(Probe.Options.w)) { OutputValue = vec4(1,1,0,0); return; }
    vec3 axis = OctahedralDirection(pixel - tile*size);
    vec2 sum = vec2(0); float totalWeight = 0;
    for (uint ray = 0u; ray < Probe.Dimensions.w; ++ray)
    {
        int x = int(ray)+SampleTableBorder;
        float cosine = clamp(dot(axis,ReadSample(ivec2(x,0)).xyz),0.0,1.0);
        float weight = pow(cosine,Probe.Visibility.y);
        float distance = clamp(ReadSample(ivec2(x,probeIndex+SampleTableBorder)).w / Probe.Visibility.z,0.0,1.0);
        sum += weight * vec2(distance,distance*distance); totalWeight += weight;
    }
    OutputValue = vec4(totalWeight > WeightEpsilon ? sum/totalWeight : vec2(1),0,1);
}
#endif
