#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
// Native ES probe gather. Data atlases use logical row zero directly; scene
// textures and gl_FragCoord both use the native bottom-left raster convention.
// The inverse camera projection retains [0,1] depth and world +Y up.
// Visibility interpolation: https://jcgt.org/published/0008/02/01/
// Irradiance SH: https://graphics.stanford.edu/papers/envmap/
#ifdef EVANS_VERTEX_STAGE
void main()
{
    vec2 uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
#else
uniform mat4 InverseViewProjection;
uniform vec4 GridMinimum, GridMaximum, Viewport, Appearance, Visibility;
uniform uvec4 Dimensions, Options;
uniform sampler2D Coefficients, DistanceMoments, SceneColor, SceneDepth;
uniform sampler2D NormalRoughness, DiffuseOpacity, AmbientMetallic;
layout(location=0) out vec4 OutColor;
const float Pi = 3.14159265358979323846, CoefficientDecode = 4.0;
const float ProjectionEpsilon = 1e-7, VectorSquaredEpsilon = 1e-12, GatherWeightFloor = 1e-6;
const float FacingFloor = 0.05, DielectricF0 = 0.04, MaximumHalf = 65504.0, TexelCenter = 0.5;
const int CoefficientCount = 9;
const uint PbrModel = 3u;
bool Finite(vec4 value) { return !any(isnan(value)) && !any(isinf(value)); }
float SH(int index, vec3 n)
{
    if (index == 0) return sqrt(1.0/(4.0*Pi));
    if (index == 1) return n.y*sqrt(3.0/(4.0*Pi));
    if (index == 2) return n.z*sqrt(3.0/(4.0*Pi));
    if (index == 3) return n.x*sqrt(3.0/(4.0*Pi));
    if (index == 4) return n.x*n.y*sqrt(15.0/(4.0*Pi));
    if (index == 5) return n.y*n.z*sqrt(15.0/(4.0*Pi));
    if (index == 6) return (3.0*n.z*n.z-1.0)*sqrt(5.0/(16.0*Pi));
    if (index == 7) return n.x*n.z*sqrt(15.0/(4.0*Pi));
    return (n.x*n.x-n.y*n.y)*sqrt(15.0/(16.0*Pi));
}
ivec2 WrapOct(ivec2 cell)
{
    int size = int(Options.z);
    if (cell.x < 0) { cell.x = -cell.x-1; cell.y = size-1-cell.y; }
    if (cell.x >= size) { cell.x = 2*size-1-cell.x; cell.y = size-1-cell.y; }
    if (cell.y < 0) { cell.y = -cell.y-1; cell.x = size-1-cell.x; }
    if (cell.y >= size) { cell.y = 2*size-1-cell.y; cell.x = size-1-cell.x; }
    return cell;
}
vec2 ReadMoments(uint probe, vec3 direction)
{
    direction /= dot(abs(direction),vec3(1.0));
    vec2 oct = direction.xy;
    if (direction.z < 0.0) oct = (1.0-abs(oct.yx))*vec2(oct.x >= 0.0 ? 1.0 : -1.0,oct.y >= 0.0 ? 1.0 : -1.0);
    vec2 pixel = (oct*0.5+0.5)*float(Options.z)-TexelCenter;
    ivec2 base = ivec2(floor(pixel)); vec2 fraction = fract(pixel);
    uint columns = uint(Visibility.w);
    ivec2 tile = ivec2(probe%columns,probe/columns)*int(Options.z);
    vec2 result = vec2(0.0);
    for (int y=0; y<2; ++y) for (int x=0; x<2; ++x)
    {
        float weight = (x == 0 ? 1.0-fraction.x : fraction.x)*(y == 0 ? 1.0-fraction.y : fraction.y);
        result += texelFetch(DistanceMoments,tile+WrapOct(base+ivec2(x,y)),0).xy*weight;
    }
    return result;
}
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec4 scene = texelFetch(SceneColor,pixel,0); OutColor = scene;
    vec4 material = texelFetch(NormalRoughness,pixel,0);
    float depth = texelFetch(SceneDepth,pixel,0).r;
    if (depth >= 1.0 || material.w < 0.0 || !Finite(material) || dot(material.xyz,material.xyz) <= VectorSquaredEpsilon) return;
    vec2 uv = gl_FragCoord.xy*Viewport.zw;
    vec4 worldH = InverseViewProjection*vec4(uv*2.0-1.0,depth,1.0);
    if (!Finite(worldH) || abs(worldH.w) <= ProjectionEpsilon) return;
    vec3 world = worldH.xyz/worldH.w, normal = normalize(material.xyz);
    vec3 grid = (world-GridMinimum.xyz)/(GridMaximum.xyz-GridMinimum.xyz)*vec3(Dimensions.xyz-1u);
    if (any(lessThan(grid,vec3(0.0))) || any(greaterThan(grid,vec3(Dimensions.xyz-1u)))) return;
    uvec3 base = min(uvec3(floor(grid)),Dimensions.xyz-2u);
    vec3 fraction = grid-vec3(base), biased = world+normal*Appearance.y;
    vec3 sum = vec3(0.0); float total = 0.0;
    for (uint z=0u; z<2u; ++z) for (uint y=0u; y<2u; ++y) for (uint x=0u; x<2u; ++x)
    {
        uvec3 cell = base+uvec3(x,y,z);
        uint index = cell.x+Dimensions.x*(cell.y+Dimensions.y*cell.z);
        if (texelFetch(Coefficients,ivec2(0,int(index)),0).w <= 0.0) continue;
        vec3 probe = mix(GridMinimum.xyz,GridMaximum.xyz,vec3(cell)/vec3(Dimensions.xyz-1u));
        vec3 fromProbe = biased-probe; float distance = length(fromProbe);
        vec2 moments = ReadMoments(index,distance > ProjectionEpsilon ? fromProbe/distance : normal);
        float delta = max((distance-Appearance.z)/Visibility.z-moments.x,0.0);
        float variance = max(moments.y-moments.x*moments.x,Visibility.x);
        float visible = pow(clamp(variance/max(variance+delta*delta,VectorSquaredEpsilon),0.0,1.0),Appearance.w);
        vec3 towardProbe = probe-world;
        float facing = dot(towardProbe,towardProbe) > VectorSquaredEpsilon ? dot(normal,normalize(towardProbe)) : 1.0;
        float wrapped = max((facing+1.0)*0.5,FacingFloor);
        vec3 trilinear = vec3(x == 0u ? 1.0-fraction.x : fraction.x,y == 0u ? 1.0-fraction.y : fraction.y,z == 0u ? 1.0-fraction.z : fraction.z);
        float weight = trilinear.x*trilinear.y*trilinear.z*wrapped*wrapped*visible;
        vec3 irradiance = vec3(0.0);
        for (int coefficient=0; coefficient<CoefficientCount; ++coefficient)
            irradiance += texelFetch(Coefficients,ivec2(coefficient,int(index)),0).rgb*SH(coefficient,normal);
        sum += max(irradiance*CoefficientDecode,vec3(0.0))*weight; total += weight;
    }
    if (total <= GatherWeightFloor) return;
    vec3 diffuse = clamp(texelFetch(DiffuseOpacity,pixel,0).rgb,0.0,1.0);
    if (Options.y == PbrModel) diffuse *= (1.0-DielectricF0)*(1.0-clamp(texelFetch(AmbientMetallic,pixel,0).w,0.0,1.0));
    OutColor = vec4(clamp(scene.rgb+sum/total*diffuse*Appearance.x,0.0,MaximumHalf),scene.a);
}
#endif
