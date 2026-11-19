#version 450
// Runtime GLSL diffusion of a separately rendered direct-diffuse signal.
// Kernel bin masses are integrated on the CPU; no material preset is baked here.
// Separable diffusion: https://www.iryoku.com/separable-sss/ (Jimenez et al., 2015).
const int KernelCapacity=25;
const uint CopyPass=0u,HorizontalPass=1u,CompositePass=3u;
const float PixelCenter=0.5,ClearDepth=1.0,MinimumW=1e-7,NormalSquaredTolerance=1e-12;
const float MinimumCoverage=1e-6,MinimumWeight=1e-7,MinimumDepthSigma=1e-6,MaximumRadiance=65504.0;
layout(set=0,binding=0,std140,row_major) uniform SubsurfaceParameters
{
    mat4 InverseViewProjection;
    vec4 Viewport;
    vec4 Diffusion; // World distance, strength, maximum pixel support, kernel support.
    vec4 Rejection; // Optical depth scale, normal exponent, reserved, reserved.
    vec4 CameraForward;
    uvec4 Options; // Pass, tap count, reserved, reserved.
    vec4 Kernel[KernelCapacity]; // RGB bin masses, signed profile-distance multiple.
} Frame;
#ifdef EVANS_VERTEX_STAGE
void main()
{
    vec2 p=vec2((gl_VertexIndex<<1)&2,gl_VertexIndex&2); gl_Position=vec4(p*2.0-1.0,0.0,1.0);
}
#else
layout(set=0,binding=1) uniform sampler2D DiffuseSignal;
layout(set=0,binding=2) uniform sampler2D SceneDepth;
layout(set=0,binding=3) uniform sampler2D NormalRoughness;
layout(set=0,binding=4) uniform sampler2D SceneColor;
layout(set=0,binding=5) uniform sampler2D OriginalDiffuse;
layout(location=0) out vec4 OutColor;
bool Unproject(vec2 uv,float depth,out vec3 world)
{
    vec4 h=Frame.InverseViewProjection*vec4(uv.x*2.0-1.0,1.0-uv.y*2.0,depth,1.0);
    world=vec3(0.0); if (abs(h.w)<MinimumW) return false;
    world=h.xyz/h.w; return !any(isnan(world)) && !any(isinf(world));
}
bool ReadSurface(ivec2 pixel,out vec3 world,out vec3 normal)
{
    world=normal=vec3(0.0);
    if (any(lessThan(pixel,ivec2(0))) || any(greaterThanEqual(pixel,ivec2(Frame.Viewport.xy)))) return false;
    vec4 n=texelFetch(NormalRoughness,pixel,0); float depth=texelFetch(SceneDepth,pixel,0).r,squared=dot(n.xyz,n.xyz);
    if (!(depth<ClearDepth && n.w>=0.0 && squared>NormalSquaredTolerance) || isnan(squared) || isinf(squared)) return false;
    normal=n.xyz*inversesqrt(squared); return Unproject((vec2(pixel)+PixelCenter)*Frame.Viewport.zw,depth,world);
}
void Gather(ivec2 pixel,float linearWeight,vec3 kernel,vec3 center,vec3 normal,float sigma,inout vec3 radiance,inout vec3 weights)
{
    if (linearWeight<=0.0) return;
    vec3 world,n; if (!ReadSurface(pixel,world,n)) return;
    vec4 signal=texelFetch(DiffuseSignal,pixel,0); if (signal.a<=MinimumCoverage) return;
    float distance=dot(world-center,Frame.CameraForward.xyz)/sigma;
    float angular=Frame.Rejection.y>0.0 ? pow(clamp(dot(normal,n),0.0,1.0),Frame.Rejection.y) : 1.0;
    vec3 weight=kernel*(linearWeight*exp(-distance*distance/2.0)*angular*signal.a);
    radiance+=signal.rgb/signal.a*weight; weights+=weight;
}
void main()
{
    ivec2 pixel=ivec2(gl_FragCoord.xy); vec4 center=texelFetch(DiffuseSignal,pixel,0);
    if (Frame.Options.x==CopyPass) { OutColor=center; return; }
    if (Frame.Options.x==CompositePass)
    {
        vec4 scene=texelFetch(SceneColor,pixel,0);
        vec3 delta=(center.rgb-texelFetch(OriginalDiffuse,pixel,0).rgb)*Frame.Diffusion.y;
        OutColor=vec4(clamp(scene.rgb+delta,0.0,MaximumRadiance),scene.a); return;
    }
    vec3 world,normal; if (center.a<=MinimumCoverage || !ReadSurface(pixel,world,normal)) { OutColor=center; return; }
    ivec2 axis=Frame.Options.x==HorizontalPass ? ivec2(1,0) : ivec2(0,1); vec3 adjacent;
    float depth=texelFetch(SceneDepth,pixel,0).r;
    if (!Unproject((vec2(pixel+axis)+PixelCenter)*Frame.Viewport.zw,depth,adjacent)) { OutColor=center; return; }
    float footprint=length(adjacent-world); if (!(footprint>0.0)) { OutColor=center; return; }
    float support=Frame.Diffusion.x*Frame.Diffusion.w/footprint;
    float scale=support>Frame.Diffusion.z && support>0.0 ? Frame.Diffusion.z/support : 1.0;
    float pixelScale=Frame.Diffusion.x*scale/footprint,sigma=max(Frame.Diffusion.x*scale*Frame.Rejection.x,MinimumDepthSigma);
    vec3 radiance=vec3(0.0),weights=vec3(0.0);
    for (uint tap=0u;tap<min(Frame.Options.y,uint(KernelCapacity));++tap)
    {
        float offset=Frame.Kernel[tap].w*pixelScale,lower=floor(offset),fraction=offset-lower;
        ivec2 q=pixel+axis*int(lower);
        Gather(q,1.0-fraction,Frame.Kernel[tap].rgb,world,normal,sigma,radiance,weights);
        Gather(q+axis,fraction,Frame.Kernel[tap].rgb,world,normal,sigma,radiance,weights);
    }
    OutColor=vec4(clamp(radiance/max(weights,vec3(MinimumWeight))*center.a,0.0,MaximumRadiance),center.a);
}
#endif
