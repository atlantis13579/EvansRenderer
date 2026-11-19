#version 310 es
precision highp float;
precision highp int;
// Independent ES diffusion of separated direct diffuse radiance. Both image
// axes use bottom-left storage; only the world-space camera matrix is shared.
// Jimenez et al. (2015): https://www.iryoku.com/separable-sss/
const int TapCapacity=25;
const uint CopyPhase=0u,HorizontalPhase=1u,CompositePhase=3u;
const float PixelCenter=0.5,BackgroundDepth=1.0,ProjectionTolerance=1e-7,NormalTolerance=1e-12;
const float CoverageTolerance=1e-6,WeightTolerance=1e-7,DepthTolerance=1e-6,HalfMaximum=65504.0;
layout(std140,binding=0) uniform DiffusionFrame
{
    vec4 CameraRows[4];
    vec4 Viewport;
    vec4 Diffusion; // World distance, strength, maximum pixel support, kernel support.
    vec4 Rejection; // Optical depth scale, normal exponent, reserved, reserved.
    vec4 CameraForward;
    uvec4 Options; // Phase, tap count, reserved, reserved.
    vec4 Kernel[TapCapacity]; // RGB bin masses, signed profile-distance multiple.
};
#ifdef EVANS_VERTEX_STAGE
void main()
{
    vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2); gl_Position=vec4(p*2.0-1.0,0.0,1.0);
}
#else
uniform highp sampler2D DiffuseSignal,SceneDepth,NormalImage,SceneColor,OriginalDiffuse;
layout(location=0) out vec4 OutColor;
bool Unproject(vec2 uv,float depth,out vec3 world)
{
    vec4 clip=vec4(uv*2.0-1.0,depth,1.0);
    vec4 h=vec4(dot(CameraRows[0],clip),dot(CameraRows[1],clip),dot(CameraRows[2],clip),dot(CameraRows[3],clip));
    world=vec3(0.0); if (abs(h.w)<ProjectionTolerance) return false;
    world=h.xyz/h.w; return !any(isnan(world)) && !any(isinf(world));
}
bool Surface(ivec2 pixel,out vec3 world,out vec3 normal)
{
    world=normal=vec3(0.0);
    if (any(lessThan(pixel,ivec2(0))) || any(greaterThanEqual(pixel,ivec2(Viewport.xy)))) return false;
    vec4 n=texelFetch(NormalImage,pixel,0); float depth=texelFetch(SceneDepth,pixel,0).r,squared=dot(n.xyz,n.xyz);
    if (!(depth<BackgroundDepth && n.w>=0.0 && squared>NormalTolerance) || isnan(squared) || isinf(squared)) return false;
    normal=n.xyz*inversesqrt(squared); return Unproject((vec2(pixel)+PixelCenter)*Viewport.zw,depth,world);
}
void Accumulate(ivec2 pixel,float fraction,vec3 kernel,vec3 center,vec3 normal,float sigma,inout vec3 sum,inout vec3 weights)
{
    if (fraction<=0.0) return;
    vec3 world,n; if (!Surface(pixel,world,n)) return;
    vec4 signal=texelFetch(DiffuseSignal,pixel,0); if (signal.a<=CoverageTolerance) return;
    float depthDelta=dot(world-center,CameraForward.xyz)/sigma;
    float normalWeight=Rejection.y>0.0 ? pow(clamp(dot(normal,n),0.0,1.0),Rejection.y) : 1.0;
    vec3 weight=kernel*(fraction*exp(-depthDelta*depthDelta/2.0)*normalWeight*signal.a);
    sum+=signal.rgb/signal.a*weight; weights+=weight;
}
void main()
{
    ivec2 pixel=ivec2(gl_FragCoord.xy); vec4 signal=texelFetch(DiffuseSignal,pixel,0);
    if (Options.x==CopyPhase) { OutColor=signal; return; }
    if (Options.x==CompositePhase)
    {
        vec4 scene=texelFetch(SceneColor,pixel,0);
        vec3 delta=(signal.rgb-texelFetch(OriginalDiffuse,pixel,0).rgb)*Diffusion.y;
        OutColor=vec4(clamp(scene.rgb+delta,0.0,HalfMaximum),scene.a); return;
    }
    vec3 world,normal; if (signal.a<=CoverageTolerance || !Surface(pixel,world,normal)) { OutColor=signal; return; }
    ivec2 axis=Options.x==HorizontalPhase ? ivec2(1,0) : ivec2(0,1);
    vec3 adjacent; float depth=texelFetch(SceneDepth,pixel,0).r;
    if (!Unproject((vec2(pixel+axis)+PixelCenter)*Viewport.zw,depth,adjacent)) { OutColor=signal; return; }
    float footprint=length(adjacent-world); if (!(footprint>0.0)) { OutColor=signal; return; }
    float support=Diffusion.x*Diffusion.w/footprint;
    float scale=support>Diffusion.z && support>0.0 ? Diffusion.z/support : 1.0;
    float pixelsPerDistance=Diffusion.x*scale/footprint,sigma=max(Diffusion.x*scale*Rejection.x,DepthTolerance);
    vec3 sum=vec3(0.0),weights=vec3(0.0);
    for (uint tap=0u;tap<min(Options.y,uint(TapCapacity));++tap)
    {
        float offset=Kernel[tap].w*pixelsPerDistance,base=floor(offset),blend=offset-base;
        ivec2 q=pixel+axis*int(base);
        Accumulate(q,1.0-blend,Kernel[tap].rgb,world,normal,sigma,sum,weights);
        Accumulate(q+axis,blend,Kernel[tap].rgb,world,normal,sigma,sum,weights);
    }
    OutColor=vec4(clamp(sum/max(weights,vec3(WeightTolerance))*signal.a,0.0,HalfMaximum),signal.a);
}
#endif
