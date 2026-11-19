#include "Techniques/LightRange.h"
#include "Rasterizer/ComputeShaderRasterRHI.h"

#include "BackendCommon.hpp"
#include "Rasterizer/D3D11GpuOutput.h"
#include "GpuGeometryFrame.h"
#include "GpuRasterCommon.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <d3d11.h>
#include "Renderer/D3DShaderCache.h"
#endif

namespace Evans::Rasterizer
{
namespace
{

constexpr const char* ComputeShaderSource = R"hlsl(
struct Vertex {
    float4 Clip; float4 Color; float4 Position; float4 Normal; float4 ShadowClip;
    float4 TexCoord;
};
struct RasterTriangleGpu {
    Vertex Vertices[3]; float4 BaseColor; float4 SpecularColor; float4 EmissiveColor;
    float Shininess; float AlphaCutoff; uint DiffuseTextureIndex; uint OpacityTextureIndex;
    uint Selected; uint NormalTextureIndex; float SubsurfaceWeight; uint ReceivesLightmap;
};
struct RasterTextureGpu { uint OffsetTexels; uint Width; uint Height; uint FirstMip; };
struct RasterLightGpu {
    float4 PositionType; float4 DirectionIntensity;
    float4 ColorCastShadows; float4 AreaSizePadding;
};
cbuffer Constants : register(b0) {
    row_major float4x4 World; row_major float4x4 View; row_major float4x4 Projection;
    row_major float4x4 LightViewProjection;
    uint Width, Height, TriangleCount, InputSpace;
    uint DepthRange, FillModeValue, ShadingValue, BackFaceCulling;
    uint FrontFaceCounterClockwise, OutlineEnabled, ShadowsEnabled, ShadowWidth;
    uint ShadowHeight, TilesX; float WireframeWidth, OutlineWidth;
    float ShadowBias, ShadowStrength; uint DepthTest, ShadowPCFRadius;
    float4 ClearColor, WireframeColor, OutlineColor, CameraPosition;
    float4 LightDirectionIntensity, LightColorAmbientStrength, AmbientColor;
    uint LightCount; int ShadowLightIndex; uint TextureCount, TextureTexelCount;
    RasterLightGpu Lights[4];
    float4 ShadowRasterBias;
    uint MaterialOptions, FrontLightmap, BackLightmap, ACESColor;
    float4 DisplayParameters;
    uint SkyTexture, SkyDiffuseTexture; float SkyIntensity, SkyDiffuseIntensity;
    float4 SubsurfaceScattering;
    float4 NormalRows[3]; uint MainTileSize, ShadowTileSize, GroupsX, CollectCounts;
};
StructuredBuffer<RasterTriangleGpu> Triangles : register(t0);
StructuredBuffer<uint> Bins : register(t1);
StructuredBuffer<float> ShadowDepths : register(t3);
StructuredBuffer<RasterTextureGpu> TextureDescriptors : register(t4);
StructuredBuffer<uint> TextureTexels : register(t5);
RWStructuredBuffer<float4> OutputColors : register(u0);
RWStructuredBuffer<float> OutputDepths : register(u1);
RWStructuredBuffer<float> OutputVisibility : register(u6);
StructuredBuffer<float> OutlineVisibility : register(t0);
RWStructuredBuffer<float> GeneratedShadowDepths : register(u2);

float Edge(float2 a,float2 b,float2 p) {
    return (b.x-a.x)*(p.y-a.y)-(b.y-a.y)*(p.x-a.x);
}
float3 NormalizeSafe(float3 value) {
    float lengthSquared=dot(value,value);
    return lengthSquared<1.0e-12?float3(0,0,1):value*rsqrt(lengthSquared);
}
float3 NormalizeOr(float3 value,float3 fallback) {
    float lengthSquared=dot(value,value);
    return lengthSquared<1.0e-12?fallback:value*rsqrt(lengthSquared);
}
)hlsl"
#define R_FN
#define R_TRI RasterTriangleGpu
#define R_TEX RasterTextureGpu
#define R_READ(T) StructuredBuffer<T>
#define R_MIN min
#define R_MAX max
#define R_FLOOR floor
#define R_CEIL ceil
#define R_ABS abs
#define R_SQRT sqrt
#define R_LOG2 log2
#define R_POW pow
#define R_EXP2 exp2
#define R_FINITE isfinite
#define R_ATAN2 atan2
#define R_ACOS acos
#define R_DOT dot
#define R_CROSS cross
#define R_FLOAT2 float2
#define R_FLOAT3 float3
#define R_FLOAT4 float4
#define R_XYZ(v) (v).xyz
#define R_NORMALIZE_OR NormalizeOr
#define EVANS_MATERIAL_STRING(...) #__VA_ARGS__ "\n"
#define EVANS_GPU_CODE(...) EVANS_MATERIAL_STRING(__VA_ARGS__)
#include "GpuMaterials.inc"
#undef EVANS_MATERIAL_STRING
#undef R_FN
#undef R_TRI
#undef R_TEX
#undef R_READ
#undef R_MIN
#undef R_MAX
#undef R_FLOOR
#undef R_CEIL
#undef R_ABS
#undef R_SQRT
#undef R_LOG2
#undef R_POW
#undef R_EXP2
#undef R_FINITE
#undef R_ATAN2
#undef R_ACOS
#undef R_DOT
#undef R_CROSS
#undef R_FLOAT2
#undef R_FLOAT3
#undef R_FLOAT4
#undef R_XYZ
#undef R_NORMALIZE_OR
#undef EVANS_GPU_CODE
R"hlsl(
float4 SampleSurface(
    RasterTriangleGpu primitive,float p0,float p1,float p2,float denominator) {
    float4 vertex=(primitive.Vertices[0].Color*p0+
                   primitive.Vertices[1].Color*p1+
                   primitive.Vertices[2].Color*p2)/denominator;
    float2 texCoord=(primitive.Vertices[0].TexCoord.xy*p0+
                     primitive.Vertices[1].TexCoord.xy*p1+
                     primitive.Vertices[2].TexCoord.xy*p2)/denominator;
    const bool aniso = (MaterialOptions & 8) != 0;
    RasterTextureGradients gradients = GpuTextureGradients(primitive, texCoord, denominator, Width, Height);
    float4 diffuse=GpuSampleTexture((MaterialOptions & 1) != 0 ? primitive.DiffuseTextureIndex : 0xffffffffu,
        texCoord,1,gradients,TextureCount,aniso,TextureDescriptors,TextureTexels);
    float opacity=GpuSampleTexture((MaterialOptions & 4) != 0 ? primitive.OpacityTextureIndex : 0xffffffffu,
        texCoord,0,gradients,TextureCount,aniso,TextureDescriptors,TextureTexels).r;
    return float4(
        vertex.rgb*primitive.BaseColor.rgb*diffuse.rgb,
        saturate(vertex.a*primitive.BaseColor.a*diffuse.a*opacity));
}
// Smooth finite support for inverse-square point and rectangle samples.
// Karis' attenuation window; Filament, Punctual lights / Attenuation function.
float LightRangeWindow(float3 toEmitter, float radius)
{
    if (radius <= 0.0f) return 1.0f;
    if (any(abs(toEmitter) >= radius)) return 0.0f;
    float3 relative = toEmitter / radius;
    float distanceRatioSquared = dot(relative, relative);
    float window = saturate(1.0f - distanceRatioSquared * distanceRatioSquared);
    return window * window;
}

void AccumulateLightSample(
    float3 lightDirection,float3 radiance,float visibility,float3 normal,
    float3 viewDirection,float shininess,uint shading, float3 scatter,
    inout float3 diffuseLight,inout float3 specularLight) {
    float directionLengthSquared=dot(lightDirection,lightDirection);
    if(directionLengthSquared<=1.0e-12)return;
    float3 direction=lightDirection*rsqrt(directionLengthSquared);
    float cosine = dot(normal, direction);
    float normalToLight = max(cosine, 0.0f);
    float3 visibleRadiance=radiance*visibility;
    diffuseLight += visibleRadiance * max(cosine + scatter, 0.0f) / (1.0f + scatter);
    if (shading == 2 && normalToLight > 0) {
        float3 halfVector=NormalizeOr(direction+viewDirection,normal);
        float specularAmount=pow(max(dot(normal,halfVector),0.0),max(shininess,1.0));
        specularLight+=visibleRadiance*specularAmount;
    }
}
void AccumulateSceneLight(
    uint lightIndex,float3 position,float3 normal,float3 viewDirection,
    float shininess,uint shading, float3 scatter,float shadowVisibility,
    inout float3 diffuseLight,inout float3 specularLight) {
    RasterLightGpu light=Lights[lightIndex];
    uint type=(uint)(light.PositionType.w+0.5);
    float visibility=(int(lightIndex)==ShadowLightIndex&&light.ColorCastShadows.w>0.5)
        ?shadowVisibility:1.0;
    float3 radiance=max(light.ColorCastShadows.rgb,0.0)*
        max(light.DirectionIntensity.w,0.0);
    if(type==0) {
        AccumulateLightSample(light.DirectionIntensity.xyz,radiance,visibility,normal,
            viewDirection,shininess,shading, scatter,diffuseLight,specularLight);
        return;
    }
    if(type==1) {
        float3 toLight=light.PositionType.xyz-position;
        float distanceSquared=max(dot(toLight,toLight),1.0e-6);
        AccumulateLightSample(toLight,radiance * LightRangeWindow(toLight, light.AreaSizePadding.z) / distanceSquared,visibility,normal,
            viewDirection,shininess,shading, scatter,diffuseLight,specularLight);
        return;
    }
    float3 emissionDirection=NormalizeOr(
        light.DirectionIntensity.xyz,float3(0,-1,0));
    float3 helper=abs(emissionDirection.y)<0.999?float3(0,1,0):float3(1,0,0);
    float3 tangent=NormalizeOr(cross(helper,emissionDirection),float3(1,0,0));
    float3 bitangent=NormalizeOr(cross(emissionDirection,tangent),float3(0,0,1));
    [unroll]for(int sampleY=0;sampleY<2;++sampleY) {
        [unroll]for(int sampleX=0;sampleX<2;++sampleX) {
            float offsetX=sampleX==0?-0.25:0.25;
            float offsetY=sampleY==0?-0.25:0.25;
            float3 samplePosition=light.PositionType.xyz+
                tangent*(offsetX*light.AreaSizePadding.x)+
                bitangent*(offsetY*light.AreaSizePadding.y);
            float3 toLight=samplePosition-position;
            float distanceSquared=max(dot(toLight,toLight),1.0e-6);
            float3 direction=toLight*rsqrt(distanceSquared);
            float emitterCosine=max(dot(emissionDirection,-direction),0.0);
            AccumulateLightSample(direction,
                radiance*(0.25*emitterCosine * LightRangeWindow(toLight, light.AreaSizePadding.z) / distanceSquared),visibility,normal,
                viewDirection,shininess,shading, scatter,diffuseLight,specularLight);
        }
    }
}
#define R_FN
#define R_TRI RasterTriangleGpu
#define R_READ(T) StructuredBuffer<T>
#define R_WRITE(T) RWStructuredBuffer<T>
#define R_INOUT(T) inout T
#define R_OUT(T) out T
#define R_FLOAT4 float4
#define R_LOWBIT firstbitlow
#define R_MIN min
#define R_MAX max
#define R_ABS abs
#define R_ADD(value,bits) InterlockedAdd(value,bits)
#define R_OR(value,bits) InterlockedOr(value,bits)
)hlsl"
#define EVANS_GPU_CODE(...) #__VA_ARGS__ "\n"
#include "GpuGeometry.inc"
#undef EVANS_GPU_CODE
R"hlsl(
StructuredBuffer<ProjectedVertex> Projected : register(t2);
RWStructuredBuffer<RasterTriangleGpu> Transformed : register(u3);
RWStructuredBuffer<ProjectedVertex> ProjectedOutput : register(u4);
RWStructuredBuffer<uint> BinOutput : register(u5);
[numthreads(64,1,1)] void TransformMain(uint3 group:SV_GroupID,uint lane:SV_GroupIndex) {
    uint id=(group.y*GroupsX+group.x)*64+lane;
    if (id>=TriangleCount) return;
    RasterTriangleGpu primitive=Triangles[id];
    [unroll]for(uint corner=0;corner<3;++corner) {
        Vertex vert=primitive.Vertices[corner];float3 normal=vert.Normal.xyz;
        if (InputSpace!=0) {
            float4 world=mul(World,vert.Position);
            if(abs(world.w)>1e-6 && abs(world.w-1.0)>1e-6) {world/=world.w;world.w=1;}
            vert.Position=world;vert.Clip=mul(Projection,mul(View,world));vert.ShadowClip=mul(LightViewProjection,world);
            normal=float3(dot(NormalRows[0].xyz,normal),dot(NormalRows[1].xyz,normal),dot(NormalRows[2].xyz,normal));
        }
        vert.Normal=float4(NormalizeSafe(normal),0);primitive.Vertices[corner]=vert;
    }
    Transformed[id]=primitive;
}
[numthreads(64,1,1)] void SetupMain(uint3 group:SV_GroupID,uint lane:SV_GroupIndex) {
    ProjectAndBin((group.y*GroupsX+group.x)*64+lane,TriangleCount,Width,Height,MainTileSize,DepthRange,false,CollectCounts!=0,FillModeValue==0,Triangles,ProjectedOutput,BinOutput);
}
[numthreads(64,1,1)] void SetupShadow(uint3 group:SV_GroupID,uint lane:SV_GroupIndex) {
    ProjectAndBin((group.y*GroupsX+group.x)*64+lane,TriangleCount,ShadowWidth,ShadowHeight,ShadowTileSize,DepthRange,true,false,true,Triangles,ProjectedOutput,BinOutput);
}
[numthreads(8,8,1)]
void ShadowMain(uint3 id:SV_DispatchThreadID) {
    uint x=id.x,y=id.y;if(x>=ShadowWidth||y>=ShadowHeight)return;
    uint pixel=y*ShadowWidth+x;
    float2 samplePosition=float2(x+0.5,y+0.5);float bestDepth=1.0e30;
    BinCursor cursor=OpenBin((y/ShadowTileSize)*((ShadowWidth+ShadowTileSize-1)/ShadowTileSize)+x/ShadowTileSize,TriangleCount);
    uint sourceId;
    while (NextCandidate(cursor,Bins,sourceId)) {
        ProjectedVertex first=Projected[sourceId*3];
        uint corners=((uint)first.Weights.w)&15u;
        for(uint fan=1;fan+1<corners;++fan) {
        ProjectedVertex second=Projected[VertexSlot(sourceId,fan,TriangleCount)],third=Projected[VertexSlot(sourceId,fan+1,TriangleCount)];
        float2 screen[3]={float2(first.Screen.x,first.Screen.y),float2(second.Screen.x,second.Screen.y),float2(third.Screen.x,third.Screen.y)};
        float inverseW[3]={first.Screen.w,second.Screen.w,third.Screen.w}, vertexDepth[3]={first.Screen.z,second.Screen.z,third.Screen.z};
        if(inverseW[0]<=0 || inverseW[1]<=0 || inverseW[2]<=0)continue;
        float area=Edge(screen[0],screen[1],screen[2]);if(abs(area)<1.0e-6)continue;
        float w0=Edge(screen[1],screen[2],samplePosition)/area;
        float w1=Edge(screen[2],screen[0],samplePosition)/area;
        float w2=1.0-w0-w1;if(w0<0||w1<0||w2<0)continue;
        float depth=w0*vertexDepth[0]+w1*vertexDepth[1]+w2*vertexDepth[2];
        if(depth<0||depth>1)continue;
        if(ShadowRasterBias.x>0) {
            float dz1=vertexDepth[1]-vertexDepth[0],dz2=vertexDepth[2]-vertexDepth[0];
            float slopeX=(dz1*(screen[2].y-screen[0].y)-dz2*(screen[1].y-screen[0].y))/area;
            float slopeY=((screen[1].x-screen[0].x)*dz2-(screen[2].x-screen[0].x)*dz1)/area;
            depth=min(depth+ShadowRasterBias.x*max(abs(slopeX),abs(slopeY)),1.0);
        }
        if(depth>=bestDepth)continue;
        RasterTriangleGpu primitive=FanAttributes(Triangles[sourceId],first,second,third);
        if((MaterialOptions&4)!=0 && primitive.AlphaCutoff>0) {
        float p0=w0*inverseW[0],p1=w1*inverseW[1],p2=w2*inverseW[2];
        float denominator=p0+p1+p2;if(abs(denominator)<1.0e-8)continue;
        float alpha=SampleSurface(primitive,p0,p1,p2,denominator).a;
        if(primitive.AlphaCutoff>0.0&&alpha<primitive.AlphaCutoff)continue;
        }
        bestDepth=depth;
    }
    }
    GeneratedShadowDepths[pixel]=bestDepth;
}

)hlsl"
R"hlsl(
[numthreads(8,8,1)]
void RasterMain(uint3 id:SV_DispatchThreadID) {
    uint x=id.x,y=id.y;if(x>=Width||y>=Height)return;
    uint pixel=y*Width+x;
    float2 samplePosition=float2(x+0.5,y+0.5);
    float visibilityDepth=1.0e30;bool selected=false;
    uint winningId=0, winningFan=0;
    float winningP0=0, winningP1=0, winningP2=0, winningDenominator=1;
    bool winningFront=false;
    float bestDepth=1.0e30,bestWireDepth=1.0e30;float4 bestColor=ClearColor;
    BinCursor cursor=OpenBin((y/MainTileSize)*((Width+MainTileSize-1)/MainTileSize)+x/MainTileSize,TriangleCount);
    uint sourceId;
    while (NextCandidate(cursor,Bins,sourceId)) {
        ProjectedVertex first=Projected[sourceId*3];
        uint corners=((uint)first.Weights.w)&15u;
        for(uint fan=1;fan+1<corners;++fan) {
        ProjectedVertex second=Projected[VertexSlot(sourceId,fan,TriangleCount)],third=Projected[VertexSlot(sourceId,fan+1,TriangleCount)];
        float2 screen[3]={float2(first.Screen.x,first.Screen.y),float2(second.Screen.x,second.Screen.y),float2(third.Screen.x,third.Screen.y)};
        float inverseW[3]={first.Screen.w,second.Screen.w,third.Screen.w}, vertexDepth[3]={first.Screen.z,second.Screen.z,third.Screen.z};
        if(inverseW[0]<=0 || inverseW[1]<=0 || inverseW[2]<=0)continue;
        float area=Edge(screen[0],screen[1],screen[2]);if(abs(area)<1.0e-6)continue;
        bool front=FrontFaceCounterClockwise!=0?area<0:area>0;
        if(BackFaceCulling!=0&&!front)continue;
        float w0=Edge(screen[1],screen[2],samplePosition)/area;
        float w1=Edge(screen[2],screen[0],samplePosition)/area,w2=1.0-w0-w1;
        bool inside=w0>=0&&w1>=0&&w2>=0;
        bool wire=false;
        float3 edgeWeights=0.0.xxx;
        if(FillModeValue!=0){
            float nearestEdge=1.0e30;
            [unroll]for(int edgeIndex=0;edgeIndex<3;++edgeIndex){
                int next=(edgeIndex+1)%3;
                float2 segment=screen[next]-screen[edgeIndex];
                float t=saturate(dot(samplePosition-screen[edgeIndex],segment)/max(dot(segment,segment),1.0e-12));
                float2 offset=samplePosition-(screen[edgeIndex]+t*segment);
                float distanceSquared=dot(offset,offset);
                if(distanceSquared<nearestEdge){
                    nearestEdge=distanceSquared;edgeWeights=0.0.xxx;
                    edgeWeights[edgeIndex]=1.0-t;edgeWeights[next]=t;
                }
            }
            float halfWidth=max(WireframeWidth,1.0)*0.5;
            wire=nearestEdge<=halfWidth*halfWidth;
        }
        if(!inside && !wire)continue;
        RasterTriangleGpu primitive=FanAttributes(Triangles[sourceId],first,second,third);
        if(wire){
            float wireDepth=dot(edgeWeights,float3(vertexDepth[0],vertexDepth[1],vertexDepth[2]));
            float3 edgePerspective=edgeWeights*float3(inverseW[0],inverseW[1],inverseW[2]);
            float edgeDenominator=edgePerspective.x+edgePerspective.y+edgePerspective.z;
            if(wireDepth>=0&&wireDepth<=1&&abs(edgeDenominator)>1.0e-8){
                float edgeAlpha=SampleSurface(primitive,edgePerspective.x,edgePerspective.y,
                    edgePerspective.z,edgeDenominator).a;
                if((MaterialOptions&4)==0 || primitive.AlphaCutoff<=0||edgeAlpha>=primitive.AlphaCutoff)
                    bestWireDepth=min(bestWireDepth,wireDepth);
            }
        }
        if(OutlineEnabled!=0 && inside) {
            float z=dot(float3(w0,w1,w2),float3(vertexDepth[0],vertexDepth[1],vertexDepth[2]));
            float3 p=float3(w0,w1,w2)*float3(inverseW[0],inverseW[1],inverseW[2]);
            float d=p.x+p.y+p.z;
            if(z>=0 && z<=1 && z<visibilityDepth && abs(d)>1e-8 &&
                ((MaterialOptions&4)==0 || primitive.AlphaCutoff<=0 || SampleSurface(primitive,p.x,p.y,p.z,d).a>=primitive.AlphaCutoff))
            {visibilityDepth=z;selected=primitive.Selected!=0;}
        }
        if(FillModeValue==1||!inside)continue;
        float depth=w0*vertexDepth[0]+w1*vertexDepth[1]+w2*vertexDepth[2];
        if(depth<0||depth>1||(DepthTest!=0&&depth>=bestDepth))continue;
        float p0=w0*inverseW[0],p1=w1*inverseW[1],p2=w2*inverseW[2];
        float denominator=p0+p1+p2;if(abs(denominator)<1.0e-8)continue;
        if((MaterialOptions&4)!=0 && primitive.AlphaCutoff>0.0 && SampleSurface(primitive,p0,p1,p2,denominator).a<primitive.AlphaCutoff)continue;
        bestDepth=depth;winningId=sourceId;winningFan=fan;winningFront=front;
        winningP0=p0;winningP1=p1;winningP2=p2;winningDenominator=denominator;
        }
    }
    if(winningFan!=0) {
        RasterTriangleGpu primitive=FanAttributes(Triangles[winningId],Projected[winningId*3],
            Projected[VertexSlot(winningId,winningFan,TriangleCount)],Projected[VertexSlot(winningId,winningFan+1,TriangleCount)]);
        float p0=winningP0,p1=winningP1,p2=winningP2,denominator=winningDenominator;
        bool front=winningFront;
        float4 color=SampleSurface(primitive,p0,p1,p2,denominator);
        float shadowVisibility=1.0;
        if(ShadowsEnabled!=0&&ShadowWidth>0&&ShadowHeight>0) {
            float3 shadowNormal = NormalizeSafe((primitive.Vertices[0].Normal.xyz*p0+
                primitive.Vertices[1].Normal.xyz*p1+primitive.Vertices[2].Normal.xyz*p2)/denominator) * (front?1.0f:-1.0f);
            float receiverBias = GpuShadowBias(ShadowBias, ShadowRasterBias.z, shadowNormal,
                Lights[clamp(ShadowLightIndex,0,3)].DirectionIntensity.xyz);
            float4 shadow=(primitive.Vertices[0].ShadowClip*p0+primitive.Vertices[1].ShadowClip*p1+primitive.Vertices[2].ShadowClip*p2)/denominator;
            if(shadow.w>1.0e-6) {
                float inverseShadowW=rcp(shadow.w);
                float shadowX=shadow.x*inverseShadowW*0.5+0.5;
                float shadowY=0.5-shadow.y*inverseShadowW*0.5;
                float shadowZ=shadow.z*inverseShadowW;
                if(DepthRange!=0)shadowZ=shadowZ*0.5+0.5;
                if(shadowX>=0&&shadowX<=1&&shadowY>=0&&shadowY<=1&&shadowZ>=0&&shadowZ<=1) {
                    int centerX=min((int)(shadowX*ShadowWidth),(int)ShadowWidth-1);
                    int centerY=min((int)(shadowY*ShadowHeight),(int)ShadowHeight-1);
                    int radius=(int)min(ShadowPCFRadius,8U);
                    uint occluded=0,validSamples=0;
                    [loop]for(int offsetY=-radius;offsetY<=radius;++offsetY) {
                        int sy=centerY+offsetY;if(sy<0||sy>=(int)ShadowHeight)continue;
                        [loop]for(int offsetX=-radius;offsetX<=radius;++offsetX) {
                            int sx=centerX+offsetX;if(sx<0||sx>=(int)ShadowWidth)continue;
                            ++validSamples;
                            if(shadowZ-receiverBias>ShadowDepths[sy*ShadowWidth+sx])++occluded;
                        }
                    }
                    if(validSamples>0)
                        shadowVisibility=1.0-saturate(ShadowStrength)*
                            ((float)occluded/(float)validSamples);
                }
            }
        }

        if(ShadingValue==0) {
            color.rgb+=primitive.EmissiveColor.rgb;
        } else {
            float3 n=NormalizeSafe((primitive.Vertices[0].Normal.xyz*p0+primitive.Vertices[1].Normal.xyz*p1+primitive.Vertices[2].Normal.xyz*p2)/denominator);
            if(!front)n=-n;
            float4 uv = (primitive.Vertices[0].TexCoord*p0+primitive.Vertices[1].TexCoord*p1+primitive.Vertices[2].TexCoord*p2)/denominator;
            RasterTextureGradients gradients = GpuTextureGradients(primitive, uv.xy, denominator, Width, Height);
            if ((MaterialOptions&2)!=0 && primitive.NormalTextureIndex<TextureCount)
                n = GpuMappedNormal(primitive,n,GpuSampleTexture(primitive.NormalTextureIndex,uv.xy,0,
                    gradients,TextureCount,(MaterialOptions&8)!=0,TextureDescriptors,TextureTexels));
            float3 position=((primitive.Vertices[0].Position*p0+primitive.Vertices[1].Position*p1+primitive.Vertices[2].Position*p2)/denominator).xyz;
            float3 viewDirection=NormalizeOr(CameraPosition.xyz-position,n);
            float3 diffuseLight=0.0.xxx,specularLight=0.0.xxx;
            [loop]for(uint lightIndex=0;lightIndex<min(LightCount,4U);++lightIndex)
                AccumulateSceneLight(lightIndex,position,n,viewDirection,primitive.Shininess,
                    ShadingValue,SubsurfaceScattering.xyz*primitive.SubsurfaceWeight,shadowVisibility,diffuseLight,specularLight);
            if (SkyDiffuseIntensity > 0 && SkyDiffuseTexture < TextureCount)
                diffuseLight += GpuSampleSky(n,SkyDiffuseTexture,TextureCount,TextureDescriptors,TextureTexels)*SkyDiffuseIntensity;
            uint atlas = front || BackLightmap>=TextureCount ? FrontLightmap : BackLightmap;
            if (primitive.ReceivesLightmap != 0 && atlas<TextureCount)
                diffuseLight += GpuSampleTexture(atlas,uv.zw,2,gradients,TextureCount,false,TextureDescriptors,TextureTexels).xyz*DisplayParameters.w;
            color.rgb=color.rgb*(AmbientColor.rgb*LightColorAmbientStrength.w+diffuseLight)+
                primitive.SpecularColor.rgb*specularLight+primitive.EmissiveColor.rgb;
        }
        color.rgb=GpuDisplayColor(color.rgb,ACESColor,DisplayParameters);
        bestColor=float4(saturate(color.rgb),saturate(color.a));
    }
    if(FillModeValue==1)bestDepth=bestWireDepth;
    if(FillModeValue!=0&&bestWireDepth<=1.0&&(DepthTest==0||bestWireDepth<=bestDepth+1.0e-4))
    {bestColor=WireframeColor;bestDepth=min(bestDepth,bestWireDepth);}
    if (bestDepth > 1.0f && FillModeValue != 1 && SkyIntensity > 0 && SkyTexture < TextureCount) {
        float3 ray = GpuSkyRay(id.x+0.5f,id.y+0.5f,Width,Height,View[0].xyz,View[1].xyz,View[2].xyz,Projection[0][0],Projection[1][1]);
        bestColor = float4(GpuDisplayColor(GpuSampleSky(ray,SkyTexture,TextureCount,TextureDescriptors,TextureTexels)*SkyIntensity,
            ACESColor,DisplayParameters),1);
    }
    OutputColors[pixel]=bestColor; OutputDepths[pixel]=bestDepth;
    if (OutlineEnabled!=0) OutputVisibility[pixel]=selected?-visibilityDepth-1:visibilityDepth;
}
[numthreads(8,8,1)]
void OutlineMain(uint3 id:SV_DispatchThreadID) {
    if(id.x<Width && id.y<Height && IsOutlinePixel(id.x,id.y,Width,Height,(int)OutlineWidth,OutlineVisibility))
        OutputColors[id.y*Width+id.x]=OutlineColor;
}
)hlsl";

#if defined(_WIN32)

template <typename Interface>
void Release(Interface*& value) noexcept
{
    if (value != nullptr)
    {
        value->Release();
        value = nullptr;
    }
}

class D3D11ComputeRuntime
{
public:
    ~D3D11ComputeRuntime() { Shutdown(); }

    [[nodiscard]] bool Initialize(std::string& error)
    {
        Shutdown();
        D3D11Library_ = LoadLibraryA("d3d11.dll");
        if (D3D11Library_ == nullptr)
        {
            error = "D3D11 system library is unavailable";
            Shutdown();
            return false;
        }
        CreateDevice_ = reinterpret_cast<CreateDeviceFunction>(
            GetProcAddress(D3D11Library_, "D3D11CreateDevice"));
        if (CreateDevice_ == nullptr)
        {
            error = "D3D11 system libraries are missing required entry points";
            Shutdown();
            return false;
        }

        const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_0};
        D3D_FEATURE_LEVEL selectedLevel = D3D_FEATURE_LEVEL_11_0;
        const HRESULT deviceResult = CreateDevice_(
            nullptr,
            D3D_DRIVER_TYPE_HARDWARE,
            nullptr,
            0,
            levels,
            1,
            D3D11_SDK_VERSION,
            &Device_,
            &selectedLevel,
            &ImmediateContext_);
        if (FAILED(deviceResult) || selectedLevel < D3D_FEATURE_LEVEL_11_0)
        {
            error = "No Direct3D 11 hardware compute device is available (HRESULT " +
                    std::to_string(static_cast<unsigned long>(deviceResult)) + ")";
            Shutdown();
            return false;
        }

        return true;
    }

    bool EnsureShaders(std::string& error)
    {
        const auto compileShader = [&](
            const char* entryPoint,
            ID3D11ComputeShader** output) -> bool
        {
            if (*output) return true;
            ID3DBlob* byteCode = nullptr;
            ID3DBlob* compilerErrors = nullptr;
            const HRESULT compileResult = Renderer::CompileCachedD3DShader(
                ComputeShaderSource,
                std::strlen(ComputeShaderSource),
                "EvansComputeRaster.hlsl",
                nullptr,
                nullptr,
                entryPoint,
                "cs_5_0",
                D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_ENABLE_STRICTNESS,
                0,
                &byteCode,
                &compilerErrors);
            if (FAILED(compileResult))
            {
                error = std::string("Failed to compile D3D11 compute entry ") +
                        entryPoint;
                if (compilerErrors != nullptr)
                {
                    error += ": " + std::string(
                        static_cast<const char*>(compilerErrors->GetBufferPointer()),
                        compilerErrors->GetBufferSize());
                }
                Release(compilerErrors);
                Release(byteCode);
                return false;
            }
            Release(compilerErrors);
            const HRESULT shaderResult = Device_->CreateComputeShader(
                byteCode->GetBufferPointer(),
                byteCode->GetBufferSize(),
                nullptr,
                output);
            Release(byteCode);
            if (FAILED(shaderResult))
            {
                error = std::string("Failed to create D3D11 compute entry ") +
                        entryPoint;
                return false;
            }
            return true;
        };
        if (!compileShader("RasterMain", &RasterShader_) ||
            !compileShader("ShadowMain", &ShadowShader_) ||
            !compileShader("TransformMain",&TransformShader_) || !compileShader("SetupMain",&SetupMain_) ||
            !compileShader("SetupShadow",&SetupShadow_) || !compileShader("OutlineMain",&OutlineShader_))
        {
            return false;
        }
        return true;
    }

    void Shutdown() noexcept
    {
        GpuOutput_.Shutdown();TextureCache_.Reset();TextureGeneration_=0;
        for(auto buffer:{&Source_,&Transformed_,&Projected_,&Bins_,&Shadow_,&Colors_,&Depths_,&Visibility_,&Descriptors_,&Texels_}) *buffer={};
        ResidentScene_.reset(); Constants_.Reset();CounterReadback_.Reset();
        Release(TransformShader_);Release(SetupMain_);Release(SetupShadow_); Release(OutlineShader_);
        Release(ShadowShader_);
        Release(RasterShader_);
        if (ImmediateContext_ != nullptr) ImmediateContext_->ClearState();
        Release(ImmediateContext_);
        Release(Device_);
        if (D3D11Library_ != nullptr) FreeLibrary(D3D11Library_);
        D3D11Library_ = nullptr;
        CreateDevice_ = nullptr;
    }

    [[nodiscard]] RasterStatus Render(
        const RasterScene& scene,
        const RasterSettings& settings,
        RenderTarget& target,
        const RasterCancellationToken& externalCancellation,
        const std::atomic<bool>& backendCancellation,
        RasterStatistics& statistics)
    {
        const auto cancelled=[&]{return Detail::IsGpuRasterCancelled(externalCancellation,backendCancellation);};
        if(cancelled())return RasterStatus::Cancelled;
        const auto start=std::chrono::steady_clock::now();
        std::string shaderError;
        if (!EnsureShaders(shaderError)) throw std::runtime_error(shaderError);
        const Detail::GpuGeometryFrame frame(scene,settings,target);
        Ensure(Source_,frame.TriangleCount,sizeof(Detail::GpuRasterTriangle),false);
        const bool upload=settings.ImmutableScene.get()!=&scene || ResidentScene_.lock().get()!=&scene;
        if(upload)
        {
            ResidentScene_.reset();
            const auto packed=Detail::BuildGpuRasterTriangles(scene);
            Upload(Source_,packed.data(),packed.size()*sizeof(Detail::GpuRasterTriangle));
            ResidentScene_=settings.ImmutableScene.get()==&scene ? settings.ImmutableScene : nullptr;
        }
        const auto& textures=TextureCache_.Get(scene);
        if(TextureGeneration_!=TextureCache_.Generation() || !Descriptors_.Resource)
        {
            Ensure(Descriptors_,textures.Descriptors.size(),sizeof(Detail::GpuRasterTexture),false);
            Ensure(Texels_,textures.TexelsRGBA8.size(),sizeof(unsigned),false);
            Upload(Descriptors_,textures.Descriptors.data(),textures.Descriptors.size()*sizeof(Detail::GpuRasterTexture));
            Upload(Texels_,textures.TexelsRGBA8.data(),textures.TexelsRGBA8.size()*sizeof(unsigned));
            TextureGeneration_=TextureCache_.Generation();
        }
        const auto pixels=std::size_t(target.Width())*target.Height();
        Ensure(Transformed_,frame.TriangleCount,sizeof(Detail::GpuRasterTriangle),true);
        Ensure(Projected_,std::size_t(frame.TriangleCount)*9,32,true);
        Ensure(Bins_,std::max(frame.MainBins.StorageWords(),frame.GenerateShadow?frame.ShadowBins.StorageWords():0),4,true);
        Ensure(Shadow_,std::size_t(frame.ShadowWidth)*frame.ShadowHeight,4,true);
        Ensure(Colors_,pixels,sizeof(Color),true); Ensure(Depths_,pixels,sizeof(float),true);
        if(settings.OutlineEnabled) Ensure(Visibility_,pixels,sizeof(float),true);
        if(frame.ExternalShadow) Upload(Shadow_,settings.SceneShadowMap->Depth.data(),settings.SceneShadowMap->Depth.size()*sizeof(float));
        if(!Constants_)
        {
            D3D11_BUFFER_DESC desc{};desc.ByteWidth=sizeof(frame.Constants);desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
            Check(Device_->CreateBuffer(&desc,nullptr,&Constants_),"Create geometry constants");
        }
        ImmediateContext_->UpdateSubresource(Constants_.Get(),0,nullptr,&frame.Constants,0,0);
        if(cancelled())return RasterStatus::Cancelled;
        const auto rasterStart=std::chrono::steady_clock::now();
        statistics.TransformMilliseconds=std::chrono::duration<double,std::milli>(rasterStart-start).count();
        auto context=ImmediateContext_;
        ID3D11Buffer* cb=Constants_.Get(); context->CSSetConstantBuffers(0,1,&cb);
        const auto dispatch=[&](ID3D11ComputeShader* shader,unsigned groupsX,unsigned groupsY,
            std::initializer_list<ID3D11ShaderResourceView*> inputs,unsigned outputSlot,
            std::initializer_list<ID3D11UnorderedAccessView*> outputs)
        {
            if(!groupsX || !groupsY)return;
            context->CSSetShader(shader,nullptr,0);
            context->CSSetShaderResources(0,static_cast<UINT>(inputs.size()),inputs.begin());
            context->CSSetUnorderedAccessViews(outputSlot,static_cast<UINT>(outputs.size()),outputs.begin(),nullptr);
            context->Dispatch(groupsX,groupsY,1);
            ID3D11ShaderResourceView* noInputs[6]{};
            ID3D11UnorderedAccessView* noOutputs[2]{};
            context->CSSetUnorderedAccessViews(outputSlot,static_cast<UINT>(outputs.size()),noOutputs,nullptr);
            context->CSSetShaderResources(0,static_cast<UINT>(inputs.size()),noInputs);
        };
        dispatch(TransformShader_,frame.GroupsX,frame.GroupsY,{Source_.Read.Get()},3,{Transformed_.Write.Get()});
        const auto setup=[&](ID3D11ComputeShader* shader) {
            const unsigned zeros[4]{};
            context->ClearUnorderedAccessViewUint(Bins_.Write.Get(),zeros);
            dispatch(shader,frame.GroupsX,frame.GroupsY,{Transformed_.Read.Get()},4,{Projected_.Write.Get(),Bins_.Write.Get()});
        };
        if(frame.GenerateShadow)
        {
            setup(SetupShadow_);
            dispatch(ShadowShader_,(frame.ShadowWidth+7)/8,(frame.ShadowHeight+7)/8,
                {Transformed_.Read.Get(),Bins_.Read.Get(),Projected_.Read.Get(),nullptr,Descriptors_.Read.Get(),Texels_.Read.Get()},
                2,{Shadow_.Write.Get()});
        }
        if(cancelled()) return RasterStatus::Cancelled;
        setup(SetupMain_);
        ID3D11UnorderedAccessView* visibility=settings.OutlineEnabled?Visibility_.Write.Get():nullptr;
        if (visibility) context->CSSetUnorderedAccessViews(6,1,&visibility,nullptr);
        dispatch(RasterShader_,(target.Width()+7)/8,(target.Height()+7)/8,
            {Transformed_.Read.Get(),Bins_.Read.Get(),Projected_.Read.Get(),Shadow_.Read.Get(),Descriptors_.Read.Get(),Texels_.Read.Get()},
            0,{Colors_.Write.Get(),Depths_.Write.Get()});
        if (visibility)
        {
            visibility=nullptr; context->CSSetUnorderedAccessViews(6,1,&visibility,nullptr);
            dispatch(OutlineShader_,(target.Width()+7)/8,(target.Height()+7)/8,{Visibility_.Read.Get()},0,{Colors_.Write.Get()});
        }
        if(settings.GpuPresentation)
        {
            if (!GpuOutput_.Device()) GpuOutput_.Initialize(Device_);
            target.GpuColor=GpuOutput_.Publish(Colors_.Resource.Get(),target.Width(),target.Height());
        }
        else
        {
            Capture(Colors_,target.MutableColors().data(),pixels*sizeof(Color));
            Capture(Depths_,target.MutableDepths().data(),pixels*sizeof(float));
            statistics.ShadedFragments=std::count_if(target.Depths().begin(),target.Depths().end(),[](float z){return z>=0 && z<=1;});
        }
        statistics.PostClipCountAvailable=frame.Constants.CollectCounts!=0;
        if(statistics.PostClipCountAvailable) {
            if(!CounterReadback_) {
                D3D11_BUFFER_DESC desc{};desc.ByteWidth=4;desc.Usage=D3D11_USAGE_STAGING;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
                Check(Device_->CreateBuffer(&desc,nullptr,&CounterReadback_),"Create primitive count readback");
            }
            const UINT offset=static_cast<UINT>(frame.MainBins.Words*4);
            D3D11_BOX box{offset,0,0,offset+4,1,1};
            context->CopySubresourceRegion(CounterReadback_.Get(),0,0,0,0,Bins_.Resource.Get(),0,&box);
            D3D11_MAPPED_SUBRESOURCE mapped{};
            Check(context->Map(CounterReadback_.Get(),0,D3D11_MAP_READ,0,&mapped),"Capture compute primitive count");
            statistics.PostClipTriangles=*static_cast<const unsigned*>(mapped.pData);context->Unmap(CounterReadback_.Get(),0);
        }
        frame.Report(statistics,upload,"Native D3D11 CS 5.0");
        statistics.RasterMilliseconds=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-rasterStart).count();
        return cancelled()?RasterStatus::Cancelled:RasterStatus::Completed;
    }

private:
    struct Buffer
    {
        std::size_t Capacity=0;
        Microsoft::WRL::ComPtr<ID3D11Buffer> Resource, Staging;
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> Read;
        Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> Write;
    };
    Buffer Source_, Transformed_, Projected_, Bins_, Shadow_, Colors_, Depths_, Visibility_, Descriptors_, Texels_;
    Microsoft::WRL::ComPtr<ID3D11Buffer> Constants_,CounterReadback_;
    std::weak_ptr<const RasterScene> ResidentScene_;
    ID3D11ComputeShader *TransformShader_=nullptr, *SetupMain_=nullptr, *SetupShadow_=nullptr, *OutlineShader_=nullptr;
    void Ensure(Buffer& buffer,std::size_t count,unsigned stride,bool writable)
    {
        count=std::max<std::size_t>(1,count);
        if(buffer.Capacity>=count)return;
        Buffer next;
        next.Resource.Attach(CreateStructuredBuffer(count,stride,nullptr,D3D11_BIND_SHADER_RESOURCE|(writable?D3D11_BIND_UNORDERED_ACCESS:0)));
        next.Read.Attach(CreateShaderView(next.Resource.Get(),static_cast<UINT>(count)));
        if(writable)next.Write.Attach(CreateUnorderedView(next.Resource.Get(),static_cast<UINT>(count)));
        next.Capacity=count;buffer=std::move(next);
    }
    void Upload(Buffer& buffer,const void* data,std::size_t bytes)
    {
        if(!bytes)return;
        D3D11_BOX box{0,0,0,static_cast<UINT>(bytes),1,1};
        ImmediateContext_->UpdateSubresource(buffer.Resource.Get(),0,&box,data,0,0);
    }
    void Capture(Buffer& buffer,void* destination,std::size_t bytes)
    {
        if(!buffer.Staging)buffer.Staging.Attach(CreateReadbackBuffer(buffer.Resource.Get()));
        ImmediateContext_->CopyResource(buffer.Staging.Get(),buffer.Resource.Get());
        D3D11_MAPPED_SUBRESOURCE mapped{};
        Check(ImmediateContext_->Map(buffer.Staging.Get(),0,D3D11_MAP_READ,0,&mapped),"Capture compute raster output");
        std::memcpy(destination,mapped.pData,bytes);
        ImmediateContext_->Unmap(buffer.Staging.Get(),0);
    }
    using CreateDeviceFunction = HRESULT(WINAPI*)(IDXGIAdapter*, D3D_DRIVER_TYPE, HMODULE, UINT,
        const D3D_FEATURE_LEVEL*, UINT, UINT, ID3D11Device**, D3D_FEATURE_LEVEL*, ID3D11DeviceContext**);

    static void Check(HRESULT result, const char* operation)
    {
        if (FAILED(result))
        {
            throw std::runtime_error(
                std::string(operation) + " failed with HRESULT " +
                std::to_string(static_cast<unsigned long>(result)));
        }
    }

    [[nodiscard]] ID3D11Buffer* CreateStructuredBuffer(
        std::size_t elementCount,
        std::size_t stride,
        const void* initialData,
        UINT bindFlags)
    {
        if (elementCount == 0 || stride == 0 ||
            elementCount >
                static_cast<std::size_t>(std::numeric_limits<UINT>::max()) / stride)
        {
            throw std::overflow_error("D3D11 structured buffer size is invalid");
        }
        D3D11_BUFFER_DESC description{};
        description.ByteWidth = static_cast<UINT>(elementCount * stride);
        description.Usage = D3D11_USAGE_DEFAULT;
        description.BindFlags = bindFlags;
        description.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        description.StructureByteStride = static_cast<UINT>(stride);
        D3D11_SUBRESOURCE_DATA data{initialData, 0, 0};
        ID3D11Buffer* buffer = nullptr;
        Check(Device_->CreateBuffer(&description, initialData == nullptr ? nullptr : &data, &buffer), "CreateStructuredBuffer");
        return buffer;
    }

    [[nodiscard]] ID3D11ShaderResourceView* CreateShaderView(
        ID3D11Buffer* buffer,
        UINT elementCount)
    {
        D3D11_SHADER_RESOURCE_VIEW_DESC description{};
        description.Format = DXGI_FORMAT_UNKNOWN;
        description.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        description.Buffer.NumElements = elementCount;
        ID3D11ShaderResourceView* view = nullptr;
        Check(
            Device_->CreateShaderResourceView(buffer, &description, &view),
            "CreateShaderResourceView");
        return view;
    }

    [[nodiscard]] ID3D11UnorderedAccessView* CreateUnorderedView(
        ID3D11Buffer* buffer,
        UINT elementCount)
    {
        D3D11_UNORDERED_ACCESS_VIEW_DESC description{};
        description.Format = DXGI_FORMAT_UNKNOWN;
        description.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        description.Buffer.NumElements = elementCount;
        ID3D11UnorderedAccessView* view = nullptr;
        Check(Device_->CreateUnorderedAccessView(buffer, &description, &view), "CreateUnorderedAccessView");
        return view;
    }

    [[nodiscard]] ID3D11Buffer* CreateReadbackBuffer(ID3D11Buffer* source)
    {
        D3D11_BUFFER_DESC description{};
        source->GetDesc(&description);
        description.Usage = D3D11_USAGE_STAGING;
        description.BindFlags = 0;
        description.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        description.MiscFlags = 0;
        description.StructureByteStride = 0;
        ID3D11Buffer* buffer = nullptr;
        Check(Device_->CreateBuffer(&description, nullptr, &buffer), "CreateReadbackBuffer");
        return buffer;
    }

    HMODULE D3D11Library_ = nullptr;
    CreateDeviceFunction CreateDevice_ = nullptr;
    std::uint64_t TextureGeneration_=0;
    Detail::GpuRasterTextureCache TextureCache_;
    Detail::D3D11GpuOutput GpuOutput_;
    ID3D11Device* Device_ = nullptr;
    ID3D11DeviceContext* ImmediateContext_ = nullptr;
    ID3D11ComputeShader* RasterShader_ = nullptr;
    ID3D11ComputeShader* ShadowShader_ = nullptr;
};

#else

class D3D11ComputeRuntime
{
public:
    [[nodiscard]] bool Initialize(std::string& error)
    {
        error = "D3D11 compute raster backend is only available on Windows";
        return false;
    }
    void Shutdown() noexcept {}
};

#endif

} // namespace

class ComputeShaderRasterRHI::Context
{
public:
    Detail::ExecutionGate Execution;
    Detail::StatisticsStore Statistics;
    D3D11ComputeRuntime Runtime;
    mutable std::mutex StateMutex;
    std::string Error;
    bool NativeAvailable = false;
};

ComputeShaderRasterRHI::ComputeShaderRasterRHI()
    : Context_(std::make_unique<Context>())
{
    (void)Initialize({});
}

ComputeShaderRasterRHI::~ComputeShaderRasterRHI()
{
    Shutdown();
}

bool ComputeShaderRasterRHI::Initialize(const RasterRHICreateInfo& createInfo)
{
    Shutdown();
    std::string error;
    const bool available = Context_->Runtime.Initialize(error);
    {
        std::lock_guard<std::mutex> lock(Context_->StateMutex);
        Context_->NativeAvailable = available;
        Context_->Error = std::move(error);
    }
    return available && Context_->Execution.Initialize(createInfo.WorkerThreads);
}

void ComputeShaderRasterRHI::Shutdown() noexcept
{
    Context_->Execution.Shutdown();
    Context_->Runtime.Shutdown();
    std::lock_guard<std::mutex> lock(Context_->StateMutex);
    Context_->NativeAvailable = false;
}

bool ComputeShaderRasterRHI::IsInitialized() const noexcept
{
    std::lock_guard<std::mutex> lock(Context_->StateMutex);
    return Context_->NativeAvailable && Context_->Execution.IsInitialized();
}

RasterBackend ComputeShaderRasterRHI::Backend() const noexcept
{
    return RasterBackend::ComputeShader;
}

std::string_view ComputeShaderRasterRHI::Name() const noexcept
{
    return "D3D11 Compute Raster";
}

RasterBackendCapabilities ComputeShaderRasterRHI::Capabilities() const
{
    std::lock_guard<std::mutex> lock(Context_->StateMutex);
    return {
        RasterBackend::ComputeShader,
        Context_->NativeAvailable,
        Context_->NativeAvailable,
        8,
        8,
        Context_->NativeAvailable
            ? "Native Direct3D 11 CS 5.0 tiled raster and shadow kernels"
            : "Unsupported: " + Context_->Error};
}

RasterStatus ComputeShaderRasterRHI::Render(
    const RasterScene& scene,
    const RasterSettings& settings,
    RenderTarget& target,
    const RasterCancellationToken& cancellation)
{
    Detail::ValidateRenderTarget(target);
    {
        std::lock_guard<std::mutex> lock(Context_->StateMutex);
        if (!Context_->NativeAvailable)
        {
            throw std::runtime_error(
                "D3D11 compute raster backend is unsupported: " + Context_->Error);
        }
    }
    RasterStatistics statistics;
    const RasterStatus status = Context_->Execution.Execute(
        [&](const std::atomic<bool>& backendCancellation)
        {
#if defined(_WIN32)
            return Context_->Runtime.Render(
                scene, settings, target, cancellation, backendCancellation, statistics);
#else
            (void)scene; (void)settings; (void)target; (void)cancellation; (void)backendCancellation;
            return RasterStatus::Cancelled;
#endif
        });
    Context_->Statistics.Set(std::move(statistics));
    return status;
}

void ComputeShaderRasterRHI::Cancel() noexcept
{
    Context_->Execution.Cancel();
}

void ComputeShaderRasterRHI::WaitIdle() noexcept
{
    Context_->Execution.WaitIdle();
}

RasterStatistics ComputeShaderRasterRHI::LastStatistics() const
{
    return Context_->Statistics.Get();
}

} // namespace Evans::Rasterizer
