using uint = unsigned int;
__device__ float4 operator+(float4 a,float4 b){return make_float4(a.x+b.x,a.y+b.y,a.z+b.z,a.w+b.w);}
__device__ float4 operator*(float4 a,float b){return make_float4(a.x*b,a.y*b,a.z*b,a.w*b);}
__device__ float4 operator/(float4 a,float b){return make_float4(a.x/b,a.y/b,a.z/b,a.w/b);}
__device__ void operator*=(float4& a,float4 b){a=make_float4(a.x*b.x,a.y*b.y,a.z*b.z,a.w*b.w);}
__device__ float3 operator+(float3 a,float3 b){return make_float3(a.x+b.x,a.y+b.y,a.z+b.z);}
__device__ float3 operator-(float3 a,float3 b){return make_float3(a.x-b.x,a.y-b.y,a.z-b.z);}
__device__ float3 operator*(float3 a,float b){return make_float3(a.x*b,a.y*b,a.z*b);}
__device__ float3 operator/(float3 a,float b){return make_float3(a.x/b,a.y/b,a.z/b);}
__device__ float3 to_float3(float4 a){return make_float3(a.x,a.y,a.z);}
struct Vertex {
    float4 Clip; float4 Color; float4 Position; float4 Normal; float4 ShadowClip;
    float4 TexCoord;
};
struct Triangle {
    Vertex Vertices[3]; float4 BaseColor; float4 SpecularColor; float4 EmissiveColor;
    float Shininess; float AlphaCutoff;
    unsigned int DiffuseTextureIndex; unsigned int OpacityTextureIndex;
    unsigned int Selected; uint NormalTextureIndex; float SubsurfaceWeight; uint ReceivesLightmap;
};
struct Texture { unsigned int OffsetTexels, Width, Height, FirstMip; };
struct Light {
    float4 PositionType; float4 DirectionIntensity;
    float4 ColorCastShadows; float4 AreaSizePadding;
};
struct Constants {
    float World[16]; float View[16]; float Projection[16]; float LightViewProjection[16];
    unsigned int Width, Height, TriangleCount, InputSpace;
    unsigned int DepthRange, FillModeValue, ShadingValue, BackFaceCulling;
    unsigned int FrontFaceCounterClockwise, OutlineEnabled, ShadowsEnabled, ShadowWidth;
    unsigned int ShadowHeight, TilesX; float WireframeWidth, OutlineWidth;
    float ShadowBias, ShadowStrength; unsigned int DepthTest, ShadowPCFRadius;
    float4 ClearColor, WireframeColor, OutlineColor, CameraPosition;
    float4 LightDirectionIntensity, LightColorAmbientStrength, AmbientColor;
    unsigned int LightCount; int ShadowLightIndex;
    unsigned int TextureCount, TextureTexelCount;
    Light Lights[4];
    float4 ShadowRasterBias;
    uint MaterialOptions, FrontLightmap, BackLightmap, ACESColor;
    float4 DisplayParameters;
    uint SkyTexture, SkyDiffuseTexture; float SkyIntensity, SkyDiffuseIntensity;
    float4 SubsurfaceScattering;
    float4 NormalRows[3]; unsigned MainTileSize, ShadowTileSize, GroupsX, CollectCounts;
};
static_assert(sizeof(Vertex)==96,"CUDA raster vertex ABI mismatch");
static_assert(sizeof(Triangle)==368,"CUDA raster triangle ABI mismatch");
static_assert(sizeof(Texture)==16,"CUDA raster texture ABI mismatch");
static_assert(sizeof(Light)==64,"CUDA raster light ABI mismatch");
static_assert(sizeof(Constants)==864,"CUDA raster constants ABI mismatch");
__device__ float4 mul_matrix(const float* m, float4 v) {
    return make_float4(
        m[0]*v.x+m[1]*v.y+m[2]*v.z+m[3]*v.w,
        m[4]*v.x+m[5]*v.y+m[6]*v.z+m[7]*v.w,
        m[8]*v.x+m[9]*v.y+m[10]*v.z+m[11]*v.w,
        m[12]*v.x+m[13]*v.y+m[14]*v.z+m[15]*v.w);
}
__device__ float edge(float2 a, float2 b, float2 p) {
    return (b.x-a.x)*(p.y-a.y)-(b.y-a.y)*(p.x-a.x);
}
__device__ float3 normalize_safe(float3 v) {
    float length2=v.x*v.x+v.y*v.y+v.z*v.z;
    if(length2<1.0e-12f) return make_float3(0,0,1);
    float inverse=rsqrtf(length2); return make_float3(v.x*inverse,v.y*inverse,v.z*inverse);
}
__device__ float dot3(float3 a,float3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
__device__ float3 cross3(float3 a,float3 b){return make_float3(
    a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);}
__device__ float3 normalize_or(float3 value,float3 fallback){
    float length2=dot3(value,value);
    return length2<1.0e-12f?fallback:value*rsqrtf(length2);
}
#define R_FN __device__
#define R_TRI Triangle
#define R_TEX Texture
#define R_READ(T) const T*
#define R_MIN fminf
#define R_MAX fmaxf
#define R_FLOOR floorf
#define R_CEIL ceilf
#define R_ABS fabsf
#define R_SQRT sqrtf
#define R_LOG2 log2f
#define R_POW powf
#define R_EXP2 exp2f
#define R_FINITE isfinite
#define R_ATAN2 atan2f
#define R_ACOS acosf
#define R_DOT dot3
#define R_CROSS cross3
#define R_FLOAT2 make_float2
#define R_FLOAT3 make_float3
#define R_FLOAT4 make_float4
#define R_XYZ(v) to_float3(v)
#define R_NORMALIZE_OR normalize_or
#define EVANS_GPU_CODE(...) __VA_ARGS__
#include "GpuMaterials.inc"
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
__device__ float4 sample_texture(const Texture* textures, const uint* texels, const Constants* constants,
    uint index, float2 uv, bool srgb, RasterTextureGradients gradients = {}) {
    const uint flag = srgb ? 1 : 4;
    return GpuSampleTexture((constants->MaterialOptions & flag) != 0 ? index : 0xffffffffu, uv, srgb ? 1 : 0,
        gradients, constants->TextureCount, (constants->MaterialOptions & 8) != 0, textures, texels);
}
__device__ float light_range_window(float3 displacement, float radius) {
    if(radius<=0.0f)return 1.0f;
    if(fmaxf(fabsf(displacement.x),fmaxf(fabsf(displacement.y),fabsf(displacement.z)))>=radius)return 0.0f;
    float3 relative=displacement/radius;
    float squaredRatio=dot3(relative,relative);
    float fade=fmaxf(1.0f-squaredRatio*squaredRatio,0.0f);
    return fade*fade;
}
__device__ void accumulate_light_sample(
    float3 lightDirection,float3 radiance,float visibility,float3 normal,
    float3 viewDirection,float shininess,unsigned int shading,float3 scatter,
    float3& diffuseLight,float3& specularLight){
    float directionLengthSquared=dot3(lightDirection,lightDirection);
    if(directionLengthSquared<=1.0e-12f)return;
    float3 direction=lightDirection*rsqrtf(directionLengthSquared);
    float cosine = dot3(normal, direction);
    float normalToLight = fmaxf(cosine, 0.0f);
    float3 visibleRadiance=radiance*visibility;
    diffuseLight = diffuseLight + make_float3(
        visibleRadiance.x * fmaxf(cosine + scatter.x, 0.0f) / (1.0f + scatter.x),
        visibleRadiance.y * fmaxf(cosine + scatter.y, 0.0f) / (1.0f + scatter.y),
        visibleRadiance.z * fmaxf(cosine + scatter.z, 0.0f) / (1.0f + scatter.z));
    if (shading == 2 && normalToLight > 0){
        float3 halfVector=normalize_or(direction+viewDirection,normal);
        float specularAmount=powf(fmaxf(dot3(normal,halfVector),0.0f),fmaxf(shininess,1.0f));
        specularLight=specularLight+visibleRadiance*specularAmount;
    }
}
__device__ void accumulate_scene_light(
    const Constants* constants,unsigned int lightIndex,float3 position,float3 normal,
    float3 viewDirection,float shininess,float shadowVisibility,float3 scatter,
    float3& diffuseLight,float3& specularLight){
    Light light=constants->Lights[lightIndex];
    unsigned int type=(unsigned int)(light.PositionType.w+0.5f);
    float visibility=((int)lightIndex==constants->ShadowLightIndex&&
        light.ColorCastShadows.w>0.5f)?shadowVisibility:1.0f;
    float intensity=fmaxf(light.DirectionIntensity.w,0.0f);
    float3 radiance=make_float3(
        fmaxf(light.ColorCastShadows.x,0.0f)*intensity,
        fmaxf(light.ColorCastShadows.y,0.0f)*intensity,
        fmaxf(light.ColorCastShadows.z,0.0f)*intensity);
    if(type==0){
        accumulate_light_sample(to_float3(light.DirectionIntensity),radiance,visibility,
            normal,viewDirection,shininess,constants->ShadingValue, scatter,diffuseLight,specularLight);
        return;
    }
    if(type==1){
        float3 toLight=to_float3(light.PositionType)-position;
        float distanceSquared=fmaxf(dot3(toLight,toLight),1.0e-6f);
        accumulate_light_sample(toLight,radiance*(light_range_window(toLight,light.AreaSizePadding.z)/distanceSquared),visibility,normal,
            viewDirection,shininess,constants->ShadingValue, scatter,diffuseLight,specularLight);
        return;
    }
    float3 emissionDirection=normalize_or(
        to_float3(light.DirectionIntensity),make_float3(0,-1,0));
    float3 helper=fabsf(emissionDirection.y)<0.999f?
        make_float3(0,1,0):make_float3(1,0,0);
    float3 tangent=normalize_or(cross3(helper,emissionDirection),make_float3(1,0,0));
    float3 bitangent=normalize_or(cross3(emissionDirection,tangent),make_float3(0,0,1));
    for(int sampleY=0;sampleY<2;++sampleY){
        for(int sampleX=0;sampleX<2;++sampleX){
            float offsetX=sampleX==0?-0.25f:0.25f;
            float offsetY=sampleY==0?-0.25f:0.25f;
            float3 samplePosition=to_float3(light.PositionType)+
                tangent*(offsetX*light.AreaSizePadding.x)+
                bitangent*(offsetY*light.AreaSizePadding.y);
            float3 toLight=samplePosition-position;
            float distanceSquared=fmaxf(dot3(toLight,toLight),1.0e-6f);
            float3 direction=toLight*rsqrtf(distanceSquared);
            float emitterCosine=fmaxf(dot3(emissionDirection,direction*-1.0f),0.0f);
            accumulate_light_sample(direction,
                radiance*(0.25f*emitterCosine*light_range_window(toLight,light.AreaSizePadding.z)/distanceSquared),visibility,normal,
                viewDirection,shininess,constants->ShadingValue, scatter,diffuseLight,specularLight);
        }
    }
}
#define EVANS_GPU_CODE(...) __VA_ARGS__
#define R_FN __device__
#define R_TRI Triangle
#define R_READ(T) const T*
#define R_WRITE(T) T*
#define R_INOUT(T) T&
#define R_OUT(T) T&
#define R_FLOAT4 make_float4
#define R_LOWBIT(bits) (__ffs(bits)-1)
#define R_MIN fminf
#define R_MAX fmaxf
#define R_ABS fabsf
#define R_ADD(value, bits) atomicAdd(&(value), bits)
#define R_OR(value, bits) atomicOr(&(value), bits)
#include "GpuGeometry.inc"

extern "C" __global__ void TransformKernel(const Triangle* source, Triangle* transformed, const Constants* constants) {
    unsigned id=(blockIdx.y*constants->GroupsX+blockIdx.x)*64+threadIdx.x;
    if (id>=constants->TriangleCount) return;
    Triangle primitive=source[id];
    for (int corner=0;corner<3;++corner) {
        Vertex& vertex=primitive.Vertices[corner]; float3 normal=to_float3(vertex.Normal);
        if (constants->InputSpace!=0) {
            float4 world=mul_matrix(constants->World,vertex.Position);
            if (fabsf(world.w)>1e-6f && fabsf(world.w-1.0f)>1e-6f) {world=world/world.w;world.w=1;}
            vertex.Position=world;vertex.Clip=mul_matrix(constants->Projection,mul_matrix(constants->View,world));
            vertex.ShadowClip=mul_matrix(constants->LightViewProjection,world);
            normal=make_float3(dot3(to_float3(constants->NormalRows[0]),normal),dot3(to_float3(constants->NormalRows[1]),normal),dot3(to_float3(constants->NormalRows[2]),normal));
        }
        normal=normalize_safe(normal);vertex.Normal=make_float4(normal.x,normal.y,normal.z,0);
    }
    transformed[id]=primitive;
}
extern "C" __global__ void SetupKernel(const Triangle* triangles, ProjectedVertex* projected, unsigned* bins, const Constants* constants, unsigned shadow) {
    unsigned id=(blockIdx.y*constants->GroupsX+blockIdx.x)*64+threadIdx.x;
    ProjectAndBin(id,constants->TriangleCount,shadow?constants->ShadowWidth:constants->Width,
        shadow?constants->ShadowHeight:constants->Height,shadow?constants->ShadowTileSize:constants->MainTileSize,
        constants->DepthRange,shadow!=0,constants->CollectCounts!=0,shadow!=0 || constants->FillModeValue==0,triangles,projected,bins);
}
extern "C" __global__ void ShadowDepthKernel(
    const Triangle* triangles, const Constants* constants,
    const unsigned int* bins, const ProjectedVertex* projected, float* shadowDepths,
    const Texture* textures, const unsigned int* textureTexels) {
    unsigned int x=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=constants->ShadowWidth||y>=constants->ShadowHeight)return;
    unsigned int pixel=y*constants->ShadowWidth+x;
    float bestDepth=1.0e30f;
    float2 sample=make_float2((float)x+0.5f,(float)y+0.5f);
    BinCursor cursor=OpenBin((y/constants->ShadowTileSize)*((constants->ShadowWidth+constants->ShadowTileSize-1)/constants->ShadowTileSize)+x/constants->ShadowTileSize,constants->TriangleCount);
    uint sourceId;
    while (NextCandidate(cursor,bins,sourceId)) {
        ProjectedVertex first=projected[sourceId*3];
        uint corners=((uint)first.Weights.w)&15u;
        for(uint fan=1;fan+1<corners;++fan) {
        ProjectedVertex second=projected[VertexSlot(sourceId,fan,constants->TriangleCount)],third=projected[VertexSlot(sourceId,fan+1,constants->TriangleCount)];
        float2 screen[3]={make_float2(first.Screen.x,first.Screen.y),make_float2(second.Screen.x,second.Screen.y),make_float2(third.Screen.x,third.Screen.y)};
        float inverseW[3]={first.Screen.w,second.Screen.w,third.Screen.w}, vertexDepth[3]={first.Screen.z,second.Screen.z,third.Screen.z};
        if(inverseW[0]<=0 || inverseW[1]<=0 || inverseW[2]<=0)continue;
        float area=edge(screen[0],screen[1],screen[2]); if(fabsf(area)<1.0e-6f)continue;
        float w0=edge(screen[1],screen[2],sample)/area;
        float w1=edge(screen[2],screen[0],sample)/area; float w2=1.0f-w0-w1;
        if(w0<0||w1<0||w2<0)continue;
        Triangle triangle=FanAttributes(triangles[sourceId],first,second,third);
        float depth=w0*vertexDepth[0]+w1*vertexDepth[1]+w2*vertexDepth[2];
        if(depth<0||depth>1)continue;
        if(constants->ShadowRasterBias.x>0){
            float dz1=vertexDepth[1]-vertexDepth[0],dz2=vertexDepth[2]-vertexDepth[0];
            float slopeX=(dz1*(screen[2].y-screen[0].y)-dz2*(screen[1].y-screen[0].y))/area;
            float slopeY=((screen[1].x-screen[0].x)*dz2-(screen[2].x-screen[0].x)*dz1)/area;
            depth=fminf(depth+constants->ShadowRasterBias.x*fmaxf(fabsf(slopeX),fabsf(slopeY)),1.0f);
        }
        if(depth>=bestDepth)continue;
        if((constants->MaterialOptions&4)!=0 && triangle.AlphaCutoff>0) {
        float p0=w0*inverseW[0],p1=w1*inverseW[1],p2=w2*inverseW[2];
        float denominator=p0+p1+p2;
        if(fabsf(denominator)<1.0e-8f)continue;
        float2 uv=make_float2(
            (triangle.Vertices[0].TexCoord.x*p0+triangle.Vertices[1].TexCoord.x*p1+
             triangle.Vertices[2].TexCoord.x*p2)/denominator,
            (triangle.Vertices[0].TexCoord.y*p0+triangle.Vertices[1].TexCoord.y*p1+
             triangle.Vertices[2].TexCoord.y*p2)/denominator);
        float4 diffuse=sample_texture(
            textures,textureTexels,constants,triangle.DiffuseTextureIndex,uv,true);
        float opacity=sample_texture(
            textures,textureTexels,constants,triangle.OpacityTextureIndex,uv,false).x;
        float vertexAlpha=(triangle.Vertices[0].Color.w*p0+
            triangle.Vertices[1].Color.w*p1+triangle.Vertices[2].Color.w*p2)/denominator;
        float alpha=fminf(fmaxf(
            vertexAlpha*triangle.BaseColor.w*diffuse.w*opacity,0.0f),1.0f);
        if(triangle.AlphaCutoff>0.0f&&alpha<triangle.AlphaCutoff)continue;
        }
        bestDepth=fminf(bestDepth,depth);
    }
    }
    shadowDepths[pixel]=bestDepth;
}
extern "C" __global__ void RasterKernel(
    const Triangle* triangles, const Constants* constants,
    const unsigned int* bins, const ProjectedVertex* projected,
    const float* shadowDepths, const Texture* textures,
    const unsigned int* textureTexels, float4* colors, float* depths, float* visibility) {
    unsigned int x=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=constants->Width||y>=constants->Height)return;
    unsigned int pixel=y*constants->Width+x;
    float visibilityDepth=1.0e30f;bool selected=false;
    uint winningId=0, winningFan=0;
    float winningP0=0, winningP1=0, winningP2=0, winningDenominator=1;
    bool winningFront=false;
    float bestDepth=1.0e30f,bestWireDepth=1.0e30f; float4 bestColor=constants->ClearColor;
    float2 sample=make_float2((float)x+0.5f,(float)y+0.5f);
    BinCursor cursor=OpenBin((y/constants->MainTileSize)*((constants->Width+constants->MainTileSize-1)/constants->MainTileSize)+x/constants->MainTileSize,constants->TriangleCount);
    uint sourceId;
    while (NextCandidate(cursor,bins,sourceId)) {
        ProjectedVertex first=projected[sourceId*3];
        uint corners=((uint)first.Weights.w)&15u;
        for(uint fan=1;fan+1<corners;++fan) {
        ProjectedVertex second=projected[VertexSlot(sourceId,fan,constants->TriangleCount)],third=projected[VertexSlot(sourceId,fan+1,constants->TriangleCount)];
        float2 screen[3]={make_float2(first.Screen.x,first.Screen.y),make_float2(second.Screen.x,second.Screen.y),make_float2(third.Screen.x,third.Screen.y)};
        float inverseW[3]={first.Screen.w,second.Screen.w,third.Screen.w}, vertexDepth[3]={first.Screen.z,second.Screen.z,third.Screen.z};
        if(inverseW[0]<=0 || inverseW[1]<=0 || inverseW[2]<=0)continue;
        float area=edge(screen[0],screen[1],screen[2]); if(fabsf(area)<1.0e-6f)continue;
        bool front=constants->FrontFaceCounterClockwise!=0?area<0:area>0;
        if(constants->BackFaceCulling&& !front)continue;
        float w0=edge(screen[1],screen[2],sample)/area;
        float w1=edge(screen[2],screen[0],sample)/area; float w2=1.0f-w0-w1;
        bool inside=w0>=0&&w1>=0&&w2>=0;
        bool wire=false;
        float edgeWeights[3]={0,0,0};
        if(constants->FillModeValue!=0){
            float nearestEdge=1.0e30f;
            for(int edgeIndex=0;edgeIndex<3;++edgeIndex){
                int next=(edgeIndex+1)%3;
                float sx=screen[next].x-screen[edgeIndex].x,sy=screen[next].y-screen[edgeIndex].y;
                float px=sample.x-screen[edgeIndex].x,py=sample.y-screen[edgeIndex].y;
                float t=fminf(fmaxf((px*sx+py*sy)/fmaxf(sx*sx+sy*sy,1.0e-12f),0.0f),1.0f);
                float dx=px-t*sx,dy=py-t*sy,distanceSquared=dx*dx+dy*dy;
                if(distanceSquared<nearestEdge){
                    nearestEdge=distanceSquared;edgeWeights[0]=edgeWeights[1]=edgeWeights[2]=0;
                    edgeWeights[edgeIndex]=1.0f-t;edgeWeights[next]=t;
                }
            }
            float halfWidth=fmaxf(constants->WireframeWidth,1.0f)*0.5f;
            wire=nearestEdge<=halfWidth*halfWidth;
        }
        if(!inside && !wire)continue;
        Triangle triangle=FanAttributes(triangles[sourceId],first,second,third);
        if(wire){
            float wireDepth=edgeWeights[0]*vertexDepth[0]+edgeWeights[1]*vertexDepth[1]+edgeWeights[2]*vertexDepth[2];
            float ep0=edgeWeights[0]*inverseW[0],ep1=edgeWeights[1]*inverseW[1],ep2=edgeWeights[2]*inverseW[2];
            float edgeDenominator=ep0+ep1+ep2;
            if(wireDepth>=0&&wireDepth<=1&&fabsf(edgeDenominator)>1.0e-8f){
                float2 edgeUV=make_float2(
                    (triangle.Vertices[0].TexCoord.x*ep0+triangle.Vertices[1].TexCoord.x*ep1+triangle.Vertices[2].TexCoord.x*ep2)/edgeDenominator,
                    (triangle.Vertices[0].TexCoord.y*ep0+triangle.Vertices[1].TexCoord.y*ep1+triangle.Vertices[2].TexCoord.y*ep2)/edgeDenominator);
                float edgeAlpha=(triangle.Vertices[0].Color.w*ep0+triangle.Vertices[1].Color.w*ep1+triangle.Vertices[2].Color.w*ep2)/edgeDenominator;
                edgeAlpha*=triangle.BaseColor.w*sample_texture(textures,textureTexels,constants,triangle.DiffuseTextureIndex,edgeUV,true).w*
                    sample_texture(textures,textureTexels,constants,triangle.OpacityTextureIndex,edgeUV,false).x;
                if((constants->MaterialOptions&4)==0 || triangle.AlphaCutoff<=0||fminf(fmaxf(edgeAlpha,0.0f),1.0f)>=triangle.AlphaCutoff)
                    bestWireDepth=fminf(bestWireDepth,wireDepth);
            }
        }
        if(constants->OutlineEnabled && inside) {
            float z=w0*vertexDepth[0]+w1*vertexDepth[1]+w2*vertexDepth[2];
            float a=w0*inverseW[0],b=w1*inverseW[1],c=w2*inverseW[2],d=a+b+c;
            if(z>=0 && z<=1 && z<visibilityDepth && fabsf(d)>1e-8f &&
                ((constants->MaterialOptions&4)==0 || triangle.AlphaCutoff<=0 || [&]() {
                    float2 uv=make_float2((triangle.Vertices[0].TexCoord.x*a+triangle.Vertices[1].TexCoord.x*b+triangle.Vertices[2].TexCoord.x*c)/d,
                        (triangle.Vertices[0].TexCoord.y*a+triangle.Vertices[1].TexCoord.y*b+triangle.Vertices[2].TexCoord.y*c)/d);
                    float alpha=(triangle.Vertices[0].Color.w*a+triangle.Vertices[1].Color.w*b+triangle.Vertices[2].Color.w*c)/d*triangle.BaseColor.w;
                    alpha*=sample_texture(textures,textureTexels,constants,triangle.DiffuseTextureIndex,uv,true).w;
                    alpha*=sample_texture(textures,textureTexels,constants,triangle.OpacityTextureIndex,uv,false).x;
                    return alpha>=triangle.AlphaCutoff;
                }()))
            {visibilityDepth=z;selected=triangle.Selected!=0;}
        }
        if(constants->FillModeValue==1||!inside)continue;
        float depth=w0*vertexDepth[0]+w1*vertexDepth[1]+w2*vertexDepth[2];
        if(depth<0||depth>1||(constants->DepthTest&&depth>=bestDepth))continue;
        float p0=w0*inverseW[0],p1=w1*inverseW[1],p2=w2*inverseW[2],denominator=p0+p1+p2;
        if(fabsf(denominator)<1.0e-8f)continue;
        if((constants->MaterialOptions&4)!=0 && triangle.AlphaCutoff>0.0f) {
        float4 vertexColor=(triangle.Vertices[0].Color*p0+
            triangle.Vertices[1].Color*p1+triangle.Vertices[2].Color*p2)/denominator;
        float2 uv=make_float2(
            (triangle.Vertices[0].TexCoord.x*p0+triangle.Vertices[1].TexCoord.x*p1+
             triangle.Vertices[2].TexCoord.x*p2)/denominator,
            (triangle.Vertices[0].TexCoord.y*p0+triangle.Vertices[1].TexCoord.y*p1+
             triangle.Vertices[2].TexCoord.y*p2)/denominator);
        float4 diffuse=sample_texture(
            textures,textureTexels,constants,triangle.DiffuseTextureIndex,uv,true);
        float opacity=sample_texture(
            textures,textureTexels,constants,triangle.OpacityTextureIndex,uv,false).x;
        float alpha=fminf(fmaxf(
            vertexColor.w*triangle.BaseColor.w*diffuse.w*opacity,0.0f),1.0f);
        if(triangle.AlphaCutoff>0.0f&&alpha<triangle.AlphaCutoff)continue;
        }
        bestDepth=depth;winningId=sourceId;winningFan=fan;winningFront=front;
        winningP0=p0;winningP1=p1;winningP2=p2;winningDenominator=denominator;
        }
    }
    if(winningFan!=0) {
        Triangle triangle=FanAttributes(triangles[winningId],projected[winningId*3],
            projected[VertexSlot(winningId,winningFan,constants->TriangleCount)],
            projected[VertexSlot(winningId,winningFan+1,constants->TriangleCount)]);
        float p0=winningP0,p1=winningP1,p2=winningP2,denominator=winningDenominator;
        bool front=winningFront;
        float4 vertexColor=(triangle.Vertices[0].Color*p0+
            triangle.Vertices[1].Color*p1+triangle.Vertices[2].Color*p2)/denominator;
        float2 uv=make_float2(
            (triangle.Vertices[0].TexCoord.x*p0+triangle.Vertices[1].TexCoord.x*p1+
             triangle.Vertices[2].TexCoord.x*p2)/denominator,
            (triangle.Vertices[0].TexCoord.y*p0+triangle.Vertices[1].TexCoord.y*p1+
             triangle.Vertices[2].TexCoord.y*p2)/denominator);
        const auto gradients = GpuTextureGradients(triangle, uv, denominator, constants->Width, constants->Height);
        float4 diffuse=sample_texture(
            textures,textureTexels,constants,triangle.DiffuseTextureIndex,uv,true,gradients);
        float opacity=sample_texture(
            textures,textureTexels,constants,triangle.OpacityTextureIndex,uv,false,gradients).x;
        float alpha=fminf(fmaxf(
            vertexColor.w*triangle.BaseColor.w*diffuse.w*opacity,0.0f),1.0f);
        float shadowVisibility=1.0f;
        if(constants->ShadowsEnabled&&constants->ShadowWidth>0&&constants->ShadowHeight>0){
            const auto shadowNormal = normalize_safe(to_float3(triangle.Vertices[0].Normal*p0+
                triangle.Vertices[1].Normal*p1+triangle.Vertices[2].Normal*p2)/denominator) * (front ? 1.0f : -1.0f);
            const float receiverBias = GpuShadowBias(constants->ShadowBias, constants->ShadowRasterBias.z, shadowNormal,
                to_float3(constants->Lights[max(0,min(constants->ShadowLightIndex,3))].DirectionIntensity));
            float4 shadow=(triangle.Vertices[0].ShadowClip*p0+triangle.Vertices[1].ShadowClip*p1+triangle.Vertices[2].ShadowClip*p2)/denominator;
            if(shadow.w>1.0e-6f){
                float inverseShadowW=1.0f/shadow.w;
                float shadowX=shadow.x*inverseShadowW*0.5f+0.5f;
                float shadowY=0.5f-shadow.y*inverseShadowW*0.5f;
                float shadowZ=shadow.z*inverseShadowW;
                if(constants->DepthRange!=0)shadowZ=shadowZ*0.5f+0.5f;
                if(shadowX>=0&&shadowX<=1&&shadowY>=0&&shadowY<=1&&shadowZ>=0&&shadowZ<=1){
                    int centerX=min((int)(shadowX*constants->ShadowWidth),(int)constants->ShadowWidth-1);
                    int centerY=min((int)(shadowY*constants->ShadowHeight),(int)constants->ShadowHeight-1);
                    int radius=(int)min(constants->ShadowPCFRadius,8U);
                    unsigned int occluded=0,validSamples=0;
                    for(int offsetY=-radius;offsetY<=radius;++offsetY){
                        int sy=centerY+offsetY;if(sy<0||sy>=(int)constants->ShadowHeight)continue;
                        for(int offsetX=-radius;offsetX<=radius;++offsetX){
                            int sx=centerX+offsetX;if(sx<0||sx>=(int)constants->ShadowWidth)continue;
                            ++validSamples;
                            if(shadowZ-receiverBias>shadowDepths[sy*constants->ShadowWidth+sx])++occluded;
                        }
                    }
                    if(validSamples>0)
                        shadowVisibility=1.0f-fminf(fmaxf(constants->ShadowStrength,0.0f),1.0f)*
                            ((float)occluded/(float)validSamples);
                }
            }
        }
        float4 color=vertexColor;
        color*=triangle.BaseColor;
        color*=diffuse;
        color.w=alpha;
        if(constants->ShadingValue==0){
            color.x+=triangle.EmissiveColor.x;
            color.y+=triangle.EmissiveColor.y;
            color.z+=triangle.EmissiveColor.z;
        }else{
            float3 n=normalize_safe((to_float3(triangle.Vertices[0].Normal)*p0+to_float3(triangle.Vertices[1].Normal)*p1+to_float3(triangle.Vertices[2].Normal)*p2)/denominator);
            if(!front)n=n*-1.0f;
            if ((constants->MaterialOptions&2)!=0 && triangle.NormalTextureIndex<constants->TextureCount)
                n = GpuMappedNormal(triangle, n, GpuSampleTexture(triangle.NormalTextureIndex,uv,0,
                    gradients,constants->TextureCount,(constants->MaterialOptions&8)!=0,textures,textureTexels));
            float3 position=to_float3((triangle.Vertices[0].Position*p0+triangle.Vertices[1].Position*p1+triangle.Vertices[2].Position*p2)/denominator);
            float3 viewDirection=normalize_or(to_float3(constants->CameraPosition)-position,n);
            float3 diffuseLight=make_float3(0,0,0),specularLight=make_float3(0,0,0);
            unsigned int lightCount=min(constants->LightCount,4U);
            for(unsigned int lightIndex=0;lightIndex<lightCount;++lightIndex)
                accumulate_scene_light(constants,lightIndex,position,n,viewDirection,
                    triangle.Shininess,shadowVisibility,to_float3(constants->SubsurfaceScattering)*triangle.SubsurfaceWeight,diffuseLight,specularLight);
            if (constants->SkyDiffuseIntensity > 0 && constants->SkyDiffuseTexture < constants->TextureCount)
                diffuseLight = diffuseLight + GpuSampleSky(n,constants->SkyDiffuseTexture,constants->TextureCount,
                    textures,textureTexels)*constants->SkyDiffuseIntensity;
            uint atlas = front || constants->BackLightmap>=constants->TextureCount ? constants->FrontLightmap : constants->BackLightmap;
            if (triangle.ReceivesLightmap && atlas < constants->TextureCount) {
                float4 uv2 = (triangle.Vertices[0].TexCoord*p0+triangle.Vertices[1].TexCoord*p1+triangle.Vertices[2].TexCoord*p2)/denominator;
                diffuseLight = diffuseLight + to_float3(GpuSampleTexture(atlas,make_float2(uv2.z,uv2.w),2,gradients,
                    constants->TextureCount,false,textures,textureTexels))*constants->DisplayParameters.w;
            }
            color.x=color.x*(constants->AmbientColor.x*constants->LightColorAmbientStrength.w+diffuseLight.x)+triangle.SpecularColor.x*specularLight.x+triangle.EmissiveColor.x;
            color.y=color.y*(constants->AmbientColor.y*constants->LightColorAmbientStrength.w+diffuseLight.y)+triangle.SpecularColor.y*specularLight.y+triangle.EmissiveColor.y;
            color.z=color.z*(constants->AmbientColor.z*constants->LightColorAmbientStrength.w+diffuseLight.z)+triangle.SpecularColor.z*specularLight.z+triangle.EmissiveColor.z;
        }
        const float3 display = GpuDisplayColor(to_float3(color),constants->ACESColor,constants->DisplayParameters);
        bestColor=make_float4(fminf(fmaxf(display.x,0.0f),1.0f),fminf(fmaxf(display.y,0.0f),1.0f),
            fminf(fmaxf(display.z,0.0f),1.0f),fminf(fmaxf(color.w,0),1));
    }
    if(constants->FillModeValue==1)bestDepth=bestWireDepth;
    if(constants->FillModeValue!=0&&bestWireDepth<=1.0f&&(!constants->DepthTest||bestWireDepth<=bestDepth+1.0e-4f))
    {bestColor=constants->WireframeColor;bestDepth=fminf(bestDepth,bestWireDepth);}
    if (bestDepth > 1.0f && constants->FillModeValue != 1 && constants->SkyIntensity > 0 && constants->SkyTexture < constants->TextureCount) {
        const float* v = constants->View; const float* p = constants->Projection;
        const float3 ray = GpuSkyRay(x+0.5f,y+0.5f,constants->Width,constants->Height,
            make_float3(v[0],v[1],v[2]),make_float3(v[4],v[5],v[6]),make_float3(v[8],v[9],v[10]),p[0],p[5]);
        const auto color = GpuDisplayColor(GpuSampleSky(ray,constants->SkyTexture,constants->TextureCount,textures,textureTexels)*
            constants->SkyIntensity,constants->ACESColor,constants->DisplayParameters);
        bestColor = make_float4(color.x,color.y,color.z,1);
    }
    colors[pixel]=bestColor;
    if (depths) depths[pixel]=bestDepth;
    if (constants->OutlineEnabled) visibility[pixel]=selected?-visibilityDepth-1:visibilityDepth;
}
extern "C" __global__ void OutlineKernel(const Constants* constants, const float* visibility, float4* colors) {
    uint x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x<constants->Width && y<constants->Height &&
        IsOutlinePixel(x,y,constants->Width,constants->Height,(int)constants->OutlineWidth,visibility))
        colors[y*constants->Width+x]=constants->OutlineColor;
}
