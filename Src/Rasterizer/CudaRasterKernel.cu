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
    unsigned int Selected; unsigned int Padding0; unsigned int Padding1; unsigned int Padding2;
};
struct Texture { unsigned int OffsetTexels, Width, Height, Padding; };
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
};
static_assert(sizeof(Vertex)==96,"CUDA raster vertex ABI mismatch");
static_assert(sizeof(Triangle)==368,"CUDA raster triangle ABI mismatch");
static_assert(sizeof(Texture)==16,"CUDA raster texture ABI mismatch");
static_assert(sizeof(Light)==64,"CUDA raster light ABI mismatch");
static_assert(sizeof(Constants)==736,"CUDA raster constants ABI mismatch");
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
__device__ float srgb_to_linear(float value) {
    return value<=0.04045f?value/12.92f:powf((value+0.055f)/1.055f,2.4f);
}
__device__ unsigned int wrap_index(int value,unsigned int size) {
    int wrapped=value%(int)size;
    return (unsigned int)(wrapped<0?wrapped+(int)size:wrapped);
}
__device__ float4 decode_texel(unsigned int packed,bool decodeSrgb) {
    constexpr float inverseByte=1.0f/255.0f;
    float4 result=make_float4(
        (float)(packed&0xffU)*inverseByte,
        (float)((packed>>8U)&0xffU)*inverseByte,
        (float)((packed>>16U)&0xffU)*inverseByte,
        (float)((packed>>24U)&0xffU)*inverseByte);
    if(decodeSrgb){
        result.x=srgb_to_linear(result.x);
        result.y=srgb_to_linear(result.y);
        result.z=srgb_to_linear(result.z);
    }
    return result;
}
__device__ float4 sample_texture(
    const Texture* textures,const unsigned int* texels,const Constants* constants,
    unsigned int textureIndex,float2 uv,bool decodeSrgb) {
    if(textureIndex>=constants->TextureCount)return make_float4(1,1,1,1);
    Texture descriptor=textures[textureIndex];
    unsigned long long texelCount=
        (unsigned long long)descriptor.Width*(unsigned long long)descriptor.Height;
    if(descriptor.Width==0||descriptor.Height==0||
       descriptor.OffsetTexels>=constants->TextureTexelCount||
       texelCount>(unsigned long long)(constants->TextureTexelCount-descriptor.OffsetTexels))
        return make_float4(1,1,1,1);
    float repeatU=uv.x-floorf(uv.x);
    float flippedV=1.0f-uv.y;
    float repeatV=flippedV-floorf(flippedV);
    float x=repeatU*(float)descriptor.Width-0.5f;
    float y=repeatV*(float)descriptor.Height-0.5f;
    int x0=(int)floorf(x),y0=(int)floorf(y);
    float blendX=x-(float)x0,blendY=y-(float)y0;
    unsigned int ix0=wrap_index(x0,descriptor.Width);
    unsigned int ix1=wrap_index(x0+1,descriptor.Width);
    unsigned int iy0=wrap_index(y0,descriptor.Height);
    unsigned int iy1=wrap_index(y0+1,descriptor.Height);
    unsigned int row0=descriptor.OffsetTexels+iy0*descriptor.Width;
    unsigned int row1=descriptor.OffsetTexels+iy1*descriptor.Width;
    float4 c00=decode_texel(texels[row0+ix0],decodeSrgb);
    float4 c10=decode_texel(texels[row0+ix1],decodeSrgb);
    float4 c01=decode_texel(texels[row1+ix0],decodeSrgb);
    float4 c11=decode_texel(texels[row1+ix1],decodeSrgb);
    float4 top=c00*(1.0f-blendX)+c10*blendX;
    float4 bottom=c01*(1.0f-blendX)+c11*blendX;
    return top*(1.0f-blendY)+bottom*blendY;
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
__device__ void accumulate_light_sample(
    float3 lightDirection,float3 radiance,float visibility,float3 normal,
    float3 viewDirection,float shininess,unsigned int shading,
    float3& diffuseLight,float3& specularLight){
    float directionLengthSquared=dot3(lightDirection,lightDirection);
    if(directionLengthSquared<=1.0e-12f)return;
    float3 direction=lightDirection*rsqrtf(directionLengthSquared);
    float normalToLight=fmaxf(dot3(normal,direction),0.0f);
    if(normalToLight<=0.0f)return;
    float3 visibleRadiance=radiance*visibility;
    diffuseLight=diffuseLight+visibleRadiance*normalToLight;
    if(shading==2){
        float3 halfVector=normalize_or(direction+viewDirection,normal);
        float specularAmount=powf(fmaxf(dot3(normal,halfVector),0.0f),fmaxf(shininess,1.0f));
        specularLight=specularLight+visibleRadiance*specularAmount;
    }
}
__device__ void accumulate_scene_light(
    const Constants* constants,unsigned int lightIndex,float3 position,float3 normal,
    float3 viewDirection,float shininess,float shadowVisibility,
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
            normal,viewDirection,shininess,constants->ShadingValue,diffuseLight,specularLight);
        return;
    }
    if(type==1){
        float3 toLight=to_float3(light.PositionType)-position;
        float distanceSquared=fmaxf(dot3(toLight,toLight),1.0e-6f);
        accumulate_light_sample(toLight,radiance/distanceSquared,visibility,normal,
            viewDirection,shininess,constants->ShadingValue,diffuseLight,specularLight);
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
                radiance*(0.25f*emitterCosine/distanceSquared),visibility,normal,
                viewDirection,shininess,constants->ShadingValue,diffuseLight,specularLight);
        }
    }
}
__device__ float4 light_clip(const Vertex& vertex,const Constants* constants) {
    if(constants->InputSpace==0) return vertex.ShadowClip;
    return mul_matrix(constants->LightViewProjection,mul_matrix(constants->World,vertex.Position));
}
extern "C" __global__ void ShadowDepthKernel(
    const Triangle* triangles, const Constants* constants,
    const unsigned int* tileOffsets, const unsigned int* tileIndices, float* shadowDepths,
    const Texture* textures, const unsigned int* textureTexels) {
    unsigned int x=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=constants->ShadowWidth||y>=constants->ShadowHeight)return;
    unsigned int pixel=y*constants->ShadowWidth+x;
    float bestDepth=1.0e30f;
    float2 sample=make_float2((float)x+0.5f,(float)y+0.5f);
    unsigned int tilesX=(constants->ShadowWidth+15U)/16U;
    unsigned int tile=(y/16U)*tilesX+x/16U;
    for(unsigned int binIndex=tileOffsets[tile];binIndex<tileOffsets[tile+1U];++binIndex){
        Triangle triangle=triangles[tileIndices[binIndex]];
        float2 screen[3]; float inverseW[3]; float vertexDepth[3]; bool valid=true;
        for(int vertexIndex=0;vertexIndex<3;++vertexIndex){
            float4 clip=light_clip(triangle.Vertices[vertexIndex],constants);
            if(clip.w<=1.0e-6f){valid=false;break;}
            inverseW[vertexIndex]=1.0f/clip.w;
            float3 ndc=make_float3(
                clip.x*inverseW[vertexIndex],clip.y*inverseW[vertexIndex],
                clip.z*inverseW[vertexIndex]);
            screen[vertexIndex]=make_float2((ndc.x*0.5f+0.5f)*constants->ShadowWidth,(0.5f-ndc.y*0.5f)*constants->ShadowHeight);
            vertexDepth[vertexIndex]=constants->DepthRange==0?ndc.z:ndc.z*0.5f+0.5f;
        }
        if(!valid)continue;
        float area=edge(screen[0],screen[1],screen[2]); if(fabsf(area)<1.0e-6f)continue;
        float w0=edge(screen[1],screen[2],sample)/area;
        float w1=edge(screen[2],screen[0],sample)/area; float w2=1.0f-w0-w1;
        if(w0<0||w1<0||w2<0)continue;
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
        float depth=w0*vertexDepth[0]+w1*vertexDepth[1]+w2*vertexDepth[2];
        if(depth<0||depth>1)continue;
        if(constants->ShadowRasterBias.x>0){
            float dz1=vertexDepth[1]-vertexDepth[0],dz2=vertexDepth[2]-vertexDepth[0];
            float slopeX=(dz1*(screen[2].y-screen[0].y)-dz2*(screen[1].y-screen[0].y))/area;
            float slopeY=((screen[1].x-screen[0].x)*dz2-(screen[2].x-screen[0].x)*dz1)/area;
            depth=fminf(depth+constants->ShadowRasterBias.x*fmaxf(fabsf(slopeX),fabsf(slopeY)),1.0f);
        }
        bestDepth=fminf(bestDepth,depth);
    }
    shadowDepths[pixel]=bestDepth;
}
extern "C" __global__ void RasterKernel(
    const Triangle* triangles, const Constants* constants,
    const unsigned int* tileOffsets, const unsigned int* tileIndices,
    const float* shadowDepths, const Texture* textures,
    const unsigned int* textureTexels, float4* colors, float* depths) {
    unsigned int x=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=constants->Width||y>=constants->Height)return;
    unsigned int pixel=y*constants->Width+x;
    float bestDepth=1.0e30f,bestWireDepth=1.0e30f; float4 bestColor=constants->ClearColor;
    float2 sample=make_float2((float)x+0.5f,(float)y+0.5f);
    unsigned int tile=(y/16U)*constants->TilesX+x/16U;
    for(unsigned int binIndex=tileOffsets[tile];binIndex<tileOffsets[tile+1U];++binIndex){
        Triangle triangle=triangles[tileIndices[binIndex]]; float4 clip[3]; float4 world[3]; float4 shadowClip[3]; float3 normal[3];
        float2 screen[3]; float inverseW[3]; float vertexDepth[3]; bool valid=true;
        for(int vertexIndex=0;vertexIndex<3;++vertexIndex){
            Vertex vertex=triangle.Vertices[vertexIndex];
            if(constants->InputSpace==0){clip[vertexIndex]=vertex.Clip;world[vertexIndex]=vertex.Position;normal[vertexIndex]=to_float3(vertex.Normal);shadowClip[vertexIndex]=vertex.ShadowClip;}
            else {world[vertexIndex]=mul_matrix(constants->World,vertex.Position);
                clip[vertexIndex]=mul_matrix(constants->Projection,mul_matrix(constants->View,world[vertexIndex]));
                shadowClip[vertexIndex]=mul_matrix(constants->LightViewProjection,world[vertexIndex]);
                normal[vertexIndex]=normalize_safe(make_float3(
                    constants->World[0]*vertex.Normal.x+constants->World[1]*vertex.Normal.y+constants->World[2]*vertex.Normal.z,
                    constants->World[4]*vertex.Normal.x+constants->World[5]*vertex.Normal.y+constants->World[6]*vertex.Normal.z,
                    constants->World[8]*vertex.Normal.x+constants->World[9]*vertex.Normal.y+constants->World[10]*vertex.Normal.z));}
            if(clip[vertexIndex].w<=1.0e-6f){valid=false;break;} inverseW[vertexIndex]=1.0f/clip[vertexIndex].w;
            float3 ndc=make_float3(clip[vertexIndex].x*inverseW[vertexIndex],clip[vertexIndex].y*inverseW[vertexIndex],clip[vertexIndex].z*inverseW[vertexIndex]);
            screen[vertexIndex]=make_float2((ndc.x*0.5f+0.5f)*constants->Width,(0.5f-ndc.y*0.5f)*constants->Height);
            vertexDepth[vertexIndex]=constants->DepthRange==0?ndc.z:ndc.z*0.5f+0.5f;
        }
        if(!valid)continue;
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
                if(triangle.AlphaCutoff<=0||fminf(fmaxf(edgeAlpha,0.0f),1.0f)>=triangle.AlphaCutoff)
                    bestWireDepth=fminf(bestWireDepth,wireDepth);
            }
        }
        if(constants->FillModeValue==1||!inside)continue;
        float depth=w0*vertexDepth[0]+w1*vertexDepth[1]+w2*vertexDepth[2];
        if(depth<0||depth>1||(constants->DepthTest&&depth>=bestDepth))continue;
        float p0=w0*inverseW[0],p1=w1*inverseW[1],p2=w2*inverseW[2],denominator=p0+p1+p2;
        if(fabsf(denominator)<1.0e-8f)continue;
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
        float shadowVisibility=1.0f;
        if(constants->ShadowsEnabled&&constants->ShadowWidth>0&&constants->ShadowHeight>0){
            float4 shadow=(shadowClip[0]*p0+shadowClip[1]*p1+shadowClip[2]*p2)/denominator;
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
                            if(shadowZ-constants->ShadowBias>shadowDepths[sy*constants->ShadowWidth+sx])++occluded;
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
            float3 n=normalize_safe((normal[0]*p0+normal[1]*p1+normal[2]*p2)/denominator);
            if(!front)n=n*-1.0f;
            float3 position=to_float3((world[0]*p0+world[1]*p1+world[2]*p2)/denominator);
            float3 viewDirection=normalize_or(to_float3(constants->CameraPosition)-position,n);
            float3 diffuseLight=make_float3(0,0,0),specularLight=make_float3(0,0,0);
            unsigned int lightCount=min(constants->LightCount,4U);
            for(unsigned int lightIndex=0;lightIndex<lightCount;++lightIndex)
                accumulate_scene_light(constants,lightIndex,position,n,viewDirection,
                    triangle.Shininess,shadowVisibility,diffuseLight,specularLight);
            color.x=color.x*(constants->AmbientColor.x*constants->LightColorAmbientStrength.w+diffuseLight.x)+triangle.SpecularColor.x*specularLight.x+triangle.EmissiveColor.x;
            color.y=color.y*(constants->AmbientColor.y*constants->LightColorAmbientStrength.w+diffuseLight.y)+triangle.SpecularColor.y*specularLight.y+triangle.EmissiveColor.y;
            color.z=color.z*(constants->AmbientColor.z*constants->LightColorAmbientStrength.w+diffuseLight.z)+triangle.SpecularColor.z*specularLight.z+triangle.EmissiveColor.z;
        }
        color.x=fmaxf(color.x,0.0f);color.y=fmaxf(color.y,0.0f);color.z=fmaxf(color.z,0.0f);
        float displayScale=1.0f/fmaxf(1.0f,fmaxf(color.x,fmaxf(color.y,color.z)));
        bestDepth=depth;bestColor=make_float4(fminf(color.x*displayScale,1.0f),fminf(color.y*displayScale,1.0f),fminf(color.z*displayScale,1.0f),fminf(fmaxf(color.w,0),1));
    }
    if(constants->FillModeValue==1)bestDepth=bestWireDepth;
    if(constants->FillModeValue!=0&&bestWireDepth<=1.0f&&(!constants->DepthTest||bestWireDepth<=bestDepth+1.0e-4f))
    {bestColor=constants->WireframeColor;bestDepth=fminf(bestDepth,bestWireDepth);}
    colors[pixel]=bestColor;depths[pixel]=bestDepth;
}
