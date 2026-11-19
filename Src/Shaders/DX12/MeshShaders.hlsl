// Native D3D12 mesh-stage implementation, compiled as Shader Model 6.5.
// A bounded workgroup builds local indices, transforms each unique position
// once, then compacts triangles. Primitive attributes retain stable IDs and
// projective interpolation planes across attribute seams. Nanite consumes
// its GPU hardware-bin queue; ordinary geometry consumes immutable world data.
// https://microsoft.github.io/DirectX-Specs/d3d/MeshShader.html
static const uint GroupWidth=32,TrianglesPerMesh=32,Corners=3,VerticesPerMesh=96;
static const uint DispatchColumns=256,WordBytes=4,HardwareHeaderWords=8,HardwareTriangleWord=6;
#if !defined(EVANS_MESH_NANITE) && !defined(EVANS_MESH_NATIVE_BARYCENTRICS)
#define EVANS_MESH_INTERPOLATED_COVERAGE 1
#endif

#if defined(EVANS_MESH_ARGUMENTS)
cbuffer Scheduling : register(b0) { uint4 Counts,Layout,Images; };
ByteAddressBuffer HardwareArguments : register(t0);
RWByteAddressBuffer MeshArguments : register(u0);
[numthreads(GroupWidth,1,1)] void CSArguments(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint bin=(group.y*DispatchColumns+group.x)*GroupWidth+lane;
    if (bin>=Counts.z) return;
    uint triangles=HardwareArguments.Load((bin*HardwareHeaderWords+HardwareTriangleWord)*WordBytes);
    uint groups=(triangles+TrianglesPerMesh-1)/TrianglesPerMesh;
    MeshArguments.Store3(bin*3*WordBytes,uint3(min(groups,DispatchColumns),max(1u,(groups+DispatchColumns-1)/DispatchColumns),1));
}
#else
#if defined(EVANS_MESH_NANITE)
cbuffer DrawParameters : register(b0)
{ uint4 Counts,Layout,Dispatch,Quality,Range; float4 Material; uint4 Images,Schedule; };
ByteAddressBuffer ClipTable : register(t1),WorkQueue : register(t2),HardwareArguments : register(t6);
static const uint ClipWords=16;
static const float PositiveNormalMinimum=1.175494351e-38f;
struct MeshPrimitive { nointerpolation uint Ref:TEXCOORD0; nointerpolation uint Priority:TEXCOORD1; };
float4 ReadPosition(uint id,uint corner)
{ id=min(id,max(Counts.z,1u)-1u); return asfloat(ClipTable.Load4((Layout.z+id*ClipWords+corner*4)*WordBytes)); }
float4 SnapInside(float4 p)
{
    if (p.w>=PositiveNormalMinimum && all(abs(p.xy)<=p.w) && p.z>=0 && p.z<=p.w)
    {
        float subpixels=float(1u<<Quality.y),halfPixel=0.5;
        float2 pixel=floor((p.xy/p.w*float2(halfPixel,-halfPixel)+halfPixel)*float2(Counts.xy)*subpixels+halfPixel)/subpixels;
        p.xy=(pixel/float2(Counts.xy)*float2(2,-2)+float2(-1,1))*p.w;
    }
    return p;
}
#else
cbuffer VisibilityFrame : register(b0)
{
    row_major float4x4 Camera;
    float4 Extent,Tint,DiffuseReflectance,AmbientReflectance,SpecularReflectance,EmittedRadiance;
    float4 MaterialParameters,AmbientLight,SurfaceParameters;
    uint4 TextureOptions,DrawRange,MaterialTiles;
};
ByteAddressBuffer Geometry : register(t0);
static const uint TriangleBytes=192,VectorBytes=16,ColorBytes=96,UVBytes=144;
struct MeshPrimitive
{
#if !defined(EVANS_MESH_NATIVE_BARYCENTRICS)
    nointerpolation float3 PlaneX:TEXCOORD0,PlaneY:TEXCOORD1,PlaneOrigin:TEXCOORD2;
#endif
    nointerpolation uint ID:TEXCOORD3;
};
float4 ReadPosition(uint id,uint corner)
{ id=min(id,max(DrawRange.x,1u)-1u); return asfloat(Geometry.Load4(id*TriangleBytes+corner*VectorBytes)); }
// Plane support is conservative for every vertex inside the world AABB.
// The absolute-product budget covers plane extraction and dot-product rounding;
// cancellation, overflow and clip-plane contact therefore retain the cluster.
static const float FloatEpsilon=1.192092896e-7f,ClipRoundingOperations=32;
bool ClusterVisible(uint cluster)
{
    uint clusters=(DrawRange.x+TrianglesPerMesh-1)/TrianglesPerMesh;
    cluster=min(cluster,max(clusters,1u)-1u); // Safe even for speculative tail loads.
    uint at=(DrawRange.x+cluster)*TriangleBytes;
    float3 low=asfloat(Geometry.Load4(at)).xyz,high=asfloat(Geometry.Load4(at+VectorBytes)).xyz;
    float4 magnitude=float4(max(abs(low),abs(high)),1);
    [unroll] for (uint side=0;side<6;++side)
    {
        uint axis=side/2; float4 row=Camera[axis],other=Camera[3];
        float4 plane=other+((side&1u)==0?row:-row),budget=abs(other)+abs(row);
        if (side==4) { plane=Camera[2]; budget=abs(plane); }
        float4 support=float4(float3(plane.x>=0?high.x:low.x,plane.y>=0?high.y:low.y,plane.z>=0?high.z:low.z),1);
        float upper=dot(plane,support),margin=dot(budget,magnitude)*(FloatEpsilon*ClipRoundingOperations);
        if (isfinite(upper) && isfinite(margin) && upper < -margin) return false;
    }
    return true;
}
bool MeshGroupVisible(uint meshGroup)
{
    uint local=meshGroup*TrianglesPerMesh;
    if (local>=DrawRange.z) return false;
    uint first=DrawRange.y+local,last=first+min(TrianglesPerMesh,DrawRange.z-local)-1;
    uint a=first/TrianglesPerMesh,b=last/TrianglesPerMesh;
    return ClusterVisible(a) || (a!=b && ClusterVisible(b));
}
#if defined(EVANS_MESH_INTERPOLATED_COVERAGE)
// Cutout UV/alpha keep native raster interpolation and participate in the local key.
float4 ReadCoverage(uint id,uint corner)
{
    id=min(id,max(DrawRange.x,1u)-1u);
    float4 packedUV=asfloat(Geometry.Load4(id*TriangleBytes+UVBytes+(corner==2?VectorBytes:0)));
    float2 uv=corner==1?packedUV.zw:packedUV.xy;
    float alpha=asfloat(Geometry.Load4(id*TriangleBytes+ColorBytes+corner*VectorBytes)).a;
    return float4(uv.x,1-uv.y,alpha,0);
}
#endif
#endif

#if defined(EVANS_MESH_TASK_STAGE) || defined(EVANS_MESH_TASK_INPUT)
struct MeshPayload { uint Groups[GroupWidth]; };
#endif
#if defined(EVANS_MESH_TASK_STAGE)
groupshared MeshPayload TaskPayload;
groupshared uint TaskMask;
// One amplification group classifies 32 candidate meshlets and emits a stable
// compact list. Culled clusters allocate no mesh-stage output storage.
[numthreads(GroupWidth,1,1)]
void ASMesh(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    if (lane==0) TaskMask=0;
    GroupMemoryBarrierWithGroupSync();
    uint candidate=(group.y*DispatchColumns+group.x)*GroupWidth+lane;
    if (MeshGroupVisible(candidate)) InterlockedOr(TaskMask,1u<<lane);
    GroupMemoryBarrierWithGroupSync();
    uint mask=TaskMask;
    if ((mask&(1u<<lane))!=0) TaskPayload.Groups[countbits(mask&((1u<<lane)-1u))]=candidate;
    GroupMemoryBarrierWithGroupSync();
    DispatchMesh(countbits(mask),1,1,TaskPayload);
}
#else
bool TriviallyOutside(float4 a,float4 b,float4 c)
{
    if (!all(isfinite(a)) || !all(isfinite(b)) || !all(isfinite(c))) return true;
    return (a.x < -a.w && b.x < -b.w && c.x < -c.w) || (a.x>a.w && b.x>b.w && c.x>c.w) ||
        (a.y < -a.w && b.y < -b.w && c.y < -c.w) || (a.y>a.w && b.y>b.w && c.y>c.w) ||
        (a.z<0 && b.z<0 && c.z<0) || (a.z>a.w && b.z>b.w && c.z>c.w);
}
// FNV-style word hashing with byte folding, followed by full position equality.
// A low load factor bounds probing even for quantized meshes. Atomic minimum
// selects the earliest corner, making local vertex numbering deterministic.
static const uint HashSlots=256,EmptyVertex=0xffffffffu,VertexMaskWords=3,WordByteBits=8;
static const uint HashOffset=2166136261u,HashPrime=16777619u;
groupshared float4 SharedInput[VerticesPerMesh],SharedClip[VerticesPerMesh];
groupshared uint VertexOwners[HashSlots],VertexMasks[VertexMaskWords],SurvivorMask;
#if defined(EVANS_MESH_INTERPOLATED_COVERAGE)
groupshared float4 SharedCoverage[VerticesPerMesh];
#endif
struct MeshVertex
{
    precise float4 Position:SV_POSITION;
#if defined(EVANS_MESH_INTERPOLATED_COVERAGE)
    float2 UV:TEXCOORD4; float Alpha:TEXCOORD5;
#endif
};
uint PositionHash(uint source)
{
    float4 position=SharedInput[source];
    uint value=HashOffset;
    [unroll] for (uint component=0;component<4;++component)
        value=(value^asuint(position[component]==0?0.0:position[component]))*HashPrime;
#if defined(EVANS_MESH_INTERPOLATED_COVERAGE)
    [unroll] for (uint component=0;component<4;++component)
        value=(value^asuint(SharedCoverage[source][component]==0?0.0:SharedCoverage[source][component]))*HashPrime;
#endif
    return (value^(value>>WordByteBits)^(value>>(WordByteBits*2))^(value>>(WordByteBits*3)))&(HashSlots-1);
}
uint InsertPosition(uint candidate)
{
    uint slot=PositionHash(candidate);
    [loop] for (uint probe=0;probe<HashSlots;++probe)
    {
        uint previous; InterlockedCompareExchange(VertexOwners[slot],EmptyVertex,candidate,previous);
        if (previous==EmptyVertex) return slot;
        bool equal=all(SharedInput[previous]==SharedInput[candidate]);
#if defined(EVANS_MESH_INTERPOLATED_COVERAGE)
        equal=equal && all(SharedCoverage[previous]==SharedCoverage[candidate]);
#endif
        if (equal) { InterlockedMin(VertexOwners[slot],candidate); return slot; }
        slot=(slot+1)&(HashSlots-1);
    }
    return EmptyVertex; // Unreachable: at most 96 positions occupy 256 slots.
}
uint VertexRank(uint source)
{
    uint word=source/GroupWidth,bit=source%GroupWidth;
    uint rank=countbits(VertexMasks[word]&((1u<<bit)-1u));
    [unroll] for (uint earlier=0;earlier<VertexMaskWords;++earlier) if (earlier<word) rank+=countbits(VertexMasks[earlier]);
    return rank;
}
uint3 BuildPositionIndex(uint lane,bool valid)
{
    for (uint slot=lane;slot<HashSlots;slot+=GroupWidth) VertexOwners[slot]=EmptyVertex;
    if (lane<VertexMaskWords) VertexMasks[lane]=0;
    if (lane==0) SurvivorMask=0;
    GroupMemoryBarrierWithGroupSync();
    uint3 slots=0,canonical=0;
    if (valid) [unroll] for (uint corner=0;corner<Corners;++corner) slots[corner]=InsertPosition(lane*Corners+corner);
    GroupMemoryBarrierWithGroupSync();
    if (valid) [unroll] for (uint corner=0;corner<Corners;++corner)
    {
        uint source=lane*Corners+corner; canonical[corner]=VertexOwners[slots[corner]];
        if (canonical[corner]==source) InterlockedOr(VertexMasks[source/GroupWidth],1u<<(source%GroupWidth));
    }
    return canonical;
}
[outputtopology("triangle")]
[numthreads(GroupWidth,1,1)]
void MSMesh(uint3 group:SV_GroupID,uint lane:SV_GroupIndex,
#if defined(EVANS_MESH_TASK_INPUT)
    in payload MeshPayload payload,
#endif
    out vertices MeshVertex vertices[VerticesPerMesh],out indices uint3 indices[TrianglesPerMesh],
    out primitives MeshPrimitive primitives[TrianglesPerMesh])
{
    uint meshGroup=group.y*DispatchColumns+group.x; bool clusterValid=true;
#if defined(EVANS_MESH_TASK_INPUT)
    meshGroup=payload.Groups[group.x];
#elif !defined(EVANS_MESH_NANITE)
    if (lane==0) SurvivorMask=MeshGroupVisible(meshGroup)?1u:0u;
    GroupMemoryBarrierWithGroupSync();
    clusterValid=SurvivorMask!=0;
#endif
    uint local=meshGroup*TrianglesPerMesh+lane,id=0,priority=0;
#if defined(EVANS_MESH_NANITE)
    uint count=HardwareArguments.Load((Schedule.x*HardwareHeaderWords+HardwareTriangleWord)*WordBytes);
    bool valid=local<count;
    if (valid)
    {
        uint first=WorkQueue.Load(Schedule.x*WordBytes); priority=WorkQueue.Load((first+local)*WordBytes);
        valid=priority<Schedule.z;
        if (valid) { id=WorkQueue.Load((Schedule.y+priority)*WordBytes); valid=id<Counts.z; }
    }
#else
    id=DrawRange.y+local; bool valid=clusterValid && local<DrawRange.z && id<DrawRange.x;
#endif
    [unroll] for (uint corner=0;corner<Corners;++corner)
    {
        float4 p=valid?ReadPosition(id,corner):0;
#if defined(EVANS_MESH_NANITE)
        p=SnapInside(p);
#endif
        SharedInput[lane*Corners+corner]=p;
#if defined(EVANS_MESH_INTERPOLATED_COVERAGE)
        SharedCoverage[lane*Corners+corner]=valid && MaterialParameters.z>0?ReadCoverage(id,corner):0;
#endif
        valid=valid && all(isfinite(p));
    }
    uint3 canonical=BuildPositionIndex(lane,valid);
    if (valid) [unroll] for (uint corner=0;corner<Corners;++corner)
    {
        uint source=lane*Corners+corner;
        if (canonical[corner]==source)
        {
#if defined(EVANS_MESH_NANITE)
            SharedClip[source]=SharedInput[source];
#else
            precise float4 projected=mul(Camera,SharedInput[source]); SharedClip[source]=projected;
#endif
        }
    }
    GroupMemoryBarrierWithGroupSync();
    float4 positions[Corners]; bool alive=valid;
    if (valid)
    {
        [unroll] for (uint corner=0;corner<Corners;++corner) positions[corner]=SharedClip[canonical[corner]];
        alive=!TriviallyOutside(positions[0],positions[1],positions[2]);
    }
    if (alive) InterlockedOr(SurvivorMask,1u<<lane);
    GroupMemoryBarrierWithGroupSync();
    uint total=countbits(SurvivorMask),vertexCount=0;
    [unroll] for (uint word=0;word<VertexMaskWords;++word) vertexCount+=countbits(VertexMasks[word]);
    SetMeshOutputCounts(total?vertexCount:0,total);
    if (total==0) return;
    if (valid) [unroll] for (uint corner=0;corner<Corners;++corner)
    {
        uint source=lane*Corners+corner;
        if (canonical[corner]==source)
        {
            MeshVertex vertex; float4 clip=SharedClip[source];
            vertex.Position=all(isfinite(clip))?clip:float4(0,0,0,1);
#if defined(EVANS_MESH_INTERPOLATED_COVERAGE)
            vertex.UV=SharedCoverage[source].xy; vertex.Alpha=SharedCoverage[source].z;
#endif
            vertices[VertexRank(source)]=vertex;
        }
    }
    if (alive)
    {
        uint output=countbits(SurvivorMask&((1u<<lane)-1u));
        indices[output]=uint3(VertexRank(canonical[0]),VertexRank(canonical[1]),VertexRank(canonical[2]));
        MeshPrimitive primitive=(MeshPrimitive)0;
#if defined(EVANS_MESH_NANITE)
        primitive.Ref=id+1u; primitive.Priority=priority;
#else
#if !defined(EVANS_MESH_NATIVE_BARYCENTRICS)
        // Adjugate rows of homogeneous XYW reconstruct the original triangle's
        // perspective barycentrics, including near-plane clipping. A common
        // scale prevents cross-product overflow; determinant division cancels.
        float3 a=positions[0].xyw,b=positions[1].xyw,c=positions[2].xyw;
        float3 bound=max(abs(a),max(abs(b),abs(c))); float scale=max(bound.x,max(bound.y,bound.z));
        if (scale>0) { a/=scale; b/=scale; c/=scale; }
        float3 rowA=cross(b,c),rowB=cross(c,a),rowC=cross(a,b);
        primitive.PlaneX=float3(rowA.x,rowB.x,rowC.x); primitive.PlaneY=float3(rowA.y,rowB.y,rowC.y);
        primitive.PlaneOrigin=float3(rowA.z,rowB.z,rowC.z);
#endif
        primitive.ID=id+1u;
#endif
        primitives[output]=primitive;
    }
}
#endif
#endif
