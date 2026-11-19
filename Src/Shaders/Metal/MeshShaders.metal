#include <metal_stdlib>
using namespace metal;

// Native Metal 3 indexed meshlets. Per-primitive references and projective
// interpolation planes preserve material/UV seams while positions are shared.
// MSL 2.17 and 5.7.2; WWDC 2022 "Transform your geometry with Metal mesh shaders".
constant uint MeshThreads=32,MeshPrimitives=32,MeshVertices=96,MeshCorners=3,MeshColumns=256;
constant uint HardwareStride=8,HardwareCountWord=6,ClipStride=16,VectorWords=4;
constant uint HashCapacity=256,MissingVertex=0xffffffffu,VertexMaskWords=3,ByteBits=8;
constant uint WordHashBasis=2166136261u,WordHashPrime=16777619u;
constant float PositiveNormalMinimum=1.175494351e-38f;
struct WorldTriangle { float4 Position[3],Normal[3],Color[3],UVAB,UVC; uint4 Source; };
struct VisibilityFrame
{
    float4 CameraRows[4],Extent,Tint,Diffuse,Ambient,Specular,Emission,Material,AmbientLighting,Surface;
    uint4 Images,Draw,MaterialTiles;
};
struct HardwareFrame { uint4 Counts,Layout,Dispatch,Quality,Range; float4 Material; uint4 Images,Schedule; };
struct HardwarePayload { uint4 MeshBase; };
struct VisibilityPayload { uint Groups[32]; };
constant uint MeshesPerObject=1024;
struct IndexedVertex { float4 Position [[position,invariant]]; };
struct VisibilityIndexedVertex
{
    float4 Position [[position,invariant]];
#if !defined(EVANS_MESH_NATIVE_BARYCENTRICS)
    float2 UV; float Opacity;
#endif
};
struct VisibilityPrimitive
{
#if !defined(EVANS_MESH_NATIVE_BARYCENTRICS)
    float3 PlaneX [[flat]],PlaneY [[flat]],PlaneOrigin [[flat]];
#endif
    uint Reference [[flat]];
};
struct HardwarePrimitive { uint Reference [[flat]],Priority [[flat]]; };
using VisibilityMesh=metal::mesh<VisibilityIndexedVertex,VisibilityPrimitive,MeshVertices,MeshPrimitives,topology::triangle>;
using HardwareMesh=metal::mesh<IndexedVertex,HardwarePrimitive,MeshVertices,MeshPrimitives,topology::triangle>;
struct MeshScratch
{
    float4 Input[MeshVertices],Clip[MeshVertices],Coverage[MeshVertices];
    atomic_uint Owners[HashCapacity],VertexMasks[VertexMaskWords],Survivors;
};
bool rejectTriangle(float4 a,float4 b,float4 c)
{
    if (!all(isfinite(a)) || !all(isfinite(b)) || !all(isfinite(c))) return true;
    return (a.x < -a.w && b.x < -b.w && c.x < -c.w) || (a.x>a.w && b.x>b.w && c.x>c.w) ||
        (a.y < -a.w && b.y < -b.w && c.y < -c.w) || (a.y>a.w && b.y>b.w && c.y>c.w) ||
        (a.z<0 && b.z<0 && c.z<0) || (a.z>a.w && b.z>b.w && c.z>c.w);
}
float4 cameraPosition(float4 p,constant VisibilityFrame& frame)
{
    return float4(dot(frame.CameraRows[0],p),dot(frame.CameraRows[1],p),
        dot(frame.CameraRows[2],p),dot(frame.CameraRows[3],p));
}
// Bound each homogeneous clip half-plane by the world AABB support point.
// A source-product error budget keeps contact/cancellation/overflow conservative.
constant float FloatEpsilon=1.192092896e-7f,ClipRoundingOperations=32.0f;
bool clusterVisible(uint cluster,constant VisibilityFrame& frame,device const WorldTriangle* geometry)
{
    uint clusters=(frame.Draw.x+MeshPrimitives-1)/MeshPrimitives;
    cluster=min(cluster,max(clusters,1u)-1u);
    WorldTriangle bounds=geometry[frame.Draw.x+cluster];
    float3 low=bounds.Position[0].xyz,high=bounds.Position[1].xyz;
    float4 magnitude=float4(max(abs(low),abs(high)),1);
    for (uint side=0;side<6;++side)
    {
        float4 row=frame.CameraRows[side/2],other=frame.CameraRows[3];
        float4 plane=other+((side&1u)==0?row:-row),budget=abs(other)+abs(row);
        if (side==4) { plane=row; budget=abs(row); }
        float4 support=float4(select(low,high,plane.xyz>=float3(0)),1);
        float upper=dot(plane,support),margin=dot(budget,magnitude)*(FloatEpsilon*ClipRoundingOperations);
        if (isfinite(upper) && isfinite(margin) && upper < -margin) return false;
    }
    return true;
}
bool meshGroupVisible(uint meshGroup,constant VisibilityFrame& frame,device const WorldTriangle* geometry)
{
    uint local=meshGroup*MeshPrimitives;
    if (local>=frame.Draw.z) return false;
    uint first=frame.Draw.y+local,last=first+min(MeshPrimitives,frame.Draw.z-local)-1;
    uint a=first/MeshPrimitives,b=last/MeshPrimitives;
    return clusterVisible(a,frame,geometry) || (a!=b && clusterVisible(b,frame,geometry));
}
// An object group classifies 32 meshlets, then publishes their stable compact
// IDs. Invisible clusters never allocate downstream mesh output storage.
[[object,max_total_threads_per_threadgroup(MeshThreads)]]
void EvansVisibilityObject(constant VisibilityFrame& frame [[buffer(0)]],
    device const WorldTriangle* geometry [[buffer(1)]],object_data VisibilityPayload& payload [[payload]],
    mesh_grid_properties grid,uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]])
{
    threadgroup atomic_uint visible;
    if (lane==0) atomic_store_explicit(&visible,0u,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint candidate=(group.y*MeshColumns+group.x)*MeshThreads+lane;
    if (meshGroupVisible(candidate,frame,geometry)) atomic_fetch_or_explicit(&visible,1u<<lane,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint mask=atomic_load_explicit(&visible,memory_order_relaxed);
    if ((mask&(1u<<lane))!=0) payload.Groups[popcount(mask&((1u<<lane)-1))]=candidate;
    threadgroup_barrier(mem_flags::mem_object_data);
    if (lane==0) grid.set_threadgroups_per_grid(uint3(popcount(mask),1,1));
}
uint positionHash(threadgroup MeshScratch& scratch,uint source,bool coverage)
{
    float4 point=scratch.Input[source];
    uint hash=WordHashBasis;
    for (uint component=0;component<4;++component)
        hash=(hash^as_type<uint>(point[component]==0?0.0f:point[component]))*WordHashPrime;
    if (coverage) for (uint component=0;component<4;++component)
        hash=(hash^as_type<uint>(scratch.Coverage[source][component]==0?0.0f:scratch.Coverage[source][component]))*WordHashPrime;
    return (hash^(hash>>ByteBits)^(hash>>(ByteBits*2))^(hash>>(ByteBits*3)))&(HashCapacity-1);
}
uint insertPosition(threadgroup MeshScratch& scratch,uint candidate,bool coverage)
{
    uint slot=positionHash(scratch,candidate,coverage),probes=0;
    while (probes<HashCapacity)
    {
        uint previous=MissingVertex;
        if (atomic_compare_exchange_weak_explicit(&scratch.Owners[slot],&previous,candidate,memory_order_relaxed,memory_order_relaxed)) return slot;
        // A weak CAS may fail spuriously; retry the empty slot rather than
        // consuming probe capacity or pretending that insertion succeeded.
        if (previous==MissingVertex) continue;
        if (all(scratch.Input[previous]==scratch.Input[candidate]) && (!coverage || all(scratch.Coverage[previous]==scratch.Coverage[candidate])))
        { atomic_fetch_min_explicit(&scratch.Owners[slot],candidate,memory_order_relaxed); return slot; }
        slot=(slot+1)&(HashCapacity-1); ++probes;
    }
    return MissingVertex; // At most 96 distinct positions occupy 256 slots.
}
uint3 indexPositions(threadgroup MeshScratch& scratch,uint lane,bool valid,bool coverage)
{
    for (uint slot=lane;slot<HashCapacity;slot+=MeshThreads) atomic_store_explicit(&scratch.Owners[slot],MissingVertex,memory_order_relaxed);
    if (lane<VertexMaskWords) atomic_store_explicit(&scratch.VertexMasks[lane],0u,memory_order_relaxed);
    if (lane==0) atomic_store_explicit(&scratch.Survivors,0u,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint3 slots=0,canonical=0;
    if (valid) for (uint corner=0;corner<MeshCorners;++corner) slots[corner]=insertPosition(scratch,lane*MeshCorners+corner,coverage);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid) for (uint corner=0;corner<MeshCorners;++corner)
    {
        uint source=lane*MeshCorners+corner; canonical[corner]=atomic_load_explicit(&scratch.Owners[slots[corner]],memory_order_relaxed);
        if (canonical[corner]==source) atomic_fetch_or_explicit(&scratch.VertexMasks[source/MeshThreads],1u<<(source%MeshThreads),memory_order_relaxed);
    }
    return canonical;
}
uint vertexRank(threadgroup MeshScratch& scratch,uint source)
{
    uint word=source/MeshThreads,bit=source%MeshThreads;
    uint rank=popcount(atomic_load_explicit(&scratch.VertexMasks[word],memory_order_relaxed)&((1u<<bit)-1));
    for (uint before=0;before<VertexMaskWords;++before) if (before<word) rank+=popcount(atomic_load_explicit(&scratch.VertexMasks[before],memory_order_relaxed));
    return rank;
}
uint2 compactPrimitives(threadgroup MeshScratch& scratch,uint lane,bool alive)
{
    if (alive) atomic_fetch_or_explicit(&scratch.Survivors,1u<<lane,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint mask=atomic_load_explicit(&scratch.Survivors,memory_order_relaxed);
    return uint2(popcount(mask&((1u<<lane)-1)),popcount(mask));
}
void emitVisibilityMesh(VisibilityMesh mesh,constant VisibilityFrame& frame,
    device const WorldTriangle* geometry,uint meshGroup,uint lane,threadgroup MeshScratch& scratch)
{
    uint local=meshGroup*MeshPrimitives+lane,id=frame.Draw.y+local;
    bool valid=local<frame.Draw.z && id<frame.Draw.x;
    for (uint corner=0;corner<MeshCorners;++corner)
    {
        float4 p=valid?geometry[min(id,max(frame.Draw.x,1u)-1u)].Position[corner]:float4(0);
        scratch.Input[lane*MeshCorners+corner]=p;
#if !defined(EVANS_MESH_NATIVE_BARYCENTRICS)
        float4 attributes=0;
        if (valid && frame.Material.z>0)
        {
            WorldTriangle item=geometry[min(id,max(frame.Draw.x,1u)-1u)]; float2 uv=corner==0?item.UVAB.xy:corner==1?item.UVAB.zw:item.UVC.xy;
            attributes=float4(uv.x,1-uv.y,item.Color[corner].a,0);
        }
        scratch.Coverage[lane*MeshCorners+corner]=attributes;
#endif
        valid=valid && all(isfinite(p));
    }
    uint3 canonical=indexPositions(scratch,lane,valid,
#if defined(EVANS_MESH_NATIVE_BARYCENTRICS)
        false
#else
        true
#endif
    );
    if (valid) for (uint corner=0;corner<MeshCorners;++corner)
    {
        uint source=lane*MeshCorners+corner;
        if (canonical[corner]==source) scratch.Clip[source]=cameraPosition(scratch.Input[source],frame);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 positions[MeshCorners]; bool alive=valid;
    if (valid)
    {
        for (uint corner=0;corner<MeshCorners;++corner) positions[corner]=scratch.Clip[canonical[corner]];
        alive=!rejectTriangle(positions[0],positions[1],positions[2]);
    }
    uint2 compact=compactPrimitives(scratch,lane,alive); if (lane==0) mesh.set_primitive_count(compact.y);
    if (compact.y==0) return;
    if (valid) for (uint corner=0;corner<MeshCorners;++corner)
    {
        uint source=lane*MeshCorners+corner;
        if (canonical[corner]==source)
        {
            float4 p=scratch.Clip[source]; if (!all(isfinite(p))) p=float4(0,0,0,1);
#if defined(EVANS_MESH_NATIVE_BARYCENTRICS)
            mesh.set_vertex(vertexRank(scratch,source),{p});
#else
            mesh.set_vertex(vertexRank(scratch,source),{p,scratch.Coverage[source].xy,scratch.Coverage[source].z});
#endif
        }
    }
    if (alive)
    {
        VisibilityPrimitive primitive;
#if !defined(EVANS_MESH_NATIVE_BARYCENTRICS)
        // Homogeneous XYW adjugate rows survive near-plane clipping. The
        // determinant cancels when the fragment normalizes its three weights.
        float3 a=positions[0].xyw,b=positions[1].xyw,c=positions[2].xyw;
        float3 bound=max(abs(a),max(abs(b),abs(c))); float scale=max(bound.x,max(bound.y,bound.z));
        if (scale>0) { a/=scale; b/=scale; c/=scale; }
        float3 row0=cross(b,c),row1=cross(c,a),row2=cross(a,b);
        primitive.PlaneX=float3(row0.x,row1.x,row2.x); primitive.PlaneY=float3(row0.y,row1.y,row2.y);
        primitive.PlaneOrigin=float3(row0.z,row1.z,row2.z);
#endif
        primitive.Reference=id+1;
        mesh.set_primitive(compact.x,primitive);
        for (uint corner=0;corner<MeshCorners;++corner) mesh.set_index(compact.x*MeshCorners+corner,uchar(vertexRank(scratch,canonical[corner])));
    }
}
[[mesh,max_total_threads_per_threadgroup(MeshThreads)]]
void EvansVisibilityMesh(VisibilityMesh mesh,constant VisibilityFrame& frame [[buffer(0)]],
    device const WorldTriangle* geometry [[buffer(1)]],uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],object_data const VisibilityPayload& payload [[payload]])
{
    threadgroup MeshScratch scratch;
    emitVisibilityMesh(mesh,frame,geometry,payload.Groups[group.x],lane,scratch);
}
[[mesh,max_total_threads_per_threadgroup(MeshThreads)]]
void EvansVisibilityDirectMesh(VisibilityMesh mesh,constant VisibilityFrame& frame [[buffer(0)]],
    device const WorldTriangle* geometry [[buffer(1)]],uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup MeshScratch scratch; uint meshGroup=group.y*MeshColumns+group.x;
    if (lane==0) atomic_store_explicit(&scratch.Survivors,meshGroupVisible(meshGroup,frame,geometry)?1u:0u,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(&scratch.Survivors,memory_order_relaxed)==0)
    { if (lane==0) mesh.set_primitive_count(0u); return; }
    emitVisibilityMesh(mesh,frame,geometry,meshGroup,lane,scratch);
}
float4 hardwareCorner(device const uint* positions,uint id,uint corner,constant HardwareFrame& frame)
{
    id=min(id,max(frame.Counts.z,1u)-1u);
    uint at=frame.Layout.z+id*ClipStride+corner*VectorWords;
    return as_type<float4>(uint4(positions[at],positions[at+1],positions[at+2],positions[at+3]));
}
float4 quantizeInside(float4 p,constant HardwareFrame& frame)
{
    if (p.w>=PositiveNormalMinimum && all(abs(p.xy)<=p.w) && p.z>=0 && p.z<=p.w)
    {
        float grid=float(1u<<frame.Quality.y); float2 extent=float2(frame.Counts.xy);
        float2 pixel=floor((p.xy/p.w*float2(0.5f,-0.5f)+0.5f)*extent*grid+0.5f)/grid;
        p.xy=(pixel/extent*float2(2,-2)+float2(-1,1))*p.w;
    }
    return p;
}
[[mesh,max_total_threads_per_threadgroup(MeshThreads)]]
void EvansNaniteMesh(HardwareMesh mesh,constant HardwareFrame& frame [[buffer(0)]],
    device const uint* projected [[buffer(2)]],device const uint* queue [[buffer(3)]],
    device const uint* arguments [[buffer(5)]],uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],object_data const HardwarePayload& payload [[payload]])
{
    uint local=(payload.MeshBase.x+group.x)*MeshPrimitives+lane,id=0,priority=0;
    uint count=arguments[frame.Schedule.x*HardwareStride+HardwareCountWord]; bool valid=local<count;
    if (valid)
    {
        priority=queue[queue[frame.Schedule.x]+local]; valid=priority<frame.Schedule.z;
        if (valid) { id=queue[frame.Schedule.y+priority]; valid=id<frame.Counts.z; }
    }
    threadgroup MeshScratch scratch;
    for (uint corner=0;corner<MeshCorners;++corner)
    {
        float4 p=valid?quantizeInside(hardwareCorner(projected,id,corner,frame),frame):float4(0);
        scratch.Input[lane*MeshCorners+corner]=p; valid=valid && all(isfinite(p));
    }
    uint3 canonical=indexPositions(scratch,lane,valid,false);
    if (valid) for (uint corner=0;corner<MeshCorners;++corner)
    {
        uint source=lane*MeshCorners+corner;
        if (canonical[corner]==source) scratch.Clip[source]=scratch.Input[source];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 positions[MeshCorners]; bool alive=valid;
    if (valid)
    {
        for (uint corner=0;corner<MeshCorners;++corner) positions[corner]=scratch.Clip[canonical[corner]];
        alive=!rejectTriangle(positions[0],positions[1],positions[2]);
    }
    uint2 compact=compactPrimitives(scratch,lane,alive); if (lane==0) mesh.set_primitive_count(compact.y);
    if (compact.y==0) return;
    if (valid) for (uint corner=0;corner<MeshCorners;++corner)
    {
        uint source=lane*MeshCorners+corner;
        if (canonical[corner]==source)
        {
            float4 p=scratch.Clip[source]; if (!all(isfinite(p))) p=float4(0,0,0,1);
            mesh.set_vertex(vertexRank(scratch,source),{p});
        }
    }
    if (alive)
    {
        mesh.set_primitive(compact.x,{id+1,priority});
        for (uint corner=0;corner<MeshCorners;++corner) mesh.set_index(compact.x*MeshCorners+corner,uchar(vertexRank(scratch,canonical[corner])));
    }
}
// Object-stage dispatch works on Apple7/8 and Mac2, which expose mesh shaders
// without Apple9's indirect mesh-draw arguments. The GPU bin count controls
// each mesh grid; empty objects emit no geometry and never reach the CPU.
[[object,max_total_threads_per_threadgroup(1)]]
void EvansNaniteObject(constant HardwareFrame& frame [[buffer(0)]],device const uint* raster [[buffer(5)]],
    object_data HardwarePayload& payload [[payload]],mesh_grid_properties grid,uint3 group [[threadgroup_position_in_grid]])
{
    uint count=raster[frame.Schedule.x*HardwareStride+HardwareCountWord];
    uint meshes=(count+MeshPrimitives-1)/MeshPrimitives,first=group.x*MeshesPerObject;
    payload.MeshBase=uint4(first,0,0,0);
    grid.set_threadgroups_per_grid(uint3(first<meshes?min(meshes-first,MeshesPerObject):0u,1,1));
}

#if defined(EVANS_MESH_NATIVE_BARYCENTRICS)
// MSL's device-negotiated perspective barycentrics match the rasterizer's
// interpolation directly. Attributes still come from the original primitive.
struct BaryVisibilityInput { float4 Position [[position]]; uint Reference [[flat]]; };
struct BaryVisibilityOutput { uint Reference [[color(0)]]; float2 Weights [[color(1)]]; };
fragment BaryVisibilityOutput EvansVisibilityBaryFragment(BaryVisibilityInput input [[stage_in]],
    float3 bary [[barycentric_coord,center_perspective]],bool frontFace [[front_facing]],
    constant VisibilityFrame& frame [[buffer(0)]],const device WorldTriangle* geometry [[buffer(1)]],
    texture2d<float> diffuse [[texture(2)]],texture2d<float> opacity [[texture(4)]],sampler repeatSampler [[sampler(0)]])
{
    constexpr uint BackFace=0x80000000u;
    if (input.Reference==0 || input.Reference>frame.Draw.x || !all(isfinite(bary))) discard_fragment();
    if (frame.Material.z>0)
    {
        uint safeTriangle=min(input.Reference-1u,max(frame.Draw.x,1u)-1u); WorldTriangle triangle=geometry[safeTriangle];
        float2 a=float2(triangle.UVAB.x,1-triangle.UVAB.y),b=float2(triangle.UVAB.z,1-triangle.UVAB.w),c=float2(triangle.UVC.x,1-triangle.UVC.y);
        float2 uv=a*bary.x+b*bary.y+c*bary.z;
        float coverage=dot(float3(triangle.Color[0].a,triangle.Color[1].a,triangle.Color[2].a),bary)*frame.Material.y;
        if (frame.Images.x!=0) coverage*=diffuse.sample(repeatSampler,uv).a;
        if (frame.Images.z!=0) coverage*=opacity.sample(repeatSampler,uv).r;
        if (saturate(coverage)<frame.Material.z) discard_fragment();
    }
    return {input.Reference|(frontFace?0u:BackFace),bary.yz};
}
#endif
