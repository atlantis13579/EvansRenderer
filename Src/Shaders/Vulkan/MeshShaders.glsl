#version 450
// Independent Vulkan EXT mesh stages. Each group exports at most 32 triangles
// with locally shared vertices and stable per-primitive references. Nanite bins
// dispatch indirectly from a GPU-written argument buffer; no CPU count
// feedback is needed.
// https://docs.vulkan.org/features/latest/features/proposals/VK_EXT_mesh_shader.html
#if !defined(EVANS_MESH_ARGUMENTS)
#extension GL_EXT_mesh_shader : require
#endif
const uint MeshWidth=32u,MeshCorners=3u,MeshColumns=256u,HardwareWords=8u,TriangleCountWord=6u;
#if !defined(EVANS_MESH_NANITE) && !defined(EVANS_MESH_NATIVE_BARYCENTRICS)
#define EVANS_MESH_INTERPOLATED_COVERAGE 1
#endif
layout(local_size_x=32,local_size_y=1,local_size_z=1) in;

#if defined(EVANS_MESH_ARGUMENTS)
layout(std140,set=0,binding=0) uniform ScheduleFrame { uvec4 counts,layoutInfo,images; };
layout(std430,set=0,binding=1) readonly buffer SourceCommands { uint sourceWords[]; };
layout(std430,set=0,binding=2) writeonly buffer MeshCommands { uint commands[]; };
void main()
{
    uint bin=(gl_WorkGroupID.y*MeshColumns+gl_WorkGroupID.x)*MeshWidth+gl_LocalInvocationIndex;
    if (bin>=counts.z) return;
    uint count=sourceWords[bin*HardwareWords+TriangleCountWord];
    uint groups=(count+MeshWidth-1u)/MeshWidth,at=bin*3u;
    commands[at]=min(groups,MeshColumns); commands[at+1u]=max(1u,(groups+MeshColumns-1u)/MeshColumns); commands[at+2u]=1u;
}
#else
#if !defined(EVANS_MESH_TASK_STAGE)
layout(triangles,max_vertices=96,max_primitives=32) out;
// Match vertex-stage depth for independently rasterized lightmap receivers.
out gl_MeshPerVertexEXT
{
    invariant precise vec4 gl_Position;
} gl_MeshVerticesEXT[];
#endif
#if defined(EVANS_MESH_NANITE)
layout(std140,set=0,binding=0) uniform HardwareFrame
{ uvec4 counts,layoutInfo,dispatchInfo,quality,range; vec4 material; uvec4 images,schedule; };
layout(std430,set=0,binding=2) readonly buffer ProjectedGeometry { uint projected[]; };
layout(std430,set=0,binding=3) readonly buffer CompactedGeometry { uint queue[]; };
layout(std430,set=0,binding=7) readonly buffer RasterCommands { uint arguments[]; };
layout(location=0) perprimitiveEXT flat out uint references[];
layout(location=1) perprimitiveEXT flat out uint priorities[];
const uint ProjectedStride=16u,VectorWords=4u;
const float PositiveFloatMinimum=1.175494351e-38;
vec4 cornerPosition(uint id,uint corner)
{
    id=min(id,max(counts.z,1u)-1u);
    uint at=layoutInfo.z+id*ProjectedStride+corner*VectorWords;
    return uintBitsToFloat(uvec4(projected[at],projected[at+1u],projected[at+2u],projected[at+3u]));
}
vec4 snapCoverage(vec4 p)
{
    if (p.w>=PositiveFloatMinimum && all(lessThanEqual(abs(p.xy),vec2(p.w))) && p.z>=0.0 && p.z<=p.w)
    {
        float grid=float(1u<<quality.y); vec2 extent=vec2(counts.xy);
        vec2 pixel=floor((p.xy/p.w*vec2(0.5,-0.5)+0.5)*extent*grid+0.5)/grid;
        p.xy=(pixel/extent*vec2(2.0,-2.0)+vec2(-1.0,1.0))*p.w;
    }
    return p;
}
#else
struct WorldTriangle { vec4 position[3],normal[3],color[3],uvAB,uvC; uvec4 source; };
layout(std140,set=0,binding=0,row_major) uniform VisibilityFrame
{
    mat4 camera;
    vec4 viewport,tint,diffuse,ambient,specular,emission,material,ambientLight,surface;
    uvec4 options,draw,materialTiles;
};
layout(std430,set=0,binding=1) readonly buffer WorldGeometry { WorldTriangle geometry[]; };
#if !defined(EVANS_MESH_TASK_STAGE)
#if !defined(EVANS_MESH_NATIVE_BARYCENTRICS)
layout(location=0) perprimitiveEXT flat out vec3 planeX[];
layout(location=1) perprimitiveEXT flat out vec3 planeY[];
layout(location=2) perprimitiveEXT flat out vec3 planeOrigin[];
layout(location=4) out vec2 coverageUV[];
layout(location=5) out float coverageAlpha[];
#endif
layout(location=3) perprimitiveEXT flat out uint references[];
#endif
vec4 cornerPosition(uint id,uint corner) { return geometry[min(id,max(draw.x,1u)-1u)].position[corner]; }
// A positive AABB support point bounds each world-space clip half-plane.
// Absolute source-row products retain near-plane contacts and cancellation.
const float FloatEpsilon=1.192092896e-7,ClipRoundingOperations=32.0;
bool clusterVisible(uint cluster)
{
    uint clusters=(draw.x+MeshWidth-1u)/MeshWidth;
    cluster=min(cluster,max(clusters,1u)-1u);
    WorldTriangle bounds=geometry[draw.x+cluster];
    vec3 low=bounds.position[0].xyz,high=bounds.position[1].xyz;
    vec4 magnitude=vec4(max(abs(low),abs(high)),1.0);
    for (uint side=0u;side<6u;++side)
    {
        uint axis=side/2u;
        vec4 row=vec4(camera[0][axis],camera[1][axis],camera[2][axis],camera[3][axis]);
        vec4 other=vec4(camera[0][3],camera[1][3],camera[2][3],camera[3][3]);
        vec4 plane=other+((side&1u)==0u?row:-row),budget=abs(other)+abs(row);
        if (side==4u) { plane=row; budget=abs(row); }
        vec4 support=vec4(mix(low,high,greaterThanEqual(plane.xyz,vec3(0.0))),1.0);
        float upper=dot(plane,support),margin=dot(budget,magnitude)*(FloatEpsilon*ClipRoundingOperations);
        if (!isnan(upper) && !isinf(upper) && !isnan(margin) && !isinf(margin) && upper < -margin) return false;
    }
    return true;
}
bool meshGroupVisible(uint meshGroup)
{
    uint local=meshGroup*MeshWidth;
    if (local>=draw.z) return false;
    uint first=draw.y+local,last=first+min(MeshWidth,draw.z-local)-1u;
    uint a=first/MeshWidth,b=last/MeshWidth;
    return clusterVisible(a) || (a!=b && clusterVisible(b));
}
#endif
#if defined(EVANS_MESH_TASK_STAGE) || defined(EVANS_MESH_TASK_INPUT)
struct MeshPayload { uint groups[32]; };
taskPayloadSharedEXT MeshPayload payload;
#endif
#if defined(EVANS_MESH_TASK_STAGE)
shared uint taskMask;
void main()
{
    uint lane=gl_LocalInvocationIndex;
    if (lane==0u) taskMask=0u;
    barrier();
    uint candidate=(gl_WorkGroupID.y*MeshColumns+gl_WorkGroupID.x)*MeshWidth+lane;
    if (meshGroupVisible(candidate)) atomicOr(taskMask,1u<<lane);
    barrier();
    uint mask=taskMask;
    if ((mask&(1u<<lane))!=0u) payload.groups[uint(bitCount(mask&((1u<<lane)-1u)))]=candidate;
    barrier();
    EmitMeshTasksEXT(uint(bitCount(mask)),1u,1u);
}
#else
bool clippedAway(vec4 a,vec4 b,vec4 c)
{
    if (any(isnan(a)) || any(isinf(a)) || any(isnan(b)) || any(isinf(b)) || any(isnan(c)) || any(isinf(c))) return true;
    return (a.x < -a.w && b.x < -b.w && c.x < -c.w) || (a.x>a.w && b.x>b.w && c.x>c.w) ||
        (a.y < -a.w && b.y < -b.w && c.y < -c.w) || (a.y>a.w && b.y>b.w && c.y>c.w) ||
        (a.z<0.0 && b.z<0.0 && c.z<0.0) || (a.z>a.w && b.z>b.w && c.z>c.w);
}
// Position equality is checked after hashing; collisions cannot weld geometry.
// Canonical earliest corners and three lane masks give deterministic dense
// indices, independently of subgroup width. Shared transforms retain UV seams
// because all remaining attributes belong to each original primitive.
const uint HashCapacity=256u,MissingVertex=0xffffffffu,VertexMaskWords=3u,ByteBits=8u;
const uint WordHashBasis=2166136261u,WordHashPrime=16777619u;
shared vec4 inputPositions[96],projectedPositions[96];
shared uint canonicalOwners[256],vertexMasks[3],survivorMask;
#if defined(EVANS_MESH_INTERPOLATED_COVERAGE)
shared vec4 coverageAttributes[96];
#endif
uint hashPosition(uint source)
{
    vec4 point=inputPositions[source];
    uint hash=WordHashBasis;
    for (uint component=0u;component<4u;++component)
        hash=(hash^floatBitsToUint(point[component]==0.0?0.0:point[component]))*WordHashPrime;
#if defined(EVANS_MESH_INTERPOLATED_COVERAGE)
    for (uint component=0u;component<4u;++component)
        hash=(hash^floatBitsToUint(coverageAttributes[source][component]==0.0?0.0:coverageAttributes[source][component]))*WordHashPrime;
#endif
    return (hash^(hash>>ByteBits)^(hash>>(ByteBits*2u))^(hash>>(ByteBits*3u)))&(HashCapacity-1u);
}
uint insertVertex(uint candidate)
{
    uint slot=hashPosition(candidate);
    for (uint probe=0u;probe<HashCapacity;++probe)
    {
        uint previous=atomicCompSwap(canonicalOwners[slot],MissingVertex,candidate);
        if (previous==MissingVertex) return slot;
        bool same=all(equal(inputPositions[previous],inputPositions[candidate]));
#if defined(EVANS_MESH_INTERPOLATED_COVERAGE)
        same=same && all(equal(coverageAttributes[previous],coverageAttributes[candidate]));
#endif
        if (same) { atomicMin(canonicalOwners[slot],candidate); return slot; }
        slot=(slot+1u)&(HashCapacity-1u);
    }
    return MissingVertex; // Capacity exceeds the maximum 96 distinct corners.
}
uint denseVertex(uint source)
{
    uint word=source/MeshWidth,bit=source%MeshWidth;
    uint rank=uint(bitCount(vertexMasks[word]&((1u<<bit)-1u)));
    for (uint before=0u;before<VertexMaskWords;++before) if (before<word) rank+=uint(bitCount(vertexMasks[before]));
    return rank;
}
uvec3 indexPositions(uint lane,bool valid)
{
    for (uint slot=lane;slot<HashCapacity;slot+=MeshWidth) canonicalOwners[slot]=MissingVertex;
    if (lane<VertexMaskWords) vertexMasks[lane]=0u;
    if (lane==0u) survivorMask=0u;
    barrier();
    uvec3 slots=uvec3(0u),canonical=uvec3(0u);
    if (valid) for (uint corner=0u;corner<MeshCorners;++corner) slots[corner]=insertVertex(lane*MeshCorners+corner);
    barrier();
    if (valid) for (uint corner=0u;corner<MeshCorners;++corner)
    {
        uint source=lane*MeshCorners+corner; canonical[corner]=canonicalOwners[slots[corner]];
        if (canonical[corner]==source) atomicOr(vertexMasks[source/MeshWidth],1u<<(source%MeshWidth));
    }
    return canonical;
}
void main()
{
    uint lane=gl_LocalInvocationIndex,meshGroup=gl_WorkGroupID.y*MeshColumns+gl_WorkGroupID.x;
#if defined(EVANS_MESH_TASK_INPUT)
    meshGroup=payload.groups[gl_WorkGroupID.x];
#elif !defined(EVANS_MESH_NANITE)
    if (lane==0u) survivorMask=meshGroupVisible(meshGroup)?1u:0u;
    barrier();
    if (survivorMask==0u) { SetMeshOutputsEXT(0u,0u); return; }
#endif
    uint local=meshGroup*MeshWidth+lane;
    uint id=0u,priority=0u;
#if defined(EVANS_MESH_NANITE)
    uint count=arguments[schedule.x*HardwareWords+TriangleCountWord]; bool valid=local<count;
    if (valid)
    {
        priority=queue[queue[schedule.x]+local]; valid=priority<schedule.z;
        if (valid) { id=queue[schedule.y+priority]; valid=id<counts.z; }
    }
#else
    id=draw.y+local; bool valid=local<draw.z && id<draw.x;
#endif
    for (uint corner=0u;corner<MeshCorners;++corner)
    {
        vec4 p=valid?cornerPosition(id,corner):vec4(0.0);
#if defined(EVANS_MESH_NANITE)
        p=snapCoverage(p);
#endif
        inputPositions[lane*MeshCorners+corner]=p;
#if defined(EVANS_MESH_INTERPOLATED_COVERAGE)
        vec4 attributes=vec4(0.0);
        if (valid && material.z>0.0)
        {
            WorldTriangle item=geometry[min(id,max(draw.x,1u)-1u)]; vec2 uv=corner==0u?item.uvAB.xy:corner==1u?item.uvAB.zw:item.uvC.xy;
            attributes=vec4(uv.x,1.0-uv.y,item.color[corner].a,0.0);
        }
        coverageAttributes[lane*MeshCorners+corner]=attributes;
#endif
        valid=valid && !any(isnan(p)) && !any(isinf(p));
    }
    uvec3 canonical=indexPositions(lane,valid);
    if (valid) for (uint corner=0u;corner<MeshCorners;++corner)
    {
        uint source=lane*MeshCorners+corner;
        if (canonical[corner]==source)
        {
#if defined(EVANS_MESH_NANITE)
            projectedPositions[source]=inputPositions[source];
#else
            precise vec4 projected=camera*inputPositions[source]; projectedPositions[source]=projected;
#endif
        }
    }
    barrier();
    vec4 positions[3]; bool alive=valid;
    if (valid)
    {
        for (uint corner=0u;corner<MeshCorners;++corner) positions[corner]=projectedPositions[canonical[corner]];
        alive=!clippedAway(positions[0],positions[1],positions[2]);
    }
    if (alive) atomicOr(survivorMask,1u<<lane);
    barrier();
    uint total=uint(bitCount(survivorMask)),vertices=0u;
    for (uint word=0u;word<VertexMaskWords;++word) vertices+=uint(bitCount(vertexMasks[word]));
    SetMeshOutputsEXT(total!=0u?vertices:0u,total);
    if (total==0u) return;
    if (valid) for (uint corner=0u;corner<MeshCorners;++corner)
    {
        uint source=lane*MeshCorners+corner;
        if (canonical[corner]==source)
        {
            vec4 p=projectedPositions[source]; if (any(isnan(p)) || any(isinf(p))) p=vec4(0,0,0,1);
            uint vertex=denseVertex(source); p.y=-p.y; gl_MeshVerticesEXT[vertex].gl_Position=p;
#if defined(EVANS_MESH_INTERPOLATED_COVERAGE)
            coverageUV[vertex]=coverageAttributes[source].xy; coverageAlpha[vertex]=coverageAttributes[source].z;
#endif
        }
    }
    if (alive)
    {
        uint primitive=uint(bitCount(survivorMask&((1u<<lane)-1u)));
        gl_PrimitiveTriangleIndicesEXT[primitive]=uvec3(denseVertex(canonical[0]),denseVertex(canonical[1]),denseVertex(canonical[2]));
        references[primitive]=id+1u;
#if defined(EVANS_MESH_NANITE)
        priorities[primitive]=priority;
#else
#if !defined(EVANS_MESH_NATIVE_BARYCENTRICS)
        // Adjugate XYW rows give unnormalized projective barycentrics. Common
        // input scaling avoids overflow; determinant division cancels when normalizing weights.
        vec3 a=positions[0].xyw,b=positions[1].xyw,c=positions[2].xyw;
        vec3 bound=max(abs(a),max(abs(b),abs(c))); float scale=max(bound.x,max(bound.y,bound.z));
        if (scale>0.0) { a/=scale; b/=scale; c/=scale; }
        vec3 row0=cross(b,c),row1=cross(c,a),row2=cross(a,b);
        planeX[primitive]=vec3(row0.x,row1.x,row2.x); planeY[primitive]=vec3(row0.y,row1.y,row2.y);
        planeOrigin[primitive]=vec3(row0.z,row1.z,row2.z);
#endif
#endif
    }
}
#endif
#endif
