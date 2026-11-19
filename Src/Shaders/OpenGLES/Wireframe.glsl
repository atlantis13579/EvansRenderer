#version 310 es
precision highp int;
precision highp float;
layout(local_size_x=64) in;
layout(std140,binding=0) uniform WireParameters { uvec4 Range; };
layout(std430,binding=0) readonly buffer Indices { uint sourceIndices[]; };
layout(std430,binding=1) readonly buffer Vertices { uint sourceVertices[]; };
layout(std430,binding=2) writeonly buffer Expanded { uint expandedVertices[]; };
layout(std430,binding=3) writeonly buffer Sequential { uint expandedIndices[]; };
void main()
{
    uint item=gl_GlobalInvocationID.x;
    if (item>=Range.y) return;
    uint triangle=item-item%3u;
    uint addresses[3]; vec3 positions[3];
    for (uint corner=0u;corner<3u;++corner)
    {
        addresses[corner]=(sourceIndices[Range.x+triangle+corner]+Range.z)*12u;
        uint address=addresses[corner];
        positions[corner]=uintBitsToFloat(uvec3(sourceVertices[address],sourceVertices[address+1u],sourceVertices[address+2u]));
    }
    vec3 normal=cross(positions[1]-positions[0],positions[2]-positions[0]);
    vec4 plane=vec4(normal,-dot(normal,positions[0]));
    uint source=addresses[item%3u],target=item*16u;
    for (uint word=0u;word<12u;++word) expandedVertices[target+word]=sourceVertices[source+word];
    for (uint word=0u;word<4u;++word) expandedVertices[target+12u+word]=floatBitsToUint(plane[word]);
    expandedIndices[item*2u]=item;
    expandedIndices[item*2u+1u]=triangle+(item+1u)%3u;
}
