// D3D12 ExecuteIndirect reads DRAW_INDEXED arguments generated from the native
// main/post decisions. Explicit bounds control is required for root SRVs.
cbuffer NaniteCommandParameters : register(b0)
{
    uint4 Counts;
    uint4 Layout;
    uint4 Source;
    uint4 Target;
};
ByteAddressBuffer ClusterRanges : register(t0);
ByteAddressBuffer VisibilityStates : register(t1);
RWByteAddressBuffer DrawCommands : register(u0);
static const uint ThreadsPerGroup=64,BytesPerWord=4;
static const uint RangeWordCount=2,CommandWordCount=5;
static const uint MainVisibleState=2,PostVisibleState=4;
[numthreads(ThreadsPerGroup,1,1)]
void CSBuildDraws(uint3 dispatchGroup:SV_GroupID,uint localIndex:SV_GroupIndex)
{
    uint candidate=(dispatchGroup.y*Target.w+dispatchGroup.x)*ThreadsPerGroup+localIndex;
    [branch] if (candidate>=Counts.x) return;
    uint inputAddress=candidate*RangeWordCount*BytesPerWord;
    uint firstIndex=ClusterRanges.Load(inputAddress);
    uint indexCount=ClusterRanges.Load(inputAddress+BytesPerWord);
    uint desiredState=Counts.w==0?MainVisibleState:PostVisibleState;
    uint instanceCount=VisibilityStates.Load(candidate*BytesPerWord)==desiredState?1u:0u;
    uint commandAddress=candidate*CommandWordCount*BytesPerWord;
    DrawCommands.Store(commandAddress,indexCount);
    DrawCommands.Store(commandAddress+BytesPerWord,instanceCount);
    DrawCommands.Store(commandAddress+2*BytesPerWord,firstIndex);
    DrawCommands.Store(commandAddress+3*BytesPerWord,0); // Absolute decoded vertex indices.
    DrawCommands.Store(commandAddress+4*BytesPerWord,0);
}

ByteAddressBuffer OriginalIndices : register(t2);
static const uint CandidateStride=4,BlockStride=4,MaterialStride=8;
groupshared uint ScanValues[ThreadsPerGroup];
void InclusivePrefix(uint lane,uint value)
{
    ScanValues[lane]=value; GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint distance=1;distance<ThreadsPerGroup;distance*=2)
    {
        uint left=0; [branch] if (lane>=distance) left=ScanValues[lane-distance];
        GroupMemoryBarrierWithGroupSync(); ScanValues[lane]+=left; GroupMemoryBarrierWithGroupSync();
    }
}
// Separate dispatches order local prefixes, bin prefixes and index scattering.
// Root descriptors do not bound loads: guard every partial group explicitly.
[numthreads(ThreadsPerGroup,1,1)]
void CSCountBins(uint3 dispatchGroup:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint blockIndex=dispatchGroup.y*Target.w+dispatchGroup.x;
    [branch] if (blockIndex>=Target.z) return;
    uint4 block=ClusterRanges.Load4((Layout.y+blockIndex*BlockStride)*BytesPerWord);
    uint query=block.x+lane,indexCount=0;
    [branch] if (lane<block.y)
    {
        uint state=VisibilityStates.Load(query*BytesPerWord);
        [branch] if (state==(Counts.w==0?MainVisibleState:PostVisibleState))
            indexCount=ClusterRanges.Load((Layout.x+query*CandidateStride+1)*BytesPerWord);
    }
    InclusivePrefix(lane,indexCount);
    if (lane<block.y) DrawCommands.Store((Source.x+query)*BytesPerWord,ScanValues[lane]-indexCount);
    if (lane+1==ThreadsPerGroup) DrawCommands.Store((Source.y+blockIndex)*BytesPerWord,ScanValues[lane]);
}
[numthreads(ThreadsPerGroup,1,1)]
void CSScanBins(uint3 dispatchGroup:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint material=dispatchGroup.y*Target.w+dispatchGroup.x;
    [branch] if (material>=Target.z) return;
    uint4 bin=ClusterRanges.Load4((Layout.z+material*MaterialStride)*BytesPerWord);
    uint indexStart=Target.x+ClusterRanges.Load((Layout.z+material*MaterialStride+4)*BytesPerWord);
    uint sum=0;
    for (uint tile=0;tile<bin.w;tile+=ThreadsPerGroup)
    {
        uint block=tile+lane,value=0;
        [branch] if (block<bin.w) value=DrawCommands.Load((Source.y+bin.z+block)*BytesPerWord);
        InclusivePrefix(lane,value);
        if (block<bin.w) DrawCommands.Store((Source.z+bin.z+block)*BytesPerWord,sum+ScanValues[lane]-value);
        uint tileTotal=ScanValues[ThreadsPerGroup-1]; GroupMemoryBarrierWithGroupSync(); sum+=tileTotal;
    }
    if (lane==0)
    {
        uint address=(Source.w+material*CommandWordCount)*BytesPerWord;
        DrawCommands.Store4(address,uint4(sum,sum>0?1u:0u,indexStart,0)); DrawCommands.Store(address+4*BytesPerWord,0);
    }
}
[numthreads(ThreadsPerGroup,1,1)]
void CSScatterBins(uint3 dispatchGroup:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint query=dispatchGroup.y*Target.w+dispatchGroup.x;
    [branch] if (query>=Target.z) return;
    [branch] if (VisibilityStates.Load(query*BytesPerWord)!=(Counts.w==0?MainVisibleState:PostVisibleState)) return;
    uint4 candidate=ClusterRanges.Load4((Layout.x+query*CandidateStride)*BytesPerWord);
    uint binStart=ClusterRanges.Load((Layout.z+candidate.w*MaterialStride+4)*BytesPerWord);
    uint prefix=DrawCommands.Load((Source.x+query)*BytesPerWord)+DrawCommands.Load((Source.z+candidate.z)*BytesPerWord);
    for (uint corner=lane;corner<candidate.y;corner+=ThreadsPerGroup)
    {
        uint index=OriginalIndices.Load((candidate.x+corner)*BytesPerWord);
        DrawCommands.Store((Target.x+binStart+prefix+corner)*BytesPerWord,index);
    }
}
