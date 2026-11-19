EVANS_GPU_CODE(
struct ProjectedVertex { float4 Screen; float4 Weights; };
struct ClipPoint { float4 Position; float4 Weights; };
R_FN float ClipDistance(float4 clip, uint plane, uint depthRange) {
    if (plane == 0) return clip.x + clip.w;
    if (plane == 1) return clip.w - clip.x;
    if (plane == 2) return clip.y + clip.w;
    if (plane == 3) return clip.w - clip.y;
    if (plane == 4) return depthRange == 0 ? clip.z : clip.z + clip.w;
    return clip.w - clip.z;
}
R_FN uint MaskWords(uint count) { return (count + 31u) >> 5; }
R_FN uint VertexSlot(uint id, uint corner, uint count) {
    return corner < 3 ? id * 3 + corner : count * corner + id;
}
R_FN uint TakeBit(R_INOUT(uint) bits) {
    uint bit = R_LOWBIT(bits); bits &= bits - 1u; return bit;
}
struct BinCursor { uint Base, FineWords, MidWords, TopWords, NextTop, MidIndex, FineIndex, Top, Mid, Fine; };
R_FN BinCursor OpenBin(uint tile, uint count) {
    BinCursor cursor;
    cursor.FineWords = MaskWords(count); cursor.MidWords = MaskWords(cursor.FineWords);
    cursor.TopWords = MaskWords(cursor.MidWords);
    cursor.Base = tile * (cursor.FineWords + cursor.MidWords + cursor.TopWords);
    cursor.NextTop = 0; cursor.Top = 0; cursor.Mid = 0; cursor.Fine = 0;
    cursor.MidIndex = 0; cursor.FineIndex = 0; return cursor;
}
R_FN bool NextCandidate(R_INOUT(BinCursor) cursor, R_READ(uint) bins, R_OUT(uint) id) {
    id = 0;
    while (cursor.Fine == 0) {
        while (cursor.Mid == 0) {
            while (cursor.Top == 0) {
                if (cursor.NextTop == cursor.TopWords) return false;
                cursor.Top = bins[cursor.Base + cursor.FineWords + cursor.MidWords + cursor.NextTop++];
            }
            cursor.MidIndex = (cursor.NextTop - 1) * 32 + TakeBit(cursor.Top);
            cursor.Mid = bins[cursor.Base + cursor.FineWords + cursor.MidIndex];
        }
        cursor.FineIndex = cursor.MidIndex * 32 + TakeBit(cursor.Mid);
        cursor.Fine = bins[cursor.Base + cursor.FineIndex];
    }
    id = cursor.FineIndex * 32 + TakeBit(cursor.Fine); return true;
}
)
EVANS_GPU_CODE(
R_FN void ProjectAndBin(uint id, uint count, uint width, uint height, uint tileSize,
    uint depthRange, bool shadow, bool collectCounts, bool solid, R_READ(R_TRI) inputTriangles, R_WRITE(ProjectedVertex) projected,
    R_WRITE(uint) bins) {
    if (id >= count) return;
    ClipPoint polygon[10], scratch[10];
    uint unionMask = 0, sharedMask = 63;
    for (uint corner = 0; corner < 3; ++corner) {
        Vertex vert = inputTriangles[id].Vertices[corner];
        float4 clip = shadow ? vert.ShadowClip : vert.Clip;
        polygon[corner].Position = clip;
        polygon[corner].Weights = R_FLOAT4(corner==0?1.0f:0.0f,corner==1?1.0f:0.0f,corner==2?1.0f:0.0f,0.0f);
        uint mask = 0;
        for (uint plane = 0; plane < 6; ++plane) if (!(ClipDistance(clip,plane,depthRange)>=0)) mask |= 1u << plane;
        unionMask |= mask; sharedMask &= mask;
    }
    if (sharedMask != 0) return;
    uint vertices = 3;
    for (uint plane = 0; plane < 6 && vertices >= 3; ++plane) {
        if ((unionMask & (1u << plane)) == 0) continue;
        ClipPoint previous = polygon[vertices-1];
        float previousDistance = ClipDistance(previous.Position,plane,depthRange);
        uint written = 0;
        for (uint corner = 0; corner < vertices; ++corner) {
            ClipPoint current = polygon[corner];
            float distance = ClipDistance(current.Position,plane,depthRange);
            if ((previousDistance < 0) != (distance < 0)) {
                float difference = previousDistance-distance;
                float amount = R_ABS(difference)>1.0e-6f ? previousDistance/difference : 0.0f;
                scratch[written].Position = previous.Position*(1.0f-amount)+current.Position*amount;
                scratch[written++].Weights = previous.Weights*(1.0f-amount)+current.Weights*amount;
            }
            if (distance >= 0) scratch[written++] = current;
            previous=current; previousDistance=distance;
        }
        vertices=written;
        for (uint corner=0; corner<vertices; ++corner) polygon[corner]=scratch[corner];
    }
    if (vertices < 3) return;
    float minX=(float)width, minY=(float)height, maxX=-1.0f, maxY=-1.0f;
    for (uint corner=0; corner<vertices; ++corner) {
        float4 clip=polygon[corner].Position;
        ProjectedVertex outVertex;
        float invW=clip.w>1.0e-6f ? 1.0f/clip.w : 0.0f;
        float x=(clip.x*invW*0.5f+0.5f)*width, y=(0.5f-clip.y*invW*0.5f)*height;
        float depth=clip.z*invW;
        outVertex.Screen=R_FLOAT4(x,y,depthRange==0?depth:depth*0.5f+0.5f,invW);
        outVertex.Weights=polygon[corner].Weights;
        outVertex.Weights.w=(float)(vertices+(unionMask!=0?16:0));
        projected[VertexSlot(id,corner,count)]=outVertex;
        polygon[corner].Position=outVertex.Screen;
        if (invW>0) { minX=R_MIN(minX,x); minY=R_MIN(minY,y); maxX=R_MAX(maxX,x); maxY=R_MAX(maxY,y); }
    }
    uint tilesX=(width+tileSize-1)/tileSize;
    if(collectCounts && !shadow) {
        BinCursor end=OpenBin(tilesX*((height+tileSize-1)/tileSize),count);
        R_ADD(bins[end.Base],vertices-2u);
    }
    if (maxX<0 || maxY<0) return;
    uint firstX=(uint)R_MAX(0.0f,floor(minX))/tileSize, firstY=(uint)R_MAX(0.0f,floor(minY))/tileSize;
    uint lastX=(uint)R_MIN((float)(width-1),ceil(maxX))/tileSize, lastY=(uint)R_MIN((float)(height-1),ceil(maxY))/tileSize;
    bool trim=solid && vertices==3 && firstX<lastX && firstY<lastY &&
        polygon[0].Position.w>0 && polygon[1].Position.w>0 && polygon[2].Position.w>0;
    float4 planes[3];
    if(trim) {
        float4 first=polygon[0].Position, second=polygon[1].Position, third=polygon[2].Position;
        float area=(second.x-first.x)*(third.y-first.y)-(second.y-first.y)*(third.x-first.x);
        float winding=area>=0?1.0f:-1.0f;
        for(uint edgeIndex=0;edgeIndex<3;++edgeIndex) {
            float4 start=polygon[edgeIndex].Position, finish=polygon[(edgeIndex+1)%3].Position;
            planes[edgeIndex]=R_FLOAT4(start.x,start.y,(start.y-finish.y)*winding,(finish.x-start.x)*winding);
        }
    }
    uint fine=id>>5, mid=fine>>5, top=mid>>5;
    for (uint tileY=firstY;tileY<=lastY;++tileY) for (uint tileX=firstX;tileX<=lastX;++tileX) {
        if(trim) {
            bool outside=false;
            for(uint edgeIndex=0;edgeIndex<3;++edgeIndex) {
                float4 plane=planes[edgeIndex];
                float farX=(float)(tileX*tileSize)+(plane.z>=0?(float)tileSize+1.0f:-1.0f);
                float farY=(float)(tileY*tileSize)+(plane.w>=0?(float)tileSize+1.0f:-1.0f);
                if(plane.z*(farX-plane.x)+plane.w*(farY-plane.y)<0)outside=true;
            }
            if(outside)continue;
        }
        BinCursor layout=OpenBin(tileY*tilesX+tileX,count);
        R_OR(bins[layout.Base+fine],1u<<(id&31));
        R_OR(bins[layout.Base+layout.FineWords+mid],1u<<(fine&31));
        R_OR(bins[layout.Base+layout.FineWords+layout.MidWords+top],1u<<(mid&31));
    }
}
)
EVANS_GPU_CODE(
R_FN Vertex BlendVertex(Vertex first,Vertex second,Vertex third,float4 weight) {
    Vertex result;
    result.Clip=first.Clip*weight.x+second.Clip*weight.y+third.Clip*weight.z;
    result.Color=first.Color*weight.x+second.Color*weight.y+third.Color*weight.z;
    result.Position=first.Position*weight.x+second.Position*weight.y+third.Position*weight.z;
    result.Normal=first.Normal*weight.x+second.Normal*weight.y+third.Normal*weight.z;
    result.ShadowClip=first.ShadowClip*weight.x+second.ShadowClip*weight.y+third.ShadowClip*weight.z;
    result.TexCoord=first.TexCoord*weight.x+second.TexCoord*weight.y+third.TexCoord*weight.z;
    return result;
}
R_FN R_TRI FanAttributes(R_TRI source, ProjectedVertex first, ProjectedVertex second, ProjectedVertex third) {
    if (first.Weights.w<16.0f) return source;
    R_TRI result=source;
    result.Vertices[0]=BlendVertex(source.Vertices[0],source.Vertices[1],source.Vertices[2],first.Weights);
    result.Vertices[1]=BlendVertex(source.Vertices[0],source.Vertices[1],source.Vertices[2],second.Weights);
    result.Vertices[2]=BlendVertex(source.Vertices[0],source.Vertices[1],source.Vertices[2],third.Weights);
    return result;
}
)
EVANS_GPU_CODE(
R_FN bool IsOutlinePixel(uint x, uint y, uint width, uint height, int radius, R_READ(float) visibility) {
    float center = visibility[y * width + x];
    if (center < 0) return false;
    float limit = center + 0.002f;
    for (int dy = -radius; dy <= radius; ++dy) {
        int row = (int)y + dy;
        if (row < 0 || row >= (int)height) continue;
        for (int dx = -radius; dx <= radius; ++dx) {
            int col = (int)x + dx;
            if (col < 0 || col >= (int)width ||
                (float)dx*dx + (float)dy*dy > (float)radius*radius) continue;
            float neighbor = visibility[(uint)row * width + (uint)col];
            if (neighbor < 0 && -neighbor - 1.0f <= limit) return true;
        }
    }
    return false;
}
)
