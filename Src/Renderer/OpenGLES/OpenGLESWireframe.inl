// Preserve native line coverage while classifying each original triangle.
GLuint WireVertexBuffer=0,WireExpandProgram=0,WireExpandUniform=0;
using WireRangeFunction=void(EVANS_GL_CALL*)(GLenum,GLuint,GLuint,GLintptr,GLsizeiptr);
using WireInteger64Function=void(EVANS_GL_CALL*)(GLenum,GLuint,std::int64_t*);
WireRangeFunction WireBindRange=nullptr;
WireInteger64Function WireGetInteger64=nullptr;
PFNGLENABLEVERTEXATTRIBARRAY WireDisableAttribute=nullptr;
struct WireVertex { Vertex Surface; Float4 Plane; };
static_assert(sizeof(WireVertex)==64,"Wireframe storage matches sixteen packed words");

void DestroyWireframe()
{
    if (WireVertexBuffer) GlDeleteBuffers(1,&WireVertexBuffer);
    if (WireExpandUniform) GlDeleteBuffers(1,&WireExpandUniform);
    if (WireExpandProgram) GlDeleteProgram(WireExpandProgram);
    WireVertexBuffer=WireExpandUniform=WireExpandProgram=0;
}
bool ReadWireframeSource(const BufferResource& resource,std::vector<std::uint8_t>& readback,const std::uint8_t*& data)
{
    const auto bytes=resource.Description.SizeInBytes;
    if (resource.Bytes.size()==bytes) { data=resource.Bytes.data(); return true; }
    // GPU-only async geometry has no host mirror. Mapping is confined to this
    // capacity fallback; normal-sized geometry stays entirely on the device.
    using MapFunction=void*(EVANS_GL_CALL*)(GLenum,GLintptr,GLsizeiptr,GLbitfield);
    using UnmapFunction=GLboolean(EVANS_GL_CALL*)(GLenum);
    MapFunction map=nullptr; UnmapFunction unmap=nullptr;
    if (!bytes || !LoadGL(map,"glMapBufferRange") || !LoadGL(unmap,"glUnmapBuffer"))
        return Fail("Wireframe source readback is unavailable");
    GLint previous=0; GlGetIntegerv(0x8894,&previous);
    readback.resize(bytes); GlBindBuffer(GL_ARRAY_BUFFER,resource.Buffer);
    const auto* mapped=map(GL_ARRAY_BUFFER,0,GLsizeiptr(bytes),0x00000001);
    if (!mapped) { GlBindBuffer(GL_ARRAY_BUFFER,GLuint(previous)); return Fail("Map wireframe source failed"); }
    std::memcpy(readback.data(),mapped,bytes);
    const bool valid=unmap(GL_ARRAY_BUFFER)!=0;
    GlBindBuffer(GL_ARRAY_BUFFER,GLuint(previous));
    if (!valid || !CheckGL("Read wireframe source")) return Fail("Wireframe source mapping was invalidated");
    data=readback.data(); return true;
}

bool PrepareWireframe(const BufferResource& vertices,const BufferResource& indices,unsigned first,unsigned count,int base)
{
    if (!WireDisableAttribute && !LoadGL(WireDisableAttribute,"glDisableVertexAttribArray")) return false;
    if (!WireVertexBuffer) GlGenBuffers(1,&WireVertexBuffer);
    const auto vertexBytes=std::uint64_t(count)*sizeof(WireVertex);
    const auto indexBytes=std::uint64_t(count)*2*sizeof(unsigned);
    if (count>unsigned(std::numeric_limits<GLsizei>::max()/2) || count%3 ||
        std::uint64_t(first)+count>indices.Description.SizeInBytes/sizeof(unsigned))
        return Fail("Wireframe expansion requires a bounded triangle range");
    GLint maximum=0,maxGroups=0;
    if (GpuDrivenSupported)
    {
        GlGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE,&maximum);
        GpuGetIntegerIndexed(0x91BE,0,&maxGroups);
    }
    // Oversized storage ranges use host expansion; they remain valid draw buffers.
    const bool useCompute=GpuDrivenSupported && maximum>0 && maxGroups>0 &&
        vertexBytes<=std::uint64_t(maximum) && indexBytes<=std::uint64_t(maximum) &&
        vertices.Description.SizeInBytes<=std::uint64_t(maximum) &&
        indices.Description.SizeInBytes<=std::uint64_t(maximum) &&
        (std::uint64_t(count)+63)/64<=std::uint64_t(maxGroups);
    if (useCompute)
    {
        if (!WireBindRange && (!LoadGL(WireBindRange,"glBindBufferRange") || !LoadGL(WireGetInteger64,"glGetInteger64i_v"))) return false;
        if (!WireExpandProgram)
        {
            std::string source;
            if (!LoadTechniqueShaderSource("OpenGLES","Wireframe.glsl",source,LastError)) return false;
            const auto program=CompileComputeProgram(source); if(!program)return false;
            WireExpandProgram=program;
        }
        GLint savedProgram=0,savedUniform=0,savedStorage=0,uniformSlot=0;
        std::array<GLint,4> storage{}; std::array<std::int64_t,4> starts{},sizes{};
        std::int64_t uniformStart=0,uniformSize=0;
        GlGetIntegerv(GpuCurrentProgram,&savedProgram); GlGetIntegerv(GpuUniformBinding,&savedUniform); GlGetIntegerv(GpuStorageBinding,&savedStorage);
        GpuGetIntegerIndexed(GpuUniformBinding,0,&uniformSlot); WireGetInteger64(0x8A29,0,&uniformStart); WireGetInteger64(0x8A2A,0,&uniformSize);
        for (GLuint slot=0;slot<4;++slot)
        { GpuGetIntegerIndexed(GpuStorageBinding,slot,&storage[slot]); WireGetInteger64(0x90D4,slot,&starts[slot]); WireGetInteger64(0x90D5,slot,&sizes[slot]); }
        const auto restore=[&]()
        {
            for (GLuint slot=0;slot<4;++slot)
                if (storage[slot] && sizes[slot]>0) WireBindRange(GL_SHADER_STORAGE_BUFFER,slot,GLuint(storage[slot]),GLintptr(starts[slot]),GLsizeiptr(sizes[slot]));
                else GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,slot,GLuint(storage[slot]));
            if (uniformSlot && uniformSize>0) WireBindRange(GL_UNIFORM_BUFFER,0,GLuint(uniformSlot),GLintptr(uniformStart),GLsizeiptr(uniformSize));
            else GlBindBufferBase(GL_UNIFORM_BUFFER,0,GLuint(uniformSlot));
            GlBindBuffer(GL_UNIFORM_BUFFER,GLuint(savedUniform)); GlBindBuffer(GL_SHADER_STORAGE_BUFFER,GLuint(savedStorage)); GlUseProgram(GLuint(savedProgram));
        };
        struct Restore { const decltype(restore)& Function; ~Restore() { Function(); } } cleanup{restore};
        if (!WireExpandUniform) GlGenBuffers(1,&WireExpandUniform);
        const ShaderUInt4 range{first,count,unsigned(base),0};
        GlBindBuffer(GL_UNIFORM_BUFFER,WireExpandUniform); GlBufferData(GL_UNIFORM_BUFFER,sizeof(range),&range,GL_DYNAMIC_DRAW);
        GlBindBufferBase(GL_UNIFORM_BUFFER,0,WireExpandUniform);
        GlBindBuffer(GL_SHADER_STORAGE_BUFFER,WireVertexBuffer); GlBufferData(GL_SHADER_STORAGE_BUFFER,GLsizeiptr(vertexBytes),nullptr,GL_DYNAMIC_DRAW);
        GlBindBuffer(GL_SHADER_STORAGE_BUFFER,ScratchIndexBuffer); GlBufferData(GL_SHADER_STORAGE_BUFFER,GLsizeiptr(indexBytes),nullptr,GL_DYNAMIC_DRAW);
        const GLuint buffers[]{indices.Buffer,vertices.Buffer,WireVertexBuffer,ScratchIndexBuffer};
        for (GLuint slot=0;slot<4;++slot) GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,slot,buffers[slot]);
        GlUseProgram(WireExpandProgram); GpuDispatch((count+63)/64,1,1);
        GpuMemoryBarrier(GpuStorageBarrier|0x00000001|0x00000002);
    }
    else
    {
        std::vector<std::uint8_t> vertexReadback,indexReadback;
        const std::uint8_t* vertexData=nullptr; const std::uint8_t* indexData=nullptr;
        if (!ReadWireframeSource(vertices,vertexReadback,vertexData) ||
            !ReadWireframeSource(indices,indexReadback,indexData)) return false;
        std::vector<WireVertex> expanded(count); std::vector<unsigned> sequential(std::size_t(count)*2);
        const auto* source=reinterpret_cast<const Vertex*>(vertexData);
        const auto* index=reinterpret_cast<const unsigned*>(indexData);
        for (unsigned triangle=0;triangle<count;triangle+=3)
        {
            std::array<std::size_t,3> addresses{};
            Float3 position[3];
            for (unsigned corner=0;corner<3;++corner)
            {
                const auto address=std::int64_t(index[first+triangle+corner])+base;
                if (address<0 || std::uint64_t(address)>=vertices.Description.SizeInBytes/sizeof(Vertex))
                    return Fail("Wireframe face vertex is out of range");
                addresses[corner]=std::size_t(address); position[corner]=source[address].Position;
            }
            const Float3 a{position[1].X-position[0].X,position[1].Y-position[0].Y,position[1].Z-position[0].Z};
            const Float3 b{position[2].X-position[0].X,position[2].Y-position[0].Y,position[2].Z-position[0].Z};
            const Float3 normal{a.Y*b.Z-a.Z*b.Y,a.Z*b.X-a.X*b.Z,a.X*b.Y-a.Y*b.X};
            const Float4 plane{normal.X,normal.Y,normal.Z,-(normal.X*position[0].X+normal.Y*position[0].Y+normal.Z*position[0].Z)};
            for (unsigned corner=0;corner<3;++corner)
            {
                const auto item=triangle+corner;
                expanded[item]={source[addresses[corner]],plane};
                sequential[item*2]=item; sequential[item*2+1]=triangle+(corner+1)%3;
            }
        }
        GlBindBuffer(GL_ARRAY_BUFFER,WireVertexBuffer); GlBufferData(GL_ARRAY_BUFFER,GLsizeiptr(vertexBytes),expanded.data(),GL_DYNAMIC_DRAW);
        GlBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ScratchIndexBuffer); GlBufferData(GL_ELEMENT_ARRAY_BUFFER,GLsizeiptr(indexBytes),sequential.data(),GL_DYNAMIC_DRAW);
    }
    GlBindBuffer(GL_ARRAY_BUFFER,WireVertexBuffer);
    GlVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE_VALUE,sizeof(WireVertex),reinterpret_cast<const void*>(offsetof(Vertex,Position)));
    GlVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE_VALUE,sizeof(WireVertex),reinterpret_cast<const void*>(offsetof(Vertex,Normal)));
    GlVertexAttribPointer(2,4,GL_FLOAT,GL_FALSE_VALUE,sizeof(WireVertex),reinterpret_cast<const void*>(offsetof(Vertex,Color)));
    GlVertexAttribPointer(3,2,GL_FLOAT,GL_FALSE_VALUE,sizeof(WireVertex),reinterpret_cast<const void*>(offsetof(Vertex,TexCoord)));
    GlEnableVertexAttribArray(15); GlVertexAttribPointer(15,4,GL_FLOAT,GL_FALSE_VALUE,sizeof(WireVertex),reinterpret_cast<const void*>(offsetof(WireVertex,Plane)));
    GlBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ScratchIndexBuffer);
    return CheckGL("Prepare native wireframe face planes");
}

bool BindWireFace(bool clockwise,bool frontCull)
{
    const GLint cullLocation=GlGetUniformLocation(Program,"WireFaceCull");
    if (cullLocation<0) return Fail("Indexed wireframe shader lacks original-face culling");
    // Cofactors of the clip X/Y/W rows give the oriented camera in object space.
    // Their dot product with a triangle plane equals its homogeneous screen area.
    double matrix[3][4]{};
    const unsigned rows[]{0,1,3};
    for (unsigned row=0;row<3;++row) for (unsigned column=0;column<4;++column)
        for (unsigned inner=0;inner<4;++inner)
            matrix[row][column]+=double(BoundViewProjection.Values[rows[row]*4+inner])*BoundWorld.Values[inner*4+column];
    float camera[4]{};
    const float sign=(clockwise?-1.0f:1.0f)*(frontCull?-1.0f:1.0f);
    for (unsigned excluded=0;excluded<4;++excluded)
    {
        double minor[3][3]{};
        for (unsigned row=0;row<3;++row) for (unsigned column=0,target=0;column<4;++column)
            if (column!=excluded) minor[row][target++]=matrix[row][column];
        const double determinant=minor[0][0]*(minor[1][1]*minor[2][2]-minor[1][2]*minor[2][1])-
            minor[0][1]*(minor[1][0]*minor[2][2]-minor[1][2]*minor[2][0])+
            minor[0][2]*(minor[1][0]*minor[2][1]-minor[1][1]*minor[2][0]);
        camera[excluded]=float(determinant*(excluded%2?-1.0:1.0))*sign;
    }
    GlUniform4f(GlGetUniformLocation(Program,"WireFaceCamera"),camera[0],camera[1],camera[2],camera[3]);
    GlUniform1i(GlGetUniformLocation(Program,"WireFaceSign"),sign<0?-1:1);
    GlUniform1i(cullLocation,1);
    return true;
}
