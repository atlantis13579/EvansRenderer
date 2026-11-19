#include "Renderer/OpenGLES/OpenGLESProgramCache.h"
#include "Renderer/ShaderCache.h"
#include <algorithm>
#include <cstring>
#if defined(_WIN32)
#define EVANS_CACHE_GL_CALL __stdcall
#else
#define EVANS_CACHE_GL_CALL
#endif
namespace Evans::Renderer
{
    unsigned CompileCachedGLESProgram(const std::function<void*(const char*)>& resolve,
        std::initializer_list<std::pair<unsigned,std::string_view>> stages,std::string& error)
    {
#define GL_ENTRY(result,name,arguments) const auto name=reinterpret_cast<result(EVANS_CACHE_GL_CALL*)arguments>(resolve("gl" #name)); \
        if(!name){error="Missing OpenGL ES entry point: gl" #name;return 0;}
        GL_ENTRY(const unsigned char*,GetString,(unsigned))
        GL_ENTRY(void,GetIntegerv,(unsigned,int*))
        GL_ENTRY(unsigned,GetError,())
        GL_ENTRY(unsigned,CreateShader,(unsigned))
        GL_ENTRY(void,ShaderSource,(unsigned,int,const char* const*,const int*))
        GL_ENTRY(void,CompileShader,(unsigned))
        GL_ENTRY(void,GetShaderiv,(unsigned,unsigned,int*))
        GL_ENTRY(void,GetShaderInfoLog,(unsigned,int,int*,char*))
        GL_ENTRY(void,DeleteShader,(unsigned))
        GL_ENTRY(unsigned,CreateProgram,())
        GL_ENTRY(void,DeleteProgram,(unsigned))
        GL_ENTRY(void,AttachShader,(unsigned,unsigned))
        GL_ENTRY(void,LinkProgram,(unsigned))
        GL_ENTRY(void,GetProgramiv,(unsigned,unsigned,int*))
        GL_ENTRY(void,GetProgramInfoLog,(unsigned,int,int*,char*))
#undef GL_ENTRY
        const auto ProgramBinary=reinterpret_cast<void(EVANS_CACHE_GL_CALL*)(unsigned,unsigned,const void*,int)>(resolve("glProgramBinary"));
        const auto GetProgramBinary=reinterpret_cast<void(EVANS_CACHE_GL_CALL*)(unsigned,int,int*,unsigned*,void*)>(resolve("glGetProgramBinary"));
        const auto ProgramParameteri=reinterpret_cast<void(EVANS_CACHE_GL_CALL*)(unsigned,unsigned,int)>(resolve("glProgramParameteri"));
        constexpr unsigned CompileStatus=0x8b81,LinkStatus=0x8b82,InfoLogLength=0x8b84,BinaryLength=0x8741,BinaryHint=0x8257;
        int formats=0;GetIntegerv(0x87fe,&formats);
        const bool cacheable=ProgramBinary && GetProgramBinary && ProgramParameteri && formats>0 && ShaderCache::Enabled();
        ShaderCache::Key key("gles-program-v1");
        for(auto name:{0x1f00u,0x1f01u,0x1f02u,0x8b8cu})
        {const auto* text=GetString(name);key.Add(text?reinterpret_cast<const char*>(text):"");}
        std::vector<int> binaryFormats(std::max(formats,0));
        if(formats>0)GetIntegerv(0x87ff,binaryFormats.data());
        for(auto format:binaryFormats)key.Value(format);
        for(const auto& stage:stages)key.Value(stage.first).Add(stage.second);
        const auto digest=key.Finish();std::vector<std::uint8_t> data;
        unsigned program=CreateProgram();if(!program){error="Cannot create OpenGL ES program";return 0;}
        if(cacheable && ShaderCache::Read("gles-program",digest,data) && data.size()>4)
        {
            unsigned format=0;std::memcpy(&format,data.data(),4);
            if(std::find(binaryFormats.begin(),binaryFormats.end(),int(format))!=binaryFormats.end())
            {
                ProgramBinary(program,format,data.data()+4,int(data.size()-4));int linked=0;GetProgramiv(program,LinkStatus,&linked);
                const auto code=GetError();if(linked && !code)return program;
            }
            DeleteProgram(program);program=CreateProgram();
            if(!program){error="Cannot recreate OpenGL ES program";return 0;}
        }
        if(cacheable)ProgramParameteri(program,BinaryHint,1);
        for(const auto& stage:stages)
        {
            const auto shader=CreateShader(stage.first);
            if(!shader){DeleteProgram(program);error="Cannot create OpenGL ES shader";return 0;}
            const auto* source=stage.second.data();const int length=int(stage.second.size());
            ShaderSource(shader,1,&source,&length);CompileShader(shader);
            int compiled=0;GetShaderiv(shader,CompileStatus,&compiled);
            if(!compiled)
            {
                int size=0;GetShaderiv(shader,InfoLogLength,&size);std::vector<char> log(std::max(size,1));
                GetShaderInfoLog(shader,int(log.size()),nullptr,log.data());error=std::string("OpenGL ES shader compilation failed: ")+log.data();
                DeleteShader(shader);DeleteProgram(program);return 0;
            }
            AttachShader(program,shader);DeleteShader(shader);
        }
        LinkProgram(program);int linked=0;GetProgramiv(program,LinkStatus,&linked);
        if(!linked)
        {
            int size=0;GetProgramiv(program,InfoLogLength,&size);std::vector<char> log(std::max(size,1));
            GetProgramInfoLog(program,int(log.size()),nullptr,log.data());error=std::string("OpenGL ES program link failed: ")+log.data();
            DeleteProgram(program);return 0;
        }
        if(cacheable)
        {
            int length=0;GetProgramiv(program,BinaryLength,&length);
            if(length>0 && length<256*1024*1024)
            {
                data.resize(std::size_t(length)+4);unsigned format=0;int written=0;
                GetProgramBinary(program,length,&written,&format,data.data()+4);
                if(written>0 && written<=length && !GetError())
                {std::memcpy(data.data(),&format,4);ShaderCache::Write("gles-program",digest,data.data(),std::size_t(written)+4);}
            }
        }
        return program;
    }
}
