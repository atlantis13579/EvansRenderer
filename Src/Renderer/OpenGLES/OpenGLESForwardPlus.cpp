#include "Renderer/OpenGLES/OpenGLESProgramCache.h"
#include "Renderer/OpenGLES/OpenGLESForwardPlus.h"
#include "Renderer/TechniqueShader.h"

#include <algorithm>
#include <limits>
#include <sstream>

#if defined(_WIN32)
#define EVANS_FORWARD_GL_CALL __stdcall
#else
#define EVANS_FORWARD_GL_CALL
#endif

namespace Evans::Renderer
{
    namespace
    {
        using UInt = unsigned int;
        using Int = int;
        using Size = std::ptrdiff_t;
        using Boolean = unsigned char;
        // Local ES enumerants keep this module independent of another platform SDK.
        constexpr UInt MajorVersion = 0x821B, MinorVersion = 0x821C;
        constexpr UInt ComputeShader = 0x91B9, CompileStatus = 0x8B81, LinkStatus = 0x8B82, InfoLogLength = 0x8B84;
        constexpr UInt Texture2D = 0x0DE1, TextureMS = 0x9100, TextureZero = 0x84C0;
        constexpr UInt TextureWidth = 0x1000, TextureHeight = 0x1001, TextureInternalFormat = 0x1003, TextureSamples = 0x9106;
        constexpr UInt Depth32F = 0x8CAC, ActiveTexture = 0x84E0, CurrentProgram = 0x8B8D;
        constexpr UInt DrawFramebuffer = 0x8CA9, ReadFramebuffer = 0x8CA8, DrawFramebufferBinding = 0x8CA6, ReadFramebufferBinding = 0x8CAA;
        constexpr UInt ShaderStorage = 0x90D2, DynamicDraw = 0x88E8, DynamicCopy = 0x88EA;
        constexpr UInt StorageBindings = 0x90DD, ComputeStorageBlocks = 0x90DB, StorageBlockBytes = 0x90DE;
        constexpr UInt GroupCountLimit = 0x91BE, GroupSizeLimit = 0x91BF, InvocationLimit = 0x90EB, SharedMemoryLimit = 0x8262;
        constexpr UInt StorageBarrier = 0x00002000, BufferUpdateBarrier = 0x00000200;
        constexpr UInt GroupLanes = 64, BoundSlot = 0, HeaderSlot = 1, IndexSlot = 2, BindingCount = 3;
        // Two 128-element uvec2 arrays, eight plane/error pairs, and reductions;
        // reserve padding beyond their 2,320-byte logical shared footprint.
        constexpr Int RequiredSharedBytes = 4096;
    }
    struct OpenGLESForwardPlus::Impl
    {
        ProcLoader CacheResolver;
        TiledLightShader Shader = TiledLightShader::ForwardPlus;
#define FORWARD_GL_FUNCTIONS(X) \
        X(UInt, GetError, ()) \
        X(void, GetIntegerv, (UInt, Int*)) \
        X(void, GetIntegeri_v, (UInt, UInt, Int*)) \
        X(void, GetInteger64v, (UInt, std::int64_t*)) \
        X(UInt, CreateShader, (UInt)) \
        X(void, ShaderSource, (UInt, Int, const char* const*, const Int*)) \
        X(void, CompileShader, (UInt)) \
        X(void, GetShaderiv, (UInt, UInt, Int*)) \
        X(void, GetShaderInfoLog, (UInt, Int, Int*, char*)) \
        X(void, DeleteShader, (UInt)) \
        X(UInt, CreateProgram, ()) \
        X(void, AttachShader, (UInt, UInt)) \
        X(void, LinkProgram, (UInt)) \
        X(void, GetProgramiv, (UInt, UInt, Int*)) \
        X(void, GetProgramInfoLog, (UInt, Int, Int*, char*)) \
        X(void, DeleteProgram, (UInt)) \
        X(void, UseProgram, (UInt)) \
        X(Int, GetUniformLocation, (UInt, const char*)) \
        X(void, Uniform4fv, (Int, Int, const float*)) \
        X(void, Uniform4ui, (Int, UInt, UInt, UInt, UInt)) \
        X(void, Uniform1i, (Int, Int)) \
        X(Boolean, IsTexture, (UInt)) \
        X(void, ActiveTexture, (UInt)) \
        X(void, BindTexture, (UInt, UInt)) \
        X(void, BindSampler, (UInt, UInt)) \
        X(void, GetTexLevelParameteriv, (UInt, Int, UInt, Int*)) \
        X(void, BindFramebuffer, (UInt, UInt)) \
        X(void, GenBuffers, (Int, UInt*)) \
        X(void, DeleteBuffers, (Int, const UInt*)) \
        X(void, BindBuffer, (UInt, UInt)) \
        X(void, BufferData, (UInt, Size, const void*, UInt)) \
        X(void, BufferSubData, (UInt, Size, Size, const void*)) \
        X(void, BindBufferBase, (UInt, UInt, UInt)) \
        X(void, DispatchCompute, (UInt, UInt, UInt)) \
        X(void, MemoryBarrier, (UInt))
#define DECLARE_FORWARD_GL(result, name, arguments) result (EVANS_FORWARD_GL_CALL* name) arguments = nullptr;
        FORWARD_GL_FUNCTIONS(DECLARE_FORWARD_GL)
#undef DECLARE_FORWARD_GL
        struct Program
        {
            UInt Handle = 0;
            Int Rows = -1, Extent = -1, Layout = -1, Depth = -1;
        };
        std::array<Program, 2> Programs;
        UInt Bounds = 0, Headers = 0, Indices = 0, AllocatedTiles = 0, AllocatedIndices = 0;
        ForwardPlusGrid CurrentGrid;
        std::array<Int, 3> GroupLimits {};
        std::int64_t BlockLimit = 0;
        bool Initialized = false, Open = false, Dispatched = false, Ready = false;

        bool Check(std::string& error, const char* operation)
        {
            const auto code = GetError(); if (!code) return true;
            std::ostringstream text; text << "OpenGL ES Forward+ " << operation << " failed: 0x" << std::hex << code;
            error = text.str(); return false;
        }
        Program* PrepareProgram(bool multisample, std::string& error)
        {
            auto& result = Programs[multisample ? 1 : 0]; if (result.Handle) return &result;
            std::string source;
            if (!LoadTechniqueShaderSource("OpenGLES", Shader == TiledLightShader::TiledDeferred ? "TiledDeferred.glsl" : "ForwardPlus.glsl", source, error)) return nullptr;
            if (multisample) source.insert(source.find('\n') + 1, Shader == TiledLightShader::TiledDeferred ? "#define EVANS_TILED_DEFERRED_MULTISAMPLE 1\n" : "#define EVANS_FORWARD_PLUS_MULTISAMPLE 1\n");
            const auto program=CompileCachedGLESProgram(CacheResolver,{{ComputeShader,source}},error);
            if(!program)return nullptr;
            Program candidate; candidate.Handle = program;
            candidate.Rows = GetUniformLocation(program, "CameraRows[0]"); candidate.Extent = GetUniformLocation(program, "ExtentAndLights");
            candidate.Layout = GetUniformLocation(program, "TileLayout"); candidate.Depth = GetUniformLocation(program, "CameraDepth");
            if (candidate.Rows < 0 || candidate.Extent < 0 || candidate.Layout < 0 || candidate.Depth < 0 || !Check(error, "program interface"))
            { if (error.empty()) error = "ES Forward+ uniform contract is incomplete"; DeleteProgram(program); return nullptr; }
            result = candidate; return &result;
        }
        bool CreateStorage(std::uint64_t bytes, UInt usage, UInt& result, std::string& error)
        {
            GenBuffers(1, &result); BindBuffer(ShaderStorage, result); BufferData(ShaderStorage, Size(bytes), nullptr, usage);
            if (!Check(error, "buffer allocation") || !result)
            { if (result) DeleteBuffers(1, &result); result = 0; if (error.empty()) error = "ES Forward+ buffer allocation failed"; return false; }
            return true;
        }
    };
    OpenGLESForwardPlus::OpenGLESForwardPlus(TiledLightShader shader) : m_Impl(std::make_unique<Impl>()) { m_Impl->Shader = shader; }
    OpenGLESForwardPlus::~OpenGLESForwardPlus() { Shutdown(); }
    bool OpenGLESForwardPlus::Initialize(const ProcLoader& loader, std::string& error)
    {
        error.clear(); auto& gl = *m_Impl; gl.CacheResolver=loader;
        if (gl.Initialized) return true;
        if (!loader) { error = "ES Forward+ requires the owning context's function loader"; return false; }
#define LOAD_FORWARD_GL(result, name, arguments) \
        gl.name = reinterpret_cast<decltype(gl.name)>(loader("gl" #name)); \
        if (!gl.name) { error = "Missing ES Forward+ entry point: gl" #name; return false; }
        FORWARD_GL_FUNCTIONS(LOAD_FORWARD_GL)
#undef LOAD_FORWARD_GL
#undef FORWARD_GL_FUNCTIONS
        Int major = 0, minor = 0; gl.GetIntegerv(MajorVersion, &major); gl.GetIntegerv(MinorVersion, &minor);
        if (major < 3 || (major == 3 && minor < 1)) { error = "Forward+ light assignment requires OpenGL ES 3.1"; return false; }
        Int bindings = 0, blocks = 0, shared = 0, invocations = 0;
        gl.GetIntegerv(StorageBindings, &bindings); gl.GetIntegerv(ComputeStorageBlocks, &blocks);
        gl.GetIntegerv(SharedMemoryLimit, &shared); gl.GetIntegerv(InvocationLimit, &invocations);
        gl.GetInteger64v(StorageBlockBytes, &gl.BlockLimit);
        for (UInt axis = 0; axis < 3; ++axis)
        {
            Int size = 0; gl.GetIntegeri_v(GroupSizeLimit, axis, &size); gl.GetIntegeri_v(GroupCountLimit, axis, &gl.GroupLimits[axis]);
            if (size < Int(axis == 0 ? GroupLanes : 1) || gl.GroupLimits[axis] < 1)
            { error = "ES Forward+ compute group limits are insufficient"; return false; }
        }
        if (!gl.Check(error, "capability query")) return false;
        if (bindings < Int(BindingCount) || blocks < Int(BindingCount) || shared < RequiredSharedBytes || invocations < Int(GroupLanes) ||
            gl.BlockLimit < std::int64_t(MaximumForwardPlusLights * sizeof(ForwardPlusGpuBounds)))
        { error = "ES Forward+ requires three compute storage blocks and sufficient shared/storage memory"; return false; }
        gl.Initialized = true; return true;
    }
    void OpenGLESForwardPlus::Shutdown()
    {
        auto& gl = *m_Impl;
        for (auto& program : gl.Programs) if (program.Handle) { gl.DeleteProgram(program.Handle); program = {}; }
        for (auto* buffer : {&gl.Bounds, &gl.Headers, &gl.Indices}) if (*buffer) { gl.DeleteBuffers(1, buffer); *buffer = 0; }
        gl.Initialized = gl.Open = gl.Dispatched = gl.Ready = false;
        gl.CurrentGrid = {}; gl.AllocatedTiles = gl.AllocatedIndices = 0;
    }
    void OpenGLESForwardPlus::BeginFrame() { m_Impl->Open = true; m_Impl->Dispatched = m_Impl->Ready = false; }
    void OpenGLESForwardPlus::AbortFrame() { m_Impl->Open = m_Impl->Dispatched = m_Impl->Ready = false; }
    std::uint32_t OpenGLESForwardPlus::Headers() const { return m_Impl->Ready ? m_Impl->Headers : 0; }
    std::uint32_t OpenGLESForwardPlus::Indices() const { return m_Impl->Ready ? m_Impl->Indices : 0; }
    ForwardPlusGrid OpenGLESForwardPlus::Grid() const { return m_Impl->Ready ? m_Impl->CurrentGrid : ForwardPlusGrid {}; }

    bool OpenGLESForwardPlus::Dispatch(std::uint32_t depth, bool multisample, const Matrix4& camera,
        const std::vector<ForwardPlusLightBounds>& lights, const ForwardPlusParameters& parameters, std::string& error)
    {
        auto& gl = *m_Impl; gl.Ready = false; error.clear();
        const auto fail = [&](const char* reason) { error = reason; return false; };
        if (!gl.Initialized || !gl.Open || gl.Dispatched) return fail("ES Forward+ requires an initialized, unused open frame");
        if (!depth || !gl.IsTexture(depth)) return fail("ES Forward+ requires an existing camera-depth texture");
        if (!gl.Check(error, "dispatch entry")) return false;
        Int oldProgram = 0, oldActive = 0, oldDraw = 0, oldRead = 0;
        gl.GetIntegerv(CurrentProgram, &oldProgram); gl.GetIntegerv(ActiveTexture, &oldActive);
        gl.GetIntegerv(DrawFramebufferBinding, &oldDraw); gl.GetIntegerv(ReadFramebufferBinding, &oldRead);
        gl.ActiveTexture(TextureZero); gl.BindSampler(0, 0);
        const auto finish = [&](bool success)
        {
            for (UInt slot = 0; slot < BindingCount; ++slot) gl.BindBufferBase(ShaderStorage, slot, 0);
            gl.BindBuffer(ShaderStorage, 0); gl.BindTexture(Texture2D, 0); gl.BindTexture(TextureMS, 0);
            gl.ActiveTexture(UInt(oldActive)); gl.UseProgram(UInt(oldProgram));
            gl.BindFramebuffer(DrawFramebuffer, UInt(oldDraw)); gl.BindFramebuffer(ReadFramebuffer, UInt(oldRead));
            const bool clean = gl.Check(error, "dispatch/restore"); gl.Ready = success && clean; return gl.Ready;
        };
        const UInt target = multisample ? TextureMS : Texture2D;
        gl.BindTexture(target, depth);
        if (!gl.Check(error, "depth binding")) return finish(false);
        Int width = 0, height = 0, samples = 1, format = 0;
        gl.GetTexLevelParameteriv(target, 0, TextureWidth, &width); gl.GetTexLevelParameteriv(target, 0, TextureHeight, &height);
        gl.GetTexLevelParameteriv(target, 0, TextureInternalFormat, &format);
        if (multisample) gl.GetTexLevelParameteriv(target, 0, TextureSamples, &samples);
        if (!gl.Check(error, "depth metadata")) return finish(false);
        if (width <= 0 || height <= 0 || samples <= 0 || format != Int(Depth32F))
        { fail("ES Forward+ needs original D32F camera-depth samples"); return finish(false); }
        ForwardPlusGrid grid; ForwardPlusConstants constants; std::vector<ForwardPlusGpuBounds> bounds;
        if (!BuildForwardPlusGrid(UInt(width), UInt(height), UInt(samples), lights.size(), parameters, grid, error) ||
            !BuildForwardPlusConstants(camera, grid, constants, error) || !BuildForwardPlusGpuBounds(lights, bounds, error)) return finish(false);
        const auto headerBytes = std::uint64_t(grid.TileCount) * sizeof(ForwardPlusTile);
        const auto indexBytes = std::uint64_t(grid.IndexCount) * sizeof(std::uint32_t);
        const auto limit = std::min(std::uint64_t(gl.BlockLimit), std::uint64_t(std::numeric_limits<Size>::max()));
        if (grid.TilesX > UInt(gl.GroupLimits[0]) || grid.TilesY > UInt(gl.GroupLimits[1]) || headerBytes > limit || indexBytes > limit)
        { fail("ES Forward+ tile lists exceed compute dispatch or storage-block limits"); return finish(false); }
        auto* program = gl.PrepareProgram(multisample, error); if (!program) return finish(false);
        if (!gl.Bounds && !gl.CreateStorage(MaximumForwardPlusLights * sizeof(ForwardPlusGpuBounds), DynamicDraw, gl.Bounds, error)) return finish(false);
        if (grid.TileCount != gl.AllocatedTiles || grid.IndexCount != gl.AllocatedIndices)
        {
            UInt headers = 0, indices = 0;
            if (!gl.CreateStorage(headerBytes, DynamicCopy, headers, error) || !gl.CreateStorage(indexBytes, DynamicCopy, indices, error))
            { if (headers) gl.DeleteBuffers(1, &headers); if (indices) gl.DeleteBuffers(1, &indices); return finish(false); }
            if (gl.Headers) gl.DeleteBuffers(1, &gl.Headers); if (gl.Indices) gl.DeleteBuffers(1, &gl.Indices);
            gl.Headers = headers; gl.Indices = indices; gl.AllocatedTiles = grid.TileCount; gl.AllocatedIndices = grid.IndexCount;
        }
        gl.BindBuffer(ShaderStorage, gl.Bounds);
        if (!bounds.empty()) gl.BufferSubData(ShaderStorage, 0, Size(bounds.size() * sizeof(ForwardPlusGpuBounds)), bounds.data());
        gl.BindFramebuffer(DrawFramebuffer, 0); gl.BindFramebuffer(ReadFramebuffer, 0); gl.UseProgram(program->Handle);
        gl.Uniform4fv(program->Rows, 4, constants.ViewProjection.Values.data());
        gl.Uniform4ui(program->Extent, grid.Width, grid.Height, grid.Samples, grid.LightCount);
        gl.Uniform4ui(program->Layout, grid.TileSize, grid.TileCapacity, grid.TilesX, grid.TilesY); gl.Uniform1i(program->Depth, 0);
        gl.BindBufferBase(ShaderStorage, BoundSlot, gl.Bounds); gl.BindBufferBase(ShaderStorage, HeaderSlot, gl.Headers);
        gl.BindBufferBase(ShaderStorage, IndexSlot, gl.Indices);
        if (!gl.Check(error, "assignment bindings")) return finish(false);
        // SSBO storage is incoherent. Order previous readers before overwriting
        // lists, then expose stores to both shader consumers and buffer mapping.
        // https://registry.khronos.org/OpenGL-Refpages/es3.1/html/glMemoryBarrier.xhtml
        gl.MemoryBarrier(StorageBarrier); gl.DispatchCompute(grid.TilesX, grid.TilesY, 1); gl.Dispatched = true;
        gl.MemoryBarrier(StorageBarrier | BufferUpdateBarrier); gl.CurrentGrid = grid;
        return finish(true);
    }
}
#undef EVANS_FORWARD_GL_CALL
