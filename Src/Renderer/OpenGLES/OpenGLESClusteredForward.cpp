#include "Renderer/OpenGLES/OpenGLESProgramCache.h"
#include "Renderer/OpenGLES/OpenGLESClusteredForward.h"
#include "Renderer/TechniqueShader.h"

#include <algorithm>
#include <limits>
#include <sstream>

#if defined(_WIN32)
#define EVANS_CLUSTER_GL_CALL __stdcall
#else
#define EVANS_CLUSTER_GL_CALL
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
        constexpr UInt CurrentProgram = 0x8B8D, UniformBuffer = 0x8A11, UniformBlockBytes = 0x8A30;
        constexpr UInt ShaderStorage = 0x90D2, DynamicDraw = 0x88E8, DynamicCopy = 0x88EA;
        constexpr UInt StorageBindings = 0x90DD, ComputeStorageBlocks = 0x90DB, StorageBlockBytes = 0x90DE;
        constexpr UInt GroupCountLimit = 0x91BE, GroupSizeLimit = 0x91BF, InvocationLimit = 0x90EB, SharedMemoryLimit = 0x8262;
        constexpr UInt StorageBarrier = 0x00002000, BufferUpdateBarrier = 0x00000200;
        constexpr UInt GroupLanes = 64, BoundSlot = 0, HeaderSlot = 1, IndexSlot = 2, BindingCount = 3;
        // Membership/scan arrays and six plane/error pairs, including padding.
        constexpr Int RequiredSharedBytes = 4096;
    }
    struct OpenGLESClusteredForward::Impl
    {
        ProcLoader CacheResolver;
#define CLUSTER_GL_FUNCTIONS(X) \
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
        X(void, GenBuffers, (Int, UInt*)) \
        X(void, DeleteBuffers, (Int, const UInt*)) \
        X(void, BindBuffer, (UInt, UInt)) \
        X(void, BufferData, (UInt, Size, const void*, UInt)) \
        X(void, BufferSubData, (UInt, Size, Size, const void*)) \
        X(void, BindBufferBase, (UInt, UInt, UInt)) \
        X(void, DispatchCompute, (UInt, UInt, UInt)) \
        X(void, MemoryBarrier, (UInt))
#define DECLARE_CLUSTER_GL(result, name, arguments) result (EVANS_CLUSTER_GL_CALL* name) arguments = nullptr;
        CLUSTER_GL_FUNCTIONS(DECLARE_CLUSTER_GL)
#undef DECLARE_CLUSTER_GL
        struct Program
        {
            UInt Handle = 0;
        };
        Program Compute;
        UInt Constants = 0, Bounds = 0, Headers = 0, Indices = 0, AllocatedClusters = 0, AllocatedIndices = 0;
        ClusteredForwardGrid CurrentGrid;
        std::array<Int, 3> GroupLimits {};
        std::int64_t BlockLimit = 0;
        bool Initialized = false, Open = false, Dispatched = false, Ready = false;

        bool Check(std::string& error, const char* operation)
        {
            const auto code = GetError(); if (!code) return true;
            std::ostringstream text; text << "OpenGL ES Clustered Forward " << operation << " failed: 0x" << std::hex << code;
            error = text.str(); return false;
        }
        Program* PrepareProgram(std::string& error)
        {
            auto& result = Compute; if (result.Handle) return &result;
            std::string source;
            if (!LoadTechniqueShaderSource("OpenGLES", "ClusteredForward.glsl", source, error)) return nullptr;
            const auto program=CompileCachedGLESProgram(CacheResolver,{{ComputeShader,source}},error);
            if(!program)return nullptr;
            Program candidate; candidate.Handle = program;
            result = candidate; return &result;
        }
        bool CreateStorage(std::uint64_t bytes, UInt usage, UInt& result, std::string& error)
        {
            GenBuffers(1, &result); BindBuffer(ShaderStorage, result); BufferData(ShaderStorage, Size(bytes), nullptr, usage);
            if (!Check(error, "buffer allocation") || !result)
            { if (result) DeleteBuffers(1, &result); result = 0; if (error.empty()) error = "ES Clustered Forward buffer allocation failed"; return false; }
            return true;
        }
    };
    OpenGLESClusteredForward::OpenGLESClusteredForward() : m_Impl(std::make_unique<Impl>()) {}
    OpenGLESClusteredForward::~OpenGLESClusteredForward() { Shutdown(); }
    bool OpenGLESClusteredForward::Initialize(const ProcLoader& loader, std::string& error)
    {
        error.clear(); auto& gl = *m_Impl; gl.CacheResolver=loader;
        if (gl.Initialized) return true;
        if (!loader) { error = "ES Clustered Forward requires the owning context's function loader"; return false; }
#define LOAD_CLUSTER_GL(result, name, arguments) \
        gl.name = reinterpret_cast<decltype(gl.name)>(loader("gl" #name)); \
        if (!gl.name) { error = "Missing ES Clustered Forward entry point: gl" #name; return false; }
        CLUSTER_GL_FUNCTIONS(LOAD_CLUSTER_GL)
#undef LOAD_CLUSTER_GL
#undef CLUSTER_GL_FUNCTIONS
        Int major = 0, minor = 0; gl.GetIntegerv(MajorVersion, &major); gl.GetIntegerv(MinorVersion, &minor);
        if (major < 3 || (major == 3 && minor < 1)) { error = "Clustered Forward light assignment requires OpenGL ES 3.1"; return false; }
        Int bindings = 0, blocks = 0, shared = 0, invocations = 0, uniformBytes = 0;
        gl.GetIntegerv(UniformBlockBytes, &uniformBytes);
        gl.GetIntegerv(StorageBindings, &bindings); gl.GetIntegerv(ComputeStorageBlocks, &blocks);
        gl.GetIntegerv(SharedMemoryLimit, &shared); gl.GetIntegerv(InvocationLimit, &invocations);
        gl.GetInteger64v(StorageBlockBytes, &gl.BlockLimit);
        for (UInt axis = 0; axis < 3; ++axis)
        {
            Int size = 0; gl.GetIntegeri_v(GroupSizeLimit, axis, &size); gl.GetIntegeri_v(GroupCountLimit, axis, &gl.GroupLimits[axis]);
            if (size < Int(axis == 0 ? GroupLanes : 1) || gl.GroupLimits[axis] < 1)
            { error = "ES Clustered Forward compute group limits are insufficient"; return false; }
        }
        if (!gl.Check(error, "capability query")) return false;
        if (uniformBytes < Int(sizeof(ClusteredForwardConstants)) || bindings < Int(BindingCount) || blocks < Int(BindingCount) || shared < RequiredSharedBytes || invocations < Int(GroupLanes) ||
            gl.BlockLimit < std::int64_t(MaximumForwardPlusLights * sizeof(ForwardPlusGpuBounds)))
        { error = "ES Clustered Forward requires three compute storage blocks and sufficient shared/storage memory"; return false; }
        gl.Initialized = true; return true;
    }
    void OpenGLESClusteredForward::Shutdown()
    {
        auto& gl = *m_Impl;
        if (gl.Compute.Handle) { gl.DeleteProgram(gl.Compute.Handle); gl.Compute = {}; }
        for (auto* buffer : {&gl.Constants, &gl.Bounds, &gl.Headers, &gl.Indices}) if (*buffer) { gl.DeleteBuffers(1, buffer); *buffer = 0; }
        gl.Initialized = gl.Open = gl.Dispatched = gl.Ready = false;
        gl.CurrentGrid = {}; gl.AllocatedClusters = gl.AllocatedIndices = 0;
    }
    void OpenGLESClusteredForward::BeginFrame() { m_Impl->Open = true; m_Impl->Dispatched = m_Impl->Ready = false; }
    void OpenGLESClusteredForward::AbortFrame() { m_Impl->Open = m_Impl->Dispatched = m_Impl->Ready = false; }
    std::uint32_t OpenGLESClusteredForward::Headers() const { return m_Impl->Ready ? m_Impl->Headers : 0; }
    std::uint32_t OpenGLESClusteredForward::Indices() const { return m_Impl->Ready ? m_Impl->Indices : 0; }
    ClusteredForwardGrid OpenGLESClusteredForward::Grid() const { return m_Impl->Ready ? m_Impl->CurrentGrid : ClusteredForwardGrid {}; }

    bool OpenGLESClusteredForward::Dispatch(std::uint32_t width, std::uint32_t height, const Matrix4& camera,
        const std::vector<ForwardPlusLightBounds>& lights, const ClusteredForwardParameters& parameters, std::string& error)
    {
        auto& gl = *m_Impl; gl.Ready = false; error.clear();
        const auto fail = [&](const char* reason) { error = reason; return false; };
        if (!gl.Initialized || !gl.Open || gl.Dispatched) return fail("ES Clustered Forward requires an initialized, unused open frame");
        if (!gl.Check(error, "dispatch entry")) return false;
        Int oldProgram = 0; gl.GetIntegerv(CurrentProgram, &oldProgram);
        const auto finish = [&](bool success)
        {
            for (UInt slot = 0; slot < BindingCount; ++slot) gl.BindBufferBase(ShaderStorage, slot, 0);
            gl.BindBufferBase(UniformBuffer, 0, 0); gl.BindBuffer(UniformBuffer, 0); gl.BindBuffer(ShaderStorage, 0);
            gl.UseProgram(UInt(oldProgram));
            const bool clean = gl.Check(error, "dispatch/restore"); gl.Ready = success && clean; return gl.Ready;
        };
        ClusteredForwardGrid grid; ClusteredForwardConstants constants; std::vector<ForwardPlusGpuBounds> bounds;
        if (!BuildClusteredForwardGrid(width, height, lights.size(), parameters, grid, error) ||
            !BuildClusteredForwardConstants(camera, grid, constants, error) || !BuildForwardPlusGpuBounds(lights, bounds, error)) return finish(false);
        const auto headerBytes = std::uint64_t(grid.ClusterCount + ClusteredBoundarySlots) * sizeof(ClusteredForwardHeader);
        const auto indexBytes = std::uint64_t(grid.IndexCount) * sizeof(std::uint32_t);
        const auto limit = std::min(std::uint64_t(gl.BlockLimit), std::uint64_t(std::numeric_limits<Size>::max()));
        if (grid.TilesX > UInt(gl.GroupLimits[0]) || grid.TilesY > UInt(gl.GroupLimits[1]) || grid.DepthSlices > UInt(gl.GroupLimits[2]) || headerBytes > limit || indexBytes > limit)
        { fail("ES Clustered Forward tile lists exceed compute dispatch or storage-block limits"); return finish(false); }
        auto* program = gl.PrepareProgram(error); if (!program) return finish(false);
        if (!gl.Constants && !gl.CreateStorage(sizeof(constants), DynamicDraw, gl.Constants, error)) return finish(false);
        if (!gl.Bounds && !gl.CreateStorage(MaximumForwardPlusLights * sizeof(ForwardPlusGpuBounds), DynamicDraw, gl.Bounds, error)) return finish(false);
        if (grid.ClusterCount != gl.AllocatedClusters || grid.IndexCount != gl.AllocatedIndices)
        {
            UInt headers = 0, indices = 0;
            if (!gl.CreateStorage(headerBytes, DynamicCopy, headers, error) || !gl.CreateStorage(indexBytes, DynamicCopy, indices, error))
            { if (headers) gl.DeleteBuffers(1, &headers); if (indices) gl.DeleteBuffers(1, &indices); return finish(false); }
            if (gl.Headers) gl.DeleteBuffers(1, &gl.Headers); if (gl.Indices) gl.DeleteBuffers(1, &gl.Indices);
            gl.Headers = headers; gl.Indices = indices; gl.AllocatedClusters = grid.ClusterCount; gl.AllocatedIndices = grid.IndexCount;
        }
        gl.BindBuffer(ShaderStorage, gl.Bounds);
        if (!bounds.empty()) gl.BufferSubData(ShaderStorage, 0, Size(bounds.size() * sizeof(ForwardPlusGpuBounds)), bounds.data());
        gl.BindBuffer(UniformBuffer, gl.Constants); gl.BufferData(UniformBuffer, sizeof(constants), &constants, DynamicDraw);
        gl.BindBufferBase(UniformBuffer, 0, gl.Constants); gl.UseProgram(program->Handle);
        gl.BindBufferBase(ShaderStorage, BoundSlot, gl.Bounds); gl.BindBufferBase(ShaderStorage, HeaderSlot, gl.Headers);
        gl.BindBufferBase(ShaderStorage, IndexSlot, gl.Indices);
        if (!gl.Check(error, "assignment bindings")) return finish(false);
        // SSBO storage is incoherent. Order previous readers before overwriting
        // lists, then expose stores to both shader consumers and buffer mapping.
        // https://registry.khronos.org/OpenGL-Refpages/es3.1/html/glMemoryBarrier.xhtml
        gl.MemoryBarrier(StorageBarrier); gl.DispatchCompute(grid.TilesX, grid.TilesY, grid.DepthSlices); gl.Dispatched = true;
        gl.MemoryBarrier(StorageBarrier | BufferUpdateBarrier); gl.CurrentGrid = grid;
        return finish(true);
    }
}
#undef EVANS_CLUSTER_GL_CALL
