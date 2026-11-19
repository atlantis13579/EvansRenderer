#include "Renderer/OpenGLES/OpenGLESProgramCache.h"
#include "Renderer/OpenGLES/OpenGLESContactShadows.h"
#include "Renderer/TechniqueShader.h"

#include <array>
#include <sstream>
#include <vector>

#if defined(_WIN32)
#define EVANS_CONTACT_GL_CALL __stdcall
#else
#define EVANS_CONTACT_GL_CALL
#endif

namespace Evans::Renderer
{
    namespace
    {
        using UInt = unsigned int;
        using Int = int;
        using Boolean = unsigned char;
        // ES 3.1 enumerants are kept local; the renderer loads its existing EGL
        // runtime dynamically and does not require a new platform SDK/library.
        constexpr UInt MajorVersion = 0x821B, MinorVersion = 0x821C;
        constexpr UInt ComputeShader = 0x91B9, CompileStatus = 0x8B81, LinkStatus = 0x8B82, InfoLogLength = 0x8B84;
        constexpr UInt Texture2D = 0x0DE1, Texture2DMS = 0x9100, TextureArray = 0x8C1A, TextureUnitZero = 0x84C0;
        constexpr UInt TextureWidth = 0x1000, TextureHeight = 0x1001, TextureInternalFormat = 0x1003;
        constexpr UInt TextureSamples = 0x9106, FixedSampleLocations = 0x9107, Depth32F = 0x8CAC;
        constexpr UInt R32F = 0x822E, Nearest = 0x2600, ClampToEdge = 0x812F;
        constexpr UInt MinFilter = 0x2801, MagFilter = 0x2800, WrapS = 0x2802, WrapT = 0x2803;
        constexpr UInt Framebuffer = 0x8D40, DrawFramebuffer = 0x8CA9, ReadFramebuffer = 0x8CA8;
        constexpr UInt DrawFramebufferBinding = 0x8CA6, ReadFramebufferBinding = 0x8CAA;
        constexpr UInt DepthAttachment = 0x8D00, CompleteFramebuffer = 0x8CD5, SamplePosition = 0x8E50;
        constexpr UInt CurrentProgram = 0x8B8D, ActiveTexture = 0x84E0, WriteOnly = 0x88B9;
        constexpr UInt GroupCountLimit = 0x91BE, GroupSizeLimit = 0x91BF, InvocationLimit = 0x90EB;
        constexpr UInt TextureFetchBarrier = 0x00000008, ImageAccessBarrier = 0x00000020, FramebufferBarrier = 0x00000400;
        constexpr UInt GroupWidth = 8;
        constexpr Int MaximumSamples = 8;
    }
    struct OpenGLESContactShadows::Impl
    {
        ProcLoader CacheResolver;
#define CONTACT_GL_FUNCTIONS(X) \
        X(UInt, GetError, ()) \
        X(void, GetIntegerv, (UInt, Int*)) \
        X(void, GetIntegeri_v, (UInt, UInt, Int*)) \
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
        X(void, UniformMatrix4fv, (Int, Int, Boolean, const float*)) \
        X(void, Uniform4fv, (Int, Int, const float*)) \
        X(void, Uniform4ui, (Int, UInt, UInt, UInt, UInt)) \
        X(void, Uniform1i, (Int, Int)) \
        X(void, GenTextures, (Int, UInt*)) \
        X(void, DeleteTextures, (Int, const UInt*)) \
        X(Boolean, IsTexture, (UInt)) \
        X(void, ActiveTexture, (UInt)) \
        X(void, BindTexture, (UInt, UInt)) \
        X(void, BindSampler, (UInt, UInt)) \
        X(void, TexParameteri, (UInt, UInt, Int)) \
        X(void, TexStorage3D, (UInt, Int, UInt, Int, Int, Int)) \
        X(void, GetTexLevelParameteriv, (UInt, Int, UInt, Int*)) \
        X(void, GenFramebuffers, (Int, UInt*)) \
        X(void, DeleteFramebuffers, (Int, const UInt*)) \
        X(void, BindFramebuffer, (UInt, UInt)) \
        X(void, FramebufferTexture2D, (UInt, UInt, UInt, UInt, Int)) \
        X(void, DrawBuffers, (Int, const UInt*)) \
        X(void, ReadBuffer, (UInt)) \
        X(UInt, CheckFramebufferStatus, (UInt)) \
        X(void, GetMultisamplefv, (UInt, UInt, float*)) \
        X(void, BindImageTexture, (UInt, UInt, Int, Boolean, Int, UInt, UInt)) \
        X(void, DispatchCompute, (UInt, UInt, UInt)) \
        X(void, MemoryBarrier, (UInt))
#define DECLARE_CONTACT_GL(result, name, arguments) result (EVANS_CONTACT_GL_CALL* name) arguments = nullptr;
        CONTACT_GL_FUNCTIONS(DECLARE_CONTACT_GL)
#undef DECLARE_CONTACT_GL
        std::array<UInt, 2> Programs {};
        UInt Visibility = 0, QueryFramebuffer = 0;
        Int Width = 0, Height = 0, Samples = 0;
        std::array<Int, 3> GroupLimits {};
        bool Initialized = false, Open = false, Recorded = false, Ready = false;

        bool Check(std::string& error, const char* operation)
        {
            const auto code = GetError();
            if (!code) return true;
            std::ostringstream message; message << "OpenGL ES contact " << operation << " failed: 0x" << std::hex << code;
            error = message.str(); return false;
        }
        UInt Program(bool multisample, std::string& error)
        {
            auto& program = Programs[multisample ? 1 : 0];
            if (program) return program;
            std::string source;
            if (!LoadTechniqueShaderSource("OpenGLES", "ContactShadows.glsl", source, error)) return 0;
            if (multisample) source.insert(source.find('\n') + 1, "#define EVANS_MULTISAMPLE 1\n");
            const auto candidate=CompileCachedGLESProgram(CacheResolver,{{ComputeShader,source}},error);
            if(!candidate)return 0;
            program = candidate; return program;
        }
        void Matrix(UInt program, const char* name, const Matrix4& value)
        {
            std::array<float, 16> columns {};
            for (unsigned row = 0; row < 4; ++row)
                for (unsigned column = 0; column < 4; ++column) columns[column * 4 + row] = value.Values[row * 4 + column];
            UniformMatrix4fv(GetUniformLocation(program, name), 1, 0, columns.data());
        }
    };

    OpenGLESContactShadows::OpenGLESContactShadows() : m_Impl(std::make_unique<Impl>()) {}
    OpenGLESContactShadows::~OpenGLESContactShadows() { Shutdown(); }
    bool OpenGLESContactShadows::Initialize(const ProcLoader& load, std::string& error)
    {
        error.clear();
        if (m_Impl->Initialized) return true;
        auto& gl = *m_Impl; gl.CacheResolver=load;
#define LOAD_CONTACT_GL(result, name, arguments) \
        gl.name = reinterpret_cast<decltype(gl.name)>(load("gl" #name)); \
        if (!gl.name) { error = "Missing ES contact entry point: gl" #name; return false; }
        CONTACT_GL_FUNCTIONS(LOAD_CONTACT_GL)
#undef LOAD_CONTACT_GL
#undef CONTACT_GL_FUNCTIONS
        Int major = 0, minor = 0; gl.GetIntegerv(MajorVersion, &major); gl.GetIntegerv(MinorVersion, &minor);
        if (major < 3 || (major == 3 && minor < 1)) { error = "Contact visibility requires OpenGL ES 3.1"; return false; }
        Int invocations = 0; gl.GetIntegerv(InvocationLimit, &invocations);
        for (UInt axis = 0; axis < 3; ++axis)
        {
            Int size = 0; gl.GetIntegeri_v(GroupSizeLimit, axis, &size);
            gl.GetIntegeri_v(GroupCountLimit, axis, &gl.GroupLimits[axis]);
            if (size < Int(axis < 2 ? GroupWidth : 1) || invocations < Int(GroupWidth * GroupWidth))
            { error = "ES contact workgroup limits are insufficient"; return false; }
        }
        if (!gl.Check(error, "capability query")) return false;
        gl.Initialized = true; return true;
    }
    void OpenGLESContactShadows::Shutdown()
    {
        auto& gl = *m_Impl;
        for (auto& program : gl.Programs) if (program) { gl.DeleteProgram(program); program = 0; }
        if (gl.Visibility) gl.DeleteTextures(1, &gl.Visibility);
        if (gl.QueryFramebuffer) gl.DeleteFramebuffers(1, &gl.QueryFramebuffer);
        gl.Visibility = gl.QueryFramebuffer = 0; gl.Width = gl.Height = gl.Samples = 0;
        gl.Initialized = gl.Open = gl.Ready = gl.Recorded = false;
    }
    void OpenGLESContactShadows::BeginFrame() { m_Impl->Open = true; m_Impl->Ready = m_Impl->Recorded = false; }
    void OpenGLESContactShadows::AbortFrame()
    {
        // ES commands are immediate: abort hides readiness, not submitted work.
        // Ordered dispatch plus image-access barriers protect later image reuse.
        m_Impl->Open = m_Impl->Ready = m_Impl->Recorded = false;
    }
    std::uint32_t OpenGLESContactShadows::Result() const { return m_Impl->Ready ? m_Impl->Visibility : 0; }

    bool OpenGLESContactShadows::Dispatch(std::uint32_t depth, bool multisample,
        const ContactShadowDispatch& desc, std::string& error)
    {
        auto& gl = *m_Impl; gl.Ready = false; error.clear();
        const auto fail = [&](const char* message) { error = message; return false; };
        if (!gl.Initialized || !gl.Open || gl.Recorded) return fail("ES contact dispatch requires an initialized, unused open frame");
        if (!depth || !gl.IsTexture(depth)) return fail("ES contact input must be an existing depth texture");
        if (!gl.Check(error, "dispatch entry")) return false;
        Int oldProgram = 0, oldActive = 0, oldDraw = 0, oldRead = 0;
        gl.GetIntegerv(CurrentProgram, &oldProgram); gl.GetIntegerv(ActiveTexture, &oldActive);
        gl.GetIntegerv(DrawFramebufferBinding, &oldDraw); gl.GetIntegerv(ReadFramebufferBinding, &oldRead);
        gl.ActiveTexture(TextureUnitZero); gl.BindSampler(0, 0);
        // All exits restore context routing; texture/image unit zero is scratch.
        const auto finish = [&](bool success)
        {
            gl.BindImageTexture(0, 0, 0, 0, 0, WriteOnly, R32F);
            gl.BindTexture(Texture2D, 0); gl.BindTexture(Texture2DMS, 0); gl.BindTexture(TextureArray, 0);
            gl.ActiveTexture(UInt(oldActive)); gl.UseProgram(UInt(oldProgram));
            gl.BindFramebuffer(DrawFramebuffer, UInt(oldDraw)); gl.BindFramebuffer(ReadFramebuffer, UInt(oldRead));
            const bool clean = gl.Check(error, "dispatch/restore"); gl.Ready = success && clean; return gl.Ready;
        };
        const UInt target = multisample ? Texture2DMS : Texture2D;
        gl.BindTexture(target, depth);
        Int width = 0, height = 0, format = 0, samples = 1, fixed = 1;
        gl.GetTexLevelParameteriv(target, 0, TextureWidth, &width);
        gl.GetTexLevelParameteriv(target, 0, TextureHeight, &height);
        gl.GetTexLevelParameteriv(target, 0, TextureInternalFormat, &format);
        if (multisample)
        {
            gl.GetTexLevelParameteriv(target, 0, TextureSamples, &samples);
            gl.GetTexLevelParameteriv(target, 0, FixedSampleLocations, &fixed);
        }
        if (!gl.Check(error, "depth texture query")) return finish(false);
        if (format != Int(Depth32F) || width <= 0 || height <= 0 || !fixed || samples < 1 || samples > MaximumSamples)
        { fail("ES contact requires finite-size D32F depth and fixed 1/2/4/8 sample locations"); return finish(false); }
        ContactShadowConstants constants;
        if (!BuildContactShadowConstants(desc, UInt(width), UInt(height), UInt(samples), constants, error)) return finish(false);
        const UInt groupsX = (UInt(width) + GroupWidth - 1) / GroupWidth, groupsY = (UInt(height) + GroupWidth - 1) / GroupWidth;
        if (groupsX > UInt(gl.GroupLimits[0]) || groupsY > UInt(gl.GroupLimits[1]) || samples > gl.GroupLimits[2])
        { fail("ES contact image exceeds dispatch limits"); return finish(false); }
        std::array<Float4, MaximumSamples> locations {}; locations[0] = {0.5f, 0.5f, 0, 0};
        if (!gl.QueryFramebuffer) gl.GenFramebuffers(1, &gl.QueryFramebuffer);
        gl.BindFramebuffer(Framebuffer, gl.QueryFramebuffer);
        gl.FramebufferTexture2D(Framebuffer, DepthAttachment, target, depth, 0);
        constexpr UInt NoColor = 0; gl.DrawBuffers(1, &NoColor); gl.ReadBuffer(NoColor);
        if (gl.CheckFramebufferStatus(Framebuffer) != CompleteFramebuffer)
        { fail("ES contact depth query framebuffer is incomplete"); return finish(false); }
        if (multisample)
            for (Int sample = 0; sample < samples; ++sample) gl.GetMultisamplefv(SamplePosition, UInt(sample), &locations[sample].X);
        gl.FramebufferTexture2D(Framebuffer, DepthAttachment, target, 0, 0);
        gl.BindFramebuffer(Framebuffer, 0);
        if (!gl.Check(error, "depth sample locations")) return finish(false);
        const auto program = gl.Program(multisample, error);
        if (!program) return finish(false);
        if (width != gl.Width || height != gl.Height || samples != gl.Samples || !gl.Visibility)
        {
            UInt candidate = 0; gl.GenTextures(1, &candidate); gl.BindTexture(TextureArray, candidate);
            gl.TexStorage3D(TextureArray, 1, R32F, width, height, samples);
            for (auto parameter : {MinFilter, MagFilter}) gl.TexParameteri(TextureArray, parameter, Int(Nearest));
            for (auto parameter : {WrapS, WrapT}) gl.TexParameteri(TextureArray, parameter, Int(ClampToEdge));
            if (!candidate || !gl.Check(error, "visibility allocation"))
            { if (candidate) gl.DeleteTextures(1, &candidate); return finish(false); }
            if (gl.Visibility) gl.DeleteTextures(1, &gl.Visibility);
            gl.Visibility = candidate; gl.Width = width; gl.Height = height; gl.Samples = samples;
        }
        gl.UseProgram(program); gl.Matrix(program, "Camera", constants.ViewProjection); gl.Matrix(program, "InverseCamera", constants.InverseViewProjection);
        gl.Uniform4fv(gl.GetUniformLocation(program, "DepthMetric"), 1, &constants.DepthMetric.X);
        gl.Uniform4fv(gl.GetUniformLocation(program, "LightPositionType"), 1, &constants.LightPositionType.X);
        gl.Uniform4fv(gl.GetUniformLocation(program, "LightDirection"), 1, &constants.LightDirection.X);
        gl.Uniform4fv(gl.GetUniformLocation(program, "TraceParameters"), 1, &constants.TraceParameters.X);
        gl.Uniform4ui(gl.GetUniformLocation(program, "ImageAndBudget"), UInt(width), UInt(height), constants.ImageAndBudget.Z, UInt(samples));
        gl.Uniform4fv(gl.GetUniformLocation(program, "SampleLocations[0]"), Int(locations.size()), &locations[0].X);
        gl.Uniform1i(gl.GetUniformLocation(program, "CameraDepth"), 0);
        gl.BindImageTexture(0, gl.Visibility, 0, 1, 0, WriteOnly, R32F);
        // ES 3.1 section 7.11.2: image stores are incoherent. Make the previous
        // consumers finish before reuse, then expose this dispatch to texture
        // fetches, later image accesses and framebuffer-layer readback.
        // https://registry.khronos.org/OpenGL/specs/es/3.1/es_spec_3.1.pdf
        gl.MemoryBarrier(ImageAccessBarrier);
        gl.DispatchCompute(groupsX, groupsY, UInt(samples)); gl.Recorded = true;
        gl.MemoryBarrier(TextureFetchBarrier | ImageAccessBarrier | FramebufferBarrier);
        return finish(true);
    }
}
#undef EVANS_CONTACT_GL_CALL
