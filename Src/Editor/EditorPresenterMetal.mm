#include "Renderer/Metal/MetalShaderCache.h"
#include "Renderer/Types.h"
#include "Editor/EditorPresenterMetal.h"

#if defined(__APPLE__)

#include "Editor/Imgui.h"

#import <Cocoa/Cocoa.h>
#import <CoreGraphics/CoreGraphics.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>
#include <limits>
#include <utility>
#include <vector>

namespace Evans::Editor
{
namespace
{

constexpr float ImguiScale = 1.0f / 8.0f;
constexpr float Pi = 3.14159265358979323846f;
constexpr float TabStops[] = {150.0f, 210.0f, 270.0f, 330.0f};

struct MetalVertex
{
    float Position[2];
    float TexCoord[2];
    std::uint32_t Color;
};

static_assert(sizeof(MetalVertex) == 20u, "Metal UI vertex layout must match the MSL packed vertex.");

struct Glyph
{
    float U0 = 0.0f;
    float V0 = 0.0f;
    float U1 = 0.0f;
    float V1 = 0.0f;
    float Width = 0.0f;
    float Height = 0.0f;
    float Advance = 0.0f;
};

constexpr const char* ShaderSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct UiVertex
{
    packed_float2 Position;
    packed_float2 TexCoord;
    uint Color;
};

struct UiOutput
{
    float4 Position [[position]];
    float2 TexCoord;
    float4 Color;
};

vertex UiOutput EvansEditorVertex(
    device const UiVertex* vertices [[buffer(0)]],
    constant float2& screenSize [[buffer(1)]],
    uint vertexId [[vertex_id]])
{
    UiVertex input = vertices[vertexId];
    float2 position = float2(input.Position);
    UiOutput output;
    output.Position = float4(
        position.x * 2.0 / screenSize.x - 1.0,
        1.0 - position.y * 2.0 / screenSize.y,
        0.0,
        1.0);
    output.TexCoord = float2(input.TexCoord);
    output.Color = float4(
        float(input.Color & 255u),
        float((input.Color >> 8u) & 255u),
        float((input.Color >> 16u) & 255u),
        float((input.Color >> 24u) & 255u)) / 255.0;
    return output;
}

fragment float4 EvansEditorFragment(
    UiOutput input [[stage_in]],
    texture2d<float> texture [[texture(0)]],
    sampler textureSampler [[sampler(0)]])
{
    return input.Color * texture.sample(textureSampler, input.TexCoord);
}
)MSL";

void ReleaseObject(id object)
{
    if (object != nil)
    {
        [object release];
    }
}

std::string ErrorDescription(NSError* error, const char* fallback)
{
    if (error != nil)
    {
        const char* description = [[error localizedDescription] UTF8String];
        if (description != nullptr)
        {
            return description;
        }
    }
    return fallback;
}

} // namespace

struct EditorPresenterMetal::Impl
{
    ~Impl()
    {
        Shutdown();
    }

    bool Fail(std::string message)
    {
        LastError = std::move(message);
        return false;
    }

    id<MTLTexture> CreateTexture(int width, int height, MTLPixelFormat format, const void* pixels)
    {
        if (Device == nil || width <= 0 || height <= 0)
        {
            return nil;
        }
        MTLTextureDescriptor* descriptor = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:format
            width:static_cast<NSUInteger>(width)
            height:static_cast<NSUInteger>(height)
            mipmapped:NO];
        descriptor.usage = MTLTextureUsageShaderRead;
        if (@available(macOS 10.15, *))
        {
            descriptor.storageMode = Device.hasUnifiedMemory
                ? MTLStorageModeShared
                : MTLStorageModeManaged;
        }
        else
        {
            descriptor.storageMode = MTLStorageModeManaged;
        }
        id<MTLTexture> texture = [Device newTextureWithDescriptor:descriptor];
        if (texture != nil && pixels != nullptr)
        {
            const MTLRegion region = MTLRegionMake2D(0, 0, width, height);
            [texture replaceRegion:region mipmapLevel:0 withBytes:pixels bytesPerRow:width * 4];
        }
        return texture;
    }

    bool CreatePipeline()
    {
        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:ShaderSource];
        id<MTLLibrary> library = Evans::Renderer::CompileCachedMetalLibrary(Device,source,nil,&error);
        if (library == nil)
        {
            return Fail(ErrorDescription(error, "Metal editor shader compilation failed."));
        }
        id<MTLFunction> vertex = [library newFunctionWithName:@"EvansEditorVertex"];
        id<MTLFunction> fragment = [library newFunctionWithName:@"EvansEditorFragment"];
        if (vertex == nil || fragment == nil)
        {
            ReleaseObject(vertex);
            ReleaseObject(fragment);
            ReleaseObject(library);
            return Fail("Metal editor shader entry points are missing.");
        }
        MTLRenderPipelineDescriptor* descriptor = [[MTLRenderPipelineDescriptor alloc] init];
        descriptor.vertexFunction = vertex;
        descriptor.fragmentFunction = fragment;
        MTLRenderPipelineColorAttachmentDescriptor* colorAttachment =
            [[descriptor colorAttachments] objectAtIndexedSubscript:0];
        colorAttachment.pixelFormat = MTLPixelFormatBGRA8Unorm;
        colorAttachment.blendingEnabled = YES;
        colorAttachment.sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        colorAttachment.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        colorAttachment.rgbBlendOperation = MTLBlendOperationAdd;
        colorAttachment.sourceAlphaBlendFactor = MTLBlendFactorOne;
        colorAttachment.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        colorAttachment.alphaBlendOperation = MTLBlendOperationAdd;
        Pipeline = Evans::Renderer::CreateCachedMetalRenderPipeline(Device,descriptor,&error);
        if (Pipeline != nil)
        {
            // The viewport is a completed image; only UI primitives need alpha blending.
            colorAttachment.blendingEnabled = NO;
            ViewportPipeline = Evans::Renderer::CreateCachedMetalRenderPipeline(Device,descriptor,&error);
        }
        ReleaseObject(descriptor);
        ReleaseObject(vertex);
        ReleaseObject(fragment);
        ReleaseObject(library);
        if (Pipeline == nil || ViewportPipeline == nil)
        {
            return Fail(ErrorDescription(error, "Metal editor pipeline creation failed."));
        }

        MTLSamplerDescriptor* samplerDescriptor = [[MTLSamplerDescriptor alloc] init];
        samplerDescriptor.minFilter = MTLSamplerMinMagFilterLinear;
        samplerDescriptor.magFilter = MTLSamplerMinMagFilterLinear;
        samplerDescriptor.sAddressMode = MTLSamplerAddressModeClampToEdge;
        samplerDescriptor.tAddressMode = MTLSamplerAddressModeClampToEdge;
        Sampler = [Device newSamplerStateWithDescriptor:samplerDescriptor];
        ReleaseObject(samplerDescriptor);
        return Sampler != nil || Fail("Metal editor sampler creation failed.");
    }

    bool CreateFontTexture()
    {
        constexpr int AtlasWidth = 512;
        constexpr int AtlasHeight = 512;
        std::vector<std::uint8_t> pixels(
            static_cast<std::size_t>(AtlasWidth) * AtlasHeight * 4u,
            0u);
        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
        if (colorSpace == nullptr)
        {
            return Fail("Metal editor could not create the font color space.");
        }
        CGContextRef context = CGBitmapContextCreate(
            pixels.data(),
            AtlasWidth,
            AtlasHeight,
            8,
            AtlasWidth * 4,
            colorSpace,
            kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
        CGColorSpaceRelease(colorSpace);
        if (context == nullptr)
        {
            return Fail("Metal editor could not allocate the font atlas.");
        }

        NSGraphicsContext* graphics = [NSGraphicsContext
            graphicsContextWithCGContext:context
            flipped:YES];
        if (graphics == nil)
        {
            CGContextRelease(context);
            return Fail("Metal editor could not create the font graphics context.");
        }
        [NSGraphicsContext saveGraphicsState];
        [NSGraphicsContext setCurrentContext:graphics];
        [graphics setShouldAntialias:YES];
        NSFont* font = [NSFont systemFontOfSize:15.0];
        NSDictionary* attributes = @{
            NSFontAttributeName: font,
            NSForegroundColorAttributeName: [NSColor whiteColor]};
        FontHeight = std::max(8, static_cast<int>(std::ceil(
            [@"M" sizeWithAttributes:attributes].height)));
        imguiSetFontHeight(FontHeight);
        int penX = 1;
        int penY = 1;
        for (int character = 32; character < 128; ++character)
        {
            const char ascii = static_cast<char>(character);
            NSString* string = [[NSString alloc]
                initWithBytes:&ascii length:1 encoding:NSASCIIStringEncoding];
            const NSSize size = [string sizeWithAttributes:attributes];
            const int glyphWidth = std::max(1, static_cast<int>(std::ceil(size.width)));
            if (penX + glyphWidth + 2 >= AtlasWidth)
            {
                penX = 1;
                penY += FontHeight + 4;
            }
            if (penY + FontHeight + 2 >= AtlasHeight)
            {
                ReleaseObject(string);
                break;
            }
            Glyph& glyph = Glyphs[character - 32];
            glyph.U0 = static_cast<float>(penX) / AtlasWidth;
            glyph.V0 = static_cast<float>(penY) / AtlasHeight;
            glyph.U1 = static_cast<float>(penX + glyphWidth) / AtlasWidth;
            glyph.V1 = static_cast<float>(penY + FontHeight) / AtlasHeight;
            glyph.Width = static_cast<float>(glyphWidth);
            glyph.Height = static_cast<float>(FontHeight);
            glyph.Advance = static_cast<float>(glyphWidth);
            [string drawAtPoint:NSMakePoint(penX, penY) withAttributes:attributes];
            ReleaseObject(string);
            penX += glyphWidth + 1;
        }
        [NSGraphicsContext restoreGraphicsState];
        CGContextRelease(context);
        // The shader and blend state expect straight alpha. Core Graphics
        // writes premultiplied white glyphs, so restore white RGB coverage.
        for (std::size_t pixel = 0; pixel < pixels.size(); pixel += 4u)
        {
            pixels[pixel] = 255u;
            pixels[pixel + 1u] = 255u;
            pixels[pixel + 2u] = 255u;
        }
        FontTexture = CreateTexture(AtlasWidth, AtlasHeight, MTLPixelFormatRGBA8Unorm, pixels.data());
        return FontTexture != nil || Fail("Metal editor font texture creation failed.");
    }

    bool AttachLayer(void* nativeWindow)
    {
        id nativeObject = (id)nativeWindow;
        if ([nativeObject isKindOfClass:[CAMetalLayer class]])
        {
            Layer = [(CAMetalLayer*)nativeObject retain];
        }
        else
        {
            if ([nativeObject isKindOfClass:[NSWindow class]])
            {
                View = [[(NSWindow*)nativeObject contentView] retain];
            }
            else if ([nativeObject isKindOfClass:[NSView class]])
            {
                View = [(NSView*)nativeObject retain];
            }
            if (View == nil)
            {
                return Fail("Metal editor requires an NSWindow, NSView, or CAMetalLayer.");
            }

            PreviousWantsLayer = View.wantsLayer;
            CALayer* existingLayer = View.layer;
            if ([existingLayer isKindOfClass:[CAMetalLayer class]])
            {
                Layer = [(CAMetalLayer*)existingLayer retain];
            }
            else
            {
                PreviousLayer = [existingLayer retain];
                CAMetalLayer* metalLayer = [[CAMetalLayer alloc] init];
                View.wantsLayer = YES;
                View.layer = metalLayer;
                Layer = metalLayer;
                InstalledLayer = true;
            }
        }

        if (Layer == nil)
        {
            return Fail("Metal editor could not attach a CAMetalLayer.");
        }
        Layer.device = Device;
        Layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
        Layer.framebufferOnly = YES;
        Layer.opaque = YES;
        Layer.presentsWithTransaction = NO;
        Layer.displaySyncEnabled = NO;
        return true;
    }

    bool Initialize(void* nativeWindow, int width, int height)
    {
        if (nativeWindow == nullptr || width <= 0 || height <= 0)
        {
            return Fail("Metal editor requires a native window and non-zero dimensions.");
        }
        if (![NSThread isMainThread])
        {
            return Fail("Metal editor presentation must be initialized on the AppKit main thread.");
        }
        Shutdown();
        LastError.clear();
        NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
        Device = MTLCreateSystemDefaultDevice();
        if (Device == nil)
        {
            [pool drain];
            return Fail("No Metal device is available.");
        }
        Queue = [Device newCommandQueue];
        const std::uint8_t white[] = {255u, 255u, 255u, 255u};
        const bool layerAttached = Queue != nil && AttachLayer(nativeWindow);
        WhiteTexture = layerAttached
            ? CreateTexture(1, 1, MTLPixelFormatRGBA8Unorm, white)
            : nil;
        const bool initialized = layerAttached && WhiteTexture != nil &&
            CreatePipeline() && CreateFontTexture() && Resize(width, height);
        [pool drain];
        if (!initialized)
        {
            const std::string error = LastError.empty()
                ? "Metal editor initialization failed."
                : LastError;
            Shutdown();
            LastError = error;
            return false;
        }
        LastError.clear();
        return true;
    }

    void ReleaseFrameResources()
    {
        for (id<MTLBuffer> buffer : FrameBuffers)
        {
            ReleaseObject(buffer);
        }
        FrameBuffers.clear();
        Encoder = nil;
        CommandBuffer = nil;
        Drawable = nil;
        if (FramePool != nil)
        {
            [FramePool drain];
            FramePool = nil;
        }
    }

    void AbortFrame()
    {
        if (Encoder != nil)
        {
            [Encoder endEncoding];
            Encoder = nil;
        }
        ReleaseFrameResources();
    }

    void Shutdown() noexcept
    {
        AbortFrame();
        ReleaseObject(ViewportTexture);
        ReleaseObject(FontTexture);
        ReleaseObject(WhiteTexture);
        ReleaseObject(Sampler);
        ReleaseObject(Pipeline);
        ReleaseObject(ViewportPipeline);
        ReleaseObject(Queue);
        if (InstalledLayer && View != nil && View.layer == Layer)
        {
            View.layer = PreviousLayer;
            View.wantsLayer = PreviousWantsLayer;
        }
        ReleaseObject(Layer);
        ReleaseObject(PreviousLayer);
        ReleaseObject(Device);
        ReleaseObject(View);
        ViewportTexture = nil;
        FontTexture = nil;
        WhiteTexture = nil;
        Sampler = nil;
        Pipeline = nil;
        ViewportPipeline = nil;
        Queue = nil;
        Layer = nil;
        PreviousLayer = nil;
        Device = nil;
        View = nil;
        InstalledLayer = false;
        PreviousWantsLayer = false;
        Width = 1;
        Height = 1;
        ViewportWidth = 0;
        ViewportHeight = 0;
        UploadedRevision = std::numeric_limits<std::uint64_t>::max();
    }

    bool Resize(int width, int height)
    {
        if (Layer == nil || width <= 0 || height <= 0)
        {
            return Fail("Metal editor resize requires an initialized layer and non-zero dimensions.");
        }
        if (View != nil && ![NSThread isMainThread])
        {
            return Fail("Metal editor layer resize must run on the AppKit main thread.");
        }
        Width = width;
        Height = height;
        CGFloat scale = Layer.contentsScale;
        if (View != nil)
        {
            NSWindow* window = View.window;
            if (window != nil)
            {
                scale = window.backingScaleFactor;
            }
            else if (View.layer != nil && View.layer.contentsScale > 0.0)
            {
                scale = View.layer.contentsScale;
            }
            [CATransaction begin];
            [CATransaction setDisableActions:YES];
            Layer.frame = View.bounds;
            [CATransaction commit];
        }
        if (scale <= 0.0)
        {
            scale = 1.0;
        }
        Layer.contentsScale = scale;
        Layer.drawableSize = CGSizeMake(
            std::max<CGFloat>(1.0, std::round(static_cast<CGFloat>(width) * scale)),
            std::max<CGFloat>(1.0, std::round(static_cast<CGFloat>(height) * scale)));
        LastError.clear();
        return true;
    }

    bool UploadViewport(const Image& image, std::uint64_t revision)
    {
        if (!image.IsValid())
        {
            return false;
        }
        if(image.GpuColor)
        {
            if(image.GpuColor->Backend!=Renderer::RHIBackend::Metal) return Fail("Incompatible GPU viewport backend");
            id<MTLTexture> texture=(id<MTLTexture>)image.GpuColor->NativeTexture;
            if(texture.device!=Device)return Fail("GPU viewport belongs to another Metal device");
            ReleaseObject(ViewportTexture);ViewportTexture=[texture retain];
            ViewportIsGpu=true;ViewportWidth=image.Width;ViewportHeight=image.Height;UploadedRevision=revision;
            const auto lease=image.GpuColor;
            [CommandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completed) { (void)completed; (void)lease.get(); }];
            return true;
        }
        if (ViewportIsGpu || ViewportTexture == nil || ViewportWidth != image.Width || ViewportHeight != image.Height)
        {
            ReleaseObject(ViewportTexture);
            ViewportTexture = CreateTexture(
                static_cast<int>(image.Width),
                static_cast<int>(image.Height),
                MTLPixelFormatRGBA8Unorm,
                image.PixelsRGBA8.data());
            ViewportIsGpu=false;ViewportWidth = image.Width;
            ViewportHeight = image.Height;
            UploadedRevision = revision;
            return ViewportTexture != nil;
        }
        if (UploadedRevision != revision)
        {
            const MTLRegion region = MTLRegionMake2D(0, 0, image.Width, image.Height);
            [ViewportTexture
                replaceRegion:region
                mipmapLevel:0
                withBytes:image.PixelsRGBA8.data()
                bytesPerRow:image.Width * 4u];
            UploadedRevision = revision;
        }
        return true;
    }

    void DrawVertices(
        const MetalVertex* vertices,
        std::size_t count,
        id<MTLTexture> texture)
    {
        if (Encoder == nil || vertices == nullptr || count == 0 || texture == nil)
        {
            return;
        }
        id<MTLBuffer> buffer = [Device
            newBufferWithBytes:vertices
            length:count * sizeof(MetalVertex)
            options:MTLResourceStorageModeShared];
        if (buffer == nil)
        {
            return;
        }
        FrameBuffers.push_back(buffer);
        [Encoder setVertexBuffer:buffer offset:0 atIndex:0];
        [Encoder setFragmentTexture:texture atIndex:0];
        [Encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:count];
    }

    [[nodiscard]] float ToScreenY(float y) const noexcept
    {
        return static_cast<float>(Height) - y;
    }

    void DrawRect(float x, float y, float width, float height, std::uint32_t color)
    {
        const float left = x;
        const float right = x + width;
        const float top = ToScreenY(y + height);
        const float bottom = ToScreenY(y);
        const MetalVertex vertices[] = {
            {{left, top}, {0.0f, 0.0f}, color},
            {{right, top}, {1.0f, 0.0f}, color},
            {{right, bottom}, {1.0f, 1.0f}, color},
            {{left, top}, {0.0f, 0.0f}, color},
            {{right, bottom}, {1.0f, 1.0f}, color},
            {{left, bottom}, {0.0f, 1.0f}, color}};
        DrawVertices(vertices, 6, WhiteTexture);
    }

    void DrawRoundedRect(
        float x,
        float y,
        float width,
        float height,
        float radius,
        std::uint32_t color)
    {
        radius = std::min(radius, std::min(width, height) * 0.5f);
        if (radius <= 0.5f)
        {
            DrawRect(x, y, width, height, color);
            return;
        }
        constexpr int Segments = 6;
        const float left = x;
        const float right = x + width;
        const float top = ToScreenY(y + height);
        const float bottom = ToScreenY(y);
        const float centers[4][2] = {
            {right - radius, top + radius},
            {right - radius, bottom - radius},
            {left + radius, bottom - radius},
            {left + radius, top + radius}};
        const float starts[4] = {-Pi * 0.5f, 0.0f, Pi * 0.5f, Pi};
        std::vector<MetalVertex> ring;
        ring.reserve(4u * (Segments + 1u));
        for (int corner = 0; corner < 4; ++corner)
        {
            for (int segment = 0; segment <= Segments; ++segment)
            {
                const float angle = starts[corner] + Pi * 0.5f *
                    static_cast<float>(segment) / Segments;
                ring.push_back({{
                    centers[corner][0] + std::cos(angle) * radius,
                    centers[corner][1] + std::sin(angle) * radius},
                    {0.5f, 0.5f}, color});
            }
        }
        const MetalVertex center = {
            {(left + right) * 0.5f, (top + bottom) * 0.5f},
            {0.5f, 0.5f},
            color};
        std::vector<MetalVertex> triangles;
        triangles.reserve(ring.size() * 3u);
        for (std::size_t index = 0; index < ring.size(); ++index)
        {
            triangles.push_back(center);
            triangles.push_back(ring[index]);
            triangles.push_back(ring[(index + 1u) % ring.size()]);
        }
        DrawVertices(triangles.data(), triangles.size(), WhiteTexture);
    }

    void DrawLine(
        float x0,
        float y0,
        float x1,
        float y1,
        float radius,
        std::uint32_t color)
    {
        y0 = ToScreenY(y0);
        y1 = ToScreenY(y1);
        const float dx = x1 - x0;
        const float dy = y1 - y0;
        const float length = std::sqrt(dx * dx + dy * dy);
        if (length <= 1.0e-5f)
        {
            return;
        }
        const float halfWidth = std::max(1.0f, radius) * 0.5f;
        const float nx = -dy * halfWidth / length;
        const float ny = dx * halfWidth / length;
        const MetalVertex vertices[] = {
            {{x0 - nx, y0 - ny}, {0.0f, 0.0f}, color},
            {{x0 + nx, y0 + ny}, {1.0f, 0.0f}, color},
            {{x1 + nx, y1 + ny}, {1.0f, 1.0f}, color},
            {{x0 - nx, y0 - ny}, {0.0f, 0.0f}, color},
            {{x1 + nx, y1 + ny}, {1.0f, 1.0f}, color},
            {{x1 - nx, y1 - ny}, {0.0f, 1.0f}, color}};
        DrawVertices(vertices, 6, WhiteTexture);
    }

    void DrawTriangle(
        float x,
        float y,
        float width,
        float height,
        int flags,
        std::uint32_t color)
    {
        MetalVertex vertices[3] = {};
        for (MetalVertex& vertex : vertices)
        {
            vertex.Color = color;
        }
        if ((flags & IMGUI_ALIGN_RIGHT) != 0)
        {
            vertices[0].Position[0] = x;
            vertices[0].Position[1] = ToScreenY(y);
            vertices[1].Position[0] = x + width;
            vertices[1].Position[1] = ToScreenY(y + height * 0.5f);
            vertices[2].Position[0] = x;
            vertices[2].Position[1] = ToScreenY(y + height);
        }
        else
        {
            vertices[0].Position[0] = x;
            vertices[0].Position[1] = ToScreenY(y + height);
            vertices[1].Position[0] = x + width * 0.5f;
            vertices[1].Position[1] = ToScreenY(y);
            vertices[2].Position[0] = x + width;
            vertices[2].Position[1] = ToScreenY(y + height);
        }
        DrawVertices(vertices, 3, WhiteTexture);
    }

    float MeasureText(const char* text) const
    {
        if (text == nullptr)
        {
            return 0.0f;
        }
        float x = 0.0f;
        float width = 0.0f;
        for (const unsigned char* current = reinterpret_cast<const unsigned char*>(text);
             *current != 0;
             ++current)
        {
            if (*current == '\t')
            {
                for (float tab : TabStops)
                {
                    if (x < tab)
                    {
                        x = tab;
                        break;
                    }
                }
                width = std::max(width, x);
            }
            else if (*current >= 32 && *current < 128)
            {
                const Glyph& glyph = Glyphs[*current - 32];
                width = std::max(width, x + glyph.Width);
                x += glyph.Advance;
            }
        }
        return width;
    }

    void DrawText(float x, float y, int align, const char* text, std::uint32_t color)
    {
        if (text == nullptr || FontTexture == nil)
        {
            return;
        }
        if (align == IMGUI_ALIGN_CENTER) x -= MeasureText(text) * 0.5f;
        else if (align == IMGUI_ALIGN_RIGHT) x -= MeasureText(text);
        const float origin = x;
        std::vector<MetalVertex> vertices;
        vertices.reserve(std::strlen(text) * 6u);
        const float top = ToScreenY(y + static_cast<float>(FontHeight));
        for (const unsigned char* current = reinterpret_cast<const unsigned char*>(text);
             *current != 0;
             ++current)
        {
            if (*current == '\t')
            {
                const float relative = x - origin;
                for (float tab : TabStops)
                {
                    if (relative < tab)
                    {
                        x = origin + tab;
                        break;
                    }
                }
                continue;
            }
            if (*current < 32 || *current >= 128) continue;
            const Glyph& glyph = Glyphs[*current - 32];
            const float right = x + glyph.Width;
            const float bottom = top + glyph.Height;
            const MetalVertex quad[] = {
                {{x, top}, {glyph.U0, glyph.V0}, color},
                {{right, top}, {glyph.U1, glyph.V0}, color},
                {{right, bottom}, {glyph.U1, glyph.V1}, color},
                {{x, top}, {glyph.U0, glyph.V0}, color},
                {{right, bottom}, {glyph.U1, glyph.V1}, color},
                {{x, bottom}, {glyph.U0, glyph.V1}, color}};
            vertices.insert(vertices.end(), std::begin(quad), std::end(quad));
            x += glyph.Advance;
        }
        DrawVertices(vertices.data(), vertices.size(), FontTexture);
    }

    void SetScissor(int x, int y, int width, int height)
    {
        if (Encoder == nil)
        {
            return;
        }
        const NSUInteger drawableWidth = Drawable != nil
            ? Drawable.texture.width
            : static_cast<NSUInteger>(std::max<CGFloat>(Layer.drawableSize.width, 1.0));
        const NSUInteger drawableHeight = Drawable != nil
            ? Drawable.texture.height
            : static_cast<NSUInteger>(std::max<CGFloat>(Layer.drawableSize.height, 1.0));
        const double scaleX = static_cast<double>(drawableWidth) / std::max(Width, 1);
        const double scaleY = static_cast<double>(drawableHeight) / std::max(Height, 1);
        const int logicalLeft = std::clamp(x, 0, Width);
        const int logicalTop = std::clamp(Height - (y + height), 0, Height);
        const int logicalRight = std::clamp(x + width, logicalLeft, Width);
        const int logicalBottom = std::clamp(Height - y, logicalTop, Height);
        NSUInteger left = static_cast<NSUInteger>(std::floor(logicalLeft * scaleX));
        NSUInteger top = static_cast<NSUInteger>(std::floor(logicalTop * scaleY));
        NSUInteger right = static_cast<NSUInteger>(std::ceil(logicalRight * scaleX));
        NSUInteger bottom = static_cast<NSUInteger>(std::ceil(logicalBottom * scaleY));
        left = std::min(left, drawableWidth - 1u);
        top = std::min(top, drawableHeight - 1u);
        right = std::clamp(right, left + 1u, drawableWidth);
        bottom = std::clamp(bottom, top + 1u, drawableHeight);
        const MTLScissorRect scissor = {left, top, right - left, bottom - top};
        [Encoder setScissorRect:scissor];
    }

    void DrawImgui()
    {
        const imguiGfxCmd* commands = imguiGetRenderQueue();
        const int commandCount = imguiGetRenderQueueSize();
        for (int commandIndex = 0; commandIndex < commandCount; ++commandIndex)
        {
            const imguiGfxCmd& command = commands[commandIndex];
            switch (command.type)
            {
            case IMGUI_GFXCMD_RECT:
            {
                const float x = command.rect.x * ImguiScale + 0.5f;
                const float y = command.rect.y * ImguiScale + 0.5f;
                const float width = command.rect.w * ImguiScale - 1.0f;
                const float height = command.rect.h * ImguiScale - 1.0f;
                const float radius = command.rect.r * ImguiScale;
                if (radius <= 0.0f) DrawRect(x, y, width, height, command.col);
                else DrawRoundedRect(x, y, width, height, radius, command.col);
                break;
            }
            case IMGUI_GFXCMD_LINE:
                DrawLine(
                    command.line.x0 * ImguiScale,
                    command.line.y0 * ImguiScale,
                    command.line.x1 * ImguiScale,
                    command.line.y1 * ImguiScale,
                    command.line.r * ImguiScale,
                    command.col);
                break;
            case IMGUI_GFXCMD_TRIANGLE:
                DrawTriangle(
                    command.rect.x * ImguiScale + 0.5f,
                    command.rect.y * ImguiScale + 0.5f,
                    command.rect.w * ImguiScale - 1.0f,
                    command.rect.h * ImguiScale - 1.0f,
                    command.flags,
                    command.col);
                break;
            case IMGUI_GFXCMD_TEXT:
                DrawText(
                    static_cast<float>(command.text.x),
                    static_cast<float>(command.text.y),
                    command.text.align,
                    command.text.text,
                    command.col);
                break;
            case IMGUI_GFXCMD_SCISSOR:
                if (command.flags != 0)
                {
                    SetScissor(command.rect.x, command.rect.y, command.rect.w, command.rect.h);
                }
                else
                {
                    SetScissor(0, 0, Width, Height);
                }
                break;
            default:
                break;
            }
        }
        SetScissor(0, 0, Width, Height);
    }

    bool BeginFrame(
        const Image& image,
        std::uint64_t revision,
        int viewportX,
        int viewportY,
        int viewportWidth,
        int viewportHeight)
    {
        if (Device == nil || Queue == nil || Layer == nil || Pipeline == nil || ViewportPipeline == nil ||
            Sampler == nil || Encoder != nil || CommandBuffer != nil || Drawable != nil)
        {
            return Fail("Metal editor frame state is invalid.");
        }
        FramePool = [[NSAutoreleasePool alloc] init];
        Drawable = [Layer nextDrawable];
        CommandBuffer = [Queue commandBuffer];
        if (Drawable == nil || CommandBuffer == nil)
        {
            ReleaseFrameResources();
            return Fail("Metal editor could not acquire a drawable.");
        }
        MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = Drawable.texture;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0.035, 0.043, 0.059, 1.0);
        Encoder = [CommandBuffer renderCommandEncoderWithDescriptor:pass];
        if (Encoder == nil)
        {
            AbortFrame();
            return Fail("Metal editor could not create a render encoder.");
        }
        const NSUInteger drawableWidth = Drawable.texture.width;
        const NSUInteger drawableHeight = Drawable.texture.height;
        if (drawableWidth == 0u || drawableHeight == 0u)
        {
            AbortFrame();
            return Fail("Metal editor acquired an empty drawable.");
        }
        const MTLViewport viewport = {
            0.0,
            0.0,
            static_cast<double>(drawableWidth),
            static_cast<double>(drawableHeight),
            0.0,
            1.0};
        [Encoder setViewport:viewport];
        const float screenSize[2] = {static_cast<float>(Width), static_cast<float>(Height)};
        [Encoder setRenderPipelineState:ViewportPipeline];
        [Encoder setVertexBytes:screenSize length:sizeof(screenSize) atIndex:1];
        [Encoder setFragmentSamplerState:Sampler atIndex:0];
        SetScissor(0, 0, Width, Height);
        if (image.IsValid())
        {
            if (!UploadViewport(image, revision))
            {
                AbortFrame();
                return Fail("Metal editor could not upload the viewport image.");
            }
            const float x = static_cast<float>(viewportX);
            const float y = static_cast<float>(viewportY);
            const float width = static_cast<float>(viewportWidth);
            const float height = static_cast<float>(viewportHeight);
            const MetalVertex vertices[] = {
                {{x, y}, {0.0f, 0.0f}, 0xffffffffu},
                {{x + width, y}, {1.0f, 0.0f}, 0xffffffffu},
                {{x + width, y + height}, {1.0f, 1.0f}, 0xffffffffu},
                {{x, y}, {0.0f, 0.0f}, 0xffffffffu},
                {{x + width, y + height}, {1.0f, 1.0f}, 0xffffffffu},
                {{x, y + height}, {0.0f, 1.0f}, 0xffffffffu}};
            DrawVertices(vertices, 6, ViewportTexture);
        }
        [Encoder setRenderPipelineState:Pipeline];
        LastError.clear();
        return true;
    }

    bool EndFrame()
    {
        if (Encoder == nil || CommandBuffer == nil || Drawable == nil)
        {
            AbortFrame();
            return Fail("Metal editor has no frame to present.");
        }
        [Encoder endEncoding];
        Encoder = nil;
        [CommandBuffer presentDrawable:Drawable];
        [CommandBuffer commit];
        ReleaseFrameResources();
        LastError.clear();
        return true;
    }

    NSView* View = nil;
    CAMetalLayer* Layer = nil;
    CALayer* PreviousLayer = nil;
    id<MTLDevice> Device = nil;
    bool ViewportIsGpu=false;
    id<MTLCommandQueue> Queue = nil;
    id<MTLRenderPipelineState> Pipeline = nil;
    id<MTLRenderPipelineState> ViewportPipeline = nil;
    id<MTLSamplerState> Sampler = nil;
    id<MTLTexture> WhiteTexture = nil;
    id<MTLTexture> FontTexture = nil;
    id<MTLTexture> ViewportTexture = nil;
    id<CAMetalDrawable> Drawable = nil;
    id<MTLCommandBuffer> CommandBuffer = nil;
    id<MTLRenderCommandEncoder> Encoder = nil;
    NSAutoreleasePool* FramePool = nil;
    std::vector<id<MTLBuffer>> FrameBuffers;
    Glyph Glyphs[96] = {};
    int FontHeight = 16;
    int Width = 1;
    int Height = 1;
    bool InstalledLayer = false;
    bool PreviousWantsLayer = false;
    std::uint32_t ViewportWidth = 0;
    std::uint32_t ViewportHeight = 0;
    std::uint64_t UploadedRevision = std::numeric_limits<std::uint64_t>::max();
    std::string LastError;
};

EditorPresenterMetal::EditorPresenterMetal()
    : Impl_(std::make_unique<Impl>())
{
}

EditorPresenterMetal::~EditorPresenterMetal() = default;

bool EditorPresenterMetal::Initialize(void* nativeWindow, int width, int height)
{
    return Impl_->Initialize(nativeWindow, width, height);
}

void EditorPresenterMetal::Shutdown() noexcept
{
    Impl_->Shutdown();
}

bool EditorPresenterMetal::Resize(int width, int height)
{
    return Impl_->Resize(width, height);
}

bool EditorPresenterMetal::BeginFrame(
    const Image& image,
    std::uint64_t revision,
    int viewportX,
    int viewportY,
    int viewportWidth,
    int viewportHeight)
{
    return Impl_->BeginFrame(
        image,
        revision,
        viewportX,
        viewportY,
        viewportWidth,
        viewportHeight);
}

void EditorPresenterMetal::DrawImgui()
{
    Impl_->DrawImgui();
}

bool EditorPresenterMetal::EndFrame()
{
    return Impl_->EndFrame();
}

bool EditorPresenterMetal::IsInitialized() const noexcept
{
    return Impl_->Device != nil && Impl_->Queue != nil && Impl_->Layer != nil &&
        Impl_->Pipeline != nil && Impl_->ViewportPipeline != nil && Impl_->Sampler != nil &&
        Impl_->WhiteTexture != nil && Impl_->FontTexture != nil;
}

const std::string& EditorPresenterMetal::LastError() const noexcept
{
    return Impl_->LastError;
}

} // namespace Evans::Editor

#endif
