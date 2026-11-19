#include "Editor/Editor.hpp"
#include "Editor/ContentRoot.hpp"
#include "Editor/EditorPresenter.h"
#include "Editor/Imgui.h"
#include "Editor/ModuleAdapters.h"
#include "Editor/TechniqueControls.h"

#if defined(__APPLE__)

#import <Cocoa/Cocoa.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace
{
using Evans::Editor::BackendDescriptor;
using Evans::Editor::DemoDescriptor;
using Evans::Editor::DemoId;
using Evans::Editor::EditorCore;
using Evans::Editor::EditorPresentationAPI;
using Evans::Editor::FillMode;
using Evans::Editor::IEditorPresenter;
using Evans::Editor::Rect;

constexpr int EditorPanelX = 10;
constexpr int EditorPanelY = 10;
constexpr int EditorPanelWidth = 360;
constexpr int EditorPanelGap = 10;
constexpr int EditorViewportMargin = 10;
constexpr int InitialWidth = 1280;
constexpr int InitialHeight = 720;

Rect EditorViewportBounds(int clientWidth, int clientHeight)
{
    const float left = static_cast<float>(
        EditorPanelX + EditorPanelWidth + EditorPanelGap);
    const float top = static_cast<float>(EditorViewportMargin);
    return {
        left,
        top,
        std::max(1.0f, static_cast<float>(clientWidth) - left - EditorViewportMargin),
        std::max(1.0f, static_cast<float>(clientHeight) - top - EditorViewportMargin)};
}

int BackendSelectionIndex(
    const std::vector<BackendDescriptor>& backends,
    const std::string& selectedBackend)
{
    for (std::size_t index = 0; index < backends.size(); ++index)
    {
        if (backends[index].Id == selectedBackend)
        {
            return static_cast<int>(index);
        }
    }
    return 0;
}

int DemoSelectionIndex(DemoId selectedDemo)
{
    const std::vector<DemoDescriptor>& demos =
        Evans::Editor::DemoCatalog::Descriptors();
    for (std::size_t index = 0; index < demos.size(); ++index)
    {
        if (demos[index].Id == selectedDemo)
        {
            return static_cast<int>(index);
        }
    }
    return 0;
}

NSString* StringFromUtf8(const std::string& text)
{
    NSString* converted = [NSString stringWithUTF8String:text.c_str()];
    return converted != nil ? converted : @"Unknown Evans error";
}

} // namespace

@interface EvansEditorView : NSView
{
@private
    std::unique_ptr<EditorCore> Editor_;
    std::unique_ptr<IEditorPresenter> Presenter_;
    std::chrono::steady_clock::time_point LastTick_;
    std::string PresentationError_;
    int ClientWidth_;
    int ClientHeight_;
    int MouseX_;
    int MouseY_;
    int MouseWheel_;
    int PanelScroll_;
    double WheelRemainder_;
    unsigned char MouseButtons_;
    BOOL CameraRotateCapture_;
    BOOL LightDragCapture_;
    BOOL Running_;
}

- (BOOL)startEditor;
- (void)stopEditor;
- (NSString*)startupErrorText;
- (void)renderNow;
- (void)resizeEditor;
- (void)drawEditorImgui;
- (void)frameTimer:(NSTimer*)timer;
- (void)updateMouseFromEvent:(NSEvent*)event;

@end

@implementation EvansEditorView

- (instancetype)initWithFrame:(NSRect)frameRect
{
    self = [super initWithFrame:frameRect];
    if (self != nil)
    {
        LastTick_ = std::chrono::steady_clock::now();
        ClientWidth_ = std::max(1, static_cast<int>(std::lround(NSWidth(frameRect))));
        ClientHeight_ = std::max(1, static_cast<int>(std::lround(NSHeight(frameRect))));
        MouseX_ = 0;
        MouseY_ = 0;
        MouseWheel_ = 0;
        PanelScroll_ = 0;
        WheelRemainder_ = 0.0;
        MouseButtons_ = 0;
        CameraRotateCapture_ = NO;
    LightDragCapture_=NO;
        Running_ = NO;
    }
    return self;
}

- (void)dealloc
{
    [self stopEditor];
    [super dealloc];
}

- (BOOL)acceptsFirstResponder
{
    return YES;
}

- (BOOL)acceptsFirstMouse:(NSEvent*)event
{
    (void)event;
    return YES;
}

- (void)viewDidMoveToWindow
{
    [super viewDidMoveToWindow];
    if ([self window] != nil)
    {
        [[self window] setAcceptsMouseMovedEvents:YES];
    }
    [self resizeEditor];
}

- (void)viewDidChangeBackingProperties
{
    [super viewDidChangeBackingProperties];
    [self resizeEditor];
}

- (void)setFrameSize:(NSSize)newSize
{
    [super setFrameSize:newSize];
    [self resizeEditor];
}

- (BOOL)startEditor
{
    if (Running_)
    {
        return YES;
    }
    if (![NSThread isMainThread])
    {
        PresentationError_ = "The macOS editor must start on the AppKit main thread.";
        return NO;
    }

    [self resizeEditor];
    try
    {
        std::unique_ptr<IEditorPresenter> presenter =
            Evans::Editor::CreateEditorPresenter(EditorPresentationAPI::Metal);
        if (presenter == nullptr)
        {
            PresentationError_ = "Metal editor presentation is unavailable.";
            return NO;
        }
        if (!presenter->Initialize(self, ClientWidth_, ClientHeight_))
        {
            PresentationError_ = presenter->LastError();
            return NO;
        }

        auto editor = std::make_unique<EditorCore>(Evans::Editor::ResolveContentRoot());
        Evans::Editor::RegisterModuleAdapters(*editor);
        editor->SetPickHandler(Evans::Editor::CreatePickHandler());

        // Metal is the first usable realtime backend on macOS. Software backend
        // selections continue to use Metal only as their presentation surface.
        editor->SelectBackend("renderer.metal");
        const Rect viewport = EditorViewportBounds(ClientWidth_, ClientHeight_);
        editor->ResizeViewport(
            static_cast<std::uint32_t>(std::max(viewport.Width, 1.0f)),
            static_cast<std::uint32_t>(std::max(viewport.Height, 1.0f)));

        Presenter_ = std::move(presenter);
        Editor_ = std::move(editor);
        PresentationError_.clear();
        LastTick_ = std::chrono::steady_clock::now();
        Running_ = YES;

        Editor_->SetContinuousRendering(true);
        Editor_->SetGpuPresentation(true);
        [self performSelector:@selector(frameTimer:) withObject:nil afterDelay:0.0 inModes:@[NSRunLoopCommonModes]];
        return YES;
    }
    catch (const std::exception& exception)
    {
        PresentationError_ = exception.what();
    }
    catch (...)
    {
        PresentationError_ = "Unknown macOS editor initialization failure.";
    }
    Presenter_.reset();
    Editor_.reset();
    return NO;
}

- (void)stopEditor
{
    Running_ = NO;
    if (Editor_ != nullptr)
    {
        Editor_->ClearCameraInput();
    }
    [NSObject cancelPreviousPerformRequestsWithTarget:self selector:@selector(frameTimer:) object:nil];

    if (Presenter_ != nullptr)
    {
        Presenter_->Shutdown();
        Presenter_.reset();
    }
    Editor_.reset();
}

- (NSString*)startupErrorText
{
    return StringFromUtf8(PresentationError_);
}

- (void)resizeEditor
{
    const NSRect bounds = [self bounds];
    ClientWidth_ = std::max(1, static_cast<int>(std::lround(NSWidth(bounds))));
    ClientHeight_ = std::max(1, static_cast<int>(std::lround(NSHeight(bounds))));
    MouseX_ = std::clamp(MouseX_, 0, ClientWidth_ - 1);
    MouseY_ = std::clamp(MouseY_, 0, ClientHeight_ - 1);
    if (Presenter_ != nullptr && Presenter_->IsInitialized() &&
        !Presenter_->Resize(ClientWidth_, ClientHeight_))
    {
        PresentationError_ = Presenter_->LastError();
    }
    if (Editor_ != nullptr)
    {
        const Rect viewport = EditorViewportBounds(ClientWidth_, ClientHeight_);
        Editor_->ResizeViewport(
            static_cast<std::uint32_t>(std::max(viewport.Width, 1.0f)),
            static_cast<std::uint32_t>(std::max(viewport.Height, 1.0f)));
    }
}

- (void)drawEditorImgui
{
    if (Editor_ == nullptr)
    {
        return;
    }

    const int panelHeight = std::max(120, ClientHeight_ - EditorPanelY * 2);
    imguiBeginScrollArea(
        "Evans",
        EditorPanelX,
        EditorPanelY,
        EditorPanelWidth,
        panelHeight,
        &PanelScroll_);

    const std::vector<BackendDescriptor> backends = Editor_->Backends();
    std::vector<const char*> backendLabels;
    backendLabels.reserve(backends.size());
    for (const BackendDescriptor& backend : backends)
    {
        backendLabels.push_back(backend.DisplayName.c_str());
    }

    // Keep the backend selector as the first control in the editor panel.
    imguiLabel("RHI Backend");
    int backendIndex = BackendSelectionIndex(backends, Editor_->SelectedBackend());
    if (!backendLabels.empty() && imguiCombo(
            "",
            &backendIndex,
            backendLabels.data(),
            static_cast<int>(backendLabels.size())))
    {
        Editor_->SelectBackend(backends[static_cast<std::size_t>(backendIndex)].Id);
    }

    imguiSeparatorLine();
    imguiLabel("Scene");
    const std::vector<DemoDescriptor>& demos =
        Evans::Editor::DemoCatalog::Descriptors();
    std::vector<const char*> demoLabels;
    demoLabels.reserve(demos.size());
    for (const DemoDescriptor& demo : demos)
    {
        demoLabels.push_back(demo.Name.data());
    }
    int demoIndex = DemoSelectionIndex(Editor_->SelectedDemo());
    if (!demoLabels.empty() && imguiCombo(
            "Demo",
            &demoIndex,
            demoLabels.data(),
            static_cast<int>(demoLabels.size())))
    {
        Editor_->SelectDemo(demos[static_cast<std::size_t>(demoIndex)].Id);
    }

    Evans::Editor::DrawRenderModeControls(*Editor_);
    Evans::Editor::DrawRenderBufferControls(*Editor_);
    Evans::Editor::DrawTechniqueControls(*Editor_);

    imguiSeparatorLine();
    imguiLabel("Status");
    if (Presenter_ != nullptr)
    {
        const std::string presentation = std::string("Presentation: ") +
            Evans::Editor::ToString(Presenter_->API());
        imguiValue(presentation.c_str());
    }
    if (!PresentationError_.empty())
    {
        const std::string error = "Presentation error: " + PresentationError_;
        imguiValue(error.c_str());
    }
    imguiValue(Editor_->StatusText().c_str());
    imguiEndScrollArea();
}

- (void)renderNow
{
    if (!Running_ || Editor_ == nullptr || Presenter_ == nullptr ||
        !Presenter_->IsInitialized())
    {
        return;
    }
    NSWindow* window = [self window];
    if (window == nil || ![window isVisible] || [window isMiniaturized] ||
        ([window occlusionState] & NSWindowOcclusionStateVisible) == 0)
    {
        return;
    }

    const Rect viewport = EditorViewportBounds(ClientWidth_, ClientHeight_);
    if (!Presenter_->BeginFrame(
            Editor_->ViewportImage(),
            Editor_->ViewportRevision(),
            static_cast<int>(viewport.X),
            static_cast<int>(viewport.Y),
            static_cast<int>(viewport.Width),
            static_cast<int>(viewport.Height)))
    {
        PresentationError_ = Presenter_->LastError();
        return;
    }

    // AppKit events use a bottom-left origin. MouseY_ is kept in the
    // top-left viewport convention and converted back for Riemann ImGui.
    imguiBeginFrame(
        MouseX_,
        ClientHeight_ - 1 - MouseY_,
        MouseButtons_,
        MouseWheel_);
    MouseWheel_ = 0;
    [self drawEditorImgui];
    Evans::Editor::DrawViewportStatus(*Editor_,viewport,ClientHeight_);
    imguiEndFrame();
    Presenter_->DrawImgui();
    if (!Presenter_->EndFrame())
    {
        PresentationError_ = Presenter_->LastError();
        return;
    }
    PresentationError_.clear();
}

- (void)frameTimer:(NSTimer*)timer
{
    (void)timer;
    if (!Running_ || Editor_ == nullptr)
    {
        return;
    }

    NSAutoreleasePool* framePool = [[NSAutoreleasePool alloc] init];
    const auto now = std::chrono::steady_clock::now();
    const float deltaSeconds = std::clamp(
        std::chrono::duration<float>(now - LastTick_).count(),
        0.0f,
        0.1f);
    LastTick_ = now;

    // Tick only polls futures and advances realtime scenes; it never waits for
    // a software rasterizer. Completed images are uploaded during presentation.
    Editor_->Tick(deltaSeconds);
    [self renderNow];
    [framePool drain];
    if (Running_) [self performSelector:@selector(frameTimer:) withObject:nil afterDelay:0.0 inModes:@[NSRunLoopCommonModes]];
}

- (void)updateMouseFromEvent:(NSEvent*)event
{
    const NSPoint point = [self convertPoint:[event locationInWindow] fromView:nil];
    MouseX_ = std::clamp(
        static_cast<int>(std::lround(point.x)),
        0,
        std::max(ClientWidth_ - 1, 0));
    MouseY_ = std::clamp(
        static_cast<int>(std::lround(ClientHeight_ - 1.0 - point.y)),
        0,
        std::max(ClientHeight_ - 1, 0));
}

- (void)mouseMoved:(NSEvent*)event
{
    [self updateMouseFromEvent:event];
    if (Editor_ != nullptr) Editor_->HandleDemoPointer(float(MouseX_),float(MouseY_),
        EditorViewportBounds(ClientWidth_,ClientHeight_).Contains(float(MouseX_),float(MouseY_)));
}

- (void)mouseDragged:(NSEvent*)event
{
    [self mouseMoved:event];
    if (Editor_ && LightDragCapture_) Editor_->HandleLightPointer(float(MouseX_),float(MouseY_),EditorViewportBounds(ClientWidth_,ClientHeight_),true);
}

- (void)rightMouseDragged:(NSEvent*)event
{
    [self updateMouseFromEvent:event];
    if (Editor_ != nullptr)
    {
        Editor_->HandleCameraPointer(
            static_cast<float>(MouseX_),
            static_cast<float>(MouseY_),
            CameraRotateCapture_ == YES);
    }
}

- (void)mouseDown:(NSEvent*)event
{
    [self updateMouseFromEvent:event];
    MouseButtons_ |= IMGUI_MBUT_LEFT;
    if (Editor_ != nullptr)
    {
        const Rect viewport = EditorViewportBounds(ClientWidth_, ClientHeight_);
        LightDragCapture_=Editor_->HandleLightPointer(float(MouseX_),float(MouseY_),viewport,true);
        if (!LightDragCapture_) Editor_->HandleViewportPointer(
            static_cast<float>(MouseX_),
            static_cast<float>(MouseY_),
            viewport);
    }
    [self renderNow];
}

- (void)mouseUp:(NSEvent*)event
{
    [self updateMouseFromEvent:event];
    if (Editor_) Editor_->HandleLightPointer(float(MouseX_),float(MouseY_),EditorViewportBounds(ClientWidth_,ClientHeight_),false);
    LightDragCapture_=NO;
    MouseButtons_ &= static_cast<unsigned char>(~IMGUI_MBUT_LEFT);
    [self renderNow];
}

- (void)rightMouseDown:(NSEvent*)event
{
    [self updateMouseFromEvent:event];
    MouseButtons_ |= IMGUI_MBUT_RIGHT;
    const Rect viewport = EditorViewportBounds(ClientWidth_, ClientHeight_);
    CameraRotateCapture_ = viewport.Contains(
        static_cast<float>(MouseX_),
        static_cast<float>(MouseY_));
    if (Editor_ != nullptr)
    {
        Editor_->HandleCameraPointer(
            static_cast<float>(MouseX_),
            static_cast<float>(MouseY_),
            CameraRotateCapture_ == YES);
    }
    [[self window] makeFirstResponder:self];
    [self renderNow];
}

- (void)rightMouseUp:(NSEvent*)event
{
    [self updateMouseFromEvent:event];
    MouseButtons_ &= static_cast<unsigned char>(~IMGUI_MBUT_RIGHT);
    CameraRotateCapture_ = NO;
    LightDragCapture_=NO;
    if (Editor_ != nullptr)
    {
        Editor_->HandleCameraPointer(
            static_cast<float>(MouseX_),
            static_cast<float>(MouseY_),
            false);
    }
    [self renderNow];
}

- (void)keyDown:(NSEvent*)event
{
    if (Editor_ == nullptr)
    {
        [super keyDown:event];
        return;
    }
    NSString* characters = [event charactersIgnoringModifiers];
    if ([characters length] != 1)
    {
        [super keyDown:event];
        return;
    }
    const unichar key = [characters characterAtIndex:0];
    if (key == 'w' || key == 'W' || key == 'a' || key == 'A' ||
        key == 's' || key == 'S' || key == 'd' || key == 'D')
    {
        Editor_->SetCameraMovementKey(static_cast<char>(key), true);
        return;
    }
    [super keyDown:event];
}

- (void)keyUp:(NSEvent*)event
{
    if (Editor_ == nullptr)
    {
        [super keyUp:event];
        return;
    }
    NSString* characters = [event charactersIgnoringModifiers];
    if ([characters length] != 1)
    {
        [super keyUp:event];
        return;
    }
    const unichar key = [characters characterAtIndex:0];
    if (key == 'w' || key == 'W' || key == 'a' || key == 'A' ||
        key == 's' || key == 'S' || key == 'd' || key == 'D')
    {
        Editor_->SetCameraMovementKey(static_cast<char>(key), false);
        return;
    }
    [super keyUp:event];
}

- (BOOL)resignFirstResponder
{
    CameraRotateCapture_ = NO;
    LightDragCapture_=NO;
    MouseButtons_ &= static_cast<unsigned char>(~IMGUI_MBUT_RIGHT);
    if (Editor_ != nullptr)
    {
        Editor_->ClearCameraInput();
    }
    return [super resignFirstResponder];
}

- (void)scrollWheel:(NSEvent*)event
{
    [self updateMouseFromEvent:event];
    WheelRemainder_ += [event scrollingDeltaY];
    const int wheelSteps = static_cast<int>(WheelRemainder_);
    if (wheelSteps != 0)
    {
        MouseWheel_ = std::clamp(MouseWheel_ + wheelSteps, -64, 64);
        WheelRemainder_ -= wheelSteps;
        [self renderNow];
    }
}

@end

@interface EvansApplicationDelegate : NSObject <NSApplicationDelegate>
{
@private
    NSWindow* Window_;
    EvansEditorView* EditorView_;
    int ExitCode_;
}

- (int)exitCode;

@end


@implementation EvansApplicationDelegate

- (instancetype)init
{
    self = [super init];
    if (self != nil)
    {
        Window_ = nil;
        EditorView_ = nil;
        ExitCode_ = 0;
    }
    return self;
}

- (void)dealloc
{
    if (EditorView_ != nil)
    {
        [EditorView_ stopEditor];
    }
    [EditorView_ release];
    [Window_ release];
    [super dealloc];
}

- (void)applicationDidFinishLaunching:(NSNotification*)notification
{
    (void)notification;
    const NSWindowStyleMask style =
        NSWindowStyleMaskTitled |
        NSWindowStyleMaskClosable |
        NSWindowStyleMaskMiniaturizable |
        NSWindowStyleMaskResizable;
    Window_ = [[NSWindow alloc]
        initWithContentRect:NSMakeRect(0.0, 0.0, InitialWidth, InitialHeight)
        styleMask:style
        backing:NSBackingStoreBuffered
        defer:NO];
    [Window_ setTitle:@"Evans Editor"];
    [Window_ setContentMinSize:NSMakeSize(720.0, 480.0)];
    [Window_ setReleasedWhenClosed:NO];

    EditorView_ = [[EvansEditorView alloc]
        initWithFrame:[[Window_ contentView] bounds]];
    [EditorView_ setAutoresizingMask:NSViewWidthSizable | NSViewHeightSizable];
    [Window_ setContentView:EditorView_];
    [Window_ makeFirstResponder:EditorView_];

    if (![EditorView_ startEditor])
    {
        ExitCode_ = 2;
        NSAlert* alert = [[NSAlert alloc] init];
        [alert setMessageText:@"Evans Editor could not start"];
        [alert setInformativeText:[EditorView_ startupErrorText]];
        [alert addButtonWithTitle:@"Quit"];
        [alert runModal];
        [alert release];
        [NSApp terminate:nil];
        return;
    }

    [Window_ center];
    [Window_ makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
    [EditorView_ renderNow];
}

- (void)applicationWillTerminate:(NSNotification*)notification
{
    (void)notification;
    [EditorView_ stopEditor];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication*)sender
{
    (void)sender;
    return YES;
}

- (BOOL)applicationShouldHandleReopen:(NSApplication*)sender
                    hasVisibleWindows:(BOOL)hasVisibleWindows
{
    (void)sender;
    if (!hasVisibleWindows && Window_ != nil)
    {
        [Window_ makeKeyAndOrderFront:nil];
    }
    return YES;
}

- (int)exitCode
{
    return ExitCode_;
}

@end

namespace
{

void InstallApplicationMenu(NSApplication* application)
{
    NSMenu* menuBar = [[NSMenu alloc] initWithTitle:@""];
    NSMenuItem* applicationItem = [[NSMenuItem alloc]
        initWithTitle:@""
        action:nil
        keyEquivalent:@""];
    [menuBar addItem:applicationItem];

    NSMenu* applicationMenu = [[NSMenu alloc] initWithTitle:@"Evans"];
    NSMenuItem* quitItem = [[NSMenuItem alloc]
        initWithTitle:@"Quit Evans"
        action:@selector(terminate:)
        keyEquivalent:@"q"];
    [applicationMenu addItem:quitItem];
    [applicationItem setSubmenu:applicationMenu];
    [application setMainMenu:menuBar];

    [quitItem release];
    [applicationMenu release];
    [applicationItem release];
    [menuBar release];
}

} // namespace

int RunMacEditor()
{
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    NSApplication* application = [NSApplication sharedApplication];
    [application setActivationPolicy:NSApplicationActivationPolicyRegular];
    InstallApplicationMenu(application);

    EvansApplicationDelegate* delegate =
        [[EvansApplicationDelegate alloc] init];
    [application setDelegate:delegate];
    [application run];
    const int exitCode = [delegate exitCode];
    [application setDelegate:nil];
    [delegate release];
    [pool drain];
    return exitCode;
}

#endif
