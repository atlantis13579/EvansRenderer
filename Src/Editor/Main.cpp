#include "Editor/AssetCooker.h"

#if defined(_WIN32)
#include "Editor/ModuleAdapters.h"
#include "Editor/ContentRoot.hpp"
#include "Editor/EvansEditor.hpp"
#include "Editor/TechniqueControls.h"
#include "Editor/EditorPresenter.h"
#include "Editor/Imgui.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <windowsx.h>
#include <shellapi.h>
#include <cstdio>
#include <fcntl.h>
#include <io.h>

namespace
{
using Evans::Editor::EditorCore;
using Evans::Editor::Rect;

void BindCommandStream(DWORD kind,FILE* target)
{
    const auto handle=GetStdHandle(kind);
    if (!handle || handle==INVALID_HANDLE_VALUE) return;
    HANDLE duplicate=nullptr;
    if (!DuplicateHandle(GetCurrentProcess(),handle,GetCurrentProcess(),&duplicate,0,FALSE,DUPLICATE_SAME_ACCESS)) return;
    const int descriptor=_open_osfhandle(reinterpret_cast<intptr_t>(duplicate),_O_TEXT);
    if (descriptor<0) { CloseHandle(duplicate); return; }
    FILE* reopened=nullptr;
    if (freopen_s(&reopened,"NUL","w",target)==0) _dup2(descriptor,_fileno(reopened));
    _close(descriptor);
}

std::unique_ptr<EditorCore> CreateEditor()
{
    auto editor = std::make_unique<EditorCore>(Evans::Editor::ResolveContentRoot());
    editor->SetPickHandler(Evans::Editor::CreatePickHandler());
    editor->SetContinuousRendering(true);
    editor->SetGpuPresentation(true);
    Evans::Editor::RegisterModuleAdapters(*editor);
    return editor;
}

constexpr wchar_t WindowClassName[] = L"EvansEditorWindow";
constexpr wchar_t SurfaceClassName[] = L"EvansPresentationSurface";
constexpr int EditorPanelX = 10;
constexpr int EditorPanelY = 10;
constexpr int EditorPanelWidth = 360;
constexpr int EditorPanelGap = 10;
constexpr int EditorViewportMargin = 10;

struct PresentationSurface
{
    HWND Window = nullptr;
    std::unique_ptr<Evans::Editor::IEditorPresenter> Presenter;
    int Width = 0;
    int Height = 0;

    void Reset()
    {
        Presenter.reset();
        if (Window) DestroyWindow(Window);
        Window = nullptr;
        Width = Height = 0;
    }

    ~PresentationSurface() { Reset(); }

    bool Resize(int width, int height)
    {
        if (Width == width && Height == height) return true;
        if (!SetWindowPos(Window, nullptr, 0, 0, width, height, SWP_NOACTIVATE | SWP_NOZORDER) ||
            !Presenter->Resize(width, height)) return false;
        Width = width;
        Height = height;
        return true;
    }
};

struct WindowState
{
    std::unique_ptr<EditorCore> Editor;
    std::array<PresentationSurface, static_cast<size_t>(Evans::Editor::EditorPresentationAPI::Vulkan) + 1> Surfaces;
    Evans::Editor::IEditorPresenter* Presenter = nullptr;
    Evans::Editor::EditorPresentationAPI LastRealtimePresentation =
        Evans::Editor::EditorPresentationAPI::DX11;
    std::string PresentationSelection;
    std::string FailedPresentationSelection;
    std::string PresentationError;
    void* NativeWindow = nullptr;
    std::chrono::steady_clock::time_point LastTick = std::chrono::steady_clock::now();
    int ClientWidth = 1280;
    int ClientHeight = 720;
    int MouseX = 0;
    int MouseY = 0;
    int MouseWheel = 0;
    int PanelScroll = 0;
    unsigned char MouseButtons = 0;
    bool CameraRotateCapture = false,LightDragCapture=false;
    bool DrawingFrame = false;
    bool ClosePending = false;
    bool Closing = false;
    bool ResizePending = false;
    bool InSizeMove = false;
};

bool IsRealtimeBackend(std::string_view backendId)
{
    return backendId.rfind("renderer.", 0) == 0;
}

bool ActivatePresenter(WindowState& state, Evans::Editor::EditorPresentationAPI api)
{
    auto& surface = state.Surfaces[static_cast<size_t>(api)];
    if (!surface.Presenter)
    {
        // Each native swap chain keeps its own HWND, including its driver pixel format.
        surface.Window = CreateWindowExW(0, SurfaceClassName, L"",
            WS_CHILD | WS_DISABLED | WS_CLIPSIBLINGS | WS_CLIPCHILDREN, 0, 0,
            state.ClientWidth, state.ClientHeight, static_cast<HWND>(state.NativeWindow),
            nullptr, GetModuleHandleW(nullptr), nullptr);
        surface.Presenter = Evans::Editor::CreateEditorPresenter(api);
        if (!surface.Window || !surface.Presenter ||
            !surface.Presenter->Initialize(surface.Window, state.ClientWidth, state.ClientHeight))
        {
            state.PresentationError = surface.Window && surface.Presenter ? surface.Presenter->LastError() :
                std::string("Cannot create editor presentation surface for ") + Evans::Editor::ToString(api);
            surface.Reset();
            return false;
        }
        surface.Width = state.ClientWidth;
        surface.Height = state.ClientHeight;
    }
    if (!surface.Resize(state.ClientWidth, state.ClientHeight))
    {
        state.PresentationError = surface.Presenter->LastError();
        surface.Reset();
        return false;
    }
    if (!SetWindowPos(surface.Window, HWND_TOP, 0, 0, 0, 0,
            SWP_NOSIZE | SWP_NOMOVE | SWP_SHOWWINDOW | SWP_NOACTIVATE))
    {
        state.PresentationError = "Cannot show the editor presentation surface";
        surface.Reset();
        return false;
    }
    if (state.Presenter && state.Presenter != surface.Presenter.get())
        ShowWindow(state.Surfaces[static_cast<size_t>(state.Presenter->API())].Window, SW_HIDE);
    state.Presenter = surface.Presenter.get();
    return true;
}

bool EnsurePresentationBackend(WindowState& state)
{
    const std::string selectedBackend = state.Editor->SelectedBackend();
    const Evans::Editor::EditorPresentationAPI desiredAPI =
        Evans::Editor::ResolvePresentationAPI(
            selectedBackend,
            state.LastRealtimePresentation);
    if (state.FailedPresentationSelection.empty() && selectedBackend == state.PresentationSelection &&
        state.Presenter != nullptr && state.Presenter->API() == desiredAPI)
    {
        return true;
    }
    if (selectedBackend == state.FailedPresentationSelection &&
        state.Presenter != nullptr)
    {
        return true;
    }
    if (state.Presenter != nullptr && state.Presenter->API() == desiredAPI)
    {
        state.PresentationSelection = selectedBackend;
        state.FailedPresentationSelection.clear();
        state.PresentationError.clear();
        if (IsRealtimeBackend(selectedBackend))
        {
            state.LastRealtimePresentation = desiredAPI;
        }
        return true;
    }

    // Keep displaying the previous frame until the selected backend publishes its first image.
    if (state.Presenter != nullptr && state.Editor->ViewportBackend() != selectedBackend)
    {
        return true;
    }
    if (ActivatePresenter(state, desiredAPI))
    {
        state.PresentationSelection = selectedBackend;
        state.FailedPresentationSelection.clear();
        state.PresentationError.clear();
        if (IsRealtimeBackend(selectedBackend)) state.LastRealtimePresentation = desiredAPI;
        return true;
    }
    state.FailedPresentationSelection = selectedBackend;
    return state.Presenter != nullptr;
}

Rect EditorViewportBounds(const WindowState& state)
{
    const float left = static_cast<float>(EditorPanelX + EditorPanelWidth + EditorPanelGap);
    const float top = static_cast<float>(EditorViewportMargin);
    return {
        left,
        top,
        std::max(1.0f, static_cast<float>(state.ClientWidth) - left - EditorViewportMargin),
        std::max(1.0f, static_cast<float>(state.ClientHeight) - top - EditorViewportMargin)};
}

int DemoSelectionIndex(Evans::Editor::DemoId demo)
{
    const std::vector<Evans::Editor::DemoDescriptor>& descriptors =
        Evans::Editor::DemoCatalog::Descriptors();
    for (std::size_t index = 0; index < descriptors.size(); ++index)
    {
        if (descriptors[index].Id == demo)
        {
            return static_cast<int>(index);
        }
    }
    return 0;
}

void DrawEditorImgui(WindowState& state)
{
    const int panelHeight = std::max(120, state.ClientHeight - EditorPanelY * 2);
    imguiBeginScrollArea(
        "Evans",
        EditorPanelX,
        EditorPanelY,
        EditorPanelWidth,
        panelHeight,
        &state.PanelScroll);

    const std::vector<Evans::Editor::DemoDescriptor>& demos =
        Evans::Editor::DemoCatalog::Descriptors();
    std::vector<const char*> demoLabels;
    demoLabels.reserve(demos.size());
    for (const Evans::Editor::DemoDescriptor& demo : demos)
    {
        demoLabels.push_back(demo.Name.data());
    }
    int demoIndex = DemoSelectionIndex(state.Editor->SelectedDemo());
    if (!demoLabels.empty() &&
        imguiCombo("Scene", &demoIndex, demoLabels.data(), static_cast<int>(demoLabels.size())))
    {
        state.Editor->SelectDemo(demos[static_cast<std::size_t>(demoIndex)].Id);
    }

    Evans::Editor::DrawTechniqueControls(*state.Editor);

    imguiSeparatorLine();
    imguiLabel("Status");
    if (state.Presenter != nullptr)
    {
        const std::string presentation = std::string("Presentation: ") +
            Evans::Editor::ToString(state.Presenter->API());
        imguiValue(presentation.c_str());
    }
    if (!state.PresentationError.empty())
    {
        const std::string error = "Presentation error: " + state.PresentationError;
        imguiValue(error.c_str());
    }
    imguiValue(state.Editor->StatusText().c_str());
    imguiEndScrollArea();
}

bool DrawEditorFrame(WindowState& state)
{
    // Native swapchain operations can synchronously dispatch window messages.
    // Never replace a presenter or begin another frame while one is recording.
    if (state.DrawingFrame)
    {
        return true;
    }
    state.DrawingFrame = true;
    struct FrameGuard
    {
        WindowState& State;
        ~FrameGuard()
        {
            State.DrawingFrame = false;
            if (State.ClosePending)
            {
                State.ClosePending = false;
                PostMessageW(static_cast<HWND>(State.NativeWindow), WM_CLOSE, 0, 0);
            }
        }
    } guard{state};
    const auto resetFailedPresenter = [&state](const std::string& message)
    {
        state.PresentationError = message;
        const auto failedSelection = std::move(state.PresentationSelection);
        state.PresentationSelection.clear();
        if (!state.Presenter) return false;
        const auto failedAPI = state.Presenter->API();
        state.Presenter = nullptr;
        state.Surfaces[static_cast<size_t>(failedAPI)].Reset();
        if (failedAPI != Evans::Editor::EditorPresentationAPI::DX11 &&
            ActivatePresenter(state, Evans::Editor::EditorPresentationAPI::DX11))
            state.FailedPresentationSelection = failedSelection;
        return false;
    };
    if (!EnsurePresentationBackend(state) || state.Presenter == nullptr ||
        !state.Presenter->IsInitialized())
    {
        return false;
    }

    if (state.ResizePending)
    {
        state.ResizePending = false;
        if (!state.Surfaces[static_cast<size_t>(state.Presenter->API())].Resize(state.ClientWidth, state.ClientHeight))
            return resetFailedPresenter(state.Presenter->LastError());
    }
    const Rect viewport = EditorViewportBounds(state);
    const auto& viewportImage=state.Editor->ViewportImage();
    const Evans::Editor::Image emptyImage;
    const bool compatible=!viewportImage.GpuColor || state.Presenter->API()==
        Evans::Editor::ResolvePresentationAPI(state.Editor->ViewportBackend(),state.LastRealtimePresentation);
    if (!state.Presenter->BeginFrame(
            compatible?viewportImage:emptyImage,
            state.Editor->ViewportRevision(),
            static_cast<int>(viewport.X),
            static_cast<int>(viewport.Y),
            static_cast<int>(viewport.Width),
            static_cast<int>(viewport.Height)))
    {
        return resetFailedPresenter(state.Presenter->LastError());
    }

    const int imguiMouseY = state.ClientHeight - 1 - state.MouseY;
    imguiBeginFrame(
        state.MouseX,
        imguiMouseY,
        state.MouseButtons,
        state.MouseWheel);
    state.MouseWheel = 0;
    DrawEditorImgui(state);
    Evans::Editor::DrawViewportStatus(*state.Editor,viewport,state.ClientHeight);
    imguiEndFrame();
    state.Presenter->DrawImgui();
    if (!state.Presenter->EndFrame())
    {
        return resetFailedPresenter(state.Presenter->LastError());
    }
    return true;
}

void RenderEditorWindow(WindowState& state)
{
    if (state.Closing || IsIconic(static_cast<HWND>(state.NativeWindow))) return;
    (void)DrawEditorFrame(state);
}

void ResizeEditor(WindowState& state, int width, int height)
{
    state.ClientWidth = std::max(width, 1);
    state.ClientHeight = std::max(height, 1);
    // Resize GPU resources at the next frame boundary, including during modal sizing.
    state.ResizePending = true;

    const Rect viewport = EditorViewportBounds(state);
    state.Editor->ResizeViewport(
        static_cast<std::uint32_t>(std::max(viewport.Width, 1.0f)),
        static_cast<std::uint32_t>(std::max(viewport.Height, 1.0f)));
}

LRESULT CALLBACK WindowProcedure(HWND window, UINT message, WPARAM wParam, LPARAM lParam)
{
    auto* state = reinterpret_cast<WindowState*>(GetWindowLongPtrW(window, GWLP_USERDATA));
    if (message == WM_NCCREATE)
    {
        const auto* create = reinterpret_cast<const CREATESTRUCTW*>(lParam);
        state = static_cast<WindowState*>(create->lpCreateParams);
        SetWindowLongPtrW(window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(state));
    }
    if (state == nullptr)
    {
        return DefWindowProcW(window, message, wParam, lParam);
    }

    switch (message)
    {
    case WM_CREATE:
    {
        RECT client = {};
        GetClientRect(window, &client);
        state->ClientWidth = std::max<LONG>(client.right - client.left, 1);
        state->ClientHeight = std::max<LONG>(client.bottom - client.top, 1);
        state->NativeWindow = window;
        if (!EnsurePresentationBackend(*state))
        {
            MessageBoxA(
                window,
                state->PresentationError.c_str(),
                "Evans Editor",
                MB_OK | MB_ICONERROR);
            return -1;
        }
        ResizeEditor(*state, state->ClientWidth, state->ClientHeight);
        return 0;
    }
    case WM_SIZE:
        if (wParam != SIZE_MINIMIZED)
        {
            ResizeEditor(*state, LOWORD(lParam), HIWORD(lParam));
            InvalidateRect(window, nullptr, FALSE);
        }
        return 0;
    case WM_MOUSEMOVE:
        state->MouseX = GET_X_LPARAM(lParam);
        state->MouseY = GET_Y_LPARAM(lParam);
        state->Editor->HandleCameraPointer(
            static_cast<float>(state->MouseX),
            static_cast<float>(state->MouseY),
            state->CameraRotateCapture && (wParam & MK_RBUTTON) != 0);
        if (state->LightDragCapture && (wParam & MK_LBUTTON)!=0) state->Editor->HandleLightPointer(float(state->MouseX),float(state->MouseY),EditorViewportBounds(*state),true);
        state->Editor->HandleDemoPointer(float(state->MouseX),float(state->MouseY),
            !state->CameraRotateCapture && EditorViewportBounds(*state).Contains(float(state->MouseX),float(state->MouseY)));
        InvalidateRect(window, nullptr, FALSE);
        return 0;
    case WM_LBUTTONDOWN:
    {
        state->MouseX = GET_X_LPARAM(lParam);
        state->MouseY = GET_Y_LPARAM(lParam);
        state->MouseButtons |= IMGUI_MBUT_LEFT;
        SetCapture(window);
        const Rect viewport = EditorViewportBounds(*state);
        if (viewport.Contains(
                static_cast<float>(state->MouseX),
                static_cast<float>(state->MouseY)))
        {
            state->LightDragCapture=state->Editor->HandleLightPointer(float(state->MouseX),float(state->MouseY),viewport,true);
            if (!state->LightDragCapture) state->Editor->HandleViewportPointer(
                static_cast<float>(state->MouseX),
                static_cast<float>(state->MouseY),
                viewport);
        }
        InvalidateRect(window, nullptr, FALSE);
        return 0;
    }
    case WM_LBUTTONUP:
        state->MouseX = GET_X_LPARAM(lParam);
        state->MouseY = GET_Y_LPARAM(lParam);
        state->Editor->HandleLightPointer(float(state->MouseX),float(state->MouseY),EditorViewportBounds(*state),false);
        state->LightDragCapture=false;
        state->MouseButtons &= static_cast<unsigned char>(~IMGUI_MBUT_LEFT);
        if (state->MouseButtons == 0)
        {
            ReleaseCapture();
        }
        InvalidateRect(window, nullptr, FALSE);
        return 0;
    case WM_RBUTTONDOWN:
        state->MouseX = GET_X_LPARAM(lParam);
        state->MouseY = GET_Y_LPARAM(lParam);
        state->MouseButtons |= IMGUI_MBUT_RIGHT;
        state->CameraRotateCapture = EditorViewportBounds(*state).Contains(
            static_cast<float>(state->MouseX),
            static_cast<float>(state->MouseY));
        state->Editor->HandleCameraPointer(
            static_cast<float>(state->MouseX),
            static_cast<float>(state->MouseY),
            state->CameraRotateCapture);
        SetCapture(window);
        InvalidateRect(window, nullptr, FALSE);
        return 0;
    case WM_RBUTTONUP:
        state->MouseX = GET_X_LPARAM(lParam);
        state->MouseY = GET_Y_LPARAM(lParam);
        state->MouseButtons &= static_cast<unsigned char>(~IMGUI_MBUT_RIGHT);
        state->CameraRotateCapture = false;
        state->LightDragCapture=false;state->Editor->HandleLightPointer(0,0,EditorViewportBounds(*state),false);
        state->Editor->HandleCameraPointer(
            static_cast<float>(state->MouseX),
            static_cast<float>(state->MouseY),
            false);
        if (state->MouseButtons == 0)
        {
            ReleaseCapture();
        }
        InvalidateRect(window, nullptr, FALSE);
        return 0;
    case WM_CAPTURECHANGED:
        state->MouseButtons = 0;
        state->CameraRotateCapture = false;
        state->LightDragCapture=false;state->Editor->HandleLightPointer(0,0,EditorViewportBounds(*state),false);
        state->Editor->HandleCameraPointer(0.0f, 0.0f, false);
        return 0;
    case WM_KEYDOWN:
        state->Editor->SetCameraMovementKey(static_cast<char>(wParam), true);
        return 0;
    case WM_KEYUP:
        state->Editor->SetCameraMovementKey(static_cast<char>(wParam), false);
        return 0;
    case WM_KILLFOCUS:
        state->MouseButtons = 0;
        state->CameraRotateCapture = false;
        state->LightDragCapture=false;state->Editor->HandleLightPointer(0,0,EditorViewportBounds(*state),false);
        state->Editor->ClearCameraInput();
        return 0;
    case WM_MOUSEWHEEL:
    {
        POINT cursor{GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)};
        ScreenToClient(window, &cursor);
        state->MouseX = cursor.x;
        state->MouseY = cursor.y;
        state->MouseWheel += GET_WHEEL_DELTA_WPARAM(wParam) > 0 ? -1 : 1;
        InvalidateRect(window, nullptr, FALSE);
        return 0;
    }
    case WM_ERASEBKGND:
        return 1;
    case WM_ENTERSIZEMOVE:
        state->InSizeMove = true;
        return 0;
    case WM_EXITSIZEMOVE:
        state->InSizeMove = false;
        return 0;
    case WM_PAINT:
    {
        PAINTSTRUCT paint = {};
        BeginPaint(window, &paint);
        EndPaint(window, &paint);
        if (state->InSizeMove) RenderEditorWindow(*state);
        return 0;
    }
    case WM_CLOSE:
        if (state->DrawingFrame)
        {
            state->ClosePending = true;
            return 0;
        }
        state->Closing = true;
        DestroyWindow(window);
        return 0;
    case WM_DESTROY:
        state->Closing = true;
        state->Presenter = nullptr;
        for (auto& surface : state->Surfaces) surface.Reset();
        PostQuitMessage(0);
        return 0;
    default:
        return DefWindowProcW(window, message, wParam, lParam);
    }
}

LRESULT CALLBACK SurfaceWindowProcedure(HWND window, UINT message, WPARAM wParam, LPARAM lParam)
{
    if (message == WM_NCHITTEST) return HTTRANSPARENT;
    return DefWindowProcW(window, message, wParam, lParam);
}

bool RegisterEditorWindowClass(HINSTANCE instance)
{
    WNDCLASSW surfaceClass{};
    surfaceClass.lpfnWndProc = SurfaceWindowProcedure;
    surfaceClass.hInstance = instance;
    surfaceClass.lpszClassName = SurfaceClassName;
    if (!RegisterClassW(&surfaceClass) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) return false;

    WNDCLASSEXW windowClass = {};
    windowClass.cbSize = sizeof(windowClass);
    windowClass.style = CS_HREDRAW | CS_VREDRAW;
    windowClass.lpfnWndProc = WindowProcedure;
    windowClass.hInstance = instance;
    windowClass.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
    windowClass.hIcon = LoadIconW(nullptr, MAKEINTRESOURCEW(32512));
    windowClass.lpszClassName = WindowClassName;
    if (RegisterClassExW(&windowClass))
    {
        return true;
    }
    return GetLastError() == ERROR_CLASS_ALREADY_EXISTS;
}

int RunWindow(HINSTANCE instance, int showCommand)
{
    WindowState state;
    state.Editor = CreateEditor();

    if (!RegisterEditorWindowClass(instance))
    {
        return 3;
    }

    HWND window = CreateWindowExW(
        0,
        WindowClassName,
        L"Evans Renderer Editor",
        WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        1440,
        900,
        nullptr,
        nullptr,
        instance,
        &state);
    if (window == nullptr)
    {
        return 4;
    }
    ShowWindow(window, showCommand);
    UpdateWindow(window);

    MSG message = {};
    bool running=true;
    while (running)
    {
        for (int i = 0; i < 64 && PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE); ++i)
        {
            if (message.message==WM_QUIT) {running=false;break;}
            TranslateMessage(&message);
            DispatchMessageW(&message);
            // The immediate-mode controls consume button transitions once per frame.
            if (message.message == WM_LBUTTONDOWN || message.message == WM_LBUTTONUP) break;
        }
        if (!running) break;
        if (IsIconic(window)) {WaitMessage();state.LastTick=std::chrono::steady_clock::now();continue;}
        const auto now=std::chrono::steady_clock::now();
        const float delta=std::chrono::duration<float>(now-state.LastTick).count();state.LastTick=now;
        state.Editor->Tick(std::min(delta,0.1f));
        RenderEditorWindow(state);
    }
    return static_cast<int>(message.wParam);
}
} // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int showCommand)
{
    int count=0;
    auto** nativeArguments=CommandLineToArgvW(GetCommandLineW(),&count);
    std::vector<std::string> arguments;
    for (int i=0;nativeArguments && i<count;++i)
    {
        const int bytes=WideCharToMultiByte(CP_UTF8,0,nativeArguments[i],-1,nullptr,0,nullptr,nullptr);
        std::string value(bytes>0?static_cast<std::size_t>(bytes):0,'\0');
        if (bytes>0) { WideCharToMultiByte(CP_UTF8,0,nativeArguments[i],-1,value.data(),bytes,nullptr,nullptr); value.pop_back(); }
        arguments.push_back(std::move(value));
    }
    if (nativeArguments) LocalFree(nativeArguments);
    if (arguments.size()>1 && (arguments[1]=="--cook-nanite" || arguments[1]=="--bake-lightmap"))
    {
        const auto output=GetStdHandle(STD_OUTPUT_HANDLE);
        if (!output || output==INVALID_HANDLE_VALUE) AttachConsole(ATTACH_PARENT_PROCESS);
        BindCommandStream(STD_OUTPUT_HANDLE,stdout); BindCommandStream(STD_ERROR_HANDLE,stderr);
        if (const auto result=Evans::Editor::RunAssetCookCommand(arguments)) return *result;
    }
    return RunWindow(instance, showCommand);
}
#elif defined(__APPLE__)
int RunMacEditor();
int main(int argc,char** argv)
{
    const std::vector<std::string> arguments(argv,argv+argc);
    if (const auto result=Evans::Editor::RunAssetCookCommand(arguments)) return *result;
    return RunMacEditor();
}
#else
#include <iostream>
int main(int argc,char** argv)
{
    const std::vector<std::string> arguments(argv,argv+argc);
    if (const auto result=Evans::Editor::RunAssetCookCommand(arguments)) return *result;
    std::cerr << "The interactive editor requires Windows or macOS.\n";
    return 1;
}
#endif
