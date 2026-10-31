#include "Editor/Editor.hpp"
#include "Editor/ContentRoot.hpp"
#include "Editor/ModuleAdapters.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

namespace
{
using namespace Evans::Editor;
using Clock = std::chrono::steady_clock;

double Milliseconds(Clock::duration duration)
{
    return std::chrono::duration<double, std::milli>(duration).count();
}

void SaveFrame(const Image& frame, const std::filesystem::path& path)
{
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot save framebuffer: " + path.string());
    file << "P6\n" << frame.Width << ' ' << frame.Height << "\n255\n";
    for (std::size_t i = 0; i < frame.PixelsRGBA8.size(); i += 4)
        file.write(reinterpret_cast<const char*>(frame.PixelsRGBA8.data() + i), 3);
    if (!file) throw std::runtime_error("Framebuffer write failed: " + path.string());
}

template <typename Predicate>
void Wait(EditorCore& editor, Predicate done)
{
    const auto deadline = Clock::now() + std::chrono::seconds(45);
    while (Clock::now() < deadline)
    {
        const auto begin = Clock::now();
        editor.Tick(0.0f);
        if (Milliseconds(Clock::now() - begin) > 250.0)
            throw std::runtime_error("Editor tick blocked for more than 250 ms");
        if (done()) return;
        if (editor.StatusText().find("failed") != std::string::npos ||
            editor.StatusText().find("invalid image") != std::string::npos)
            throw std::runtime_error(editor.StatusText());
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    throw std::runtime_error("Timed out: " + editor.StatusText());
}

std::uint64_t Hash(const Image& image)
{
    if (!image.IsValid()) throw std::runtime_error("Invalid framebuffer");
    bool varied = false;
    std::uint64_t hash = 14695981039346656037ULL;
    for (std::size_t i = 0; i < image.PixelsRGBA8.size(); ++i)
    {
        hash = (hash ^ image.PixelsRGBA8[i]) * 1099511628211ULL;
        if (i >= 4 && i % 4 != 3 && image.PixelsRGBA8[i] != image.PixelsRGBA8[i % 4])
            varied = true;
    }
    if (!varied) throw std::runtime_error("Framebuffer contains only a clear color");
    return hash;
}

double Render(EditorCore& editor)
{
    const auto revision = editor.ViewportRevision();
    const auto begin = Clock::now();
    editor.RequestRender();
    Wait(editor, [&] {
        return editor.ViewportRevision() > revision &&
            editor.StatusText().rfind("Frame completed with ", 0) == 0;
    });
    const auto& image = editor.ViewportImage();
    if (editor.ViewportBackend() != editor.SelectedBackend())
        throw std::runtime_error("Framebuffer belongs to a stale backend");
    if (image.Width != editor.Settings().Width || image.Height != editor.Settings().Height)
        throw std::runtime_error("Stale framebuffer dimensions after resize");
    return Milliseconds(Clock::now() - begin);
}

void Settings(EditorCore& editor, int mode, bool shadows, bool outline)
{
    editor.SetFillMode(static_cast<FillMode>(mode));
    editor.SetShadowsEnabled(shadows);
    editor.SetOutlineEnabled(outline);
}

bool CompareFamily(
    std::unordered_map<std::string, Image>& references,
    const std::string& backend,
    int mode,
    int flags,
    const Image& image)
{
    const bool ray = backend == "renderer.dxr";
    const bool raster = backend.rfind("rasterizer.", 0) == 0;
    const std::string family = ray ? "ray" : raster ? "raster" : "realtime";
    const std::string key = family + '/' + std::to_string(mode) + '/' + std::to_string(flags);
    const auto inserted = references.emplace(key, image);
    if (inserted.second) return false;
    const Image& baseline = inserted.first->second;
    if (baseline.Width != image.Width || baseline.Height != image.Height)
        throw std::runtime_error("Backend family framebuffer dimensions differ");
    std::uint64_t absolute = 0;
    std::size_t large = 0;
    const bool lineCoverage = !ray && mode != 0;
    const auto differenceAt = [](const Image& a, std::size_t ai, const Image& b, std::size_t bi)
    {
        std::pair<int, int> difference{};
        for (std::size_t channel = 0; channel < 3; ++channel)
        {
            const int value = std::abs(int(a.PixelsRGBA8[ai + channel]) - int(b.PixelsRGBA8[bi + channel]));
            difference.first += value;
            difference.second += value > 4 ? 1 : 0;
        }
        return difference;
    };
    const auto isLine = [](const Image& frame, std::size_t i) {
        return frame.PixelsRGBA8[i] == 31 && frame.PixelsRGBA8[i + 1] == 209 && frame.PixelsRGBA8[i + 2] == 255;
    };
    const auto nearbyLine = [&](const Image& frame, int x, int y)
    {
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
                if (x + dx >= 0 && y + dy >= 0 && x + dx < int(frame.Width) && y + dy < int(frame.Height) &&
                    isLine(frame, (static_cast<std::size_t>(y + dy) * frame.Width + x + dx) * 4))
                    return true;
        return false;
    };
    std::size_t referenceLines = 0;
    std::size_t candidateLines = 0;
    for (std::size_t i = 0; i < image.PixelsRGBA8.size(); i += 4)
    {
        auto difference = differenceAt(image, i, baseline, i);
        if (lineCoverage)
        {
            const int x = int((i / 4) % image.Width);
            const int y = int((i / 4) / image.Width);
            const bool referenceLine = isLine(baseline, i);
            const bool candidateLine = isLine(image, i);
            if ((referenceLine && nearbyLine(image, x, y)) ||
                (candidateLine && nearbyLine(baseline, x, y)))
                difference = {};
            referenceLines += referenceLine ? 1 : 0;
            candidateLines += candidateLine ? 1 : 0;
        }
        absolute += difference.first;
        large += difference.second;
    }
    const double channels = static_cast<double>(image.Width) * image.Height * 3;
    double maximumMean = ray ? 0.01 : 0.25;
    double maximumLargeFraction = ray ? 0.0001 : 0.003;
    // Compare native/software lines within one pixel, while retaining a coverage
    // budget so missing dense topology cannot pass as a subpixel difference.
    const double lineDifference = static_cast<double>(std::max(referenceLines, candidateLines) -
        std::min(referenceLines, candidateLines)) / std::max<std::size_t>(1, std::max(referenceLines, candidateLines));
    if (lineCoverage) { maximumMean = 0.75; maximumLargeFraction = 0.006; }
    if (absolute / channels > maximumMean || large / channels > maximumLargeFraction || lineDifference > 0.20)
    {
        const auto outputDirectory = ResolveContentRoot().parent_path() / "Intermediate" / "Validation";
        std::filesystem::create_directories(outputDirectory);
        SaveFrame(baseline, outputDirectory / "matrix-reference.ppm");
        SaveFrame(image, outputDirectory / "matrix-candidate.ppm");
        throw std::runtime_error("Family parity failed: " + backend + '/' + key +
            " mean=" + std::to_string(absolute / channels) +
            " fraction>4=" + std::to_string(large / channels) +
            " line-coverage=" + std::to_string(lineDifference));
    }
    return true;
}


}

int main(int argc, char** argv)
{
    try
    {
        bool benchmark = false;
        std::string benchmarkBackend;
        std::ofstream report;
        std::filesystem::path captureDirectory;
        for (int i = 1; i < argc; ++i)
        {
            const std::string argument = argv[i];
            if (argument == "--benchmark") benchmark = true;
            else if (argument == "--backend" && i + 1 < argc) benchmarkBackend = argv[++i];
            else if (argument == "--report" && i + 1 < argc)
            {
                report.open(argv[++i]);
                if (!report) throw std::runtime_error("Cannot open report");
            }
            else if (argument == "--capture-dir" && i + 1 < argc)
                captureDirectory = argv[++i];
            else throw std::runtime_error(
                "Usage: EvansRendererMatrix [--benchmark [--backend ID]] [--report PATH] [--capture-dir PATH]");
        }
        if (!benchmarkBackend.empty() && !benchmark)
            throw std::runtime_error("--backend requires --benchmark; validation always covers all available backends");
        if (!captureDirectory.empty()) std::filesystem::create_directories(captureDirectory);
        EditorCore editor{ResolveContentRoot()};
        RegisterModuleAdapters(editor);
        editor.ResizeViewport(320, 180);
        editor.SetTeapotRotationSpeed(0.0f);
        auto backends = editor.Backends();
        if (backends.empty()) throw std::runtime_error("No renderer available");
        for (const auto& descriptor : StandardBackendCatalog())
        {
            const bool available = std::any_of(backends.begin(), backends.end(), [&](const auto& b) {
                return b.Id == descriptor.Id;
            });
            std::cout << (available ? "AVAILABLE " : "SKIP ") << descriptor.Id << std::endl;
        }
        if (!benchmarkBackend.empty())
        {
            backends.erase(std::remove_if(backends.begin(), backends.end(),
                [&](const auto& backend) { return backend.Id != benchmarkBackend; }), backends.end());
            if (backends.empty()) throw std::runtime_error("Requested benchmark backend is unavailable");
        }
        std::size_t frames = 0;
        std::size_t comparisons = 0;
        std::unordered_map<std::string, std::uint64_t> triangleHashes;
        const std::array<DemoId, 3> demos = {
            DemoId::Triangle, DemoId::UtahTeapot, DemoId::SponzaAtrium};
        for (const auto demo : demos)
        {
            std::unordered_map<std::string, Image> references;
            if (benchmark && demo != DemoId::SponzaAtrium) continue;
            editor.SelectDemo(demo);
            Wait(editor, [&] { return !editor.IsDemoLoading() && editor.Scene().Demo == demo; });
            if (editor.Scene().Nodes.empty()) throw std::runtime_error("Empty scene");
            editor.SelectNode(editor.Scene().Nodes.front().Id);
            for (const auto& backend : backends)
            {
                std::cout << "RUN " << static_cast<int>(demo) << '/' << backend.Id << std::endl;
                editor.SelectBackend(backend.Id);
                Settings(editor, 0, false, false);
                Render(editor);
                const auto baseline = Hash(editor.ViewportImage());
                if (demo == DemoId::Triangle) triangleHashes[backend.Id] = baseline;
                std::array<std::array<std::uint64_t, 4>, 3> effectHashes{};
                for (int mode = 0; mode < (benchmark ? 1 : 3); ++mode)
                {
                    std::uint64_t unshadowedWireHash = 0;
                    for (int flags = 0; flags < (benchmark ? 8 : 4); ++flags)
                    {
                        Settings(editor, mode, (flags & 1) != 0, (flags & 2) != 0);
                        if (benchmark)
                        {
                            editor.SetCameraMovementKey('w', true);
                            editor.Tick(0.0001f);
                            editor.ClearCameraInput();
                        }
                        const double elapsed = Render(editor);
                        const auto hash = Hash(editor.ViewportImage());
                        if (!captureDirectory.empty())
                            SaveFrame(editor.ViewportImage(), captureDirectory /
                                (std::to_string(static_cast<int>(demo)) + "-" + backend.Id + "-" +
                                 std::to_string(mode) + "-" + std::to_string(flags) + ".ppm"));
                        if (!benchmark) effectHashes[mode][flags] = hash;
                        if (mode == 1)
                        {
                            if ((flags & 1) == 0) unshadowedWireHash = hash;
                            else if (hash != unshadowedWireHash)
                                throw std::runtime_error("Shadow toggle changed pure wireframe: " + backend.Id);
                        }
                        if (!benchmark && CompareFamily(references, backend.Id, mode, flags, editor.ViewportImage()))
                            ++comparisons;
                        if (report) report << static_cast<int>(demo) << ' ' << backend.Id << ' '
                            << mode << ' ' << flags << ' ' << hash << ' ' << elapsed << '\n';
                        ++frames;
                    }
                }
                if (!benchmark)
                {
                    // Sponza contains visible shadow receivers, selectable geometry,
                    // and dense topology for every rendering family.
                    if (demo == DemoId::SponzaAtrium)
                    {
                        for (int mode = 0; mode < 3; ++mode)
                        {
                            for (int shadow = 0; shadow < 2; ++shadow)
                                if (effectHashes[mode][shadow] == effectHashes[mode][shadow | 2])
                                    throw std::runtime_error("Outline toggle had no visible effect: " + backend.Id);
                            if (mode != 1)
                                for (int outline : {0, 2})
                                    if (effectHashes[mode][outline] == effectHashes[mode][outline | 1])
                                        throw std::runtime_error("Shadow toggle had no visible effect: " + backend.Id);
                        }
                        for (int flags = 0; flags < 4; ++flags)
                            if (effectHashes[0][flags] == effectHashes[1][flags] ||
                                effectHashes[0][flags] == effectHashes[2][flags] ||
                                effectHashes[1][flags] == effectHashes[2][flags])
                                throw std::runtime_error("Fill modes produced identical images: " + backend.Id);
                    }
                    // Restore settings after all combinations to detect stale resource state.
                    Settings(editor, 0, false, false);
                    Render(editor);
                    if (Hash(editor.ViewportImage()) != baseline)
                        throw std::runtime_error("Settings round trip changed the image: " + backend.Id);
                    editor.ResizeViewport(257, 145);
                    Render(editor);
                    Hash(editor.ViewportImage());
                    editor.ResizeViewport(320, 180);
                    Render(editor);
                    if (Hash(editor.ViewportImage()) != baseline)
                        throw std::runtime_error("Resize round trip changed the image: " + backend.Id);
                }
            }
        }
        std::size_t switches = 0;
        std::size_t dynamicFrames = 0;
        if (!benchmark)
        {
            editor.SelectDemo(DemoId::Triangle);
            Wait(editor, [&] { return !editor.IsDemoLoading() && editor.Scene().Demo == DemoId::Triangle; });
            editor.SelectNode(editor.Scene().Nodes.front().Id);
            for (const auto& source : backends)
            {
                for (const auto& destination : backends)
                {
                    if (source.Id == destination.Id) continue;
                    editor.SelectBackend(source.Id);
                    Settings(editor, 2, true, true);
                    editor.RequestRender();
                    // Allow the source worker to enter the backend before switching away.
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    const auto begin = Clock::now();
                    editor.SelectBackend(destination.Id);
                    Settings(editor, 0, false, false);
                    if (Milliseconds(Clock::now() - begin) > 250.0)
                        throw std::runtime_error("Backend selection blocked");
                    Render(editor);
                    if (Hash(editor.ViewportImage()) != triangleHashes.at(destination.Id))
                        throw std::runtime_error("Stale frame after " + source.Id + " -> " + destination.Id);
                    ++switches;
                }
            }
            editor.SelectDemo(DemoId::UtahTeapot);
            Wait(editor, [&] { return !editor.IsDemoLoading() && editor.Scene().Demo == DemoId::UtahTeapot; });
            for (const auto& backend : backends)
            {
                editor.SelectBackend(backend.Id);
                Settings(editor, 0, true, false);
                Render(editor);
                const auto stationary = Hash(editor.ViewportImage());
                editor.SetTeapotRotationSpeed(90.0f);
                editor.Tick(0.2f);
                editor.SetTeapotRotationSpeed(0.0f);
                Render(editor);
                const auto rotated = Hash(editor.ViewportImage());
                if (rotated == stationary)
                    throw std::runtime_error("Scene animation did not invalidate " + backend.Id);
                editor.SetCameraMovementKey('w', true);
                editor.Tick(0.02f);
                editor.ClearCameraInput();
                Render(editor);
                if (Hash(editor.ViewportImage()) == rotated)
                    throw std::runtime_error("Camera movement did not invalidate " + backend.Id);
                dynamicFrames += 2;
            }
        }
        std::cout << "PASS " << backends.size() << " backends, " << frames
                  << " effect frames, " << switches << " directed switches, "
                  << dynamicFrames << " dynamic frames, " << comparisons
                  << " family comparisons" << std::endl;
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL " << error.what() << std::endl;
        return 1;
    }
}
