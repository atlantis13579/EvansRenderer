#include "Renderer/BakedLighting/BakedLightmap.h"
#include "Renderer/BakedLighting/BakedDiffuseLighting.h"
#include "Renderer/Lighting/SceneLightCollection.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <thread>

namespace Evans::Renderer
{
    /*
    Surface lightmap producer
    -------------------------
    Triangle/pixel clipping supplies an area-weighted barycentric centroid for
    every nonzero geometric footprint, including subpixel triangles. Integrate
    one diffuse bounce at that centroid using cosine-weighted Hammersley rays.
    Since p(w) = cos(theta)/pi, mean incident radiance directly estimates E/pi.
    The atlas stores illumination, not receiver color or camera-dependent terms.
    PBRT, "2D Sampling with Multidimensional Transformations", section 13.6.3:
    https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations

    A separable squared Euclidean distance transform copies the nearest covered
    texel into each chart's unused interior and reserved gutter. Sources never
    cross chart boundaries. The column pass constructs a lower envelope of
    parabolas after exact horizontal nearest-source scans, in linear image work.
    Felzenszwalb and Huttenlocher, "Distance Transforms of Sampled Functions"
    (Theory of Computing, 2012): https://cs.brown.edu/people/pfelzens/papers/dt-final.pdf

    Front and back receiver hemispheres have separate images. Secondary hit
    backfaces and misses contribute black, as in the probe producer. No ambient
    injection, recursion or specular paths are baked. Material/alpha/normal-map
    evaluation uses one centroid per triangle footprint and mip zero; geometric
    coverage is conservative but tiny material features can still alias.
    */
    namespace
    {
        using Maths::Vector3;
        constexpr double Pi = 3.14159265358979323846;
        constexpr double HalfSample = 0.5;
        constexpr float ParallelAxisLimit = 0.999f;
        constexpr std::uint32_t MinimumAtlasResolution = 64, MaximumAtlasResolution = 4096;
        constexpr std::uint32_t MinimumAtlasPadding = 1, MaximumAtlasPadding = 16;
        constexpr std::size_t MaximumClippedVertices = 8; // Triangle intersected with a rectangle has at most seven.
        constexpr double UVRoundingTolerance = 0.00025; // Pixel units after FP32 UV storage, at the largest atlas size.

        Vector3 Vector(Float3 v) { return {v.X,v.Y,v.Z}; }
        Float3 Native(Vector3 v) { return {v.x,v.y,v.z}; }
        bool Finite(Vector3 v) { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z); }
        bool Cancelled(const std::function<bool()>& cancelled, std::string& error)
        {
            if (cancelled && cancelled()) { error = "Lightmap bake cancelled"; return true; }
            return false;
        }

        bool ValidateAtlas(const LightmapAtlas& atlas, std::string& error, const std::function<bool()>& cancelled)
        {
            const auto size = atlas.Settings.Resolution, padding = atlas.Settings.Padding;
            if (!atlas.Scene || size < MinimumAtlasResolution || size > MaximumAtlasResolution ||
                padding < MinimumAtlasPadding || padding > MaximumAtlasPadding ||
                atlas.Triangles.size() != atlas.Scene->GetTriangleCount())
            { error = "Invalid lightmap source, dimensions or triangle mapping"; return false; }
            std::vector<std::uint8_t> owners(static_cast<std::size_t>(size)*size);
            std::vector<bool> seen(atlas.Triangles.size());
            for (std::size_t chartIndex = 0; chartIndex < atlas.Charts.size(); ++chartIndex)
            {
                if (Cancelled(cancelled,error)) return false;
                const auto& chart = atlas.Charts[chartIndex];
                if (chart.X > size || chart.Y > size || chart.Width > size-chart.X || chart.Height > size-chart.Y ||
                    chart.Width < padding*2+2 || chart.Height < padding*2+2 || chart.Triangles.empty())
                { error = "Invalid lightmap chart rectangle"; return false; }
                for (std::uint32_t y = chart.Y; y < chart.Y+chart.Height; ++y)
                    for (std::uint32_t x = chart.X; x < chart.X+chart.Width; ++x)
                    {
                        auto& owner = owners[static_cast<std::size_t>(y)*size+x];
                        if (owner) { error = "Lightmap chart rectangles overlap"; return false; }
                        owner = 1;
                    }
                for (auto id : chart.Triangles)
                {
                    if (id >= atlas.Triangles.size() || seen[id] || atlas.Triangles[id].Chart != chartIndex)
                    { error = "Invalid or duplicate lightmap triangle ownership"; return false; }
                    seen[id] = true;
                    const auto& uv = atlas.Triangles[id].UV;
                    for (const auto p : uv)
                    {
                        const double x = double(p.X)*size, y = double(p.Y)*size;
                        if (!std::isfinite(x) || !std::isfinite(y) ||
                            x < chart.X+padding+HalfSample-UVRoundingTolerance ||
                            x > chart.X+chart.Width-padding-HalfSample+UVRoundingTolerance ||
                            y < chart.Y+padding+HalfSample-UVRoundingTolerance ||
                            y > chart.Y+chart.Height-padding-HalfSample+UVRoundingTolerance)
                        { error = "Lightmap coordinates escape their chart interior"; return false; }
                    }
                    const double area = (double(uv[1].X)-uv[0].X)*(double(uv[2].Y)-uv[0].Y)-
                        (double(uv[1].Y)-uv[0].Y)*(double(uv[2].X)-uv[0].X);
                    if (!(area > 0)) { error = "Folded or collapsed lightmap triangle"; return false; }
                }
            }
            for (std::size_t i = 0; i < seen.size(); ++i)
                if (!seen[i] && atlas.Triangles[i].Chart != InvalidLightmapChart)
                { error = "Mapped lightmap triangle has no chart"; return false; }
            return true;
        }

        struct ClipVertex
        {
            double X, Y;
            std::array<double,3> Barycentric;
        };
        struct Footprint { double Area = 0; Float3 Barycentric; };

        Footprint PixelFootprint(const std::array<Float2,3>& uv, std::uint32_t size, int x, int y)
        {
            std::array<ClipVertex,MaximumClippedVertices> polygon{}, clipped{};
            std::size_t count = 3;
            for (std::size_t i = 0; i < count; ++i)
            {
                polygon[i] = {double(uv[i].X)*size-x, double(uv[i].Y)*size-y, {0,0,0}};
                polygon[i].Barycentric[i] = 1;
            }
            // Pixel-relative coordinates avoid subtracting large products in
            // area tests. Carry barycentrics through clipping instead of
            // inverting a potentially almost-degenerate UV triangle.
            for (unsigned plane = 0; plane < 4 && count; ++plane)
            {
                auto distance = [plane](const ClipVertex& v) {
                    const double coordinate = plane < 2 ? v.X : v.Y;
                    return plane%2 == 0 ? coordinate : 1-coordinate;
                };
                std::size_t nextCount = 0;
                auto previous = polygon[count-1]; double prior = distance(previous);
                for (std::size_t i = 0; i < count; ++i)
                {
                    const auto current = polygon[i]; const double next = distance(current);
                    if ((prior >= 0) != (next >= 0))
                    {
                        const double fraction = prior/(prior-next);
                        ClipVertex intersection {previous.X+(current.X-previous.X)*fraction,
                            previous.Y+(current.Y-previous.Y)*fraction, {}};
                        for (unsigned component = 0; component < 3; ++component)
                            intersection.Barycentric[component] = previous.Barycentric[component]+
                                (current.Barycentric[component]-previous.Barycentric[component])*fraction;
                        clipped[nextCount++] = intersection;
                    }
                    if (next >= 0) clipped[nextCount++] = current;
                    previous = current; prior = next;
                }
                polygon = clipped; count = nextCount;
            }
            double areaTwice = 0; std::array<double,3> moment{};
            for (std::size_t i = 1; i+1 < count; ++i)
            {
                const auto& a = polygon[0]; const auto& b = polygon[i]; const auto& c = polygon[i+1];
                const double weight = std::abs((b.X-a.X)*(c.Y-a.Y)-(b.Y-a.Y)*(c.X-a.X));
                areaTwice += weight;
                for (unsigned component = 0; component < 3; ++component)
                    moment[component] += weight*(a.Barycentric[component]+b.Barycentric[component]+c.Barycentric[component])/3;
            }
            if (!(areaTwice > 0)) return {};
            return {areaTwice*HalfSample,{float(moment[0]/areaTwice),float(moment[1]/areaTwice),float(moment[2]/areaTwice)}};
        }

        std::vector<Vector3> HemisphereDirections(std::uint32_t count)
        {
            std::vector<Vector3> result; result.reserve(count);
            for (std::uint32_t sample = 0; sample < count; ++sample)
            {
                double radicalInverse = 0, bitWeight = HalfSample;
                for (auto bits = sample; bits; bits >>= 1, bitWeight *= HalfSample)
                    if (bits&1) radicalInverse += bitWeight;
                const double radiusSquared = (sample+HalfSample)/count;
                const double radius = std::sqrt(radiusSquared), angle = 2*Pi*radicalInverse;
                result.emplace_back(float(radius*std::cos(angle)),float(radius*std::sin(angle)),float(std::sqrt(1-radiusSquared)));
            }
            return result;
        }

        struct TraceStatistics
        {
            std::uint64_t SurfaceSampleCount=0,SurfaceHitCount=0,ShadowQueryCount=0;
        };
        bool Integrate(const SceneRayHit& receiver, const SceneRayQuery& scene, const std::vector<SceneLight>& lights,
            const BakedLightmapSettings& settings, const std::vector<Vector3>& directions, double rotation,
            Vector3& irradiance, TraceStatistics& result, std::string& error, const std::function<bool()>& cancelled)
        {
            const auto normal = Vector(receiver.Normal), geometric = Vector(receiver.GeometricNormal);
            const auto helper = std::abs(normal.y) < ParallelAxisLimit ? Vector3::UnitY() : Vector3::UnitX();
            const auto tangent = helper.Cross(normal).Normalized(), bitangent = normal.Cross(tangent);
            const float cosine = float(std::cos(rotation)), sine = float(std::sin(rotation));
            const auto rotatedTangent = tangent*cosine+bitangent*sine;
            const auto rotatedBitangent = bitangent*cosine-tangent*sine;
            const auto origin = Vector(receiver.Position)+geometric*(receiver.FrontFace ? settings.SurfaceBias : -settings.SurfaceBias);
            std::array<double,3> sum{};
            for (auto local : directions)
            {
                if (Cancelled(cancelled,error)) return false;
                const auto direction = rotatedTangent*local.x+rotatedBitangent*local.y+normal*local.z;
                SceneRayHit hit;
                if (!scene.Trace(Native(origin),Native(direction),settings.SurfaceBias,settings.MaximumDistance,hit)) continue;
                ++result.SurfaceHitCount;
                if (!hit.FrontFace) continue;
                auto radiance = EvaluateBakedDiffuse(scene,hit,lights,{settings.SurfaceBias,settings.Shadows},result.ShadowQueryCount);
                if (!Finite(radiance)) { error = "Lightmap radiance overflow; reduce material or emitter intensity"; return false; }
                radiance = Vector3::Min(Vector3::Max(radiance,Vector3::Zero()),Vector3(MaximumBakedRadiance));
                sum[0] += radiance.x; sum[1] += radiance.y; sum[2] += radiance.z;
            }
            irradiance = {float(sum[0]/directions.size()),float(sum[1]/directions.size()),float(sum[2]/directions.size())};
            return true;
        }

        bool FillChart(BakedLightmap& result, const LightmapAtlasChart& chart,
            std::string& error, const std::function<bool()>& cancelled)
        {
            const int width = static_cast<int>(chart.Width), height = static_cast<int>(chart.Height);
            const auto size = result.Atlas->Settings.Resolution;
            auto index = [&](int x, int y) { return static_cast<std::size_t>(chart.Y+y)*size+chart.X+x; };
            auto surface = [&](int x, int y) { return result.TexelKinds[index(x,y)] == LightmapTexelKind::Surface; };
            std::vector<int> nearestX(static_cast<std::size_t>(width)*height,-1);
            for (int y = 0; y < height; ++y)
            {
                if (Cancelled(cancelled,error)) return false;
                const auto row = static_cast<std::size_t>(y)*width;
                int left = -1, right = -1;
                for (int x = 0; x < width; ++x) { if (surface(x,y)) left = x; nearestX[row+x] = left; }
                for (int x = width-1; x >= 0; --x)
                {
                    if (surface(x,y)) right = x;
                    auto& source = nearestX[row+x];
                    if (right >= 0 && (source < 0 || right-x < x-source)) source = right;
                }
            }
            std::vector<int> sites(height);
            std::vector<double> boundaries(height+1);
            for (int x = 0; x < width; ++x)
            {
                if (Cancelled(cancelled,error)) return false;
                int last = -1;
                auto horizontalSquared = [&](int y) {
                    const double delta = x-nearestX[static_cast<std::size_t>(y)*width+x]; return delta*delta;
                };
                for (int y = 0; y < height; ++y)
                {
                    if (nearestX[static_cast<std::size_t>(y)*width+x] < 0) continue;
                    double intersection = -std::numeric_limits<double>::infinity();
                    while (last >= 0)
                    {
                        const int prior = sites[last];
                        intersection = (horizontalSquared(y)+double(y)*y-horizontalSquared(prior)-double(prior)*prior)/(2*(y-prior));
                        if (intersection > boundaries[last]) break;
                        --last;
                    }
                    sites[++last] = y;
                    boundaries[last] = last == 0 ? -std::numeric_limits<double>::infinity() : intersection;
                    boundaries[last+1] = std::numeric_limits<double>::infinity();
                }
                if (last < 0) continue; // Entire chart rejected by alpha testing.
                int selected = 0;
                for (int y = 0; y < height; ++y)
                {
                    while (selected < last && boundaries[selected+1] < y) ++selected;
                    const auto destination = index(x,y);
                    if (result.TexelKinds[destination] == LightmapTexelKind::Surface) continue;
                    const int sourceY = sites[selected], sourceX = nearestX[static_cast<std::size_t>(sourceY)*width+x];
                    const auto source = index(sourceX,sourceY);
                    result.FrontIrradiance[destination] = result.FrontIrradiance[source];
                    if (result.Settings.BakeBackFaces) result.BackIrradiance[destination] = result.BackIrradiance[source];
                    result.TexelKinds[destination] = LightmapTexelKind::Gutter; ++result.GutterTexelCount;
                }
            }
            return true;
        }
    }

    bool BakeLightmap(std::shared_ptr<const LightmapAtlas> atlas, const SceneRayQuery& transportScene,
        const LightingState& lighting, const BakedLightmapSettings& settings, BakedLightmap& output,
        std::string& error, const std::function<bool()>& cancelled)
    {
        error.clear();
        if (Cancelled(cancelled,error)) return false;
        if (!atlas || settings.RayCount < MinimumLightmapRayCount || settings.RayCount > MaximumLightmapRayCount ||
            !std::isfinite(settings.MaximumDistance) || settings.MaximumDistance <= 0 ||
            !std::isfinite(settings.SurfaceBias) || settings.SurfaceBias <= 0 || settings.SurfaceBias >= settings.MaximumDistance)
        { error = "Invalid lightmap source, ray count, sampling distance or bias"; return false; }
        if (!ValidateAtlas(*atlas,error,cancelled)) return false;
        auto lights = lighting.SceneLights;
        if (!lights)
        {
            if (lighting.LightCount > MaximumSceneLights) { error = "Inline lightmap lighting exceeds its light capacity"; return false; }
            lights = SceneLightCollection::Create({lighting.Lights.begin(),lighting.Lights.begin()+lighting.LightCount},-1,error);
            if (!lights) return false;
        }
        BakedLightmap result; result.Atlas = std::move(atlas); result.Settings = settings;
        const auto size = result.Atlas->Settings.Resolution;
        const auto pixelCount = static_cast<std::size_t>(size)*size;
        result.FrontIrradiance.resize(pixelCount,Float4{0,0,0,0});
        if (settings.BakeBackFaces) result.BackIrradiance.resize(pixelCount,Float4{0,0,0,0});
        result.TexelKinds.resize(pixelCount,LightmapTexelKind::Empty);
        const auto directions = HemisphereDirections(settings.RayCount);
        const double inverseGoldenRatio = (std::sqrt(5.0)-1)/2;
        const double trianglePhase = std::sqrt(2.0);
        auto bakeChart=[&](const LightmapAtlasChart& chart,TraceStatistics& statistics,
            std::string& workerError,const std::function<bool()>& stop)
        {
            for (auto triangleIndex : chart.Triangles)
            {
                if (Cancelled(stop,workerError)) return false;
                const auto& uv = result.Atlas->Triangles[triangleIndex].UV;
                const int minimumX = int(std::floor(double(std::min({uv[0].X,uv[1].X,uv[2].X}))*size));
                const int minimumY = int(std::floor(double(std::min({uv[0].Y,uv[1].Y,uv[2].Y}))*size));
                const int maximumX = int(std::ceil(double(std::max({uv[0].X,uv[1].X,uv[2].X}))*size));
                const int maximumY = int(std::ceil(double(std::max({uv[0].Y,uv[1].Y,uv[2].Y}))*size));
                for (int y = minimumY; y < maximumY; ++y)
                {
                    if (Cancelled(stop,workerError)) return false;
                    for (int x = minimumX; x < maximumX; ++x)
                    {
                        const auto footprint = PixelFootprint(uv,size,x,y);
                        if (!(footprint.Area > 0)) continue;
                        SceneRayHit receiver;
                        if (!result.Atlas->Scene->SampleSurface(triangleIndex,footprint.Barycentric,true,receiver)) continue;
                        const auto pixel = static_cast<std::size_t>(y)*size+x;
                        const double phase = pixel*inverseGoldenRatio+triangleIndex*trianglePhase;
                        const double rotation = 2*Pi*(phase-std::floor(phase));
                        Vector3 front,back;
                        if (!Integrate(receiver,transportScene,lights->GetLights(),settings,directions,rotation,front,statistics,workerError,stop)) return false;
                        if (settings.BakeBackFaces)
                        {
                            // Re-evaluate the back-face tangent frame: negating
                            // a perturbed front normal would also flip its
                            // tangent-space lateral components incorrectly.
                            if (!result.Atlas->Scene->SampleSurface(triangleIndex,footprint.Barycentric,false,receiver))
                            { workerError = "Lightmap front/back surface coverage disagrees"; return false; }
                            if (!Integrate(receiver,transportScene,lights->GetLights(),settings,directions,rotation,back,statistics,workerError,stop)) return false;
                        }
                        auto accumulate = [&](Float4& texel, Vector3 value) {
                            const float weight = float(footprint.Area);
                            texel.X += value.x*weight; texel.Y += value.y*weight; texel.Z += value.z*weight; texel.W += weight;
                        };
                        accumulate(result.FrontIrradiance[pixel],front);
                        if (settings.BakeBackFaces) accumulate(result.BackIrradiance[pixel],back);
                        result.TexelKinds[pixel] = LightmapTexelKind::Surface; ++statistics.SurfaceSampleCount;
                    }
                }
            }
            return true;
        };
        // Validated charts own disjoint texels. Parallelize whole charts so
        // every texel retains its serial triangle/ray accumulation order.
        // Cancellation callbacks stay on the calling thread; workers only read
        // an atomic stop flag and immutable scene/material snapshots.
        constexpr unsigned MaximumBakeWorkers=8;
        const auto workerCount=std::min<std::size_t>(result.Atlas->Charts.size(),
            std::clamp(std::thread::hardware_concurrency(),1u,MaximumBakeWorkers));
        struct Worker { TraceStatistics Statistics; std::string Error; };
        std::vector<Worker> workers(workerCount);
        std::vector<std::future<void>> tasks;
        tasks.reserve(workerCount);
        std::atomic<std::size_t> nextChart{0};
        std::atomic<bool> stopped{false};
        const std::function<bool()> stop=[&] { return stopped.load(std::memory_order_relaxed); };
        try
        {
            for (auto& worker:workers) tasks.push_back(std::async(std::launch::async,[&,state=&worker] {
                try
                {
                    while (!stop())
                    {
                        const auto index=nextChart.fetch_add(1,std::memory_order_relaxed);
                        if (index>=result.Atlas->Charts.size()) break;
                        if (!bakeChart(result.Atlas->Charts[index],state->Statistics,state->Error,stop))
                        { stopped.store(true,std::memory_order_relaxed); break; }
                    }
                }
                catch (const std::exception& exception)
                { state->Error=exception.what(); stopped.store(true,std::memory_order_relaxed); }
            }));
        }
        catch (const std::exception& exception)
        { error=std::string("Cannot start lightmap workers: ")+exception.what(); stopped.store(true,std::memory_order_relaxed); }
        for (auto& task:tasks)
        {
            while (task.wait_for(std::chrono::milliseconds(10))!=std::future_status::ready)
                if (Cancelled(cancelled,error)) stopped.store(true,std::memory_order_relaxed);
            task.get();
        }
        if (stopped.load(std::memory_order_relaxed))
        {
            if (error.empty()) for (const auto& worker:workers)
                if (!worker.Error.empty() && worker.Error!="Lightmap bake cancelled") { error=worker.Error; break; }
            if (error.empty()) error="Lightmap bake cancelled";
            return false;
        }
        for (const auto& worker:workers)
        {
            result.SurfaceSampleCount+=worker.Statistics.SurfaceSampleCount;
            result.SurfaceHitCount+=worker.Statistics.SurfaceHitCount;
            result.ShadowQueryCount+=worker.Statistics.ShadowQueryCount;
        }
        for (std::uint32_t y = 0; y < size; ++y)
        {
            if (Cancelled(cancelled,error)) return false;
            for (std::uint32_t x = 0; x < size; ++x)
            {
                const auto pixel = static_cast<std::size_t>(y)*size+x;
                if (result.TexelKinds[pixel] != LightmapTexelKind::Surface) continue;
                auto normalize = [](Float4& texel) { texel.X /= texel.W; texel.Y /= texel.W; texel.Z /= texel.W; texel.W = 1; };
                normalize(result.FrontIrradiance[pixel]);
                if (settings.BakeBackFaces) normalize(result.BackIrradiance[pixel]);
                ++result.CoveredTexelCount;
            }
        }
        for (const auto& chart : result.Atlas->Charts)
            if (!FillChart(result,chart,error,cancelled)) return false;
        if (Cancelled(cancelled,error)) return false;
        output = std::move(result);
        return true;
    }
}
