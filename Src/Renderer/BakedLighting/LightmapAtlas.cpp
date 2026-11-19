#include "Renderer/BakedLighting/LightmapAtlas.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <numeric>
#include <tuple>
#include <unordered_map>

namespace Evans::Renderer
{
    /*
    Generated UV2 for lightmap transport
    -----------------------------------
    Material UVs may overlap, repeat or be absent. A separate chart atlas owns
    unique surface coordinates. The partition/parameterize/pack workflow and
    gutter motivation are described by Microsoft, "Using UVAtlas":
    https://learn.microsoft.com/en-us/windows/win32/direct3d9/using-uvatlas

    This implementation uses bounded planar region growing, not UVAtlas's
    stretch optimizer or LSCM. Oppositely directed manifold edges establish
    adjacency after exact position welding within each source mesh. A chart's
    seed plane stays fixed: its normal cone bounds projection area compression,
    and a 2D separating-axis test rejects overlapping projected interiors.
    Nonmanifold edges, folds and disconnected components form chart boundaries.
    The triangle cap bounds the quadratic overlap checks within each chart.

    Rectangle skyline packing allows quarter turns and reserves a pixel gutter.
    A bounded density search fits charts at a shared world-space texel density,
    with a minimum one-texel span per axis for tiny charts. UV32 quantization
    failures split only the affected charts before repacking. This heuristic
    need not find the globally densest arrangement. Padding is
    reserved here; the later bake must fill it without blending other charts.
    All calculations before final UV storage use doubles to avoid world-scale
    sensitivity. Source geometry and its first UV set remain immutable.
    */
    namespace
    {
        constexpr double Pi = 3.14159265358979323846;
        constexpr double ComparisonEpsilon = 64 * std::numeric_limits<double>::epsilon();
        constexpr double TexelCenter = 0.5;
        constexpr unsigned DensitySearchIterations = 32;
        constexpr unsigned MinimumResolution = 64, MaximumResolution = 4096, MaximumPadding = 16;
        constexpr unsigned MaximumChartTriangleLimit = 4096, MinimumInteriorPixels = 2;
        constexpr double MinimumChartSpan = MinimumInteriorPixels-1;
        constexpr float MaximumProjectionAngle = 80;

        struct Point2 { double X = 0, Y = 0; };
        struct Point3
        {
            double X = 0, Y = 0, Z = 0;
            Point3 operator-(Point3 v) const { return {X-v.X,Y-v.Y,Z-v.Z}; }
            Point3 operator/(double s) const { return {X/s,Y/s,Z/s}; }
            double Dot(Point3 v) const { return X*v.X+Y*v.Y+Z*v.Z; }
            Point3 Cross(Point3 v) const { return {Y*v.Z-Z*v.Y,Z*v.X-X*v.Z,X*v.Y-Y*v.X}; }
            double Length() const { return std::sqrt(Dot(*this)); }
        };
        Point3 Point(Float3 p) { return {p.X,p.Y,p.Z}; }
        struct Bounds2
        {
            Point2 Minimum {std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()};
            Point2 Maximum {-std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()};
            void Add(Point2 p)
            {
                Minimum.X = std::min(Minimum.X,p.X); Minimum.Y = std::min(Minimum.Y,p.Y);
                Maximum.X = std::max(Maximum.X,p.X); Maximum.Y = std::max(Maximum.Y,p.Y);
            }
            bool Intersects(const Bounds2& b) const
            { return Minimum.X < b.Maximum.X && Maximum.X > b.Minimum.X && Minimum.Y < b.Maximum.Y && Maximum.Y > b.Minimum.Y; }
        };
        struct ProjectedTriangle
        {
            std::array<Point2,3> Corners;
            Bounds2 Bounds;
        };
        struct Triangle
        {
            std::array<Point3,3> Corners;
            std::array<std::uint32_t,3> Neighbors {InvalidLightmapChart,InvalidLightmapChart,InvalidLightmapChart};
            std::array<std::uint32_t,3> Welded {};
            Point3 Normal;
            double DoubleArea = 0;
        };
        struct Chart
        {
            LightmapAtlasChart Output;
            Point3 Origin, U, V, Normal;
            Bounds2 Bounds;
            std::vector<ProjectedTriangle> Projected;
        };
        struct Edge
        {
            std::uint32_t Triangle = 0, Side = 0, Start = 0, Count = 0;
            std::uint32_t SecondTriangle = 0, SecondSide = 0;
        };
        using PositionKey = std::array<std::uint32_t,4>;
        PositionKey Key(Float3 position, std::uint32_t mesh)
        {
            PositionKey key {mesh,0,0,0};
            const float values[] {position.X == 0 ? 0 : position.X, position.Y == 0 ? 0 : position.Y, position.Z == 0 ? 0 : position.Z};
            for (unsigned axis = 0; axis < 3; ++axis) std::memcpy(&key[axis+1],&values[axis],sizeof(float));
            return key;
        }
        bool Overlap(const ProjectedTriangle& a, const ProjectedTriangle& b)
        {
            if (!a.Bounds.Intersects(b.Bounds)) return false;
            const auto origin = a.Corners[0];
            for (const auto* triangle : {&a,&b}) for (unsigned side = 0; side < 3; ++side)
            {
                const auto p = triangle->Corners[side], q = triangle->Corners[(side+1)%3];
                const Point2 axis {p.Y-q.Y,q.X-p.X};
                auto interval = [&](const ProjectedTriangle& candidate)
                {
                    std::pair<double,double> range {std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()};
                    for (auto corner : candidate.Corners)
                    {
                        const double value = (corner.X-origin.X)*axis.X+(corner.Y-origin.Y)*axis.Y;
                        range.first = std::min(range.first,value); range.second = std::max(range.second,value);
                    }
                    return range;
                };
                const auto x = interval(a), y = interval(b);
                const double tolerance = ComparisonEpsilon * std::max({std::abs(x.first),std::abs(x.second),std::abs(y.first),std::abs(y.second)});
                if (std::min(x.second,y.second)-std::max(x.first,y.first) <= tolerance) return false;
            }
            return true;
        }
        ProjectedTriangle Project(const Triangle& triangle, const Chart& chart)
        {
            ProjectedTriangle result;
            for (unsigned corner = 0; corner < 3; ++corner)
            {
                const auto offset = triangle.Corners[corner]-chart.Origin;
                result.Corners[corner] = {offset.Dot(chart.U),offset.Dot(chart.V)};
                result.Bounds.Add(result.Corners[corner]);
            }
            return result;
        }
        void ConnectEdges(std::vector<Triangle>& triangles)
        {
            std::unordered_map<std::uint64_t,Edge> edges;
            for (std::uint32_t index = 0; index < triangles.size(); ++index)
            {
                if (triangles[index].DoubleArea == 0) continue;
                for (unsigned side = 0; side < 3; ++side)
                {
                    const auto a = triangles[index].Welded[side], b = triangles[index].Welded[(side+1)%3];
                    const auto key = (std::uint64_t(std::min(a,b)) << std::numeric_limits<std::uint32_t>::digits) | std::max(a,b);
                    auto& edge = edges[key];
                    if (edge.Count == 0) { edge.Triangle = index; edge.Side = side; edge.Start = a; }
                    else if (edge.Count == 1)
                    {
                        edge.SecondTriangle = index; edge.SecondSide = side;
                        if (edge.Start == b)
                        {
                            triangles[edge.Triangle].Neighbors[edge.Side] = index;
                            triangles[index].Neighbors[side] = edge.Triangle;
                        }
                    }
                    else if (edge.Count == 2)
                    {
                        triangles[edge.Triangle].Neighbors[edge.Side] = InvalidLightmapChart;
                        triangles[edge.SecondTriangle].Neighbors[edge.SecondSide] = InvalidLightmapChart;
                    }
                    ++edge.Count;
                }
            }
        }
        Chart SeedChart(const Triangle& first)
        {
            Chart chart; chart.Origin=first.Corners[0]; chart.Normal=first.Normal;
            auto longest=first.Corners[1]-first.Corners[0];
            for (unsigned side=1;side<3;++side)
            {
                const auto edge=first.Corners[(side+1)%3]-first.Corners[side];
                if (edge.Dot(edge)>longest.Dot(longest)) longest=edge;
            }
            chart.U=longest/longest.Length(); chart.V=chart.Normal.Cross(chart.U); return chart;
        }

        std::vector<Chart> Partition(const std::vector<Triangle>& triangles, const LightmapAtlasSettings& settings,
            std::vector<LightmapAtlasTriangle>& output)
        {
            const double minimumCosine = std::cos(settings.MaximumNormalAngleDegrees*Pi/180);
            std::vector<std::uint32_t> order(triangles.size()), visited(triangles.size(),InvalidLightmapChart);
            std::iota(order.begin(),order.end(),0);
            std::stable_sort(order.begin(),order.end(),[&](auto a, auto b) { return triangles[a].DoubleArea > triangles[b].DoubleArea; });
            std::vector<Chart> charts;
            for (auto seed : order)
            {
                if (output[seed].Chart != InvalidLightmapChart || triangles[seed].DoubleArea == 0) continue;
                const auto id = static_cast<std::uint32_t>(charts.size());
                const auto& first = triangles[seed];
                Chart chart=SeedChart(first);
                std::vector<std::uint32_t> pending {seed}; visited[seed] = id;
                for (std::size_t next = 0; next < pending.size() && chart.Output.Triangles.size() < settings.MaximumChartTriangles; ++next)
                {
                    const auto index = pending[next]; const auto& triangle = triangles[index];
                    if (output[index].Chart != InvalidLightmapChart || triangle.DoubleArea == 0 ||
                        triangle.Normal.Dot(chart.Normal)+ComparisonEpsilon < minimumCosine) continue;
                    auto projected = Project(triangle,chart);
                    if (std::any_of(chart.Projected.begin(),chart.Projected.end(),[&](const auto& prior) { return Overlap(projected,prior); })) continue;
                    output[index].Chart = id; chart.Output.Triangles.push_back(index); chart.Projected.push_back(projected);
                    chart.Bounds.Add(projected.Bounds.Minimum); chart.Bounds.Add(projected.Bounds.Maximum);
                    for (auto neighbor : triangle.Neighbors)
                        if (neighbor != InvalidLightmapChart && visited[neighbor] != id && output[neighbor].Chart == InvalidLightmapChart)
                        { visited[neighbor] = id; pending.push_back(neighbor); }
                }
                charts.push_back(std::move(chart));
            }
            return charts;
        }

        struct Skyline { std::uint32_t X = 0, Y = 0, Width = 0; };
        struct Rectangle { std::uint32_t X = 0, Y = 0, Width = 0, Height = 0; bool Rotated = false; };
        bool Pack(const std::vector<Chart>& charts, const LightmapAtlasSettings& settings, double density,
            std::vector<Rectangle>& output)
        {
            std::vector<Rectangle> rectangles(charts.size());
            std::vector<std::uint32_t> order(charts.size()); std::iota(order.begin(),order.end(),0);
            for (std::size_t i = 0; i < charts.size(); ++i)
            {
                const auto& bounds = charts[i].Bounds;
                const double width = std::ceil((bounds.Maximum.X-bounds.Minimum.X)*density)+1;
                const double height = std::ceil((bounds.Maximum.Y-bounds.Minimum.Y)*density)+1;
                const unsigned gutter = settings.Padding*2;
                if (width+gutter > settings.Resolution || height+gutter > settings.Resolution) return false;
                rectangles[i].Width = std::max(MinimumInteriorPixels,unsigned(width))+gutter;
                rectangles[i].Height = std::max(MinimumInteriorPixels,unsigned(height))+gutter;
            }
            std::stable_sort(order.begin(),order.end(),[&](auto a, auto b)
            {
                const auto& x = rectangles[a]; const auto& y = rectangles[b];
                return std::make_pair(std::max(x.Width,x.Height),std::min(x.Width,x.Height)) >
                    std::make_pair(std::max(y.Width,y.Height),std::min(y.Width,y.Height));
            });
            std::vector<Skyline> skyline {{0,0,settings.Resolution}};
            for (auto id : order)
            {
                auto& rectangle = rectangles[id];
                std::tuple<unsigned,std::uint64_t,unsigned> best {std::numeric_limits<unsigned>::max(),0,0};
                Rectangle chosen;
                for (unsigned rotation = 0; rotation < 2; ++rotation)
                {
                    const unsigned width = rotation ? rectangle.Height : rectangle.Width;
                    const unsigned height = rotation ? rectangle.Width : rectangle.Height;
                    for (std::size_t start = 0; start < skyline.size(); ++start)
                    {
                        if (skyline[start].X+width > settings.Resolution) continue;
                        unsigned remaining = width, y = 0;
                        std::uint64_t coveredArea = 0;
                        for (std::size_t segment = start; segment < skyline.size() && remaining; ++segment)
                        {
                            const unsigned overlap = std::min(remaining,skyline[segment].Width);
                            y = std::max(y,skyline[segment].Y); coveredArea += std::uint64_t(overlap)*skyline[segment].Y;
                            remaining -= overlap;
                        }
                        if (remaining || y+height > settings.Resolution) continue;
                        const auto score = std::make_tuple(y+height,std::uint64_t(width)*y-coveredArea,skyline[start].X);
                        if (score < best) { best = score; chosen = {skyline[start].X,y,width,height,rotation != 0}; }
                    }
                }
                if (std::get<0>(best) == std::numeric_limits<unsigned>::max()) return false;
                rectangle = chosen;
                std::vector<Skyline> updated;
                const unsigned right = chosen.X+chosen.Width;
                for (auto segment : skyline)
                {
                    const unsigned end = segment.X+segment.Width;
                    if (end <= chosen.X || segment.X >= right) { updated.push_back(segment); continue; }
                    if (segment.X < chosen.X) updated.push_back({segment.X,segment.Y,chosen.X-segment.X});
                    if (end > right) updated.push_back({right,segment.Y,end-right});
                }
                updated.push_back({chosen.X,chosen.Y+chosen.Height,chosen.Width});
                std::sort(updated.begin(),updated.end(),[](auto a, auto b) { return a.X < b.X; });
                skyline.clear();
                for (auto segment : updated)
                {
                    if (!skyline.empty() && skyline.back().Y == segment.Y) skyline.back().Width += segment.Width;
                    else skyline.push_back(segment);
                }
            }
            output = std::move(rectangles); return true;
        }
    }

    bool BuildLightmapAtlas(std::shared_ptr<const SceneRayQuery> scene, const LightmapAtlasSettings& settings,
        LightmapAtlas& output, std::string& error)
    {
        error.clear();
        if (!scene || settings.Resolution < MinimumResolution || settings.Resolution > MaximumResolution ||
            settings.Padding == 0 || settings.Padding > MaximumPadding || settings.Padding*2+MinimumInteriorPixels > settings.Resolution ||
            !std::isfinite(settings.MaximumNormalAngleDegrees) || settings.MaximumNormalAngleDegrees < 0 ||
            settings.MaximumNormalAngleDegrees > MaximumProjectionAngle || settings.MaximumChartTriangles == 0 ||
            settings.MaximumChartTriangles > MaximumChartTriangleLimit || scene->GetTriangleCount() > std::numeric_limits<std::uint32_t>::max()/3)
        { error = "Invalid lightmap scene, resolution, gutter, normal angle or chart limit"; return false; }
        LightmapAtlas result; result.Scene = std::move(scene); result.Settings = settings;
        result.Triangles.resize(result.Scene->GetTriangleCount());
        std::vector<Triangle> triangles(result.Triangles.size());
        std::map<PositionKey,std::uint32_t> welded;
        for (std::uint32_t index = 0; index < triangles.size(); ++index)
        {
            SceneRayTriangle source;
            if (!result.Scene->GetTriangle(index,source)) { error = "Lightmap source triangle is missing"; return false; }
            auto& triangle = triangles[index];
            for (unsigned corner = 0; corner < 3; ++corner)
            {
                const auto position = source.Corners[corner].Position;
                triangle.Corners[corner] = Point(position);
                const auto entry = welded.emplace(Key(position,source.MeshIndex),static_cast<std::uint32_t>(welded.size()));
                triangle.Welded[corner] = entry.first->second;
            }
            const auto cross = (triangle.Corners[1]-triangle.Corners[0]).Cross(triangle.Corners[2]-triangle.Corners[0]);
            triangle.DoubleArea = cross.Length();
            if (triangle.DoubleArea > 0) triangle.Normal = cross/triangle.DoubleArea;
            else ++result.DegenerateTriangleCount;
        }
        ConnectEdges(triangles);
        auto charts = Partition(triangles,settings,result.Triangles);
        if (charts.empty()) { output = std::move(result); return true; }
        // Quantization can collapse a small LOD sliver inside a large chart.
        // Refine only failing charts, then repack; never relax the overlap test
        // or drop the source triangle. Each refinement removes a multi-face
        // chart, so this process terminates even on adversarial geometry.
        for (;;)
        {
            double maximumExtent=0;
            for (const auto& chart:charts)
                maximumExtent=std::max({maximumExtent,chart.Bounds.Maximum.X-chart.Bounds.Minimum.X,chart.Bounds.Maximum.Y-chart.Bounds.Minimum.Y});
            std::vector<Rectangle> rectangles;
            if (!Pack(charts,settings,0,rectangles))
            { error="Lightmap atlas cannot fit "+std::to_string(charts.size())+" charts and their gutters; increase resolution or reduce chart splitting"; return false; }
            double low=0,high=(settings.Resolution-settings.Padding*2-1)/maximumExtent;
            for (unsigned iteration=0;iteration<DensitySearchIterations;++iteration)
            {
                const double candidate=(low+high)*0.5; std::vector<Rectangle> packed;
                if (Pack(charts,settings,candidate,packed)) { low=candidate; rectangles=std::move(packed); }
                else high=candidate;
            }
            if (!(low>0)) { error="Lightmap packing could not retain positive texel density"; return false; }
            result.TexelsPerWorldUnit=low;
            std::vector<bool> refine(charts.size(),false); bool needsRefinement=false;
            for (std::size_t id=0;id<charts.size();++id)
            {
                auto& chart=charts[id]; const auto& rectangle=rectangles[id];
                chart.Output.X=rectangle.X; chart.Output.Y=rectangle.Y;
                chart.Output.Width=rectangle.Width; chart.Output.Height=rectangle.Height; chart.Output.Rotated=rectangle.Rotated;
                // Packing already reserves two interior texel centers per axis.
                // Use that minimum span for tiny/needle charts; otherwise a
                // positive double-precision projection may become zero in UV32.
                const double scaleX=std::max(low,MinimumChartSpan/(chart.Bounds.Maximum.X-chart.Bounds.Minimum.X));
                const double scaleY=std::max(low,MinimumChartSpan/(chart.Bounds.Maximum.Y-chart.Bounds.Minimum.Y));
                std::vector<ProjectedTriangle> rounded;
                for (std::size_t triangle=0;triangle<chart.Output.Triangles.size();++triangle)
                {
                    auto& target=result.Triangles[chart.Output.Triangles[triangle]]; target.Chart=static_cast<std::uint32_t>(id);
                    ProjectedTriangle stored;
                    for (unsigned corner=0;corner<3;++corner)
                    {
                        const auto projected=chart.Projected[triangle].Corners[corner];
                        const double u=rectangle.Rotated ? (projected.Y-chart.Bounds.Minimum.Y)*scaleY : (projected.X-chart.Bounds.Minimum.X)*scaleX;
                        const double v=rectangle.Rotated ? (chart.Bounds.Maximum.X-projected.X)*scaleX : (projected.Y-chart.Bounds.Minimum.Y)*scaleY;
                        target.UV[corner]={float((rectangle.X+settings.Padding+TexelCenter+u)/settings.Resolution),
                            float((rectangle.Y+settings.Padding+TexelCenter+v)/settings.Resolution)};
                        stored.Corners[corner]={target.UV[corner].X,target.UV[corner].Y}; stored.Bounds.Add(stored.Corners[corner]);
                    }
                    const auto a=stored.Corners[0],b=stored.Corners[1],c=stored.Corners[2];
                    if ((b.X-a.X)*(c.Y-a.Y)-(b.Y-a.Y)*(c.X-a.X)<=0 ||
                        std::any_of(rounded.begin(),rounded.end(),[&](const auto& prior) { return Overlap(stored,prior); }))
                    {
                        if (chart.Output.Triangles.size()==1)
                        { error="A lightmap triangle cannot retain positive float UV area after precision refinement"; return false; }
                        refine[id]=needsRefinement=true; break;
                    }
                    rounded.push_back(stored);
                }
            }
            if (!needsRefinement)
            {
                for (auto& chart:charts) result.Charts.push_back(std::move(chart.Output));
                break;
            }
            std::vector<Chart> refined;
            for (std::size_t id=0;id<charts.size();++id)
            {
                if (!refine[id]) { refined.push_back(std::move(charts[id])); continue; }
                for (auto triangle:charts[id].Output.Triangles)
                {
                    auto chart=SeedChart(triangles[triangle]); const auto projected=Project(triangles[triangle],chart);
                    chart.Output.Triangles.push_back(triangle); chart.Projected.push_back(projected);
                    chart.Bounds.Add(projected.Bounds.Minimum); chart.Bounds.Add(projected.Bounds.Maximum); refined.push_back(std::move(chart));
                }
            }
            charts=std::move(refined);
        }
        output = std::move(result); return true;
    }
}
