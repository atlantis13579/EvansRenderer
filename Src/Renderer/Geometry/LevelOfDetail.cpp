#include "Renderer/Geometry/LevelOfDetail.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <tuple>

namespace Evans::Renderer
{
    /*
    LOD geometry and selection contracts
    -----------------------------------
    Garland and Heckbert, "Surface Simplification Using Quadric Error Metrics"
    (SIGGRAPH 1997), https://www.cs.cmu.edu/~garland/Papers/quadrics.pdf, motivates
    summing squared incident-plane distances to rank edge contractions. This
    implementation restricts contraction to existing endpoints and preserves
    local topology rather than aggregating disconnected components.

    A separate radius follows each cluster: R(to) = max(R(to), R(from) + |edge|).
    The triangle inequality bounds every original vertex's displacement to its
    surviving representative. Barycentric correspondence extends the bound over
    triangles, including triangles collapsed onto retained edges/vertices. This
    is deliberately more conservative than the quadric fitting error. Attribute
    radii independently bound UV, normalized-normal and RGBA changes.

    Renderer builds/uploads chains outside an open GPU frame and selects the
    finest level required by the main, shadow and reflection views. All passes
    share that geometry. Selection history advances only after presentation;
    changing level invalidates that mesh's rigid velocity history. Each view
    projects the displacement bound at its own resolution against the same pixel
    budget. CPU outlines consume the same selected indices. The original bounds
    remain valid because endpoint contraction never moves vertices outside them.
    */
    namespace
    {
        constexpr std::uint32_t TriangleCorners = 3;
        constexpr std::uint32_t MaximumGeneratedLevels = 8;
        constexpr double Pi = 3.14159265358979323846;
        constexpr double DegreesPerHalfTurn = 180.0;
        constexpr double FeatureNormalCosine = 0.5; // Preserve folds sharper than 60 degrees.
        constexpr double MinimumFaceNormalSquared = 1e-24; // Coordinates normalized by mesh extent.
        constexpr double MinimumRemainingAreaRatio = 1e-6;
        constexpr std::size_t HeapRebuildMultiplier = 16;
        constexpr double RelativeClipTolerance = 1e-12;

        struct Vector
        {
            double X = 0, Y = 0, Z = 0;
            Vector operator-(const Vector& b) const { return {X - b.X, Y - b.Y, Z - b.Z}; }
        };
        double Dot(const Vector& a, const Vector& b) { return a.X * b.X + a.Y * b.Y + a.Z * b.Z; }
        Vector Cross(const Vector& a, const Vector& b)
        {
            return {a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X};
        }
        double Length(const Vector& v) { return std::sqrt(Dot(v, v)); }
        Vector Unit(const Float3& value)
        {
            const Vector v{value.X, value.Y, value.Z};
            const double length = Length(v);
            return length > 0 ? Vector{v.X / length, v.Y / length, v.Z / length} : Vector{};
        }
        struct Quadric
        {
            // Full symmetric rows are kept explicit for teaching/readability.
            std::array<double, 16> Values{};
            void AddPlane(const Vector& normal, double offset)
            {
                const std::array<double, 4> plane{normal.X, normal.Y, normal.Z, offset};
                for (std::size_t row = 0; row < plane.size(); ++row)
                    for (std::size_t column = 0; column < plane.size(); ++column)
                        Values[row * plane.size() + column] += plane[row] * plane[column];
            }
            void Add(const Quadric& b)
            {
                for (std::size_t i = 0; i < Values.size(); ++i) Values[i] += b.Values[i];
            }
            double Evaluate(const Vector& point) const
            {
                const std::array<double, 4> p{point.X, point.Y, point.Z, 1};
                double result = 0;
                for (std::size_t row = 0; row < p.size(); ++row)
                    for (std::size_t column = 0; column < p.size(); ++column)
                        result += p[row] * Values[row * p.size() + column] * p[column];
                return std::max(0.0, result); // Remove cancellation noise, not geometric error.
            }
        };
        struct VertexState
        {
            Vector Position, Normal;
            Quadric Cost;
            std::set<std::uint32_t> Faces;
            double PositionRadius = 0, NormalRadius = 0, TexCoordRadius = 0, ColorRadius = 0;
            std::uint64_t Revision = 0;
            bool Locked = false, Alive = true;
        };
        struct Face
        {
            std::array<std::uint32_t, TriangleCorners> Vertices;
            bool Alive = true;
        };
        struct Edge
        {
            std::vector<std::uint32_t> Faces;
            int OrientationSum = 0;
        };
        struct Contraction
        {
            double Cost = 0, Radius = 0;
            std::uint32_t From = 0, To = 0;
            std::uint64_t FromRevision = 0, ToRevision = 0;
        };
        struct HigherCost
        {
            bool operator()(const Contraction& a, const Contraction& b) const
            {
                return std::tie(a.Cost, a.Radius, a.From, a.To) > std::tie(b.Cost, b.Radius, b.From, b.To);
            }
        };
        using Queue = std::priority_queue<Contraction, std::vector<Contraction>, HigherCost>;

        class Simplifier
        {
        public:
            Simplifier(const std::vector<Vertex>& source, const LodBuildSettings& settings)
                : Source(source), Settings(settings), Vertices(source.size()) {}

            void Initialize(const std::vector<std::uint32_t>& indices,const std::vector<std::uint32_t>& lockedVertices,
                const std::vector<double>& initialNormalRadii)
            {
                Vector minimum{Source.front().Position.X, Source.front().Position.Y, Source.front().Position.Z};
                Vector maximum = minimum;
                for (const auto& v : Source)
                {
                    minimum.X = std::min(minimum.X, double(v.Position.X)); maximum.X = std::max(maximum.X, double(v.Position.X));
                    minimum.Y = std::min(minimum.Y, double(v.Position.Y)); maximum.Y = std::max(maximum.Y, double(v.Position.Y));
                    minimum.Z = std::min(minimum.Z, double(v.Position.Z)); maximum.Z = std::max(maximum.Z, double(v.Position.Z));
                }
                Scale = std::max({maximum.X - minimum.X, maximum.Y - minimum.Y, maximum.Z - minimum.Z});
                if (Scale == 0) Scale = 1;
                std::map<std::tuple<float, float, float>, std::vector<std::uint32_t>> coincident;
                for (std::uint32_t i = 0; i < Source.size(); ++i)
                {
                    auto& vertex = Vertices[i];
                    const auto& p = Source[i].Position;
                    vertex.Position = {(double(p.X) - minimum.X) / Scale, (double(p.Y) - minimum.Y) / Scale,
                        (double(p.Z) - minimum.Z) / Scale};
                    vertex.Normal = Unit(Source[i].Normal);
                    if (!initialNormalRadii.empty()) vertex.NormalRadius=initialNormalRadii[i];
                    vertex.Locked = Dot(vertex.Normal, vertex.Normal) == 0;
                    coincident[{p.X, p.Y, p.Z}].push_back(i);
                }
                // Do not weld coincident attributes: separate shells can share a
                // location without sharing topology, and UV seams need fixed rims.
                for (const auto& group : coincident)
                    if (group.second.size() > 1) for (auto index : group.second) Vertices[index].Locked = true;
                for (auto index : lockedVertices) Vertices[index].Locked = true;

                std::map<std::pair<std::uint32_t, std::uint32_t>, Edge> edges;
                std::map<std::array<std::uint32_t, TriangleCorners>, std::uint32_t> uniqueFaces;
                for (std::size_t offset = 0; offset < indices.size(); offset += TriangleCorners)
                {
                    const auto faceId = static_cast<std::uint32_t>(Faces.size());
                    Face face{{indices[offset], indices[offset + 1], indices[offset + 2]}};
                    Faces.push_back(face);
                    auto key = face.Vertices; std::sort(key.begin(), key.end());
                    const bool duplicate = !uniqueFaces.emplace(key, faceId).second;
                    Vector normal = FaceNormal(face);
                    const double lengthSquared = Dot(normal, normal);
                    const bool degenerate = lengthSquared <= MinimumFaceNormalSquared;
                    const double length = degenerate ? 1 : std::sqrt(lengthSquared);
                    normal = {normal.X / length, normal.Y / length, normal.Z / length};
                    for (std::uint32_t corner = 0; corner < TriangleCorners; ++corner)
                    {
                        const auto a = face.Vertices[corner], b = face.Vertices[(corner + 1) % TriangleCorners];
                        Vertices[a].Faces.insert(faceId);
                        Vertices[a].Locked = Vertices[a].Locked || degenerate || duplicate;
                        if (!degenerate) Vertices[a].Cost.AddPlane(normal, -Dot(normal, Vertices[a].Position));
                        auto& edge = edges[std::minmax(a, b)];
                        edge.Faces.push_back(faceId); edge.OrientationSum += a < b ? 1 : -1;
                    }
                }
                for (const auto& pair : edges)
                {
                    const auto& edge = pair.second;
                    bool fixed = edge.Faces.size() != 2 || edge.OrientationSum != 0;
                    if (!fixed)
                    {
                        const auto a = FaceNormal(Faces[edge.Faces[0]]), b = FaceNormal(Faces[edge.Faces[1]]);
                        fixed = Dot(a, b) < FeatureNormalCosine * Length(a) * Length(b);
                    }
                    if (fixed) Vertices[pair.first.first].Locked = Vertices[pair.first.second].Locked = true;
                }
                // Edge valence alone does not detect a bow-tie vertex. Its link
                // must also be one connected cycle before any incident collapse.
                for (std::uint32_t i = 0; i < Vertices.size(); ++i)
                    if (!Vertices[i].Locked && !ConnectedLink(i)) Vertices[i].Locked = true;
                NormalLimit = 2 * std::sin(Settings.MaximumNormalAngleDegrees * Pi / DegreesPerHalfTurn / 2);
                FaceNormalCosine = std::cos(Settings.MaximumNormalAngleDegrees * Pi / DegreesPerHalfTurn);
                AliveFaces = Faces.size();
                RebuildQueue();
                HeapLimit = std::max<std::size_t>(1, Pending.size()) * HeapRebuildMultiplier;
            }

            MeshLodChain Generate(const std::vector<std::uint32_t>& original)
            {
                MeshLodChain result;
                result.Levels.push_back({original, 0});
                for (const auto& vertex : Vertices) result.LockedVertexCount += vertex.Locked ? 1 : 0;
                while (result.Levels.size() < Settings.MaximumLevels && AliveFaces > Settings.MinimumTriangles)
                {
                    const auto before = AliveFaces;
                    const auto target = std::max<std::size_t>(Settings.MinimumTriangles,
                        static_cast<std::size_t>(std::floor(double(before) * Settings.TriangleRatio)));
                    while (AliveFaces > target && !Pending.empty())
                    {
                        const auto edge = Pending.top(); Pending.pop();
                        if (!Vertices[edge.From].Alive || !Vertices[edge.To].Alive ||
                            edge.FromRevision != Vertices[edge.From].Revision || edge.ToRevision != Vertices[edge.To].Revision) continue;
                        if (!ValidTopology(edge.From, edge.To)) continue;
                        Contract(edge.From, edge.To);
                        ++result.CollapseCount;
                        if (Pending.size() > HeapLimit) RebuildQueue();
                    }
                    if (AliveFaces == before) break;
                    MeshLodLevel level;
                    level.MaximumDisplacement = MaximumRadius * Scale;
                    level.Indices.reserve(AliveFaces * TriangleCorners);
                    for (const auto& face : Faces)
                        if (face.Alive) level.Indices.insert(level.Indices.end(), face.Vertices.begin(), face.Vertices.end());
                    result.Levels.push_back(std::move(level));
                }
                return result;
            }

        private:
            Vector FaceNormal(const Face& face) const
            {
                const auto& p = Vertices[face.Vertices[0]].Position;
                return Cross(Vertices[face.Vertices[1]].Position - p, Vertices[face.Vertices[2]].Position - p);
            }
            std::set<std::uint32_t> Neighbors(std::uint32_t vertex) const
            {
                std::set<std::uint32_t> result;
                for (auto faceId : Vertices[vertex].Faces)
                    for (auto other : Faces[faceId].Vertices) if (other != vertex) result.insert(other);
                return result;
            }
            bool ConnectedLink(std::uint32_t vertex) const
            {
                std::map<std::uint32_t, std::set<std::uint32_t>> link;
                for (auto faceId : Vertices[vertex].Faces)
                {
                    const auto& corners = Faces[faceId].Vertices;
                    for (std::uint32_t i = 0; i < TriangleCorners; ++i) if (corners[i] == vertex)
                    {
                        auto a = corners[(i + 1) % TriangleCorners], b = corners[(i + 2) % TriangleCorners];
                        link[a].insert(b); link[b].insert(a);
                    }
                }
                if (link.empty()) return false;
                for (const auto& entry : link) if (entry.second.size() != 2) return false;
                std::set<std::uint32_t> visited, frontier{link.begin()->first};
                while (!frontier.empty())
                {
                    auto next = *frontier.begin(); frontier.erase(frontier.begin());
                    if (!visited.insert(next).second) continue;
                    for (auto neighbor : link[next]) if (!visited.count(neighbor)) frontier.insert(neighbor);
                }
                return visited.size() == link.size();
            }
            double TexCoordDistance(std::uint32_t a, std::uint32_t b) const
            {
                return std::hypot(double(Source[a].TexCoord.X) - Source[b].TexCoord.X,
                    double(Source[a].TexCoord.Y) - Source[b].TexCoord.Y);
            }
            double ColorDistance(std::uint32_t a, std::uint32_t b) const
            {
                const auto& x = Source[a].Color; const auto& y = Source[b].Color;
                return std::max({std::abs(double(x.X) - y.X), std::abs(double(x.Y) - y.Y),
                    std::abs(double(x.Z) - y.Z), std::abs(double(x.W) - y.W)});
            }
            void Enqueue(std::uint32_t from, std::uint32_t to)
            {
                const auto& a = Vertices[from]; const auto& b = Vertices[to];
                if (!a.Alive || !b.Alive || a.Locked || b.Locked ||
                    a.NormalRadius + Length(a.Normal - b.Normal) > NormalLimit ||
                    a.TexCoordRadius + TexCoordDistance(from, to) > Settings.MaximumTexCoordDisplacement ||
                    a.ColorRadius + ColorDistance(from, to) > Settings.MaximumColorDisplacement) return;
                const double radius = std::max(b.PositionRadius, a.PositionRadius + Length(a.Position - b.Position));
                Quadric sum = a.Cost; sum.Add(b.Cost);
                Pending.push({sum.Evaluate(b.Position), radius, from, to, a.Revision, b.Revision});
            }
            void RebuildQueue()
            {
                Pending = {};
                for (std::uint32_t i = 0; i < Vertices.size(); ++i)
                    if (Vertices[i].Alive && !Vertices[i].Locked) for (auto neighbor : Neighbors(i)) Enqueue(i, neighbor);
            }
            bool ValidTopology(std::uint32_t from, std::uint32_t to) const
            {
                auto a = Neighbors(from), b = Neighbors(to);
                if (!a.count(to)) return false;
                std::set<std::uint32_t> common, opposite;
                std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::inserter(common, common.end()));
                std::uint32_t sharedFaces = 0;
                for (auto id : Vertices[from].Faces)
                {
                    const auto& f = Faces[id].Vertices;
                    if (std::find(f.begin(), f.end(), to) == f.end()) continue;
                    ++sharedFaces;
                    for (auto index : f) if (index != from && index != to) opposite.insert(index);
                }
                // The link condition prevents pinching a handle or merging fans.
                if (sharedFaces != 2 || opposite.size() != 2 || common != opposite ||
                    AliveFaces < Settings.MinimumTriangles + sharedFaces) return false;
                auto affected = Vertices[from].Faces;
                affected.insert(Vertices[to].Faces.begin(), Vertices[to].Faces.end());
                std::set<std::array<std::uint32_t, TriangleCorners>> unique;
                for (auto id : affected)
                {
                    const auto& old = Faces[id];
                    Face next = old;
                    for (auto& corner : next.Vertices) if (corner == from) corner = to;
                    auto key = next.Vertices; std::sort(key.begin(), key.end());
                    if (key[0] == key[1] || key[1] == key[2]) continue; // Exactly the two edge faces disappear.
                    // Also reject duplicate surviving faces (e.g. a tetrahedron).
                    if (!unique.insert(key).second) return false;
                    if (!Vertices[from].Faces.count(id)) continue;
                    const auto oldNormal = FaceNormal(old), newNormal = FaceNormal(next);
                    const double oldArea = Length(oldNormal), newArea = Length(newNormal);
                    if (newArea <= oldArea * MinimumRemainingAreaRatio ||
                        Dot(oldNormal, newNormal) <= FeatureNormalCosine * oldArea * newArea) return false;
                    // Endpoint attributes do not bound the new face plane. Keep
                    // coarse facets within the retained shading-normal cone so
                    // curved surfaces do not fold away from incoming light.
                    for (auto corner : next.Vertices)
                        if (Dot(newNormal, Vertices[corner].Normal) < FaceNormalCosine * newArea) return false;
                }
                return true;
            }
            void Contract(std::uint32_t from, std::uint32_t to)
            {
                auto affected = Neighbors(from), other = Neighbors(to);
                affected.insert(other.begin(), other.end()); affected.insert(from); affected.insert(to);
                auto& a = Vertices[from]; auto& b = Vertices[to];
                b.Cost.Add(a.Cost);
                b.PositionRadius = std::max(b.PositionRadius, a.PositionRadius + Length(a.Position - b.Position));
                b.NormalRadius = std::max(b.NormalRadius, a.NormalRadius + Length(a.Normal - b.Normal));
                b.TexCoordRadius = std::max(b.TexCoordRadius, a.TexCoordRadius + TexCoordDistance(from, to));
                b.ColorRadius = std::max(b.ColorRadius, a.ColorRadius + ColorDistance(from, to));
                MaximumRadius = std::max(MaximumRadius, b.PositionRadius);
                const auto faces = a.Faces;
                for (auto id : faces)
                {
                    auto& face = Faces[id];
                    for (auto index : face.Vertices) Vertices[index].Faces.erase(id);
                    for (auto& index : face.Vertices) if (index == from) index = to;
                    const auto& f = face.Vertices;
                    if (f[0] == f[1] || f[1] == f[2] || f[2] == f[0]) { face.Alive = false; --AliveFaces; }
                    else for (auto index : f) Vertices[index].Faces.insert(id);
                }
                a.Alive = false;
                for (auto index : affected) ++Vertices[index].Revision;
                // A neighbor's link can change even when its quadric is unchanged.
                // Refresh all incident candidates after revision invalidation.
                for (auto index : affected) if (Vertices[index].Alive)
                    for (auto neighbor : Neighbors(index))
                    {
                        Enqueue(index, neighbor);
                        if (!affected.count(neighbor)) Enqueue(neighbor, index);
                    }
            }

            const std::vector<Vertex>& Source;
            const LodBuildSettings& Settings;
            std::vector<VertexState> Vertices;
            std::vector<Face> Faces;
            Queue Pending;
            std::size_t AliveFaces = 0, HeapLimit = 0;
            double Scale = 1, NormalLimit = 0, FaceNormalCosine = 1, MaximumRadius = 0;
        };
    }

    bool BuildMeshLods(const std::vector<Vertex>& vertices, const std::vector<std::uint32_t>& indices,
        const LodBuildSettings& settings, MeshLodChain& output, std::string& error,const std::vector<std::uint32_t>& lockedVertices)
    { return BuildMeshLods(vertices,indices,settings,output,error,lockedVertices,{}); }

    bool BuildMeshLods(const std::vector<Vertex>& vertices,const std::vector<std::uint32_t>& indices,
        const LodBuildSettings& settings,MeshLodChain& output,std::string& error,const std::vector<std::uint32_t>& lockedVertices,
        const std::vector<double>& initialNormalRadii)
    {
        if (vertices.empty() || vertices.size() > std::numeric_limits<std::uint32_t>::max() || indices.empty() ||
            indices.size() % TriangleCorners || indices.size() / TriangleCorners > std::numeric_limits<std::uint32_t>::max())
        { error = "LOD requires indexed triangle-list geometry"; return false; }
        if (!settings.MaximumLevels || settings.MaximumLevels > MaximumGeneratedLevels || !settings.MinimumTriangles ||
            !std::isfinite(settings.TriangleRatio) || settings.TriangleRatio <= 0 || settings.TriangleRatio >= 1 ||
            !std::isfinite(settings.MaximumNormalAngleDegrees) || settings.MaximumNormalAngleDegrees < 0 || settings.MaximumNormalAngleDegrees > DegreesPerHalfTurn ||
            !std::isfinite(settings.MaximumTexCoordDisplacement) || settings.MaximumTexCoordDisplacement < 0 ||
            !std::isfinite(settings.MaximumColorDisplacement) || settings.MaximumColorDisplacement < 0)
        { error = "LOD build settings are outside their finite ranges"; return false; }
        if (!initialNormalRadii.empty() && initialNormalRadii.size()!=vertices.size())
        { error="LOD initial normal bounds must match the vertex array"; return false; }
        const double normalLimit=2*std::sin(settings.MaximumNormalAngleDegrees*Pi/DegreesPerHalfTurn/2);
        for (auto radius:initialNormalRadii) if (!std::isfinite(radius) || radius<0 || radius>normalLimit)
        { error="LOD initial normal bounds exceed the angular budget"; return false; }
        for (auto index : indices) if (index >= vertices.size())
        { error = "LOD index is outside the vertex array"; return false; }
        for (auto index : lockedVertices) if (index >= vertices.size())
        { error = "LOD fixed vertex is outside the vertex array"; return false; }
        for (const auto& v : vertices)
            for (float value : {v.Position.X, v.Position.Y, v.Position.Z, v.Normal.X, v.Normal.Y, v.Normal.Z,
                v.TexCoord.X, v.TexCoord.Y, v.Color.X, v.Color.Y, v.Color.Z, v.Color.W})
                if (!std::isfinite(value)) { error = "LOD vertex attributes must be finite"; return false; }
        Simplifier simplifier(vertices, settings);
        simplifier.Initialize(indices,lockedVertices,initialNormalRadii);
        output = simplifier.Generate(indices);
        error.clear();
        return true;
    }

    double ProjectLodError(const LodView& view, double localDisplacement)
    {
        const double invalid = std::numeric_limits<double>::infinity();
        if (!view.Width || !view.Height || !std::isfinite(localDisplacement) || localDisplacement < 0) return invalid;
        for (const auto* matrix : {&view.World, &view.ViewProjection})
            for (auto value : matrix->Values) if (!std::isfinite(value)) return invalid;
        std::array<double, 16> transform{};
        constexpr std::size_t MatrixSide = 4;
        for (std::size_t row = 0; row < MatrixSide; ++row)
            for (std::size_t column = 0; column < MatrixSide; ++column)
                for (std::size_t k = 0; k < MatrixSide; ++k)
                    transform[row * MatrixSide + column] += double(view.ViewProjection.Values[row * MatrixSide + k]) * view.World.Values[k * MatrixSide + column];
        const std::array<double, 3> minimum{view.BoundsMinimum.X, view.BoundsMinimum.Y, view.BoundsMinimum.Z};
        const std::array<double, 3> maximum{view.BoundsMaximum.X, view.BoundsMaximum.Y, view.BoundsMaximum.Z};
        for (std::size_t axis = 0; axis < minimum.size(); ++axis)
            if (!std::isfinite(minimum[axis]) || !std::isfinite(maximum[axis]) || minimum[axis] > maximum[axis]) return invalid;
        double minimumW = invalid, maximumNdcX = 0, maximumNdcY = 0;
        constexpr unsigned BoxCorners = 8;
        for (unsigned corner = 0; corner < BoxCorners; ++corner)
        {
            const std::array<double, 4> p{corner & 1 ? maximum[0] : minimum[0], corner & 2 ? maximum[1] : minimum[1],
                corner & 4 ? maximum[2] : minimum[2], 1};
            std::array<double, 4> clip{};
            for (std::size_t row = 0; row < MatrixSide; ++row)
                for (std::size_t k = 0; k < MatrixSide; ++k) clip[row] += transform[row * MatrixSide + k] * p[k];
            const double magnitude = std::max({std::abs(clip[0]), std::abs(clip[1]), std::abs(clip[2]), std::abs(clip[3])});
            const double tolerance = magnitude * RelativeClipTolerance;
            if (clip[3] <= tolerance || clip[2] <= tolerance) return invalid;
            minimumW = std::min(minimumW, clip[3]);
            maximumNdcX = std::max(maximumNdcX, std::abs(clip[0] / clip[3]));
            maximumNdcY = std::max(maximumNdcY, std::abs(clip[1] / clip[3]));
        }
        const auto rowLength = [&](std::size_t row)
        {
            const auto offset = row * MatrixSide;
            return std::hypot(transform[offset], transform[offset + 1], transform[offset + 2]);
        };
        const double deltaW = rowLength(3) * localDisplacement;
        if (deltaW >= minimumW) return invalid;
        // For q = x/w, |q' - q| <= (|dx| + |q| |dw|) / (w - |dw|).
        // A positive-W rational linear function reaches its box extrema at corners.
        const double denominator = minimumW - deltaW;
        const double x = (rowLength(0) * localDisplacement + maximumNdcX * deltaW) / denominator;
        const double y = (rowLength(1) * localDisplacement + maximumNdcY * deltaW) / denominator;
        return std::hypot(x * view.Width / 2, y * view.Height / 2);
    }

    std::uint32_t SelectMeshLod(const MeshLodChain& chain, const LodView& view,
        double maximumErrorPixels, double hysteresisFraction, std::uint32_t previousLevel)
    {
        if (chain.Levels.empty() || chain.Levels.front().MaximumDisplacement != 0 ||
            !std::isfinite(maximumErrorPixels) || maximumErrorPixels <= 0 ||
            !std::isfinite(hysteresisFraction) || hysteresisFraction < 0 || hysteresisFraction >= 1) return 0;
        double previousError = 0;
        for (std::size_t index = 0; index < chain.Levels.size(); ++index)
        {
            const auto& level = chain.Levels[index];
            if (level.Indices.empty() || level.Indices.size() % TriangleCorners ||
                (index && level.Indices.size() >= chain.Levels[index - 1].Indices.size()) ||
                !std::isfinite(level.MaximumDisplacement) || level.MaximumDisplacement < previousError) return 0;
            previousError = level.MaximumDisplacement;
        }
        if (previousLevel >= chain.Levels.size()) previousLevel = 0;
        std::uint32_t selected = 0;
        for (std::uint32_t i = 1; i < chain.Levels.size(); ++i)
        {
            const double limit = i > previousLevel ? maximumErrorPixels * (1 - hysteresisFraction) : maximumErrorPixels;
            if (ProjectLodError(view, chain.Levels[i].MaximumDisplacement) <= limit) selected = i;
        }
        return selected;
    }
}
