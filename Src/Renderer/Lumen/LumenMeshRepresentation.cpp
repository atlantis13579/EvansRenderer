#include "Renderer/Lumen/LumenMeshRepresentation.h"
#include "Renderer/Lumen/LumenSurfaceLookup.h"

#include <algorithm>
#include <cmath>

namespace Evans::Renderer
{
    namespace
    {
        bool HasArea(Float3 a,Float3 b,Float3 c)
        {
            const double ab[]{double(b.X)-a.X,double(b.Y)-a.Y,double(b.Z)-a.Z},ac[]{double(c.X)-a.X,double(c.Y)-a.Y,double(c.Z)-a.Z};
            return ab[1]*ac[2]!=ab[2]*ac[1] || ab[2]*ac[0]!=ab[0]*ac[2] || ab[0]*ac[1]!=ab[1]*ac[0];
        }
    }
    bool BuildLumenMeshRepresentation(const SceneRayMeshView& source,const Techniques::Settings& techniques,
        const LumenMeshBuildSettings& settings,LumenMeshRepresentation& output,std::string& error,const LumenPartitionCancellation& cancelled)
    {
        if (!source.Vertices || !source.Indices || !source.Visible || !settings.MaximumSurfels || !settings.MaximumDistanceBytes ||
            settings.MaximumSurfels>16*1024*1024 || settings.MaximumDistanceBytes>LumenDistanceMaximumWords*sizeof(std::uint32_t))
        { error="Lumen mesh preprocessing requires visible geometry and bounded aggregate storage"; return false; }
        LumenMeshRepresentation result;
        if (!BuildLumenMeshPartition(*source.Vertices,*source.Indices,settings.Partition,result.Geometry,error,cancelled)) return false;
        result.Cards.resize(result.Geometry.Parts.size()); result.Fields.resize(result.Geometry.Parts.size());
        if (settings.HardwareRayTracing) result.Hardware.resize(result.Geometry.Parts.size());
        for (unsigned index=0;index<result.Geometry.Parts.size();++index)
        {
            if (cancelled && cancelled()) { error="Lumen mesh representation cancelled"; return false; }
            const auto& part=result.Geometry.Parts[index]; if (part.SurfaceArea==0) continue;
            auto local=source; local.Vertices=&part.Vertices; local.Indices=&part.Indices; local.World={};
            std::vector<unsigned> surfaceTriangles;
            for (unsigned triangle=0;triangle<part.SourceTriangles.size();++triangle)
            {
                const auto first=std::size_t(triangle)*3;
                if (HasArea(part.Vertices[part.Indices[first]].Position,part.Vertices[part.Indices[first+1]].Position,part.Vertices[part.Indices[first+2]].Position)) surfaceTriangles.push_back(triangle);
            }
            std::vector<Vertex> surfaceVertices; std::vector<std::uint32_t> surfaceIndices;
            const bool filtered=surfaceTriangles.size()!=part.SourceTriangles.size();
            if (filtered)
            {
                // A budget may group a degenerate component with a real shell.
                // Keep its source mapping/draw, but do not turn a line or point
                // into an SDF obstacle or let unused positions enlarge the grid.
                std::vector<unsigned> remap(part.Vertices.size(),UINT32_MAX);
                for (const auto triangle:surfaceTriangles) for (unsigned corner=0;corner<3;++corner)
                {
                    const auto original=part.Indices[std::size_t(triangle)*3+corner]; auto& mapped=remap[original];
                    if (mapped==UINT32_MAX) { mapped=unsigned(surfaceVertices.size()); surfaceVertices.push_back(part.Vertices[original]); }
                    surfaceIndices.push_back(mapped);
                }
                local.Vertices=&surfaceVertices; local.Indices=&surfaceIndices;
            }
            const auto query=SceneRayQuery::Create({local},techniques,error); if (!query) return false;
            if (!query->GetTriangleCount()) { error="Lumen geometric proxies require an opaque or cutout material"; return false; }
            if (settings.HardwareRayTracing)
            {
                auto hardware=std::make_shared<LumenHardwareMesh>();
                if (!BuildLumenHardwareMesh(*query,*hardware,error)) return false;
                result.Hardware[index]=std::move(hardware);
            }
            if (!BuildLumenMeshCards(*query,settings.Cards,result.Cards[index],error,cancelled)) return false;
            if (filtered) for (auto& surfel:result.Cards[index].Surfels) surfel.Triangle=surfaceTriangles[surfel.Triangle];
            result.SurfelCount+=result.Cards[index].Surfels.size(); result.CardCount+=unsigned(result.Cards[index].Cards.size());
            if (result.SurfelCount>settings.MaximumSurfels || result.CardCount>LumenSurfaceMaximumCards)
            { error="Lumen mesh parts exceed aggregate surfel/card storage"; return false; }
            auto field=std::make_shared<LumenDistanceField>();
            if (!BuildLumenDistanceField(*query,settings.Distance,*field,error,cancelled)) return false;
            result.DistanceBytes+=field->Directory.size()*sizeof(std::uint32_t)+field->Bricks.size()*sizeof(std::int16_t);
            if (result.DistanceBytes>settings.MaximumDistanceBytes) { error="Lumen mesh parts exceed aggregate distance storage"; return false; }
            result.Fields[index]=std::move(field);
        }
        output=std::move(result); error.clear(); return true;
    }
    bool BuildLumenMeshPlacement(const LumenMeshRepresentation& mesh,const Matrix4& world,
        const std::vector<LumenSurfaceDraw>& draws,unsigned cardSide,LumenMeshPlacement& output,std::string& error)
    {
        const auto count=mesh.Geometry.Parts.size(); const auto& m=world.Values;
        if (count>LumenPartitionMaximumParts || draws.size()!=count || mesh.Cards.size()!=count || mesh.Fields.size()!=count ||
            !std::all_of(m.begin(),m.end(),[](float v){return std::isfinite(v);}) || m[12] || m[13] || m[14] || m[15]!=1)
        { error="Lumen mesh instance needs matching part arrays and a finite affine transform"; return false; }
        // Invert only the linear placement in double. Capture-relative origins
        // keep large translations out of the thin-depth projection product.
        const double a[]{m[0],m[1],m[2],m[4],m[5],m[6],m[8],m[9],m[10]};
        const double inverse[]{a[4]*a[8]-a[5]*a[7],a[2]*a[7]-a[1]*a[8],a[1]*a[5]-a[2]*a[4],
            a[5]*a[6]-a[3]*a[8],a[0]*a[8]-a[2]*a[6],a[2]*a[3]-a[0]*a[5],
            a[3]*a[7]-a[4]*a[6],a[1]*a[6]-a[0]*a[7],a[0]*a[4]-a[1]*a[3]};
        const double determinant=a[0]*inverse[0]+a[1]*inverse[3]+a[2]*inverse[6];
        if (!std::isfinite(determinant) || determinant==0) { error="Lumen mesh instance has a singular linear transform"; return false; }
        LumenMeshPlacement result; result.Surface.CardSide=cardSide; result.Parts.resize(count); auto& distance=result.Distance;
        for (unsigned index=0;index<count;++index)
        {
            auto& range=result.Parts[index]; range.FirstCard=unsigned(result.Surface.Cards.size());
            const auto& cards=mesh.Cards[index].Cards; const auto& geometry=mesh.Geometry.Parts[index];
            if (geometry.SurfaceArea==0)
            {
                if (mesh.Fields[index] || !cards.empty()) { error="Lumen zero-area part unexpectedly owns a surface proxy"; return false; }
                continue;
            }
            if (!mesh.Fields[index] || cards.size()>LumenLookupMaximumCandidates ||
                result.Surface.Cards.size()+cards.size()>LumenSurfaceMaximumCards)
            { error="Lumen mesh instance has missing distance fields or excessive card ranges"; return false; }
            range.DistanceInstance=unsigned(distance.Instances.size());
            distance.Instances.push_back({unsigned(distance.Fields.size()),index,world}); distance.Fields.push_back(mesh.Fields[index]);
            auto draw=draws[index]; draw.World=world;
            if (draw.IndexCount!=geometry.Indices.size()) { error="Lumen mesh capture draw must contain every triangle of its compact part"; return false; }
            const auto drawIndex=unsigned(result.Surface.Draws.size()); result.Surface.Draws.push_back(draw);
            for (const auto& card:cards)
            {
                auto capture=LumenCardSurfaceCapture(card,drawIndex,1); const auto projection=capture.ViewProjection;
                for (unsigned row=0;row<3;++row)
                {
                    for (unsigned column=0;column<3;++column)
                    {
                        double value=0; for (unsigned k=0;k<3;++k) value+=double(projection.Values[row*4+k])*inverse[k*3+column]/determinant;
                        capture.ViewProjection.Values[row*4+column]=float(value);
                    }
                }
                const double p[]{card.Origin.X,card.Origin.Y,card.Origin.Z}; float origin[3]{};
                for (unsigned row=0;row<3;++row) { double value=m[row*4+3]; for (unsigned k=0;k<3;++k) value+=double(m[row*4+k])*p[k]; origin[row]=float(value); }
                capture.ViewOrigin={origin[0],origin[1],origin[2]}; result.Surface.Cards.push_back(capture);
            }
            range.CardCount=unsigned(cards.size());
        }
        if (!result.Surface.Cards.empty())
        {
            LumenSurfaceLayout layout; std::vector<LumenLookupCard> lookupCards;
            if (!ValidateLumenSurfaceUpload(result.Surface,layout,error) || !BuildLumenLookupCards(layout,result.Surface.Cards,lookupCards,error)) return false;
        }
        output=std::move(result); error.clear(); return true;
    }
    bool BuildLumenMeshInstance(const LumenMeshRepresentation& mesh,const Matrix4& world,
        const std::vector<LumenSurfaceDraw>& draws,unsigned cardSide,LumenMeshInstanceData& output,std::string& error)
    {
        LumenMeshPlacement placement;
        if (!BuildLumenMeshPlacement(mesh,world,draws,cardSide,placement,error)) return false;
        LumenMeshInstanceData result; result.Surface=std::move(placement.Surface); result.Parts=std::move(placement.Parts);
        // Empty ranges retain explicit misses while the atlas still has a
        // valid reserved tile; no card is manufactured for an uncovered part.
        if (result.Surface.Cards.empty()) result.Surface.Cards.resize(1);
        LumenSurfaceLayout layout; std::vector<LumenLookupCard> lookupCards;
        if (!ValidateLumenSurfaceUpload(result.Surface,layout,error) || !BuildLumenLookupCards(layout,result.Surface.Cards,lookupCards,error) ||
            !BuildLumenDistanceScene(placement.Distance,result.Distance,error)) return false;
        output=std::move(result); error.clear(); return true;
    }
}
