#include "Renderer/Lumen/LumenSurfaceLookup.h"
#include "Renderer/Lumen/LumenDistanceScene.h"

#include <cstring>

namespace Evans::Renderer
{
    namespace
    {
        bool Finite(Float4 v) { return std::isfinite(v.X) && std::isfinite(v.Y) && std::isfinite(v.Z) && std::isfinite(v.W); }
        bool Unit(float v) { return std::isfinite(v) && v>=0 && v<=1; }
        bool ValidLayout(const LumenSurfaceLayout& layout,std::string& error)
        {
            LumenSurfaceLayout expected;
            if (!BuildLumenSurfaceLayout(layout.CardCount,layout.CardSide,expected,error)) return false;
            if (layout.Columns!=expected.Columns || layout.Rows!=expected.Rows || layout.Width!=expected.Width || layout.Height!=expected.Height)
            { error="Lumen lookup needs the original canonical atlas layout"; return false; }
            return true;
        }
        bool InvertProjection(const Matrix4& source,Matrix4& output)
        {
            const auto& m=source.Values;
            if (!std::all_of(m.begin(),m.end(),[](float x){return std::isfinite(x);}) || m[12] || m[13] || m[14] || m[15]!=1) return false;
            const double a[]{m[0],m[1],m[2],m[4],m[5],m[6],m[8],m[9],m[10]};
            const double adjugate[]{a[4]*a[8]-a[5]*a[7],a[2]*a[7]-a[1]*a[8],a[1]*a[5]-a[2]*a[4],
                a[5]*a[6]-a[3]*a[8],a[0]*a[8]-a[2]*a[6],a[2]*a[3]-a[0]*a[5],
                a[3]*a[7]-a[4]*a[6],a[1]*a[6]-a[0]*a[7],a[0]*a[4]-a[1]*a[3]};
            const double determinant=a[0]*adjugate[0]+a[1]*adjugate[3]+a[2]*adjugate[6];
            if (!std::isfinite(determinant) || determinant==0) return false;
            Matrix4 inverse;
            for (unsigned row=0;row<3;++row)
            {
                double translation=0;
                for (unsigned column=0;column<3;++column)
                { const auto value=adjugate[row*3+column]/determinant; inverse.Values[row*4+column]=float(value); translation-=value*m[column*4+3]; }
                inverse.Values[row*4+3]=float(translation);
            }
            if (!std::all_of(inverse.Values.begin(),inverse.Values.end(),[](float x){return std::isfinite(x);})) return false;
            // A finite inverse can still be unusable after float conversion.
            // Check both products before allowing native unprojection to use it.
            constexpr double MaximumInverseResidual=0.001;
            for (unsigned order=0;order<2;++order) for (unsigned row=0;row<3;++row) for (unsigned column=0;column<4;++column)
            {
                const auto& left=order?inverse.Values:m; const auto& right=order?m:inverse.Values; double value=0,magnitude=0;
                for (unsigned k=0;k<4;++k) { const double product=double(left[row*4+k])*right[k*4+column]; value+=product; magnitude+=std::abs(product); }
                const double expected=row==column?1:0;
                if (std::abs(value-expected)>MaximumInverseResidual*std::max(1.0,magnitude)) return false;
            }
            output=inverse; return true;
        }
    }
    bool BuildLumenLookupCards(const LumenSurfaceLayout& layout,const std::vector<LumenSurfaceCapture>& captures,std::vector<LumenLookupCard>& output,std::string& error)
    {
        if (!ValidLayout(layout,error)) return false;
        if (captures.size()!=layout.CardCount) { error="Lumen lookup lost the capture projection metadata"; return false; }
        std::vector<LumenLookupCard> cards; cards.reserve(captures.size());
        const unsigned stride=layout.CardSide+LumenSurfaceBorder*2;
        for (unsigned index=0;index<captures.size();++index)
        {
            const auto& capture=captures[index]; LumenLookupCard card; card.Projection=capture.ViewProjection;
            card.Origin={capture.ViewOrigin.X,capture.ViewOrigin.Y,capture.ViewOrigin.Z,0};
            if (!Finite(card.Origin) || capture.ProjectionGroup>=LumenCardDirectionCount || !InvertProjection(card.Projection,card.InverseProjection))
            { error="Lumen lookup requires finite invertible affine card projections and valid direction groups"; return false; }
            const auto& inverse=card.InverseProjection.Values;
            const double length=std::hypot(double(inverse[2]),double(inverse[6]),double(inverse[10]));
            float scale=0; for (unsigned row=0;row<3;++row) for (unsigned column=0;column<4;++column) scale=std::max(scale,std::abs(inverse[row*4+column]));
            if (!length || !std::isfinite(length) || !scale) { error="Lumen lookup projection has no finite depth direction"; return false; }
            card.BackwardScale={float(-inverse[2]/length),float(-inverse[6]/length),float(-inverse[10]/length),scale};
            card.Tile={(index%layout.Columns)*stride,(index/layout.Columns)*stride,capture.ProjectionGroup,index}; cards.push_back(card);
        }
        output=std::move(cards); error.clear(); return true;
    }
    bool BuildLumenRadianceMapping(const LumenSurfaceLayout& layout,const std::vector<LumenSurfaceCapture>& captures,
        unsigned instanceCount,const std::vector<unsigned>& cardInstances,LumenRadianceMapping& output,std::string& error)
    {
        if (instanceCount>LumenDistanceMaximumInstances || cardInstances.size()!=layout.CardCount)
        { error="Lumen radiance mapping needs one bounded owner per card"; return false; }
        std::vector<LumenLookupCard> cards; if (!BuildLumenLookupCards(layout,captures,cards,error)) return false;
        std::vector<std::vector<unsigned>> ownership(instanceCount);
        for (unsigned card=0;card<cardInstances.size();++card)
        {
            const auto instance=cardInstances[card]; if (instance==LumenDistanceInvalid) continue;
            if (instance>=ownership.size()) { error="Lumen radiance card addresses an absent source instance"; return false; }
            auto& group=ownership[instance]; group.push_back(card);
            if (group.size()>LumenLookupMaximumCandidates) { error="Lumen radiance instance exceeds its bounded card candidates"; return false; }
        }
        constexpr unsigned VectorWords=sizeof(ShaderUInt4)/sizeof(std::uint32_t); LumenRadianceMapping result;
        result.Words.resize(ownership.size()*VectorWords);
        for (unsigned instance=0;instance<ownership.size();++instance)
        {
            const auto& group=ownership[instance]; result.Words[instance*VectorWords]=unsigned(result.Words.size()); result.Words[instance*VectorWords+1]=unsigned(group.size());
            result.Words.insert(result.Words.end(),group.begin(),group.end());
        }
        result.Words.resize((result.Words.size()+VectorWords-1)/VectorWords*VectorWords);
        const auto cardOffset=result.Words.size(); result.CardVectorOffset=unsigned(cardOffset/VectorWords);
        result.Words.resize(cardOffset+cards.size()*sizeof(LumenLookupCard)/sizeof(std::uint32_t));
        std::memcpy(result.Words.data()+cardOffset,cards.data(),cards.size()*sizeof(LumenLookupCard));
        output=std::move(result); error.clear(); return true;
    }
    bool BuildLumenSurfaceLookupFrame(const LumenSurfaceLayout& layout,const LumenSurfaceLookupUpload& upload,LumenSurfaceLookupFrame& output,std::string& error)
    {
        if (!ValidLayout(layout,error)) return false;
        if (upload.Queries.empty() || upload.Queries.size()>LumenLookupMaximumQueries || !Unit(upload.MinimumFacingCosine) || upload.MinimumFacingCosine==0)
        { error="Lumen lookup needs bounded queries and a positive facing threshold"; return false; }
        constexpr double UnitNormalTolerance=0.0001;
        for (const auto& query:upload.Queries)
        {
            const auto& p=query.PositionRadius; const auto& n=query.NormalMinimumCosine;
            const double squared=double(n.X)*n.X+double(n.Y)*n.Y+double(n.Z)*n.Z;
            if (!Finite(p) || p.W<0 || !Finite(n) || !Unit(n.W) || std::abs(squared-1)>UnitNormalTolerance ||
                query.Range.Y>LumenLookupMaximumCandidates || std::uint64_t(query.Range.X)+query.Range.Y>layout.CardCount || query.Range.Z || query.Range.W)
            { error="Lumen lookup needs finite world radii, unit normals and bounded per-instance card ranges"; return false; }
        }
        LumenSurfaceLookupFrame frame; frame.Counts={unsigned(upload.Queries.size()),layout.CardSide,layout.Columns,layout.CardSide+LumenSurfaceBorder*2};
        frame.Atlas={layout.Width,layout.Height,LumenLookupThreads*LumenLookupColumns,layout.CardCount};
        constexpr float CoordinateRoundoff=16*std::numeric_limits<float>::epsilon();
        // World hit positions lose up to half an ULP on conversion to float,
        // even when capture-relative geometry preserves thinner depth layers.
        constexpr float WorldPositionRoundoff=std::numeric_limits<float>::epsilon()*0.5f;
        frame.Parameters={upload.MinimumFacingCosine,CoordinateRoundoff,WorldPositionRoundoff,0}; output=frame; error.clear(); return true;
    }
    bool ValidateLumenSurfaceLookupResult(const LumenSurfaceIdentity& identity,const LumenSurfaceLookupUpload& upload,const LumenSurfaceLookupResult& result,std::string& error)
    {
        if (identity.Epoch!=result.Identity.Epoch || identity.Serial!=result.Identity.Serial || identity.Geometry!=result.Identity.Geometry || identity.Material!=result.Identity.Material ||
            upload.Queries.size()!=result.Samples.size()) { error="Lumen surface lookup returned a different cache version or query count"; return false; }
        constexpr double NormalTolerance=0.002;
        for (unsigned index=0;index<result.Samples.size();++index)
        {
            const auto& sample=result.Samples[index]; const auto& a=sample.AlbedoMetallic; const auto& n=sample.NormalCoverage; const auto& e=sample.EmissionRoughness;
            const auto& metadata=sample.Metadata; const auto& range=upload.Queries[index].Range;
            if (!Finite(a) || !Finite(n) || !Finite(e) || metadata.W || metadata.Z>1 || !Unit(a.X) || !Unit(a.Y) || !Unit(a.Z) || !Unit(a.W) ||
                !Unit(n.W) || e.X<0 || e.Y<0 || e.Z<0 || !Unit(e.W)) { error="Lumen surface lookup returned invalid physical material data"; return false; }
            const double squared=double(n.X)*n.X+double(n.Y)*n.Y+double(n.Z)*n.Z;
            if (!metadata.Z)
            {
                if (metadata.X!=LumenLookupMissingCard || metadata.Y || squared || n.W || a.X || a.Y || a.Z || a.W || e.X || e.Y || e.Z || e.W)
                { error="Uncovered Lumen lookup retained a stale material"; return false; }
            }
            else if (metadata.X<range.X || std::uint64_t(metadata.X)>=std::uint64_t(range.X)+range.Y || !metadata.Y || metadata.Y>std::min(range.Y,LumenCardDirectionCount) ||
                std::abs(squared-1)>NormalTolerance || !n.W) { error="Lumen surface lookup returned an invalid card or normal"; return false; }
        }
        error.clear(); return true;
    }
}
