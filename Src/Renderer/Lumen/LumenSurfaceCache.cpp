#include "Renderer/Lumen/LumenSurfaceCache.h"

#include <atomic>
#include <cstring>

namespace Evans::Renderer
{
    namespace
    {
        std::atomic<std::uint64_t> NextSurfaceIdentity{1};
        bool Finite(const Matrix4& value)
        { return std::all_of(value.Values.begin(),value.Values.end(),[](float x){return std::isfinite(x);}); }
        bool Finite(Float3 value) { return std::isfinite(value.X) && std::isfinite(value.Y) && std::isfinite(value.Z); }
        bool Unit(float value) { return std::isfinite(value) && value>=0 && value<=1; }
        bool Positive(float value) { return std::isfinite(value) && value>=0; }
        bool ValidMaterial(const LumenSurfaceDraw& draw)
        {
            return Finite(draw.World) && Unit(draw.BaseColor.X) && Unit(draw.BaseColor.Y) && Unit(draw.BaseColor.Z) && Unit(draw.BaseColor.W) &&
                Positive(draw.Emission.X) && Positive(draw.Emission.Y) && Positive(draw.Emission.Z) && Unit(draw.Roughness) && Unit(draw.Metallic) && Unit(draw.Cutoff);
        }
    }
    LumenSurfaceCapture LumenCardSurfaceCapture(const LumenMeshCard& card,unsigned firstDraw,unsigned drawCount) noexcept
    {
        LumenSurfaceCapture result; result.ViewProjection=LumenCardViewProjection(card);
        result.ViewProjection.Values[3]=result.ViewProjection.Values[7]=result.ViewProjection.Values[11]=0;
        result.ViewOrigin=card.Origin; result.FirstDraw=firstDraw; result.DrawCount=drawCount; result.ProjectionGroup=card.Direction; return result;
    }
    bool BuildLumenSurfaceLayout(unsigned cardCount,unsigned cardSide,LumenSurfaceLayout& output,std::string& error)
    {
        if (!cardCount || cardCount>LumenSurfaceMaximumCards || !cardSide || cardSide>LumenSurfaceMaximumSide)
        { error="Lumen surface cache needs a bounded nonzero card count and texel side"; return false; }
        LumenSurfaceLayout result; result.CardCount=cardCount; result.CardSide=cardSide; result.Columns=1;
        while (result.Columns*result.Columns<cardCount) ++result.Columns;
        result.Rows=(cardCount+result.Columns-1)/result.Columns;
        const auto stride=cardSide+LumenSurfaceBorder*2; result.Width=result.Columns*stride; result.Height=result.Rows*stride;
        if (result.Width>LumenSurfaceMaximumAtlasSide || result.Height>LumenSurfaceMaximumAtlasSide || std::size_t(result.Width)*result.Height>LumenSurfaceMaximumTexels)
        { error="Lumen surface cache exceeds its native atlas storage budget"; return false; }
        output=result; error.clear(); return true;
    }
    bool ValidateLumenSurfaceUpload(const LumenSurfaceUpload& upload,LumenSurfaceLayout& output,std::string& error)
    {
        LumenSurfaceLayout layout;
        if (!upload.Identity.Epoch || !upload.Identity.Serial || !upload.Identity.Geometry || !upload.Identity.Material ||
            upload.Cards.size()>LumenSurfaceMaximumCards || upload.Draws.size()>LumenSurfaceMaximumDraws)
        { error="Lumen surface capture requires nonzero version identities and bounded draw ranges"; return false; }
        if (!BuildLumenSurfaceLayout(unsigned(upload.Cards.size()),upload.CardSide,layout,error)) return false;
        std::uint64_t commands=0;
        for (const auto& card:upload.Cards)
        {
            commands+=card.DrawCount;
            if (!Finite(card.ViewProjection) || !Finite(card.ViewOrigin) || card.ProjectionGroup>=LumenCardDirectionCount || std::uint64_t(card.FirstDraw)+card.DrawCount>upload.Draws.size())
            { error="Lumen surface card references an invalid capture projection or draw range"; return false; }
        }
        if (commands>LumenSurfaceMaximumDraws) { error="Lumen surface capture exceeds its draw-command budget"; return false; }
        for (const auto& draw:upload.Draws)
            if (!draw.Vertices || !draw.Indices || !draw.IndexCount || draw.IndexCount%3 || !ValidMaterial(draw))
            { error="Lumen surface draw requires native triangles, finite transforms and physical material values"; return false; }
        output=layout; error.clear(); return true;
    }
    bool ValidateLumenSurfaceDraw(const LumenSurfaceDraw& draw,const BufferDesc& vertices,const BufferDesc& indices,
        const std::vector<std::uint8_t>& indexMirror,std::string& error)
    {
        if (!draw.Vertices || !draw.Indices || !draw.IndexCount || draw.IndexCount%3 || !ValidMaterial(draw) ||
            vertices.Usage!=BufferUsage::Vertex || !vertices.SizeInBytes || vertices.SizeInBytes%sizeof(Vertex) ||
            indices.Usage!=BufferUsage::Index || indices.SizeInBytes%sizeof(std::uint32_t) || indexMirror.size()!=indices.SizeInBytes ||
            std::uint64_t(draw.FirstIndex)+draw.IndexCount>indices.SizeInBytes/sizeof(std::uint32_t))
        { error="Lumen surface draw exceeds its native triangle buffers"; return false; }
        for (std::uint64_t index=draw.FirstIndex;index<std::uint64_t(draw.FirstIndex)+draw.IndexCount;++index)
        {
            std::uint32_t vertex=0; std::memcpy(&vertex,indexMirror.data()+std::size_t(index)*sizeof(vertex),sizeof(vertex));
            if (vertex>=vertices.SizeInBytes/sizeof(Vertex)) { error="Lumen surface index exceeds its vertex buffer"; return false; }
        }
        error.clear(); return true;
    }
    bool BuildLumenSurfaceConstants(const LumenSurfaceLayout& layout,const LumenSurfaceCapture& capture,
        const LumenSurfaceDraw& draw,LumenSurfaceConstants& output,std::string& error)
    {
        if (!layout.CardSide || !ValidMaterial(draw) || !Finite(capture.ViewProjection) || !Finite(capture.ViewOrigin))
        { error="Lumen surface constants require a valid capture and material"; return false; }
        LumenSurfaceConstants result; result.RelativeWorld=draw.World; result.NormalWorld=BuildRealtimeNormalMatrix(draw.World);
        // Keep origin subtraction before the projection multiply. Combining both
        // transforms loses depth precision for thin cards at large coordinates.
        const double origin[]{capture.ViewOrigin.X,capture.ViewOrigin.Y,capture.ViewOrigin.Z};
        for (unsigned row=0;row<3;++row) for (unsigned column=0;column<4;++column)
            result.RelativeWorld.Values[row*4+column]=float(double(draw.World.Values[row*4+column])-origin[row]*draw.World.Values[12+column]);
        result.ViewProjection=capture.ViewProjection;
        // A tile's one-texel gutter is rendered from the same capture plane.
        // Shrink clip XY so interior pixels retain the card's original footprint.
        const float fraction=float(layout.CardSide)/(layout.CardSide+LumenSurfaceBorder*2);
        for (unsigned component=0;component<8;++component) result.ViewProjection.Values[component]*=fraction;
        if (!Finite(result.ViewProjection) || !Finite(result.RelativeWorld) || !Finite(result.NormalWorld))
        { error="Lumen surface transform exceeds native float precision"; return false; }
        result.BaseColor=draw.BaseColor; result.EmissionRoughness={draw.Emission.X,draw.Emission.Y,draw.Emission.Z,draw.Roughness};
        result.Surface={draw.Metallic,draw.Cutoff,0,0}; result.Textures={draw.Diffuse?1u:0u,draw.Normal?1u:0u,draw.Opacity?1u:0u,0};
        output=result; error.clear(); return true;
    }
    bool ValidateLumenSurfaceTexels(const LumenSurfaceLayout& layout,const std::vector<LumenSurfaceTexel>& texels,std::string& error)
    {
        LumenSurfaceLayout expected;
        if (!BuildLumenSurfaceLayout(layout.CardCount,layout.CardSide,expected,error) || expected.Width!=layout.Width || expected.Height!=layout.Height ||
            expected.Columns!=layout.Columns || expected.Rows!=layout.Rows || texels.size()!=std::size_t(layout.Width)*layout.Height)
        { error="Lumen surface readback does not match its atlas extent"; return false; }
        constexpr float NormalTolerance=0.002f;
        for (const auto& texel:texels)
        {
            const auto& a=texel.AlbedoMetallic; const auto& n=texel.NormalDepth; const auto& e=texel.EmissionRoughness;
            const float length=n.X*n.X+n.Y*n.Y+n.Z*n.Z;
            if (!Unit(a.X) || !Unit(a.Y) || !Unit(a.Z) || !Unit(a.W) || !Unit(n.W) || !Positive(e.X) || !Positive(e.Y) || !Positive(e.Z) || !Unit(e.W) ||
                !std::isfinite(length) || (length!=0 && std::abs(length-1)>NormalTolerance))
            {
                error="Lumen surface readback has invalid values at texel "+std::to_string(&texel-texels.data())+
                    " (albedo="+std::to_string(a.X)+","+std::to_string(a.Y)+","+std::to_string(a.Z)+", metallic="+std::to_string(a.W)+
                    ", depth="+std::to_string(n.W)+", normal squared="+std::to_string(length)+", emission="+std::to_string(e.X)+","+std::to_string(e.Y)+","+std::to_string(e.Z)+", roughness="+std::to_string(e.W)+")";
                return false;
            }
        }
        error.clear(); return true;
    }
    LumenSurfaceHandle NewLumenSurfaceHandle() noexcept
    {
        auto value=NextSurfaceIdentity.load(std::memory_order_relaxed);
        while (value && value!=UINT64_MAX)
            if (NextSurfaceIdentity.compare_exchange_weak(value,value+1,std::memory_order_relaxed)) return {value};
        return {};
    }
}
