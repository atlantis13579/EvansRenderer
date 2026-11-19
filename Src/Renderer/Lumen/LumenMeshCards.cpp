#include "Renderer/Lumen/LumenMeshCards.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace Evans::Renderer
{
    /* Lumen separates ray intersection from hit lighting. Axis-aligned mesh
       cards parameterize visible surface layers; later capture stores material
       properties, amortized relighting fills radiance, and mesh/global distance
       fields locate hits for screen probes and reflections. These stages need
       matching geometry/material generations before temporal reconstruction.

       This independent builder samples six orthographic directions and records
       the unobstructed near-plane interval preceding each front-facing hit.
       A weighted interval sweep chooses a capture plane, then fits its rectangle
       to the covered surfels. Repeating under a card budget also reaches hidden
       layers instead of collapsing every direction onto an outer box face.
       Coverage-weighted near-plane clusters follow the surface-cache design.
       This variant additionally measures coverage at the requested capture
       resolution. Median/gap splits compete with new layers for the same card
       budget; only a measured gain replaces a parent rectangle. Finite samples
       cannot establish complete mesh coverage or reproduce raster edge rules.
       Wright et al., "Lumen: Real-time Global Illumination in Unreal Engine 5"
       (SIGGRAPH 2022); surface-cache overview:
       https://dev.epicgames.com/documentation/en-us/unreal-engine/lumen-technical-details-in-unreal-engine
    */
    namespace
    {
        constexpr unsigned MaximumResolution=128,MaximumCards=64,MaximumRayLayers=128;
        constexpr std::size_t MaximumSurfels=4*1024*1024,MaximumCoverageRays=4*1024*1024;
        constexpr float RayStepCellFraction=0.0001f;
        constexpr float CapturePaddingCells=1;
        float Dot(Float3 a,Float3 b) { return a.X*b.X+a.Y*b.Y+a.Z*b.Z; }
        Float3 Scale(Float3 a,float x) { return {a.X*x,a.Y*x,a.Z*x}; }
        Float3 Add(Float3 a,Float3 b) { return {a.X+b.X,a.Y+b.Y,a.Z+b.Z}; }
        Float3 Cross(Float3 a,Float3 b) { return {a.Y*b.Z-a.Z*b.Y,a.Z*b.X-a.X*b.Z,a.X*b.Y-a.Y*b.X}; }
        Float3 Axis(unsigned axis) { return {axis==0?1.0f:0.0f,axis==1?1.0f:0.0f,axis==2?1.0f:0.0f}; }
        bool Finite(Float3 a) { return std::isfinite(a.X)&&std::isfinite(a.Y)&&std::isfinite(a.Z); }
        struct Direction
        {
            Float3 Right,Up,Forward,Origin;
            float StepX=0,StepY=0,Depth=0;
            unsigned Width=0,Height=0;
        };
        struct Event { float Plane; double Weight; bool Start; };
        struct Candidate { float Plane=0; double Weight=0; unsigned Direction=0; };
        bool Interrupted(const LumenCardCancellation& cancelled,std::string& error)
        { if (cancelled && cancelled()) { error="Lumen mesh-card construction cancelled"; return true; } return false; }
        Candidate BestPlane(const std::vector<LumenCardSurfel>& surfels)
        {
            Candidate best;
            for (unsigned direction=0;direction<LumenCardDirectionCount;++direction)
            {
                std::vector<Event> events;
                for (const auto& surfel:surfels) if (surfel.Card==LumenUnassignedCard && surfel.Direction==direction)
                { events.push_back({surfel.MinimumPlane,surfel.Weight,true}); events.push_back({surfel.MaximumPlane,surfel.Weight,false}); }
                std::stable_sort(events.begin(),events.end(),[](const auto& a,const auto& b)
                { return a.Plane!=b.Plane?a.Plane<b.Plane:a.Start>b.Start; });
                double weight=0;
                for (const auto& event:events)
                {
                    // Starts precede ends at a shared position: intervals are
                    // closed, with their numerical guard already removed.
                    weight+=event.Start?event.Weight:-event.Weight;
                    if (event.Start && weight>best.Weight) best={event.Plane,weight,direction};
                }
            }
            return best;
        }
        struct CardFit
        {
            LumenMeshCard Card;
            float Plane=0;
            std::vector<unsigned> Members;
            std::vector<std::uint8_t> Coverage; // Unknown, traced miss, traced matching surface.
            double Resolved=0,Unchecked=0;
        };
        struct CardSplit { std::optional<CardFit> First,Second; double Gain=0; };
        CardFit FitCard(const std::vector<LumenCardSurfel>& surfels,const Direction& direction,
            unsigned directionIndex,float plane,float step,std::vector<unsigned> members)
        {
            CardFit fit; fit.Plane=plane; fit.Members=std::move(members); auto& card=fit.Card;
            unsigned minX=direction.Width,minY=direction.Height,maxX=0,maxY=0; float farDepth=plane;
            card.Direction=directionIndex; card.Right=direction.Right; card.Up=direction.Up; card.Forward=direction.Forward;
            for (auto index:fit.Members)
            {
                const auto& surfel=surfels[index]; ++card.SurfelCount; card.Weight+=surfel.Weight;
                minX=std::min(minX,surfel.PixelX); minY=std::min(minY,surfel.PixelY); maxX=std::max(maxX,surfel.PixelX); maxY=std::max(maxY,surfel.PixelY);
                farDepth=std::max(farDepth,surfel.SurfaceDepth+step);
            }
            card.Origin=Add(direction.Origin,Add(Scale(direction.Forward,plane),
                Add(Scale(direction.Right,(minX+maxX+1)*0.5f*direction.StepX),Scale(direction.Up,(minY+maxY+1)*0.5f*direction.StepY))));
            card.Extent={(maxX+1-minX)*direction.StepX*0.5f,(maxY+1-minY)*direction.StepY*0.5f,farDepth-plane};
            return fit;
        }
        bool MeasureCoverage(const SceneRayQuery& mesh,const LumenCardBuildSettings& settings,float cell,float step,
            LumenMeshCards& result,CardFit& fit,std::string& error,const LumenCardCancellation& cancelled)
        {
            struct Pixel { Float3 Normal; float Depth=0; bool Known=false,Hit=false; };
            const auto side=settings.CaptureResolution; std::vector<Pixel> pixels(std::size_t(side)*side);
            const auto& card=fit.Card; const float width=card.Extent.X*2,height=card.Extent.Y*2;
            constexpr float NormalAgreement=0.9f,DepthToleranceCells=0.25f,DepthGuardSteps=4;
            constexpr unsigned CancellationStride=256;
            const float depthTolerance=std::max(cell*DepthToleranceCells,step*DepthGuardSteps);
            fit.Coverage.resize(fit.Members.size());
            for (unsigned ordinal=0;ordinal<fit.Members.size();++ordinal)
            {
                if (ordinal%CancellationStride==0 && Interrupted(cancelled,error)) return false;
                const auto& surfel=result.Surfels[fit.Members[ordinal]];
                const Float3 relative{surfel.Position.X-card.Origin.X,surfel.Position.Y-card.Origin.Y,surfel.Position.Z-card.Origin.Z};
                const float u=Dot(relative,card.Right),v=Dot(relative,card.Up);
                const auto x=unsigned(std::clamp(int(std::floor((u/width+0.5f)*side)),0,int(side)-1));
                const auto y=unsigned(std::clamp(int(std::floor((v/height+0.5f)*side)),0,int(side)-1));
                const float pixelU=(x+0.5f)*width/side-card.Extent.X,pixelV=(y+0.5f)*height/side-card.Extent.Y;
                auto& pixel=pixels[std::size_t(y)*side+x];
                if (!pixel.Known && result.CoverageRays<settings.MaximumCoverageRays)
                {
                    ++result.CoverageRays; pixel.Known=true;
                    const auto origin=Add(card.Origin,Add(Scale(card.Right,pixelU),Scale(card.Up,pixelV))); SceneRayHit hit;
                    pixel.Hit=mesh.Trace(origin,card.Forward,0,card.Extent.Z,hit) && hit.FrontFace;
                    if (pixel.Hit) { pixel.Depth=hit.Distance; pixel.Normal=hit.GeometricNormal; }
                }
                if (!pixel.Known) { result.CoverageBudgetExhausted=true; fit.Unchecked+=surfel.Weight; continue; }
                fit.Coverage[ordinal]=1;
                // Compare at the texel ray, using the surfel tangent plane. A
                // sloped continuous face is not an occluder merely because its
                // neighboring sample has a different depth.
                const float facing=-Dot(surfel.Normal,card.Forward);
                const float expected=Dot(relative,card.Forward)+(Dot(surfel.Normal,card.Right)*(pixelU-u)+Dot(surfel.Normal,card.Up)*(pixelV-v))/facing;
                if (pixel.Hit && Dot(pixel.Normal,surfel.Normal)>=NormalAgreement && std::abs(pixel.Depth-expected)<=depthTolerance)
                { fit.Coverage[ordinal]=2; fit.Resolved+=surfel.Weight; }
            }
            return true;
        }
        bool FindSplit(const SceneRayQuery& mesh,const LumenCardBuildSettings& settings,const std::array<Direction,LumenCardDirectionCount>& directions,
            float cell,float step,LumenMeshCards& result,const CardFit& parent,CardSplit& split,std::string& error,const LumenCardCancellation& cancelled)
        {
            if (!settings.RefineCoverage || parent.Resolved>=parent.Card.Weight*settings.TargetCoverage || result.CoverageBudgetExhausted) return true;
            constexpr double MinimumGainFraction=0.001;
            const auto& direction=directions[parent.Card.Direction];
            for (unsigned axis=0;axis<2;++axis)
            {
                std::vector<double> weights(axis?direction.Height:direction.Width);
                for (auto index:parent.Members) { const auto& s=result.Surfels[index]; weights[axis?s.PixelY:s.PixelX]+=s.Weight; }
                unsigned previous=UINT32_MAX,largestGap=0,gapCut=UINT32_MAX,medianCut=UINT32_MAX; double accumulated=0;
                for (unsigned coordinate=0;coordinate<weights.size();++coordinate) if (weights[coordinate]>0)
                {
                    if (previous!=UINT32_MAX && coordinate-previous>largestGap)
                    { largestGap=coordinate-previous; gapCut=(coordinate+previous)/2; }
                    accumulated+=weights[coordinate];
                    if (medianCut==UINT32_MAX && accumulated>=parent.Card.Weight/2) medianCut=coordinate;
                    previous=coordinate;
                }
                std::vector<unsigned> cuts;
                if (medianCut!=UINT32_MAX) cuts.push_back(medianCut);
                if (largestGap>1 && gapCut!=medianCut) cuts.push_back(gapCut);
                for (auto cut:cuts)
                {
                    std::vector<unsigned> left,right;
                    for (auto index:parent.Members)
                    { const auto& s=result.Surfels[index]; ((axis?s.PixelY:s.PixelX)<=cut?left:right).push_back(index); }
                    if (left.empty() || right.empty()) continue;
                    auto first=FitCard(result.Surfels,direction,parent.Card.Direction,parent.Plane,step,std::move(left));
                    auto second=FitCard(result.Surfels,direction,parent.Card.Direction,parent.Plane,step,std::move(right));
                    if (!MeasureCoverage(mesh,settings,cell,step,result,first,error,cancelled) || !MeasureCoverage(mesh,settings,cell,step,result,second,error,cancelled)) return false;
                    if (result.CoverageBudgetExhausted) return true;
                    const double gain=first.Resolved+second.Resolved-parent.Resolved;
                    if (gain>split.Gain && gain>parent.Card.Weight*MinimumGainFraction)
                    { split.Gain=gain; split.First=std::move(first); split.Second=std::move(second); }
                }
            }
            return true;
        }
    }
    bool BuildLumenMeshCards(const SceneRayQuery& mesh,const LumenCardBuildSettings& settings,
        LumenMeshCards& output,std::string& error,const LumenCardCancellation& cancelled)
    {
        if (!settings.Resolution || settings.Resolution>MaximumResolution || !settings.MaximumCards || settings.MaximumCards>MaximumCards ||
            !settings.MaximumRayLayers || settings.MaximumRayLayers>MaximumRayLayers || !settings.MaximumSurfels || settings.MaximumSurfels>MaximumSurfels ||
            !settings.CaptureResolution || settings.CaptureResolution>MaximumResolution || !settings.MaximumCoverageRays || settings.MaximumCoverageRays>MaximumCoverageRays ||
            !std::isfinite(settings.TargetCoverage) || settings.TargetCoverage<=0 || settings.TargetCoverage>1 ||
            !std::isfinite(settings.MinimumFacingCosine) || settings.MinimumFacingCosine<=0 || settings.MinimumFacingCosine>1)
        { error="Lumen mesh-card settings exceed finite sampling, layer or storage limits"; return false; }
        if (Interrupted(cancelled,error)) return false;
        LumenMeshCards result;
        if (!mesh.GetTriangleCount()) { output=std::move(result); error.clear(); return true; }
        const auto& bounds=mesh.GetBounds();
        const Float3 minimum{bounds.Min.x,bounds.Min.y,bounds.Min.z},maximum{bounds.Max.x,bounds.Max.y,bounds.Max.z};
        const Float3 extent{maximum.X-minimum.X,maximum.Y-minimum.Y,maximum.Z-minimum.Z};
        const float longest=std::max({extent.X,extent.Y,extent.Z});
        if (!bounds.IsValid() || !Finite(minimum) || !Finite(maximum) || !Finite(extent) || longest<=0)
        { error="Lumen mesh-card geometry needs finite nonempty bounds"; return false; }
        const float cell=longest/settings.Resolution;
        const float scale=std::max({longest,std::abs(minimum.X),std::abs(minimum.Y),std::abs(minimum.Z),
            std::abs(maximum.X),std::abs(maximum.Y),std::abs(maximum.Z)});
        // A mesh should normally be supplied in local coordinates. Also account
        // for representable spacing when a caller supplies translated geometry.
        const float step=std::max(cell*RayStepCellFraction,scale*std::numeric_limits<float>::epsilon()*2);
        const float padding=std::max(cell*CapturePaddingCells,step*2);
        if (!std::isfinite(cell+padding+step) || step<=0 || cell<=0)
        { error="Lumen mesh-card bounds exceed representable sampling precision"; return false; }
        std::array<Direction,LumenCardDirectionCount> directions;
        for (unsigned index=0;index<LumenCardDirectionCount;++index)
        {
            if (Interrupted(cancelled,error)) return false;
            auto& direction=directions[index]; const auto normal=Scale(Axis(index/2),(index&1)?1.0f:-1.0f);
            direction.Right=Axis((index/2+1)%3); direction.Up=Cross(normal,direction.Right); direction.Forward=Scale(normal,-1);
            float minU=std::numeric_limits<float>::infinity(),minV=minU,minD=minU,maxU=-minU,maxV=-minU,maxD=-minU;
            for (unsigned corner=0;corner<8;++corner)
            {
                const Float3 point{(corner&1)?maximum.X:minimum.X,(corner&2)?maximum.Y:minimum.Y,(corner&4)?maximum.Z:minimum.Z};
                const float u=Dot(point,direction.Right),v=Dot(point,direction.Up),d=Dot(point,direction.Forward);
                minU=std::min(minU,u); maxU=std::max(maxU,u); minV=std::min(minV,v); maxV=std::max(maxV,v); minD=std::min(minD,d); maxD=std::max(maxD,d);
            }
            if (maxU<=minU || maxV<=minV) continue;
            direction.Width=std::min(settings.Resolution,std::max(1u,unsigned(std::ceil((maxU-minU)/cell))));
            direction.Height=std::min(settings.Resolution,std::max(1u,unsigned(std::ceil((maxV-minV)/cell))));
            direction.StepX=(maxU-minU)/direction.Width; direction.StepY=(maxV-minV)/direction.Height;
            direction.Depth=maxD-minD+padding*2;
            direction.Origin=Add(Add(Scale(direction.Right,minU),Scale(direction.Up,minV)),Scale(direction.Forward,minD-padding));
            if (!Finite(direction.Origin) || !std::isfinite(direction.Depth))
            { error="Lumen mesh-card capture bounds overflowed"; return false; }
            for (unsigned y=0;y<direction.Height;++y)
            {
                if (Interrupted(cancelled,error)) return false;
                for (unsigned x=0;x<direction.Width;++x)
                {
                    const auto origin=Add(direction.Origin,Add(Scale(direction.Right,(x+0.5f)*direction.StepX),Scale(direction.Up,(y+0.5f)*direction.StepY)));
                    ++result.TracedRays; float next=0; unsigned layer=0;
                    for (;layer<settings.MaximumRayLayers;++layer)
                    {
                        SceneRayHit hit; if (!mesh.Trace(origin,direction.Forward,next,direction.Depth,hit)) break;
                        ++result.TracedHits; const float facing=Dot(hit.GeometricNormal,normal),lastPlane=hit.Distance-step;
                        if (hit.FrontFace && facing>=settings.MinimumFacingCosine && lastPlane>=next)
                        {
                            if (result.Surfels.size()>=settings.MaximumSurfels)
                            { error="Lumen mesh-card surfel budget exhausted"; return false; }
                            LumenCardSurfel surfel; surfel.Position=hit.Position; surfel.Normal=hit.GeometricNormal; surfel.Triangle=hit.TriangleIndex;
                            surfel.Direction=index; surfel.PixelX=x; surfel.PixelY=y; surfel.MinimumPlane=next; surfel.MaximumPlane=lastPlane; surfel.SurfaceDepth=hit.Distance;
                            surfel.Weight=double(direction.StepX)*direction.StepY*facing*facing;
                            result.TotalWeight+=surfel.Weight; result.Surfels.push_back(surfel);
                        }
                        // Backfaces do not become cards, but still bound the next
                        // layer's unobstructed capture interval.
                        next=hit.Distance+step;
                        if (!std::isfinite(next) || next>=direction.Depth) break;
                    }
                    if (layer==settings.MaximumRayLayers)
                    { SceneRayHit remaining; if (mesh.Trace(origin,direction.Forward,next,direction.Depth,remaining)) ++result.TruncatedRays; }
                }
            }
        }
        std::vector<CardFit> fits; std::vector<CardSplit> splits;
        std::optional<CardFit> nextLayer; bool searchedLayer=false;
        const auto assign=[&](const CardFit& fit,unsigned index)
        {
            for (unsigned i=0;i<fit.Members.size();++i)
            {
                auto& surfel=result.Surfels[fit.Members[i]]; surfel.Card=index;
                surfel.TexelCoverageKnown=fit.Coverage[i]!=0; surfel.TexelCovered=fit.Coverage[i]==2;
            }
        };
        while (fits.size()<settings.MaximumCards)
        {
            if (Interrupted(cancelled,error)) return false;
            if (!searchedLayer)
            {
                searchedLayer=true; const auto candidate=BestPlane(result.Surfels);
                if (candidate.Weight>0)
                {
                    std::vector<unsigned> members;
                    for (unsigned index=0;index<result.Surfels.size();++index)
                    {
                        const auto& s=result.Surfels[index];
                        if (s.Card==LumenUnassignedCard && s.Direction==candidate.Direction && s.MinimumPlane<=candidate.Plane && s.MaximumPlane>=candidate.Plane) members.push_back(index);
                    }
                    if (members.empty()) { error="Lumen card plane has no visible surfels"; return false; }
                    nextLayer=FitCard(result.Surfels,directions[candidate.Direction],candidate.Direction,candidate.Plane,step,std::move(members));
                    if (!MeasureCoverage(mesh,settings,cell,step,result,*nextLayer,error,cancelled)) return false;
                }
            }
            unsigned best=UINT32_MAX; double gain=0;
            for (unsigned index=0;index<splits.size();++index) if (splits[index].Gain>gain) { gain=splits[index].Gain; best=index; }
            if (best!=UINT32_MAX && (!nextLayer || gain>nextLayer->Resolved))
            {
                auto split=std::move(splits[best]); fits[best]=std::move(*split.First); fits.push_back(std::move(*split.Second));
                splits[best]={}; splits.emplace_back(); ++result.SpatialSplits;
                assign(fits[best],best); assign(fits.back(),unsigned(fits.size()-1));
                if (fits.size()<settings.MaximumCards &&
                    (!FindSplit(mesh,settings,directions,cell,step,result,fits[best],splits[best],error,cancelled) ||
                     !FindSplit(mesh,settings,directions,cell,step,result,fits.back(),splits.back(),error,cancelled))) return false;
            }
            else if (nextLayer)
            {
                fits.push_back(std::move(*nextLayer)); splits.emplace_back(); nextLayer.reset(); searchedLayer=false;
                assign(fits.back(),unsigned(fits.size()-1));
                if (fits.size()<settings.MaximumCards && !FindSplit(mesh,settings,directions,cell,step,result,fits.back(),splits.back(),error,cancelled)) return false;
            }
            else break;
        }
        for (const auto& fit:fits)
        { result.Cards.push_back(fit.Card); result.CoveredWeight+=fit.Card.Weight; result.ResolvedWeight+=fit.Resolved; result.UncheckedWeight+=fit.Unchecked; }
        output=std::move(result); error.clear(); return true;
    }
    Matrix4 LumenCardViewProjection(const LumenMeshCard& card) noexcept
    {
        Matrix4 result;
        const Float3 axes[]{Scale(card.Right,1/card.Extent.X),Scale(card.Up,1/card.Extent.Y),Scale(card.Forward,1/card.Extent.Z)};
        for (unsigned row=0;row<3;++row)
        {
            result.Values[row*4]=axes[row].X; result.Values[row*4+1]=axes[row].Y; result.Values[row*4+2]=axes[row].Z;
            result.Values[row*4+3]=-Dot(axes[row],card.Origin);
        }
        return result;
    }
}
