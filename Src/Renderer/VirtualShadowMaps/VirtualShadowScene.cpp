#include "Renderer/VirtualShadowMaps/VirtualShadowScene.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace Evans::Renderer
{
    namespace
    {
        bool Finite(const Matrix4& value)
        { return std::all_of(value.Values.begin(),value.Values.end(),[](float x){return std::isfinite(x);}); }
        bool Finite(Float3 value) { return std::isfinite(value.X)&&std::isfinite(value.Y)&&std::isfinite(value.Z); }
        bool Same(Float3 a,Float3 b) { return a.X==b.X && a.Y==b.Y && a.Z==b.Z; }
        bool Same(Float4 a,Float4 b) { return a.X==b.X && a.Y==b.Y && a.Z==b.Z && a.W==b.W; }
        bool Same(const VirtualShadowSceneCaster& a,const VirtualShadowSceneCaster& b)
        {
            const auto& x=a.Draw; const auto& y=b.Draw;
            return Same(x.Wind,y.Wind) && Same(x.Gust,y.Gust) && Same(x.VegetationHistory,y.VegetationHistory) &&
                a.GeometryRevision==b.GeometryRevision && a.PreparationRevision==b.PreparationRevision &&
                Same(a.BoundsMinimum,b.BoundsMinimum) && Same(a.BoundsMaximum,b.BoundsMaximum) &&
                x.Vertices.Value==y.Vertices.Value && x.Indices.Value==y.Indices.Value && x.FirstIndex==y.FirstIndex && x.IndexCount==y.IndexCount &&
                x.World.Values==y.World.Values && x.Diffuse.Value==y.Diffuse.Value && x.Opacity.Value==y.Opacity.Value &&
                x.Coverage==y.Coverage && x.Cutoff==y.Cutoff && x.Dithered==y.Dithered && x.BackFaceCulling==y.BackFaceCulling;
        }
        std::array<double,4> Transform(const Matrix4& matrix,const std::array<double,4>& point)
        {
            std::array<double,4> result{};
            for (unsigned row=0;row<4;++row)
                for (unsigned column=0;column<4;++column) result[row]+=double(matrix.Values[row*4+column])*point[column];
            return result;
        }
        void Invalidate(const Matrix4& light,const VirtualShadowSceneCaster& caster,std::vector<VirtualShadowRectangle>& output)
        {
            constexpr double HomogeneousEpsilon=1.0e-7;
            double minimumX=std::numeric_limits<double>::infinity(),minimumY=minimumX,minimumZ=minimumX;
            double maximumX=-minimumX,maximumY=-minimumX,maximumZ=-minimumX;
            for (unsigned corner=0;corner<8;++corner)
            {
                const std::array<double,4> local{(corner&1)?caster.BoundsMaximum.X:caster.BoundsMinimum.X,
                    (corner&2)?caster.BoundsMaximum.Y:caster.BoundsMinimum.Y,(corner&4)?caster.BoundsMaximum.Z:caster.BoundsMinimum.Z,1};
                const auto clip=Transform(light,Transform(caster.Draw.World,local));
                // A box crossing the projection eye plane cannot be bounded by
                // divided corners. Conservatively dirty the complete image.
                if (clip[3]<=HomogeneousEpsilon)
                { output.push_back({{0,0},{1,1}}); return; }
                const double x=clip[0]/clip[3],y=clip[1]/clip[3],z=clip[2]/clip[3];
                minimumX=std::min(minimumX,x); maximumX=std::max(maximumX,x);
                minimumY=std::min(minimumY,y); maximumY=std::max(maximumY,y);
                minimumZ=std::min(minimumZ,z); maximumZ=std::max(maximumZ,z);
            }
            if (minimumX>1 || maximumX<-1 || minimumY>1 || maximumY<-1 || minimumZ>1 || maximumZ<0) return;
            const auto unit=[](double x){return float(std::clamp(x,0.0,1.0));};
            // Canonical shadow UVs have a top-left origin. Page planning expands
            // these bounds by the physical tile gutter at each hierarchy level.
            output.push_back({{unit((minimumX+1)*0.5),unit((1-maximumY)*0.5)},
                {unit((maximumX+1)*0.5),unit((1-minimumY)*0.5)}});
        }
        bool Validate(const std::vector<VirtualShadowSceneCaster>& casters)
        {
            std::string previous;
            for (const auto& caster:casters)
            {
                const auto& draw=caster.Draw;
                if (caster.Key.empty() || (!previous.empty() && caster.Key<=previous) || !Finite(draw.World) ||
                    !Finite(caster.BoundsMinimum) || !Finite(caster.BoundsMaximum) ||
                    caster.BoundsMinimum.X>caster.BoundsMaximum.X || caster.BoundsMinimum.Y>caster.BoundsMaximum.Y || caster.BoundsMinimum.Z>caster.BoundsMaximum.Z ||
                    !draw.Vertices || !draw.Indices || !draw.IndexCount || draw.IndexCount%3 ||
                    !std::isfinite(draw.Coverage) || !std::isfinite(draw.Cutoff) || draw.Coverage<0 || draw.Coverage>1 || draw.Cutoff<0 || draw.Cutoff>1) return false;
                previous=caster.Key;
            }
            return true;
        }
    }
    bool BuildVirtualShadowSceneUpdate(const Matrix4& light,std::vector<VirtualShadowSceneCaster> casters,
        const VirtualShadowSceneState* presented,VirtualShadowSceneUpdate& output,std::string& error)
    {
        std::sort(casters.begin(),casters.end(),[](const auto& a,const auto& b){return a.Key<b.Key;});
        if (!Finite(light) || !Validate(casters))
        { error="Virtual shadow scene requires unique caster keys, finite ordered bounds and canonical material coverage"; return false; }
        if (presented && (!Validate(presented->Casters) || !Finite(presented->Identity.LightViewProjection) ||
            !presented->Identity.Epoch || !presented->Identity.Serial || presented->Identity.PreviousSerial>=presented->Identity.Serial ||
            !presented->Identity.CasterRevision || presented->Identity.Serial==UINT64_MAX || presented->Identity.CasterRevision==UINT64_MAX))
        { error="Virtual shadow scene cannot continue an invalid or exhausted presented version"; return false; }
        VirtualShadowSceneUpdate result; result.Scene.Casters=std::move(casters);
        auto& identity=result.Scene.Identity; identity.LightViewProjection=light;
        if (presented)
        {
            identity.Epoch=presented->Identity.Epoch; identity.PreviousSerial=presented->Identity.Serial;
            identity.Serial=identity.PreviousSerial+1; identity.CasterRevision=presented->Identity.CasterRevision;
            const auto& previous=presented->Casters; const auto& current=result.Scene.Casters;
            std::size_t oldIndex=0,newIndex=0;
            while (oldIndex<previous.size() || newIndex<current.size())
            {
                if (newIndex==current.size() || (oldIndex<previous.size() && previous[oldIndex].Key<current[newIndex].Key))
                    Invalidate(light,previous[oldIndex++],result.Invalidations);
                else if (oldIndex==previous.size() || current[newIndex].Key<previous[oldIndex].Key)
                    Invalidate(light,current[newIndex++],result.Invalidations);
                else
                {
                    if (!Same(previous[oldIndex],current[newIndex]))
                    { Invalidate(light,previous[oldIndex],result.Invalidations); Invalidate(light,current[newIndex],result.Invalidations); }
                    ++oldIndex; ++newIndex;
                }
            }
            // Wholly off-light changes alter no depth texel. Retain the content
            // revision, but still publish their new records for future movement.
            if (!result.Invalidations.empty()) ++identity.CasterRevision;
        }
        output=std::move(result); error.clear(); return true;
    }
}
