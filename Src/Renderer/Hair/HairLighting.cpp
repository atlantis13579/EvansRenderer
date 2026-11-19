#include "Renderer/Hair/HairLighting.h"
#include "Renderer/Lighting/AreaLightSampling.h"

#include <cmath>
#include <limits>

namespace Evans::Renderer
{
    namespace
    {
        bool Finite(Float4 value)
        { return std::isfinite(value.X) && std::isfinite(value.Y) && std::isfinite(value.Z) && std::isfinite(value.W); }
        Float3 Unit(Float3 value)
        {
            const double length=std::hypot(double(value.X),double(value.Y),double(value.Z));
            return length>0 ? Float3{float(value.X/length),float(value.Y/length),float(value.Z/length)}:Float3{0,1,0};
        }
    }
    bool BuildHairLightSamples(const LightingState& lighting,HairLightSamples& output,std::string& error)
    {
        error.clear(); auto source=lighting.SceneLights;
        if (!source)
        {
            const auto count=std::min(lighting.LightCount,MaximumSceneLights);
            std::vector<SceneLight> lights(lighting.Lights.begin(),lighting.Lights.begin()+count);
            source=SceneLightCollection::Create(std::move(lights),count ? lighting.ShadowLightIndex:-1,error);
            if (!source) return false;
        }
        HairLightSamples result; result.SourceLights=source;
        const auto append=[&](const SceneLight& light,Float3 position,Float3 direction,float weight)
        {
            const double intensity=std::max(double(light.Intensity),0.0)*weight;
            const double radiance[]{std::max(double(light.Color.X),0.0)*intensity,std::max(double(light.Color.Y),0.0)*intensity,
                std::max(double(light.Color.Z),0.0)*intensity};
            for (double value:radiance) if (!std::isfinite(value) || value>std::numeric_limits<float>::max())
            { error="Hair emitted radiance exceeds native floating-point storage"; return false; }
            const HairLightSample sample{{position.X,position.Y,position.Z,float(light.Type)},
                {direction.X,direction.Y,direction.Z,light.Range},
                {float(radiance[0]),float(radiance[1]),float(radiance[2]),0}};
            if (!Finite(sample.PositionKind) || !Finite(sample.DirectionRange) || !Finite(sample.Radiance) ||
                result.Samples.size()==std::numeric_limits<std::uint32_t>::max())
            { error="Hair lighting sample count or radiance is unrepresentable"; return false; }
            result.Samples.push_back(sample); return true;
        };
        for (std::size_t index=0;index<source->GetLights().size();++index)
        {
            const auto& light=source->GetLights()[index]; const auto first=static_cast<std::uint32_t>(result.Samples.size());
            if (light.Type==LightType::Directional)
            { if (!append(light,{},Unit(light.Direction),1)) return false; }
            else if (light.Type==LightType::Point)
            { if (!append(light,light.Position,{},1)) return false; }
            else
            {
                const auto quad=BuildRectangleEmitterQuadrature(light);
                for (std::uint32_t y=0;y<quad.SamplesPerAxis;++y) for (std::uint32_t x=0;x<quad.SamplesPerAxis;++x)
                {
                    const auto p=quad.Position(x,y),n=quad.Normal;
                    if (!append(light,{p.x,p.y,p.z},{n.x,n.y,n.z},quad.Weight)) return false;
                }
            }
            if (static_cast<std::int32_t>(index)==source->GetShadowLightIndex())
            { result.ShadowBegin=first; result.ShadowEnd=static_cast<std::uint32_t>(result.Samples.size()); }
        }
        output=std::move(result); return true;
    }
    bool PreparedHairLighting::Prepare(const LightingState& lighting,std::string& error)
    {
        error.clear(); if (Output && SameSceneLights(Previous,lighting)) return true;
        auto result=std::make_shared<HairLightSamples>(); if (!BuildHairLightSamples(lighting,*result,error)) return false;
        Previous=lighting; Output=std::move(result); ++Builds; return true;
    }
}
