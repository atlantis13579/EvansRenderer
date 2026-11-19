#include "Editor/TechniqueControls.h"
#include "Editor/Editor.hpp"
#include "Editor/Imgui.h"

namespace Evans::Editor
{
    bool DrawHairControls(EditorCore& editor,Techniques::Settings& settings)
    {
        if (editor.Scene().Demo!=DemoId::FurCube || editor.IsDemoLoading() || !imguiIsEnabled() ||
            !editor.QueryTechniqueSupport(Techniques::Technique::HairFur).Available() ||
            Techniques::UsesPathTracing(settings.Pipeline) || Techniques::UsesHybridRendering(settings.Pipeline))
        {
            imguiSetDisabledReason("Native fur requires the Fur demo and a compatible raster pipeline.");
            imguiCheck("Render fur",settings.Hair.Enabled,false);
            imguiSetDisabledReason(nullptr);
            return false;
        }
        auto& hair=settings.Hair;
        bool changed=false;
        if (imguiCheck("Render fur",hair.Enabled))
        {
            hair.Enabled=!hair.Enabled; changed=true;
            if (hair.Enabled && (Techniques::UsesPathTracing(settings.Pipeline) || Techniques::UsesHybridRendering(settings.Pipeline)))
                settings.Pipeline=Techniques::RenderingPipeline::Forward;
        }
        if (!hair.Enabled) return changed;
        // Keep preparation and failures visible beside the demo, rather than
        // burying them in the global status section below all technique groups.
        if (editor.Settings().Mode==FillMode::Wireframe)
            imguiLabel("Select Solid to display fur.");
        else
        {
            const auto& status=editor.ViewportHairFurStatus();
            imguiLabel(status.empty() || status=="Idle" ? "Preparing fur..." : status.c_str());
        }
        if (imguiCheck("Animate fur",hair.Animate)) { hair.Animate=!hair.Animate; changed=true; }
        if (hair.Animate) changed|=imguiSlider("Wind strength",&hair.WindStrength,0,1,0.01f);
        static bool showShape=false,showMaterial=false;
        if (imguiCollapse("Fur shape","",showShape)) showShape=!showShape;
        if (showShape)
        {
            imguiIndent();
            float strands=float(hair.StrandCount),segments=float(hair.Segments),seed=float(hair.Seed);
            if (imguiSlider("Strands",&strands,1024,float(Techniques::MaximumHairStrands),1024))
            { hair.StrandCount=std::uint32_t(strands); changed=true; }
            if (imguiSlider("Curve segments",&segments,1,float(Techniques::MaximumHairSegments),1))
            { hair.Segments=std::uint32_t(segments); changed=true; }
            changed|=imguiSlider("Length / cube size",&hair.LengthFraction,0.01f,0.25f,0.005f);
            changed|=imguiSlider("Radius / fur length",&hair.RadiusRatio,0.002f,0.04f,0.001f);
            changed|=imguiSlider("Bend",&hair.Bend,0,2,0.01f);
            changed|=imguiSlider("Length variation",&hair.LengthVariation,0,0.8f,0.01f);
            if (imguiSlider("Random seed",&seed,1,100,1)) { hair.Seed=std::uint32_t(seed); changed=true; }
            imguiUnindent();
        }
        if (imguiCollapse("Fur material","",showMaterial)) showMaterial=!showMaterial;
        if (showMaterial)
        {
            imguiIndent();
            changed|=imguiSlider("Longitudinal roughness",&hair.LongitudinalRoughness,0.05f,1,0.01f);
            changed|=imguiSlider("Azimuthal roughness",&hair.AzimuthalRoughness,0.05f,1,0.01f);
            changed|=imguiSlider("Cuticle tilt (degrees)",&hair.CuticleTiltDegrees,-15,15,0.1f);
            changed|=imguiSlider("Fiber refractive index",&hair.IndexOfRefraction,1.01f,2.5f,0.01f);
            changed|=imguiSlider("Fiber intensity",&hair.Intensity,0,4,0.01f);
            const char* roots[]{"Root red tint","Root green tint","Root blue tint"};
            const char* tips[]{"Tip red tint","Tip green tint","Tip blue tint"};
            for (unsigned channel=0;channel<3;++channel)
            {
                changed|=imguiSlider(roots[channel],&hair.RootTint[channel],0,1,0.01f);
                changed|=imguiSlider(tips[channel],&hair.TipTint[channel],0,1,0.01f);
            }
            imguiUnindent();
        }
        return changed;
    }
}
