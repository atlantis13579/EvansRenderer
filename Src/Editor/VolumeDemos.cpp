#include "Editor/Demos.hpp"
#include "Editor/BigWorld.hpp"

namespace Evans::Editor
{
bool DemoCatalog::IsVolumeDemo(DemoId demo) noexcept
{
    return demo==DemoId::BigWorld;
}

SceneSnapshot DemoCatalog::LoadVolume(DemoId demo) const
{
    SceneSnapshot scene; scene.Demo=demo; scene.Revision=1;
    auto world=std::make_shared<BigWorldState>();
    scene.World=world;
    const auto& s=world->Settings;
    const float bank=LakeRadius+15;
    scene.SceneCamera.Position={20,BigWorldHeight(s,20,bank)+9.0f,bank};
    scene.SceneCamera.Target={-32,LakeElevation+2,65};
    scene.SceneCamera.VerticalFieldOfViewDegrees=65;
    BuildBigWorld(scene);
    Techniques::Settings settings; ApplyBigWorldAppearance(scene,settings);
    UpdateBigWorld(scene,settings,true);
    return scene;
}

void DemoCatalog::ApplyVolumeAppearance(const SceneSnapshot& scene,Techniques::Settings& settings)
{
    ApplyBigWorldAppearance(scene,settings);
}
}
