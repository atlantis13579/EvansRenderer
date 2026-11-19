#pragma once

#include "Editor/Types.hpp"
#include <array>
#include <complex>

namespace Evans::Renderer { class AtmosphereModel; }

namespace Evans::Editor
{
enum class WorldWeather { Clear, Rain, Snow };
enum class WorldFog { None, Exponential, Height, Linear };

inline constexpr float LakeRadius=190, LakeElevation=8, LakeDepth=14;
inline constexpr double BigWorldUpdateInterval=1.0/60;

struct BigWorldSettings
{
    unsigned Seed=18;
    float Extent=1200, MountainHeight=160;
    unsigned TerrainSegments=192, TreeCount=400, GrassCount=90000;
    bool Terrain=true, Trees=true, Grass=true, AnimateVegetation=true;
    float WindSpeed=6, WindDirection=35, GrassHeight=0.85f, TreeHeight=14;
    float GrassWidth=0.07f, GrassDistance=120, GrassLodDistance=28;
    float GustStrength=0.75f, GustWavelength=32, LeafFlutter=0.22f;
    float TreeLodDistance=140;
    bool DayNight=true;
    float Hour=10, DayDuration=600, SunIntensity=4, MoonIntensity=0.06f;
    float CameraSpeed=40, IBLIntensity=0.8f;
    Techniques::AtmosphereSettings Atmosphere{true};
    WorldWeather Weather=WorldWeather::Clear;
    float Precipitation=0.7f, SnowCover=0.35f;
    WorldFog Fog=WorldFog::Height;
    float FogDensity=0.0004f, FogHeight=30, FogFalloff=0.016f, FogDistance=2500;
    bool Clouds=true;
    float CloudAltitude=240, CloudThickness=180, CloudDensity=0.04f, CloudSpeed=1;
    bool Water=true, AnimateWater=true, FFT=true, Gerstner=true;
    unsigned FFTSize=64;
    float FFTPatch=128, WaveAmplitude=0.55f, Choppiness=0.8f;
    float SwellAmplitude=0.36f, SwellLength=26, SwellSteepness=0.45f, WaterSpeed=1;
    float WaterRoughness=0.095f, WaterIOR=1.333f, Absorption=0.105f;
    std::array<float,3> WaterTint{0.095f,0.235f,0.26f};
    float TileSize=32, WaterLodDistance=48;
    unsigned WaterSegments=64;
    bool Caustics=true, UnderwaterLighting=true;
    float CausticIntensity=1.4f, CausticDepth=18;
};

struct WaterSample { Float3 Displacement, Normal{0,1,0}; };

// Tessendorf's Phillips spectrum uses conjugate pairs and a separable radix-2
// inverse FFT. Gerstner waves add independently controlled long-period swell.
// https://jtessen.people.clemson.edu/reports/papers_files/waterslides2001.pdf
class WaterSpectrum
{
public:
    explicit WaterSpectrum(const BigWorldSettings& settings);
    void Evaluate(double seconds);
    [[nodiscard]] WaterSample Sample(float x,float z) const;
    [[nodiscard]] float Caustic(float x,float z,float depth,Float3 sun) const;
    [[nodiscard]] std::array<std::array<float,4>,2> LightWaves() const;
private:
    struct Swell { float X=0,Z=0,K=0,Amplitude=0,Horizontal=0,Phase=0; };
    BigWorldSettings Settings_;
    std::vector<std::complex<float>> Initial_;
    std::array<std::vector<float>,3> Field_;
    std::array<std::array<std::vector<float>,3>,2> Derivatives_;
    std::array<Swell,4> Swells_;
    std::array<Swell,2> WindWaves_;
    double Time_=0;
};

struct BigWorldGeometry;
struct BigWorldSkyJob;
struct BigWorldState
{
    BigWorldSettings Settings;
    std::shared_ptr<const WaterSpectrum> Waves;
    std::shared_ptr<const Renderer::AtmosphereModel> Atmosphere;
    std::shared_ptr<const BigWorldGeometry> Geometry;
    std::shared_ptr<BigWorldSkyJob> SkyJob;
    double LastUpdate=0, WaterTime=0, VegetationTime=0;
    std::uint64_t CameraRevision=0, LightingRevision=0;
    bool SettingsChanged=false, CameraUnderwater=false, VegetationLod=true;
    float SkyHour=-1;
    Float3 SkyPosition;
};

[[nodiscard]] BigWorldSettings SanitizeBigWorldSettings(BigWorldSettings settings);
[[nodiscard]] float BigWorldHeight(const BigWorldSettings& settings,float x,float z);
void BuildBigWorld(SceneSnapshot& scene);
void ConfigureBigWorld(SceneSnapshot& scene,const BigWorldSettings& settings);
void UpdateBigWorld(SceneSnapshot& scene,Techniques::Settings& techniques,bool force=false);
void ApplyBigWorldAppearance(const SceneSnapshot& scene,Techniques::Settings& techniques);
}
