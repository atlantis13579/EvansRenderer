#pragma once
#include "Editor/Types.hpp"
#include "Renderer/Neural/Json.h"
#include <array>
#include <functional>

namespace Evans::Neural
{
struct Vec3
{
    float X=0,Y=0,Z=0;
    Vec3 operator+(Vec3 b) const { return {X+b.X,Y+b.Y,Z+b.Z}; }
    Vec3 operator-(Vec3 b) const { return {X-b.X,Y-b.Y,Z-b.Z}; }
    Vec3 operator*(float s) const { return {X*s,Y*s,Z*s}; }
};
float Dot(Vec3 a,Vec3 b);
Vec3 Cross(Vec3 a,Vec3 b);
Vec3 Normalize(Vec3 a);
struct View
{
    Vec3 Eye,Right,Down,Forward;
    int Width=96,Height=96;
    float Focal=100;
};
View MakeView(const Editor::Camera& camera,int width,int height);
// Inverse rigid model rotation transforms both centers and covariance axes
// without changing the immutable Gaussian asset or the editor camera.
View ModelView(View view,float yawDegrees);
View OrbitView(float angle,float elevation,int width,int height);
// Mean xyz, log scale xyz, quaternion wxyz, opacity logit, RGB logits.
struct Gaussian { std::array<float,14> Values{}; };
struct GaussianScene
{
    std::vector<Gaussian> Points;
    bool SortByCameraDistance=false;
    float SupportCutoff=18;
};
struct FloatImage { int Width=0,Height=0; std::vector<float> RGB; };
struct Array { std::vector<int> Shape; std::vector<float> Data; };
void SaveArray(const std::filesystem::path& path,const Array& array);
Array LoadArray(const std::filesystem::path& path);
GaussianScene LoadScene(const std::filesystem::path& path);
GaussianScene LoadGlb(const std::filesystem::path& path);
void SaveGlb(const std::filesystem::path& path,const GaussianScene& scene);
FloatImage Render(const GaussianScene& scene,const View& view,const std::function<bool()>& cancelled={});
Editor::Image DisplayImage(const FloatImage& image);
inline constexpr std::array<float,3> Background{0.035f,0.045f,0.065f};
}
