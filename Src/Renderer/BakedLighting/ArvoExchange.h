#pragma once

// Arvo lightmap exchange v1. This header is mirrored in EvansRenderer.
// Little-endian IEEE754, fixed-width scalar streams; no pointers or C++ bools.
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ArvoLightmap
{
struct Vec2 { float X=0,Y=0; };
struct Vec3 { float X=0,Y=0,Z=0; };
struct Vec4 { float X=0,Y=0,Z=0,W=0; };
constexpr uint32_t Missing=0xffffffffu;
struct Vertex { Vec3 Position,Normal; Vec2 UV; Vec4 Color; };
struct Triangle { std::array<Vertex,3> Corners; uint32_t Material=0,CastShadow=1; };
struct Material
{
    Vec4 Base={1,1,1,1}; Vec3 Emission;
    float AlphaCutoff=0,DiffuseScale=1,Metallic=0;
    uint32_t Diffuse=Missing,Opacity=Missing,Normal=Missing,MetallicMap=Missing;
};
struct Texture { uint32_t Width=0,Height=0; std::vector<uint8_t> Pixels; };
struct Light
{
    uint32_t Type=0,CastShadow=1;
    Vec3 Position,Direction,Color;
    float Intensity=1,Range=0;
    Vec2 Size={1,1};
    // Rectangles carry emitted radiance, point lights radiant intensity,
    // directional lights perpendicular irradiance. All RGB values are linear.
};
struct Receiver { Triangle Surface; std::array<Vec2,3> UV; uint32_t Chart=Missing; };
struct Chart { uint32_t X=0,Y=0,Width=0,Height=0; };
struct Settings
{
    uint32_t Resolution=512,Samples=128,Bounces=3,BackFaces=1,Threads=0,Seed=1;
    uint32_t IndirectOnly=1;
    float Bias=0.0001f;
    Vec3 Environment;
};
struct Job
{
    Settings Options;
    std::vector<Texture> Textures;
    std::vector<Material> Materials;
    std::vector<Triangle> Triangles;
    // Linear HDR cube: +X,-X,+Y,-Y,+Z,-Z; cube lookup reflects world Z.
    uint32_t EnvironmentSide=0;
    std::vector<Vec3> EnvironmentFaces;
    std::vector<Light> Lights;
    std::vector<Receiver> Receivers;
    std::vector<Chart> Charts;
};
struct Result
{
    uint32_t Resolution=0;
    std::vector<Vec4> Front,Back;
    std::vector<uint8_t> Kinds;
};
static_assert(sizeof(Vertex)==48 && sizeof(Triangle)==152 && sizeof(Material)==56 && sizeof(Settings)==44);
class Stream
{
    std::fstream File;
    uint64_t Remaining=0;
    bool Reading;
public:
    Stream(const std::filesystem::path& path,bool reading,uint64_t magic):Reading(reading)
    {
        const uint32_t endian=1;
        if (*reinterpret_cast<const uint8_t*>(&endian)!=1) throw std::runtime_error("Lightmap exchange requires little endian");
        File.open(path,std::ios::binary|(reading?std::ios::in:std::ios::out|std::ios::trunc));
        if (!File) throw std::runtime_error("Cannot open lightmap exchange: "+path.u8string());
        if (reading) { File.seekg(0,std::ios::end); Remaining=static_cast<uint64_t>(File.tellg()); File.seekg(0); }
        uint64_t header=magic; Scalar(header);
        if (header!=magic) throw std::runtime_error("Unsupported lightmap exchange version");
    }
    void Bytes(void* data,size_t size)
    {
        if (Reading)
        {
            if (size>Remaining) throw std::runtime_error("Truncated lightmap exchange");
            if(size) File.read(static_cast<char*>(data),size); Remaining-=size;
        }
        else if(size) File.write(static_cast<const char*>(data),size);
        if (!File) throw std::runtime_error("Lightmap exchange I/O failed");
    }
    template<class T> void Scalar(T& value) { static_assert(std::is_trivially_copyable_v<T>); Bytes(&value,sizeof(T)); }
    template<class T> void Vector(std::vector<T>& values)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        uint64_t count=values.size(); Scalar(count);
        if(count>uint64_t(2)*1024*1024*1024/sizeof(T) || (Reading && count>Remaining/sizeof(T)))
            throw std::runtime_error("Invalid lightmap stream length");
        if(Reading) values.resize(static_cast<size_t>(count));
        Bytes(values.data(),values.size()*sizeof(T));
    }
    void Finish() { if(Reading && Remaining) throw std::runtime_error("Trailing lightmap data"); File.flush(); if(!File) throw std::runtime_error("Lightmap write failed"); }
};
inline void TransferJob(const std::filesystem::path& path,Job& job,bool reading)
{
    Stream stream(path,reading,0x31424f4a4d4c5241ull);
    stream.Scalar(job.Options);
    uint32_t textures=static_cast<uint32_t>(job.Textures.size()); stream.Scalar(textures);
    if(textures>65536) throw std::runtime_error("Too many lightmap textures");
    if(reading) job.Textures.resize(textures);
    for(auto& texture:job.Textures) { stream.Scalar(texture.Width); stream.Scalar(texture.Height); stream.Vector(texture.Pixels); }
    stream.Vector(job.Materials); stream.Vector(job.Triangles);
    stream.Scalar(job.EnvironmentSide); stream.Vector(job.EnvironmentFaces); stream.Vector(job.Lights);
    stream.Vector(job.Receivers); stream.Vector(job.Charts); stream.Finish();
}
inline void TransferResult(const std::filesystem::path& path,Result& result,bool reading)
{
    Stream stream(path,reading,0x315345524d4c5241ull);
    stream.Scalar(result.Resolution); stream.Vector(result.Front); stream.Vector(result.Back); stream.Vector(result.Kinds); stream.Finish();
}
}
