#include "Renderer/BakedLighting/BakedLightingAssets.h"
#include "Renderer/Geometry/MeshGeometry.h"
#include "Renderer/Lighting/SceneLightCollection.h"
#include "Maths/Matrix4.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#endif

namespace Evans::Renderer
{
    namespace
    {
        // Increment with transport, UV2, LOD or archive layout changes. These
        // are Evans resources, not a serialization of C++ ownership or a BVH.
        constexpr std::uint64_t ArchiveVersion=1,LightmapMagic=0x50414d4c4b425645ull,ProbeMagic=0x45424f5250425645ull;
        constexpr std::uint64_t LightmapArchiveVersion=3;
        constexpr std::uint64_t HashSeed=14695981039346656037ull,HashPrime=1099511628211ull;
        constexpr std::size_t MaximumArchiveBytes=512ull*1024*1024,HeaderWords=5,HeaderBytes=HeaderWords*sizeof(std::uint64_t);
        constexpr std::uint32_t RepeatedTexelsBit=1u<<31,MaximumTexelBlock=RepeatedTexelsBit-1;
        struct ReceiverCoordinates { Float3 Position; Float2 MaterialUV,AtlasUV; };
        bool SameTexel(const Float4& a,const Float4& b) { return std::memcmp(&a,&b,sizeof(Float4))==0; }
        bool LittleEndian() { const unsigned value=1; return *reinterpret_cast<const unsigned char*>(&value)==1; }
        struct Hash
        {
            std::uint64_t Value=HashSeed;
            void Bytes(const void* data,std::size_t size)
            { const auto* p=static_cast<const unsigned char*>(data); for (std::size_t i=0;i<size;++i) Value=(Value^p[i])*HashPrime; }
            template<class T> void Scalar(const T& value) { Bytes(&value,sizeof(value)); }
            void Flag(bool value) { Scalar(std::uint32_t(value)); }
            template<class T> void Vector(const std::vector<T>& values)
            { Scalar(std::uint64_t(values.size())); Bytes(values.data(),values.size()*sizeof(T)); }
        };
        struct Writer
        {
            std::vector<unsigned char> Bytes;
            template<class T> void Scalar(const T& value)
            {
                static_assert(std::is_trivially_copyable_v<T>);
                const auto* p=reinterpret_cast<const unsigned char*>(&value); Bytes.insert(Bytes.end(),p,p+sizeof(value));
            }
            void Flag(bool value) { Scalar(std::uint32_t(value)); }
            template<class T> void Vector(const std::vector<T>& values)
            {
                Scalar(std::uint64_t(values.size()));
                if (!values.empty())
                { const auto* p=reinterpret_cast<const unsigned char*>(values.data()); Bytes.insert(Bytes.end(),p,p+values.size()*sizeof(T)); }
            }
            void Float4Runs(const std::vector<Float4>& values)
            {
                // Gutters and unused atlas regions repeat exact texels. Run
                // blocks reduce disk size without quantizing irradiance.
                Scalar(std::uint64_t(values.size()));
                for (std::size_t at=0;at<values.size();)
                {
                    std::size_t run=1;
                    while (at+run<values.size() && run<MaximumTexelBlock && SameTexel(values[at],values[at+run])) ++run;
                    if (run>1) { Scalar(std::uint32_t(run)|RepeatedTexelsBit); Scalar(values[at]); at+=run; }
                    else
                    {
                        const auto begin=at++;
                        while (at<values.size() && at-begin<MaximumTexelBlock &&
                            (at+1==values.size() || !SameTexel(values[at],values[at+1]))) ++at;
                        Scalar(std::uint32_t(at-begin));
                        const auto* p=reinterpret_cast<const unsigned char*>(values.data()+begin);
                        Bytes.insert(Bytes.end(),p,p+(at-begin)*sizeof(Float4));
                    }
                }
            }
        };
        struct Reader
        {
            const std::vector<unsigned char>& Bytes; std::size_t At=0; bool Valid=true;
            template<class T> void Scalar(T& value)
            {
                static_assert(std::is_trivially_copyable_v<T>);
                if (!Valid || sizeof(T)>Bytes.size()-At) { Valid=false; return; }
                std::memcpy(&value,Bytes.data()+At,sizeof(T)); At+=sizeof(T);
            }
            void Flag(bool& value) { std::uint32_t flag=0; Scalar(flag); if (flag>1) Valid=false; value=flag!=0; }
            template<class T> void Vector(std::vector<T>& values)
            {
                std::uint64_t count=0; Scalar(count);
                if (!Valid || count>(Bytes.size()-At)/sizeof(T)) { Valid=false; return; }
                values.resize(static_cast<std::size_t>(count));
                if (count) std::memcpy(values.data(),Bytes.data()+At,values.size()*sizeof(T));
                At+=values.size()*sizeof(T);
            }
            void Float4Runs(std::vector<Float4>& values,std::size_t maximumCount)
            {
                std::uint64_t count=0; Scalar(count);
                if (!Valid || count>maximumCount) { Valid=false; return; }
                values.resize(static_cast<std::size_t>(count));
                for (std::size_t at=0;at<values.size();)
                {
                    std::uint32_t block=0; Scalar(block); const auto length=block&MaximumTexelBlock;
                    if (!Valid || !length || length>values.size()-at) { Valid=false; return; }
                    if (block&RepeatedTexelsBit)
                    {
                        Float4 value{}; Scalar(value); if (!Valid) return;
                        std::fill_n(values.begin()+at,length,value);
                    }
                    else
                    {
                        if (length>(Bytes.size()-At)/sizeof(Float4)) { Valid=false; return; }
                        std::memcpy(values.data()+at,Bytes.data()+At,length*sizeof(Float4)); At+=length*sizeof(Float4);
                    }
                    at+=length;
                }
            }
            bool Complete() const { return Valid && At==Bytes.size(); }
        };
        static_assert(sizeof(Float4)==16 && sizeof(ReceiverCoordinates)==28 && sizeof(LightmapReceiverVertex)==44 && sizeof(LightmapReceiverRange)==20,
            "Baked lighting archives require packed float/index streams");

        template<class IO,class S> void Sampling(IO& io,S& s)
        { io.Scalar(s.RayCount); io.Scalar(s.MaximumDistance); io.Scalar(s.SurfaceBias); io.Flag(s.Shadows); io.Flag(s.BakeBackFaces); }
        template<class IO,class S> void AtlasSettings(IO& io,S& s)
        { io.Scalar(s.Resolution); io.Scalar(s.Padding); io.Scalar(s.MaximumNormalAngleDegrees); io.Scalar(s.MaximumChartTriangles); }
        template<class IO,class S> void ProbeSettings(IO& io,S& s)
        {
            io.Scalar(s.Minimum); io.Scalar(s.Maximum); io.Scalar(s.Counts); io.Scalar(s.RayCount);
            io.Scalar(s.MaximumDistance); io.Scalar(s.SurfaceBias); io.Scalar(s.MaximumBackfaceFraction); io.Flag(s.Shadows);
        }
        void Materials(Hash& h,const Techniques::Settings& s)
        {
            h.Flag(s.TextureMapping); h.Flag(s.NormalMapping); h.Flag(s.AlphaTesting); h.Flag(s.AlphaBlending);
            h.Flag(s.Refraction.Enabled); h.Scalar(s.Shading); h.Scalar(s.Metallic);
        }
        void Lights(Hash& h,const LightingState& lighting)
        {
            const auto count=lighting.SceneLights?lighting.SceneLights->GetLights().size():lighting.LightCount;
            h.Scalar(std::uint64_t(count));
            for (std::size_t index=0;index<count;++index)
            {
                const auto& l=lighting.SceneLights?lighting.SceneLights->GetLights()[index]:lighting.Lights[index];
                h.Scalar(l.Type); h.Scalar(l.Position); h.Scalar(l.Direction); h.Scalar(l.Color); h.Scalar(l.Intensity);
                h.Scalar(l.AreaSize); h.Flag(l.CastShadows); h.Scalar(l.Range); h.Scalar(l.AreaSampleAxisCount);
            }
        }
        struct SceneHash
        {
            Hash Digest;
            std::unordered_map<const ImageData*,std::uint64_t> Images;
            void Image(const std::shared_ptr<const ImageData>& image)
            {
                if (!image) { Digest.Scalar(std::uint64_t(0)); return; }
                const auto cached=Images.find(image.get());
                if (cached!=Images.end()) { Digest.Scalar(cached->second); return; }
                Hash h; h.Scalar(image->Width); h.Scalar(image->Height); h.Vector(image->PixelsRGBA8);
                Images.emplace(image.get(),h.Value); Digest.Scalar(h.Value);
            }
            void Mesh(const SceneRayMeshView& mesh)
            {
                Digest.Flag(mesh.Vertices && mesh.Indices); if (!mesh.Vertices || !mesh.Indices) return;
                Digest.Flag(mesh.Visible); Digest.Flag(mesh.CastShadow); Digest.Scalar(mesh.World); Digest.Scalar(mesh.Color);
                Digest.Scalar(HashMeshGeometry(*mesh.Vertices,*mesh.Indices));
                const auto& m=mesh.Material;
                Digest.Scalar(m.BaseColor); Digest.Scalar(m.AmbientColor); Digest.Scalar(m.SpecularColor); Digest.Scalar(m.EmissiveColor);
                Digest.Scalar(m.Shininess); Digest.Scalar(m.AlphaCutoff); Digest.Flag(m.Refractive);
                Image(m.DiffuseImage); Image(m.NormalImage); Image(m.OpacityImage);
                if (m.Metallic) Digest.Scalar(*m.Metallic);
                if (m.MetallicRoughnessImage) Image(m.MetallicRoughnessImage);
            }
        };
        bool ReadArchive(const std::filesystem::path& path,std::uint64_t magic,std::uint64_t key,std::vector<unsigned char>& payload,
            std::uint64_t version=ArchiveVersion)
        {
            if (path.empty() || !LittleEndian()) return false;
            std::ifstream input(path,std::ios::binary|std::ios::ate); if (!input) return false;
            const auto size=input.tellg(); if (size<std::streamoff(HeaderBytes) || size>std::streamoff(MaximumArchiveBytes)) return false;
            input.seekg(0); std::array<std::uint64_t,HeaderWords> header{};
            input.read(reinterpret_cast<char*>(header.data()),HeaderBytes);
            if (!input || header[0]!=magic || header[1]!=version || header[2]!=key ||
                header[3]!=std::uint64_t(size)-HeaderBytes) return false;
            payload.resize(static_cast<std::size_t>(header[3]));
            input.read(reinterpret_cast<char*>(payload.data()),static_cast<std::streamsize>(payload.size()));
            Hash hash; hash.Bytes(payload.data(),payload.size()); return bool(input) && hash.Value==header[4];
        }
        bool WriteArchive(const std::filesystem::path& path,std::uint64_t magic,std::uint64_t key,const Writer& writer,std::string& error,
            std::uint64_t version=ArchiveVersion)
        {
            std::filesystem::path temporary;
            try
            {
                if (!LittleEndian() || writer.Bytes.size()>MaximumArchiveBytes-HeaderBytes) throw std::runtime_error("Unsupported archive size or host byte order");
                std::filesystem::create_directories(path.parent_path());
                static std::atomic<unsigned> sequence{0}; temporary=path;
                temporary+=".tmp-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+"-"+std::to_string(++sequence);
                Hash hash; hash.Bytes(writer.Bytes.data(),writer.Bytes.size());
                const std::array<std::uint64_t,HeaderWords> header{magic,version,key,writer.Bytes.size(),hash.Value};
                std::ofstream output(temporary,std::ios::binary|std::ios::trunc);
                output.write(reinterpret_cast<const char*>(header.data()),HeaderBytes);
                output.write(reinterpret_cast<const char*>(writer.Bytes.data()),static_cast<std::streamsize>(writer.Bytes.size()));
                output.close(); if (!output) throw std::runtime_error("Cannot write complete lighting archive");
#ifdef _WIN32
                if (!MoveFileExW(temporary.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))
                    throw std::runtime_error("Cannot publish lighting archive");
#else
                std::filesystem::rename(temporary,path);
#endif
                error.clear(); return true;
            }
            catch (const std::exception& e)
            { if (!temporary.empty()) { std::error_code ignored; std::filesystem::remove(temporary,ignored); } error=e.what(); return false; }
        }
        bool Finite(Float4 v) { return std::isfinite(v.X)&&std::isfinite(v.Y)&&std::isfinite(v.Z)&&std::isfinite(v.W); }
        bool Irradiance(const std::vector<Float4>& values)
        { return std::all_of(values.begin(),values.end(),[](Float4 v) { return Finite(v)&&v.X>=0&&v.Y>=0&&v.Z>=0&&(v.W==0||v.W==1); }); }
    }

    LightmapBakeRequest MakeLightmapBakeRequest(std::vector<LightmapSourceInstance> instances,
        const Techniques::Settings& settings,const LightingState& lighting,bool shadows,float maximumDistance)
    {
        LightmapBakeRequest request; request.Instances=std::move(instances); request.Materials=settings; request.Lighting=lighting;
        const auto& atlas=settings.GlobalIllumination.Lightmaps;
        request.Atlas.Resolution=atlas.Resolution; request.Atlas.Padding=atlas.Padding; request.Atlas.MaximumNormalAngleDegrees=atlas.MaximumNormalAngleDegrees;
        Maths::AABB bounds;
        if (maximumDistance<=0) for (const auto& instance:request.Instances)
        {
            if (!instance.Visible || !instance.Geometry) continue;
            Maths::AABB local;
            for (const auto& vertex:instance.Geometry->Vertices) local.Expand({vertex.Position.X,vertex.Position.Y,vertex.Position.Z});
            if (!local.IsValid()) continue;
            const Maths::Matrix4 world(instance.World.Values);
            for (unsigned corner=0;corner<8;++corner)
                bounds.Expand(world.TransformPoint({corner&1?local.Max.x:local.Min.x,corner&2?local.Max.y:local.Min.y,corner&4?local.Max.z:local.Min.z}));
        }
        constexpr float TraceExtentMultiplier=2,MinimumTraceDistance=0.001f,RelativeBias=0.000001f,MinimumBias=0.00001f;
        request.Sampling.MaximumDistance=maximumDistance>0?maximumDistance:
            bounds.IsValid()?std::max((bounds.Max-bounds.Min).Length()*TraceExtentMultiplier,MinimumTraceDistance):1;
        request.Sampling.SurfaceBias=std::max(request.Sampling.MaximumDistance*RelativeBias,MinimumBias);
        request.Sampling.RayCount=atlas.RayCount; request.Sampling.Shadows=shadows; return request;
    }

    std::uint64_t LightmapAssetKey(const LightmapBakeRequest& request)
    {
        SceneHash hash; auto& h=hash.Digest; h.Scalar(ArchiveVersion); Materials(h,request.Materials); Lights(h,request.Lighting);
        AtlasSettings(h,request.Atlas); Sampling(h,request.Sampling); h.Scalar(std::uint64_t(request.Instances.size()));
        for (const auto& instance:request.Instances)
        {
            h.Flag(instance.Geometry && !instance.Geometry->Levels.empty());
            if (!instance.Geometry || instance.Geometry->Levels.empty()) continue;
            const auto& geometry=*instance.Geometry;
            SceneRayMeshView mesh; mesh.Vertices=&geometry.Vertices; mesh.Indices=&geometry.Levels.front();
            mesh.World=instance.World; mesh.Color=instance.Color; mesh.Material=instance.Material; mesh.Visible=instance.Visible; mesh.CastShadow=instance.CastShadow;
            hash.Mesh(mesh); h.Scalar(std::uint64_t(geometry.Levels.size()));
            for (const auto& level:geometry.Levels) h.Vector(level);
            h.Scalar(std::uint64_t(geometry.Segments.size())); for (const auto& segments:geometry.Segments) h.Vector(segments);
        }
        return h.Value;
    }

    std::uint64_t LightmapReceiverKey(const LightmapBakeRequest& request)
    {
        // Light edits remain realtime, but transport materials and geometry must
        // still match the immutable bake. Changing them requires an explicit rebake.
        SceneHash hash; auto& h=hash.Digest; h.Scalar(LightmapArchiveVersion); AtlasSettings(h,request.Atlas);
        Materials(h,request.Materials);
        h.Scalar(std::uint64_t(request.Instances.size()));
        for (const auto& instance:request.Instances)
        {
            h.Flag(instance.Geometry && !instance.Geometry->Levels.empty());
            h.Flag(instance.Visible); h.Scalar(instance.World);
            if (!instance.Geometry || instance.Geometry->Levels.empty()) continue;
            const auto& geometry=*instance.Geometry;
            SceneRayMeshView mesh; mesh.Vertices=&geometry.Vertices; mesh.Indices=&geometry.Levels.front();
            mesh.World=instance.World; mesh.Color=instance.Color; mesh.Material=instance.Material;
            mesh.Visible=instance.Visible; mesh.CastShadow=instance.CastShadow; hash.Mesh(mesh);
            h.Scalar(std::uint64_t(geometry.Levels.size()));
            for (const auto& level:geometry.Levels) h.Vector(level);
            h.Scalar(std::uint64_t(geometry.Segments.size()));
            for (const auto& segments:geometry.Segments) h.Vector(segments);
        }
        return h.Value;
    }

    std::uint64_t ProbeAssetKey(const std::vector<SceneRayMeshView>& meshes,const Techniques::Settings& settings,
        const LightingState& lighting,bool shadows)
    {
        SceneHash hash; auto& h=hash.Digest; h.Scalar(ArchiveVersion); Materials(h,settings); Lights(h,lighting); h.Flag(shadows);
        const auto& s=settings.GlobalIllumination.Probes;
        h.Flag(s.AutomaticBounds); h.Scalar(s.BoundsMinimum); h.Scalar(s.BoundsMaximum); h.Scalar(s.GridAxisCount); h.Scalar(s.RayCount); h.Scalar(s.MaximumBackfaceFraction);
        h.Scalar(std::uint64_t(meshes.size())); for (const auto& mesh:meshes) hash.Mesh(mesh); return h.Value;
    }

    bool SaveLightmapAsset(const std::filesystem::path& path,const LightmapBakeRequest& request,const LightmapBakeResult& result,std::string& error)
    {
        if (!result.Receivers || !result.Irradiance || !result.Irradiance->Atlas) { error="No complete lightmap to save"; return false; }
        const auto& bake=*result.Irradiance; Writer w;
        w.Scalar(LightmapAssetKey(request)); AtlasSettings(w,bake.Atlas->Settings); Sampling(w,bake.Settings);
        // Most receiver colors are constant. Separate their run stream from
        // positions and UVs without changing the native interleaved vertex ABI.
        std::vector<ReceiverCoordinates> coordinates; std::vector<Float4> colors;
        coordinates.reserve(result.Receivers->Vertices.size()); colors.reserve(result.Receivers->Vertices.size());
        for (const auto& vertex:result.Receivers->Vertices)
        { coordinates.push_back({vertex.Position,vertex.MaterialUV,vertex.AtlasUV}); colors.push_back(vertex.Color); }
        w.Vector(coordinates); w.Float4Runs(colors); w.Vector(result.Receivers->Ranges);
        w.Float4Runs(bake.FrontIrradiance); w.Float4Runs(bake.BackIrradiance); w.Vector(bake.TexelKinds);
        w.Scalar(bake.CoveredTexelCount); w.Scalar(bake.GutterTexelCount); w.Scalar(bake.SurfaceSampleCount); w.Scalar(bake.SurfaceHitCount); w.Scalar(bake.ShadowQueryCount);
        return WriteArchive(path,LightmapMagic,LightmapReceiverKey(request),w,error,LightmapArchiveVersion);
    }

    std::shared_ptr<const LightmapBakeResult> LoadLightmapAsset(const std::filesystem::path& path,const LightmapBakeRequest& request)
    {
        std::vector<unsigned char> bytes;
        if (!ReadArchive(path,LightmapMagic,LightmapReceiverKey(request),bytes,LightmapArchiveVersion)) return {};
        Reader r{bytes}; auto atlas=std::make_shared<LightmapAtlas>(); auto receivers=std::make_shared<LightmapReceivers>(); auto bake=std::make_shared<BakedLightmap>();
        std::uint64_t bakedSourceKey=0; r.Scalar(bakedSourceKey);
        AtlasSettings(r,atlas->Settings); Sampling(r,bake->Settings);
        std::vector<ReceiverCoordinates> coordinates; std::vector<Float4> colors;
        r.Vector(coordinates); r.Float4Runs(colors,coordinates.size()); r.Vector(receivers->Ranges);
        if (!r.Valid || colors.size()!=coordinates.size()) return {};
        receivers->Vertices.reserve(coordinates.size());
        for (std::size_t index=0;index<coordinates.size();++index)
        { const auto& v=coordinates[index]; receivers->Vertices.push_back({v.Position,v.MaterialUV,v.AtlasUV,colors[index]}); }
        const auto maximumTexels=std::size_t(request.Atlas.Resolution)*request.Atlas.Resolution;
        r.Float4Runs(bake->FrontIrradiance,maximumTexels); r.Float4Runs(bake->BackIrradiance,maximumTexels); r.Vector(bake->TexelKinds);
        r.Scalar(bake->CoveredTexelCount); r.Scalar(bake->GutterTexelCount); r.Scalar(bake->SurfaceSampleCount); r.Scalar(bake->SurfaceHitCount); r.Scalar(bake->ShadowQueryCount);
        const auto side=atlas->Settings.Resolution; const auto count=std::size_t(side)*side;
        if (!r.Complete() || side!=request.Atlas.Resolution || bake->FrontIrradiance.size()!=count ||
            bake->BackIrradiance.size()!=(bake->Settings.BakeBackFaces?count:0) || bake->TexelKinds.size()!=count ||
            !Irradiance(bake->FrontIrradiance) || !Irradiance(bake->BackIrradiance)) return {};
        for (const auto kind:bake->TexelKinds) if (kind>LightmapTexelKind::Gutter) return {};
        std::uint32_t next=0; std::uint32_t previousInstance=0;
        for (const auto& range:receivers->Ranges)
        {
            if (range.InstanceIndex>=request.Instances.size() || range.InstanceIndex<previousInstance ||
                !request.Instances[range.InstanceIndex].Geometry ||
                range.LodLevel>=request.Instances[range.InstanceIndex].Geometry->Levels.size() || range.FirstVertex!=next ||
                range.VertexCount%3 || range.VertexCount>receivers->Vertices.size()-next) return {};
            next+=range.VertexCount; previousInstance=range.InstanceIndex;
        }
        if (next!=receivers->Vertices.size()) return {};
        for (const auto& vertex:receivers->Vertices)
            if (!Finite({vertex.Position.X,vertex.Position.Y,vertex.Position.Z,0}) || !Finite(vertex.Color) ||
                !Finite({vertex.MaterialUV.X,vertex.MaterialUV.Y,vertex.AtlasUV.X,vertex.AtlasUV.Y}) ||
                vertex.AtlasUV.X<0 || vertex.AtlasUV.X>1 || vertex.AtlasUV.Y<0 || vertex.AtlasUV.Y>1) return {};
        receivers->Atlas=atlas; bake->Atlas=atlas;
        auto result=std::make_shared<LightmapBakeResult>(); result->Receivers=receivers; result->Irradiance=bake; return result;
    }

    bool SaveProbeAsset(const std::filesystem::path& path,std::uint64_t key,const BakedProbeSamples& samples,std::string& error)
    {
        Writer w; ProbeSettings(w,samples.Settings); w.Scalar(samples.ActiveProbeCount); w.Scalar(samples.SurfaceHitCount); w.Scalar(samples.ShadowQueryCount); w.Vector(samples.Texels);
        return WriteArchive(path,ProbeMagic,key,w,error);
    }

    std::shared_ptr<const BakedProbeSamples> LoadProbeAsset(const std::filesystem::path& path,std::uint64_t key)
    {
        std::vector<unsigned char> bytes; if (!ReadArchive(path,ProbeMagic,key,bytes)) return {};
        Reader r{bytes}; auto result=std::make_shared<BakedProbeSamples>(); auto& s=result->Settings;
        ProbeSettings(r,s); r.Scalar(result->ActiveProbeCount); r.Scalar(result->SurfaceHitCount); r.Scalar(result->ShadowQueryCount); r.Vector(result->Texels);
        if (!r.Complete() || !std::all_of(s.Counts.begin(),s.Counts.end(),[](unsigned n) { return n>=MinimumProbeGridAxis&&n<=MaximumProbeGridAxis; }) ||
            s.RayCount<MinimumProbeRayCount || s.RayCount>MaximumProbeRayCount || result->Texels.size()!=std::size_t(result->Width())*result->Height() ||
            result->ActiveProbeCount>result->ProbeCount() || !std::isfinite(s.MaximumDistance) || s.MaximumDistance<=0 ||
            !Finite({s.Minimum.X,s.Minimum.Y,s.Minimum.Z,s.SurfaceBias}) || !Finite({s.Maximum.X,s.Maximum.Y,s.Maximum.Z,s.MaximumBackfaceFraction}) ||
            s.Minimum.X>=s.Maximum.X || s.Minimum.Y>=s.Maximum.Y || s.Minimum.Z>=s.Maximum.Z ||
            s.SurfaceBias<=0 || s.SurfaceBias>=s.MaximumDistance || s.MaximumBackfaceFraction<0 || s.MaximumBackfaceFraction>1) return {};
        if (!std::all_of(result->Texels.begin(),result->Texels.end(),Finite)) return {};
        return result;
    }
}
