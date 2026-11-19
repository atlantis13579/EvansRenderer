#include "Renderer/Lumen/LumenHardwareRayTracing.h"
#include "Renderer/RayTracing/SceneRayQuery.h"

#include <cmath>
#include <cstring>
#include <type_traits>
#include <limits>
#include <unordered_map>

namespace Evans::Renderer
{
    namespace
    {
        constexpr unsigned VectorWords=4;
        bool NormalTransform(const Matrix4& world,std::array<Float4,3>& output)
        {
            const auto& m=world.Values;
            for (float v:m) if (!std::isfinite(v)) return false;
            if (m[12] || m[13] || m[14] || m[15]!=1) return false;
            const double a[]{m[0],m[1],m[2],m[4],m[5],m[6],m[8],m[9],m[10]};
            // Cofactors are det(A)*transpose(inverse(A)). Removing the positive
            // magnitude avoids overflow under tiny or highly anisotropic poses;
            // retain the determinant sign so mirrored shells keep outward normals.
            const double c[]{a[4]*a[8]-a[5]*a[7],a[5]*a[6]-a[3]*a[8],a[3]*a[7]-a[4]*a[6],
                a[2]*a[7]-a[1]*a[8],a[0]*a[8]-a[2]*a[6],a[1]*a[6]-a[0]*a[7],
                a[1]*a[5]-a[2]*a[4],a[2]*a[3]-a[0]*a[5],a[0]*a[4]-a[1]*a[3]};
            const double determinant=a[0]*c[0]+a[1]*c[1]+a[2]*c[2];
            double scale=0; for (double v:c) scale=std::max(scale,std::abs(v));
            if (!std::isfinite(determinant) || determinant==0 || !std::isfinite(scale) || scale==0) return false;
            const double orientation=determinant<0?-1:1;
            for (unsigned row=0;row<3;++row)
                output[row]={float(c[row*3]/scale*orientation),float(c[row*3+1]/scale*orientation),float(c[row*3+2]/scale*orientation),0};
            // A double-precision inverse may exist even when its float normal
            // matrix loses rank. Reject that pose instead of publishing bad hits.
            const auto& x=output[0]; const auto& y=output[1]; const auto& z=output[2];
            const double rank=double(x.X)*(double(y.Y)*z.Z-double(y.Z)*z.Y)-double(x.Y)*(double(y.X)*z.Z-double(y.Z)*z.X)+double(x.Z)*(double(y.X)*z.Y-double(y.Y)*z.X);
            return std::isfinite(rank) && rank!=0;
        }
        void MeshBounds(const LumenHardwareMesh& mesh,Float3& minimum,Float3& maximum)
        {
            const auto first=mesh.Triangles.front().A;
            minimum={first.X,first.Y,first.Z}; maximum=minimum;
            for (const auto& triangle:mesh.Triangles) for (auto p:{triangle.A,triangle.B,triangle.C})
            {
                minimum={std::min(minimum.X,p.X),std::min(minimum.Y,p.Y),std::min(minimum.Z,p.Z)};
                maximum={std::max(maximum.X,p.X),std::max(maximum.Y,p.Y),std::max(maximum.Z,p.Z)};
            }
        }
        bool MeshValid(const LumenHardwareMesh& mesh,std::string& error)
        {
            if (mesh.Triangles.empty() || mesh.Triangles.size()>LumenDistanceMaximumTriangles)
            { error="Lumen hardware meshes need a bounded nonempty triangle array"; return false; }
            if (!ValidateRaySceneGeometry(mesh.Triangles,mesh.Materials,mesh.CoverageTexels,error)) return false;
            Float3 minimum,maximum; MeshBounds(mesh,minimum,maximum);
            if (minimum.X!=mesh.Minimum.X || minimum.Y!=mesh.Minimum.Y || minimum.Z!=mesh.Minimum.Z ||
                maximum.X!=mesh.Maximum.X || maximum.Y!=mesh.Maximum.Y || maximum.Z!=mesh.Maximum.Z)
            { error="Lumen hardware mesh bounds do not cover its exact triangle array"; return false; }
            return true;
        }
        bool WorldFits(const LumenHardwareMesh& mesh,const Matrix4& world)
        {
            const auto& m=world.Values;
            for (unsigned corner=0;corner<8;++corner)
            {
                const double p[]{corner&1?mesh.Maximum.X:mesh.Minimum.X,corner&2?mesh.Maximum.Y:mesh.Minimum.Y,corner&4?mesh.Maximum.Z:mesh.Minimum.Z};
                for (unsigned row=0;row<3;++row)
                {
                    const double value=p[0]*m[row*4]+p[1]*m[row*4+1]+p[2]*m[row*4+2]+m[row*4+3];
                    if (!std::isfinite(value) || std::abs(value)>std::numeric_limits<float>::max()) return false;
                }
            }
            return true;
        }
        template<class T> bool Append(std::vector<std::uint32_t>& words,const std::vector<T>& data,unsigned& offset)
        {
            static_assert(sizeof(T)%sizeof(std::uint32_t)==0,"Hardware payload must use complete words");
            const auto first=(words.size()+VectorWords-1)/VectorWords*VectorWords;
            const auto count=data.size()*(sizeof(T)/sizeof(std::uint32_t));
            if (first>LumenHardwareMaximumWords || count>LumenHardwareMaximumWords-first) return false;
            offset=unsigned(first); words.resize(first+count);
            if (count) std::memcpy(words.data()+first,data.data(),count*sizeof(std::uint32_t));
            return true;
        }
        LumenHardwareGPUInstance InstanceRecord(const LumenHardwareScene& scene,const LumenHardwareInstance& instance,const std::array<Float4,3>& normal)
        {
            LumenHardwareGPUInstance record;
            const auto& range=scene.Geometry[instance.Mesh];
            record.Storage={range.Triangles,range.Materials,range.Coverage,unsigned(scene.Meshes[instance.Mesh]->Triangles.size())};
            record.Normal=normal; return record;
        }
        template<class T> bool Matches(const std::vector<std::uint32_t>& words,unsigned offset,const std::vector<T>& data)
        {
            const auto size=data.size()*(sizeof(T)/sizeof(std::uint32_t));
            return offset<=words.size() && size<=words.size()-offset &&
                (!size || std::memcmp(words.data()+offset,data.data(),size*sizeof(std::uint32_t))==0);
        }
    }
    bool BuildLumenHardwareMesh(const SceneRayQuery& query,LumenHardwareMesh& output,std::string& error)
    {
        RaySceneUpload visibility; if (!query.Export(visibility,error)) return false;
        LumenHardwareMesh result; result.Triangles=std::move(visibility.Triangles);
        result.Materials=std::move(visibility.Materials); result.CoverageTexels=std::move(visibility.CoverageTexels);
        if (!result.Triangles.empty()) MeshBounds(result,result.Minimum,result.Maximum);
        if (!MeshValid(result,error)) return false;
        output=std::move(result); error.clear(); return true;
    }
    bool BuildLumenHardwareScene(const std::vector<LumenHardwarePlacement>& placements,LumenDistanceSceneIdentity identity,
        LumenHardwareScene& output,std::string& error)
    {
        if (!identity.Epoch || !identity.Serial || !identity.Geometry || placements.size()>LumenDistanceMaximumInstances)
        { error="Lumen hardware scenes need nonzero identities and bounded instances"; return false; }
        LumenHardwareScene result; result.Identity=identity; result.Instances.resize(placements.size());
        result.Words.resize(placements.size()*LumenHardwareInstanceWords);
        std::unordered_map<const LumenHardwareMesh*,unsigned> unique; std::vector<bool> seen(placements.size());
        for (const auto& placement:placements)
        {
            std::array<Float4,3> normal;
            if (!placement.Mesh || placement.DistanceInstance>=placements.size() || seen[placement.DistanceInstance] || !NormalTransform(placement.World,normal))
            { error="Lumen hardware placements require unique dense distance IDs and invertible affine poses"; return false; }
            seen[placement.DistanceInstance]=true;
            auto found=unique.find(placement.Mesh.get());
            if (found==unique.end())
            {
                if (result.Meshes.size()>=LumenDistanceMaximumFields)
                { error="Lumen hardware scene exceeds its unique mesh budget"; return false; }
                if (!MeshValid(*placement.Mesh,error)) return false;
                LumenHardwareGeometry range;
                if (!Append(result.Words,placement.Mesh->Triangles,range.Triangles) ||
                    !Append(result.Words,placement.Mesh->Materials,range.Materials) || !Append(result.Words,placement.Mesh->CoverageTexels,range.Coverage))
                { error="Lumen hardware scene exceeds its immutable geometry/coverage storage budget"; return false; }
                found=unique.emplace(placement.Mesh.get(),unsigned(result.Meshes.size())).first;
                result.Meshes.push_back(placement.Mesh); result.Geometry.push_back(range);
            }
            if (!WorldFits(*placement.Mesh,placement.World))
            { error="Lumen hardware placement exceeds finite world-space bounds"; return false; }
            result.Instances[placement.DistanceInstance]={placement.World,found->second,placement.CastShadow};
            const auto record=InstanceRecord(result,result.Instances[placement.DistanceInstance],normal);
            std::memcpy(result.Words.data()+std::size_t(placement.DistanceInstance)*LumenHardwareInstanceWords,&record,sizeof(record));
        }
        output=std::move(result); error.clear(); return true;
    }
    bool ValidateLumenHardwareScene(const LumenHardwareScene& scene,std::string& error)
    {
        const auto fail=[&](const char* message) { error=message; return false; };
        if (!scene.Identity.Epoch || !scene.Identity.Serial || !scene.Identity.Geometry || scene.Meshes.size()>LumenDistanceMaximumFields ||
            scene.Instances.size()>LumenDistanceMaximumInstances || scene.Geometry.size()!=scene.Meshes.size() ||
            scene.Words.size()>LumenHardwareMaximumWords || scene.Words.size()<scene.Instances.size()*LumenHardwareInstanceWords ||
            scene.Instances.empty()!=scene.Meshes.empty()) return fail("Lumen hardware scene has invalid identity, counts or storage");
        std::size_t cursor=scene.Instances.size()*LumenHardwareInstanceWords;
        std::unordered_map<const LumenHardwareMesh*,unsigned> unique;
        for (unsigned index=0;index<scene.Meshes.size();++index)
        {
            if (!scene.Meshes[index] || !unique.emplace(scene.Meshes[index].get(),index).second)
                return fail("Lumen hardware scene has null or repeated shared meshes");
            const auto& mesh=*scene.Meshes[index]; const auto& range=scene.Geometry[index];
            if (!MeshValid(mesh,error)) return false;
            const auto matched=[&](unsigned offset,const auto& values) {
                const auto first=(cursor+VectorWords-1)/VectorWords*VectorWords;
                if (offset!=first || !Matches(scene.Words,offset,values)) return false;
                for (auto word=cursor;word<first;++word) if (scene.Words[word]) return false;
                cursor=first+values.size()*(sizeof(typename std::decay_t<decltype(values)>::value_type)/sizeof(std::uint32_t)); return true;
            };
            if (!matched(range.Triangles,mesh.Triangles) || !matched(range.Materials,mesh.Materials) || !matched(range.Coverage,mesh.CoverageTexels))
                return fail("Lumen hardware geometry has stale payloads, aliases or invalid word ranges");
        }
        if (cursor!=scene.Words.size()) return fail("Lumen hardware scene has unowned trailing storage");
        std::vector<bool> used(scene.Meshes.size());
        for (unsigned index=0;index<scene.Instances.size();++index)
        {
            const auto& instance=scene.Instances[index]; std::array<Float4,3> normal;
            if (instance.Mesh>=scene.Meshes.size() || !NormalTransform(instance.World,normal) || !WorldFits(*scene.Meshes[instance.Mesh],instance.World))
                return fail("Lumen hardware instance has invalid geometry or pose");
            used[instance.Mesh]=true; const auto record=InstanceRecord(scene,instance,normal);
            if (std::memcmp(scene.Words.data()+std::size_t(index)*LumenHardwareInstanceWords,&record,sizeof(record)))
                return fail("Lumen hardware instance lost its primitive addressing or normal transform");
        }
        if (std::find(used.begin(),used.end(),false)!=used.end()) return fail("Lumen hardware scene retains an unreferenced mesh");
        error.clear(); return true;
    }
}
