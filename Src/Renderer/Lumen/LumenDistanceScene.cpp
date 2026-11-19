#include "Renderer/Lumen/LumenDistanceScene.h"
#include "Renderer/Lumen/LumenGlobalDistance.h"
#include "Renderer/Lumen/LumenHardwareRayTracing.h"

#include <atomic>
#include <cstring>
#include <numeric>

namespace Evans::Renderer
{
    namespace
    {
        constexpr unsigned FieldWords=sizeof(LumenDistanceGPUField)/sizeof(std::uint32_t);
        constexpr unsigned InstanceWords=sizeof(LumenDistanceGPUInstance)/sizeof(std::uint32_t);
        constexpr unsigned NodeWords=sizeof(LumenDistanceGPUNode)/sizeof(std::uint32_t);
        constexpr float MaximumToleranceCells=2;
        std::atomic<std::uint64_t> NextSceneIdentity{1};
        bool Finite(Float4 v) { return std::isfinite(v.X) && std::isfinite(v.Y) && std::isfinite(v.Z) && std::isfinite(v.W); }
        bool IdentityValid(const LumenDistanceSceneIdentity& identity) { return identity.Epoch && identity.Serial && identity.Geometry; }
        float FieldError(float cell,float band)
        {
            const double error=std::sqrt(3.0)*cell*0.5+double(band)/(LumenDistanceQuantization*2)+4*std::numeric_limits<float>::epsilon();
            return std::nextafter(float(error),std::numeric_limits<float>::infinity());
        }
        float MaximumTolerance(const LumenDistanceGPUField& field)
        { return std::nextafter((field.Distances.X-field.Distances.Y)/field.MinimumCell.W,0.0f); }
        template<class T> T Read(const std::vector<std::uint32_t>& words,std::size_t offset)
        { T result; std::memcpy(&result,words.data()+offset,sizeof(T)); return result; }
        template<class T> void Write(std::vector<std::uint32_t>& words,std::size_t offset,const T& value)
        { std::memcpy(words.data()+offset,&value,sizeof(T)); }
        bool Inverse(const double a[9],double inverse[9])
        {
            const double cofactor[]{a[4]*a[8]-a[5]*a[7],a[2]*a[7]-a[1]*a[8],a[1]*a[5]-a[2]*a[4],
                a[5]*a[6]-a[3]*a[8],a[0]*a[8]-a[2]*a[6],a[2]*a[3]-a[0]*a[5],
                a[3]*a[7]-a[4]*a[6],a[1]*a[6]-a[0]*a[7],a[0]*a[4]-a[1]*a[3]};
            const double determinant=a[0]*cofactor[0]+a[1]*cofactor[3]+a[2]*cofactor[6];
            if (!std::isfinite(determinant) || determinant==0) return false;
            for (unsigned index=0;index<9;++index) { inverse[index]=cofactor[index]/determinant; if (!std::isfinite(inverse[index])) return false; }
            return true;
        }
        bool InstanceData(const LumenDistanceField& field,const LumenDistanceInstance& source,LumenDistanceGPUInstance& result,Maths::AABB& bounds)
        {
            const auto& m=source.World.Values;
            if (!std::all_of(m.begin(),m.end(),[](float x){return std::isfinite(x);}) || m[12]!=0 || m[13]!=0 || m[14]!=0 || m[15]!=1 || source.UserId==LumenDistanceInvalid) return false;
            const double linear[]{m[0],m[1],m[2],m[4],m[5],m[6],m[8],m[9],m[10]}; double inverse[9]{};
            if (!Inverse(linear,inverse)) return false;
            for (unsigned row=0;row<3;++row)
            {
                auto& target=result.WorldToUnit[row]; target={float(inverse[row*3]/field.MeshScale),float(inverse[row*3+1]/field.MeshScale),float(inverse[row*3+2]/field.MeshScale),0};
                if (!Finite(target) || double(std::abs(target.X))+std::abs(target.Y)+std::abs(target.Z)>std::numeric_limits<float>::max()/4) return false;
            }
            const double center[]{field.Center.X,field.Center.Y,field.Center.Z}; double worldCenter[3]{};
            for (unsigned row=0;row<3;++row) worldCenter[row]=linear[row*3]*center[0]+linear[row*3+1]*center[1]+linear[row*3+2]*center[2]+m[row*4+3];
            result.Center={float(worldCenter[0]),float(worldCenter[1]),float(worldCenter[2]),0}; if (!Finite(result.Center)) return false;
            result.Metadata={source.Field,source.UserId,source.CastShadow?LumenDistanceShadowCaster:0u,0};
            const double minimum[]{field.Minimum.X,field.Minimum.Y,field.Minimum.Z};
            double low[]{INFINITY,INFINITY,INFINITY},high[]{-INFINITY,-INFINITY,-INFINITY};
            for (unsigned corner=0;corner<8;++corner)
            {
                double p[3]{};
                for (unsigned axis=0;axis<3;++axis) p[axis]=(minimum[axis]+((corner&(1u<<axis))?double(field.Dimensions[axis])*field.CellSize:0))*field.MeshScale;
                for (unsigned axis=0;axis<3;++axis)
                {
                    const double value=worldCenter[axis]+linear[axis*3]*p[0]+linear[axis*3+1]*p[1]+linear[axis*3+2]*p[2];
                    low[axis]=std::min(low[axis],value); high[axis]=std::max(high[axis],value);
                }
            }
            for (unsigned axis=0;axis<3;++axis)
            {
                bounds.Min[axis]=std::nextafter(float(low[axis]),-std::numeric_limits<float>::infinity());
                bounds.Max[axis]=std::nextafter(float(high[axis]),std::numeric_limits<float>::infinity());
                if (!std::isfinite(bounds.Min[axis]) || !std::isfinite(bounds.Max[axis])) return false;
            }
            return true;
        }
        bool Encloses(const LumenDistanceGPUNode& parent,const LumenDistanceGPUNode& child)
        {
            return parent.Minimum.X<=child.Minimum.X && parent.Minimum.Y<=child.Minimum.Y && parent.Minimum.Z<=child.Minimum.Z &&
                parent.Maximum.X>=child.Maximum.X && parent.Maximum.Y>=child.Maximum.Y && parent.Maximum.Z>=child.Maximum.Z;
        }
    }
    bool BuildLumenDistanceScene(const LumenDistanceSceneUpload& upload,LumenDistanceSceneData& output,std::string& error)
    {
        if (!IdentityValid(upload.Identity) || upload.Fields.size()>LumenDistanceMaximumFields || upload.Instances.size()>LumenDistanceMaximumInstances)
        { error="Lumen distance scene needs nonzero versions and bounded assets/instances"; return false; }
        const auto fieldCount=unsigned(upload.Fields.size()),instanceCount=unsigned(upload.Instances.size()),nodeCount=instanceCount?instanceCount*2-1:0;
        std::size_t wordCount=std::size_t(fieldCount)*FieldWords+std::size_t(instanceCount)*InstanceWords+std::size_t(nodeCount)*NodeWords;
        for (const auto& field:upload.Fields)
        {
            if (!field || !ValidateLumenDistanceField(*field,error)) { if (!field) error="Lumen distance scene references a missing field"; return false; }
            wordCount+=field->Directory.size()+field->Bricks.size()/2;
            if (wordCount>LumenDistanceMaximumWords) { error="Lumen distance scene exceeds its immutable storage budget"; return false; }
        }
        LumenDistanceSceneData result; result.Info.Identity=upload.Identity;
        result.Info.Counts={fieldCount,instanceCount,nodeCount,unsigned(wordCount)};
        result.Info.Offsets={0,fieldCount*FieldWords,fieldCount*FieldWords+instanceCount*InstanceWords,0};
        result.Words.resize(wordCount); std::size_t payload=result.Info.Offsets.Z+std::size_t(nodeCount)*NodeWords;
        for (unsigned index=0;index<fieldCount;++index)
        {
            const auto& source=*upload.Fields[index]; LumenDistanceGPUField field;
            field.Dimensions={source.Dimensions[0],source.Dimensions[1],source.Dimensions[2],0};
            field.BrickDimensions={source.BrickDimensions[0],source.BrickDimensions[1],source.BrickDimensions[2],0};
            field.Storage={unsigned(payload),unsigned(payload+source.Directory.size()),unsigned(source.Bricks.size()/LumenDistanceBrickSamples),0};
            field.MinimumCell={source.Minimum.X,source.Minimum.Y,source.Minimum.Z,source.CellSize};
            field.Distances={source.MaximumDistance,FieldError(source.CellSize,source.MaximumDistance),0,0};
            result.Info.MaximumHitToleranceCells=std::min(result.Info.MaximumHitToleranceCells,MaximumTolerance(field));
            Write(result.Words,std::size_t(index)*FieldWords,field);
            std::copy(source.Directory.begin(),source.Directory.end(),result.Words.begin()+payload); payload+=source.Directory.size();
            for (std::size_t sample=0;sample<source.Bricks.size();sample+=2)
                result.Words[payload++]=std::uint16_t(source.Bricks[sample])|(std::uint32_t(std::uint16_t(source.Bricks[sample+1]))<<16);
        }
        std::vector<Maths::AABB> bounds(instanceCount); result.Info.UserIds.reserve(instanceCount);
        for (unsigned index=0;index<instanceCount;++index)
        {
            const auto& source=upload.Instances[index]; LumenDistanceGPUInstance instance;
            if (source.Field>=fieldCount || !InstanceData(*upload.Fields[source.Field],source,instance,bounds[index]))
            { error="Lumen distance instance has a missing field, singular/nonfinite affine transform or reserved identifier"; return false; }
            Write(result.Words,result.Info.Offsets.Y+std::size_t(index)*InstanceWords,instance); result.Info.UserIds.push_back(source.UserId);
            result.Info.InstanceFlags.push_back(instance.Metadata.Z);
        }
        std::vector<LumenDistanceGPUNode> nodes; nodes.reserve(nodeCount); std::vector<unsigned> order(instanceCount); std::iota(order.begin(),order.end(),0);
        auto build=[&](auto&& self,unsigned begin,unsigned end)->void
        {
            Maths::AABB combined; for (unsigned item=begin;item<end;++item) combined.Expand(bounds[order[item]]);
            const auto index=unsigned(nodes.size()); LumenDistanceGPUNode node;
            node.Minimum={combined.Min.x,combined.Min.y,combined.Min.z,0}; node.Maximum={combined.Max.x,combined.Max.y,combined.Max.z,0}; nodes.push_back(node);
            if (end-begin==1) nodes[index].Links={LumenDistanceInvalid,order[begin],1,index+1};
            else
            {
                unsigned axis=0; for (unsigned candidate=1;candidate<3;++candidate)
                    if (double(combined.Max[candidate])-combined.Min[candidate]>double(combined.Max[axis])-combined.Min[axis]) axis=candidate;
                const unsigned middle=begin+(end-begin)/2;
                std::nth_element(order.begin()+begin,order.begin()+middle,order.begin()+end,[&](unsigned a,unsigned b)
                {
                    const double ca=double(bounds[a].Min[axis])+bounds[a].Max[axis],cb=double(bounds[b].Min[axis])+bounds[b].Max[axis];
                    return ca!=cb?ca<cb:a<b;
                });
                self(self,begin,middle); self(self,middle,end); nodes[index].Links={index+1,LumenDistanceInvalid,0,unsigned(nodes.size())};
            }
        };
        if (instanceCount) build(build,0,instanceCount);
        for (unsigned index=0;index<nodes.size();++index) Write(result.Words,result.Info.Offsets.Z+std::size_t(index)*NodeWords,nodes[index]);
        if (upload.Hair && !AppendLumenHairVolume(*upload.Hair,LumenDistanceMaximumWords,result.Words,result.Info.HairHeaderWord,error)) return false;
        result.Info.Counts.W=unsigned(result.Words.size());
        if (!ValidateLumenDistanceScene(result,error)) return false; output=std::move(result); error.clear(); return true;
    }
    bool ValidateLumenDistanceScene(const LumenDistanceSceneData& scene,std::string& error)
    {
        const auto& info=scene.Info; const auto& count=info.Counts; const auto& offsets=info.Offsets; const auto& words=scene.Words;
        auto fail=[&](const char* reason){error=reason; return false;};
        if (info.Global || !IdentityValid(info.Identity) || count.X>LumenDistanceMaximumFields || count.Y>LumenDistanceMaximumInstances || count.Z!=(count.Y?count.Y*2-1:0) ||
            count.W!=words.size() || words.size()>LumenDistanceMaximumWords || info.UserIds.size()!=count.Y || info.InstanceFlags.size()!=count.Y || offsets.X || offsets.W ||
            offsets.Y!=count.X*FieldWords || offsets.Z!=offsets.Y+count.Y*InstanceWords || offsets.Z+std::size_t(count.Z)*NodeWords>words.size())
            return fail("Lumen distance scene has invalid versions, counts or record offsets");
        std::size_t payload=offsets.Z+std::size_t(count.Z)*NodeWords; float maximumTolerance=MaximumToleranceCells;
        for (unsigned index=0;index<count.X;++index)
        {
            const auto field=Read<LumenDistanceGPUField>(words,std::size_t(index)*FieldWords);
            const unsigned dimensions[]{field.Dimensions.X,field.Dimensions.Y,field.Dimensions.Z};
            const unsigned bricks[]{field.BrickDimensions.X,field.BrickDimensions.Y,field.BrickDimensions.Z}; std::size_t directoryCount=1,voxelCount=1;
            for (unsigned axis=0;axis<3;++axis)
            {
                if (!dimensions[axis] || dimensions[axis]>LumenDistanceMaximumAxis || bricks[axis]!=(dimensions[axis]+LumenDistanceBrickInterior-1)/LumenDistanceBrickInterior)
                    return fail("Lumen distance field has invalid grid or brick dimensions");
                directoryCount*=bricks[axis]; voxelCount*=dimensions[axis];
            }
            const double cell=field.MinimumCell.W,band=field.Distances.X;
            if (voxelCount>LumenDistanceMaximumVoxels || field.Dimensions.W || field.BrickDimensions.W || field.Storage.W || !Finite(field.MinimumCell) ||
                !Finite(field.Distances) || cell<1.0/128 || cell>0.25 || band<cell || band>cell*8 || field.Distances.Z || field.Distances.W ||
                field.Distances.Y!=FieldError(float(cell),float(band)) || field.Storage.X!=payload || field.Storage.Y!=payload+directoryCount || field.Storage.Z>directoryCount)
                return fail("Lumen distance field has invalid normalized metrics or payload offsets");
            const std::size_t end=field.Storage.Y+std::size_t(field.Storage.Z)*LumenDistanceBrickSamples/2;
            if (end>words.size()) return fail("Lumen distance brick payload exceeds immutable storage");
            for (std::size_t entry=payload;entry<payload+directoryCount;++entry)
                if (words[entry]>=LumenDistanceFirstBrick && words[entry]-LumenDistanceFirstBrick>=field.Storage.Z)
                    return fail("Lumen distance directory references a missing packed brick");
            constexpr std::uint32_t SignedHalfMask=0xffff,InvalidSignedHalf=0x8000;
            for (std::size_t sample=field.Storage.Y;sample<end;++sample)
                if ((words[sample]&SignedHalfMask)==InvalidSignedHalf || (words[sample]>>16)==InvalidSignedHalf)
                    return fail("Lumen distance payload contains a reserved signed sample");
            payload=end; maximumTolerance=std::min(maximumTolerance,MaximumTolerance(field));
        }
        if (!ValidateLumenHairWords(words,info.HairHeaderWord,payload,error)) return false;
        if (info.MaximumHitToleranceCells!=maximumTolerance || maximumTolerance<=0)
            return fail("Lumen distance scene has trailing storage or an invalid surface tolerance limit");
        for (unsigned index=0;index<count.Y;++index)
        {
            const auto instance=Read<LumenDistanceGPUInstance>(words,offsets.Y+std::size_t(index)*InstanceWords);
            if (!Finite(instance.Center) || instance.Center.W || instance.Metadata.X>=count.X || instance.Metadata.Y==LumenDistanceInvalid ||
                instance.Metadata.Y!=info.UserIds[index] || instance.Metadata.Z!=info.InstanceFlags[index] || (instance.Metadata.Z&~LumenDistanceShadowCaster) || instance.Metadata.W)
                return fail("Lumen distance instance has invalid fields or identifiers");
            double matrix[9]{},inverse[9]{};
            for (unsigned row=0;row<3;++row)
            {
                const auto& value=instance.WorldToUnit[row];
                if (!Finite(value) || value.W || double(std::abs(value.X))+std::abs(value.Y)+std::abs(value.Z)>std::numeric_limits<float>::max()/4)
                    return fail("Lumen distance instance exceeds finite transform arithmetic");
                matrix[row*3]=value.X; matrix[row*3+1]=value.Y; matrix[row*3+2]=value.Z;
            }
            if (!Inverse(matrix,inverse)) return fail("Lumen distance instance has a singular normalized transform");
        }
        std::vector<bool> seen(count.Y); const auto node=[&](unsigned index){return Read<LumenDistanceGPUNode>(words,offsets.Z+std::size_t(index)*NodeWords);};
        for (unsigned index=0;index<count.Z;++index)
        {
            const auto current=node(index); const auto links=current.Links;
            if (!Finite(current.Minimum) || !Finite(current.Maximum) || current.Minimum.W || current.Maximum.W ||
                current.Minimum.X>current.Maximum.X || current.Minimum.Y>current.Maximum.Y || current.Minimum.Z>current.Maximum.Z ||
                links.Z>1 || links.W<=index || links.W>count.Z || (!index && links.W!=count.Z)) return fail("Lumen distance BVH has invalid bounds or escape links");
            if (links.Z)
            {
                if (links.X!=LumenDistanceInvalid || links.W!=index+1 || links.Y>=count.Y || seen[links.Y]) return fail("Lumen distance BVH has an invalid or repeated leaf");
                seen[links.Y]=true;
            }
            else
            {
                if (links.X!=index+1 || links.Y!=LumenDistanceInvalid || links.X>=count.Z) return fail("Lumen distance BVH has an invalid first child");
                const auto left=node(links.X); if (left.Links.W<=links.X || left.Links.W>=links.W) return fail("Lumen distance BVH has no valid second child");
                const auto right=node(left.Links.W);
                if (right.Links.W!=links.W || !Encloses(current,left) || !Encloses(current,right)) return fail("Lumen distance BVH children escape their parent");
            }
        }
        if (std::any_of(seen.begin(),seen.end(),[](bool found){return !found;})) return fail("Lumen distance BVH omits an instance");
        if (info.Hardware)
        {
            const auto& hardware=*info.Hardware;
            if (!ValidateLumenHardwareScene(hardware,error)) return false;
            if (hardware.Identity.Epoch!=info.Identity.Epoch || hardware.Identity.Serial!=info.Identity.Serial ||
                hardware.Identity.Geometry!=info.Identity.Geometry || hardware.Instances.size()!=count.Y)
                return fail("Lumen hardware transport snapshot belongs to another distance-scene generation");
            for (unsigned index=0;index<count.Y;++index)
                if (hardware.Instances[index].CastShadow!=bool(info.InstanceFlags[index]&LumenDistanceShadowCaster))
                    return fail("Lumen hardware transport changed instance shadow visibility");
        }
        error.clear(); return true;
    }
    bool BuildLumenDistanceTraceFrame(const LumenDistanceSceneInfo& scene,const LumenDistanceTraceUpload& upload,LumenDistanceTraceFrame& output,std::string& error)
    {
        if (!IdentityValid(scene.Identity) || scene.Counts.X>LumenDistanceMaximumFields || scene.Counts.Y>LumenDistanceMaximumInstances ||
            scene.Counts.Z!=(scene.Counts.Y?scene.Counts.Y*2-1:0) || scene.Counts.W>LumenDistanceMaximumWords ||
            scene.Offsets.X || scene.Offsets.Y!=scene.Counts.X*FieldWords || scene.Offsets.Z!=scene.Offsets.Y+scene.Counts.Y*InstanceWords ||
            scene.Offsets.W || scene.Offsets.Z+scene.Counts.Z*NodeWords>scene.Counts.W ||
            (scene.HairHeaderWord && (scene.HairHeaderWord%4 || scene.HairHeaderWord>=scene.Counts.W || scene.Counts.W-scene.HairHeaderWord<sizeof(LumenHairHeader)/sizeof(unsigned))) ||
            !std::isfinite(upload.HairSky.X) || !std::isfinite(upload.HairSky.Y) || !std::isfinite(upload.HairSky.Z) || upload.HairSky.X<0 || upload.HairSky.Y<0 || upload.HairSky.Z<0 ||
            !std::isfinite(scene.MaximumHitToleranceCells) || scene.MaximumHitToleranceCells<=0 || scene.MaximumHitToleranceCells>2 ||
            scene.UserIds.size()!=scene.Counts.Y || scene.InstanceFlags.size()!=scene.Counts.Y ||
            std::any_of(scene.InstanceFlags.begin(),scene.InstanceFlags.end(),[](unsigned flags){return (flags&~LumenDistanceShadowCaster)!=0;}) ||
            upload.Rays.empty() || upload.Rays.size()>LumenDistanceMaximumRays || !upload.MaximumSteps || upload.MaximumSteps>LumenDistanceMaximumTraceSteps ||
            !std::isfinite(upload.HitToleranceCells) || upload.HitToleranceCells<0 || upload.HitToleranceCells>=scene.MaximumHitToleranceCells ||
            !std::isfinite(upload.StepScale) || upload.StepScale<=0 || upload.StepScale>1)
        { error="Lumen distance trace needs bounded rays/steps and a surface threshold below the saturated band"; return false; }
        constexpr double DirectionTolerance=0.0001;
        for (const auto& ray:upload.Rays)
        {
            const auto& o=ray.OriginMinimum; const auto& d=ray.DirectionMaximum;
            const double length=double(d.X)*d.X+double(d.Y)*d.Y+double(d.Z)*d.Z;
            if (!Finite(o) || !Finite(d) || o.W<0 || d.W<o.W || std::abs(length-1)>DirectionTolerance)
            { error="Lumen distance rays require finite unit directions and nonnegative ordered intervals"; return false; }
            const double origin[]{o.X,o.Y,o.Z},direction[]{d.X,d.Y,d.Z};
            for (unsigned axis=0;axis<3;++axis) for (float distance:{o.W,d.W})
                if (std::abs(origin[axis]+direction[axis]*distance)>std::numeric_limits<float>::max())
                { error="Lumen distance ray endpoint exceeds native coordinate precision"; return false; }
        }
        LumenDistanceTraceFrame result;
        result.Counts={unsigned(upload.Rays.size()),scene.Counts.Z,scene.Counts.Y,scene.Counts.X}; result.Offsets=scene.Offsets;
        result.Traversal={upload.MaximumSteps,LumenDistanceTraceThreads*LumenDistanceTraceColumns,upload.ShadowCastersOnly?LumenDistanceShadowCaster:0u,upload.HairTransport?scene.HairHeaderWord<<LumenHairPointerShift:0u};
        result.HairSky={upload.HairSky.X,upload.HairSky.Y,upload.HairSky.Z,0};
        result.Parameters={upload.HitToleranceCells,upload.StepScale,0,0};
        if (scene.Global && upload.HitToleranceCells<=scene.Global->Frame.Header.Numeric.X)
        {
            if (!ValidateLumenGlobalDistancePlan(scene,*scene.Global,error)) return false;
            if (!std::isfinite(upload.GlobalNearDistance) || upload.GlobalNearDistance<0 || !upload.GlobalMaximumSteps || upload.GlobalMaximumSteps>LumenDistanceMaximumTraceSteps)
            { error="Global distance tracing requires finite near distance and bounded steps"; return false; }
            result.Offsets.W=scene.Global->Frame.Header.Layout.W;
            result.Parameters.Z=upload.GlobalNearDistance; result.Parameters.W=float(upload.GlobalMaximumSteps);
        }
        output=result; error.clear(); return true;
    }
    bool ValidateLumenDistanceTraceResult(const LumenDistanceSceneInfo& scene,const LumenDistanceTraceUpload& upload,const LumenDistanceTraceResult& result,std::string& error)
    {
        if (result.Identity.Epoch!=scene.Identity.Epoch || result.Identity.Serial!=scene.Identity.Serial || result.Identity.Geometry!=scene.Identity.Geometry || result.Hits.size()!=upload.Rays.size())
        { error="Lumen distance feedback has mismatched scene identity or ray count"; return false; }
        if (result.HardwareTracing && !scene.Hardware)
        { error="Lumen hardware feedback has no matching immutable triangle snapshot"; return false; }
        constexpr float NormalTolerance=0.002f,AbsoluteRoundoff=0.00001f,RelativeRoundoff=8*std::numeric_limits<float>::epsilon();
        for (std::size_t index=0;index<result.Hits.size();++index)
        {
            const auto& hit=result.Hits[index]; const auto& ray=upload.Rays[index]; const auto status=hit.Metadata.Z;
            if (!Finite(hit.PositionDistance) || !Finite(hit.NormalResidual) || !Finite(hit.Uncertainty) || !Finite(hit.Medium) ||
                hit.Medium.X<0 || hit.Medium.Y<0 || hit.Medium.Z<0 || hit.Medium.W<0 || hit.Medium.W>1 || status>unsigned(LumenDistanceTraceStatus::Unresolved) ||
                hit.PositionDistance.W<ray.OriginMinimum.W || hit.PositionDistance.W>ray.DirectionMaximum.W ||
                hit.Metadata.W>std::uint64_t(upload.MaximumSteps)*scene.Counts.Y*(scene.Global?2:1) || hit.Uncertainty.X<0 || hit.Uncertainty.Y<hit.Uncertainty.X || hit.Uncertainty.Z || hit.Uncertainty.W<0 || hit.Uncertainty.W>float(scene.Global?upload.GlobalMaximumSteps:0) || std::floor(hit.Uncertainty.W)!=hit.Uncertainty.W)
            { error="Lumen distance feedback has invalid finite metrics, interval or traversal status"; return false; }
            const float actual[]{hit.PositionDistance.X,hit.PositionDistance.Y,hit.PositionDistance.Z};
            const float origin[]{ray.OriginMinimum.X,ray.OriginMinimum.Y,ray.OriginMinimum.Z},direction[]{ray.DirectionMaximum.X,ray.DirectionMaximum.Y,ray.DirectionMaximum.Z};
            for (unsigned axis=0;axis<3;++axis)
            {
                const double expected=double(origin[axis])+double(direction[axis])*hit.PositionDistance.W;
                const double tolerance=AbsoluteRoundoff+RelativeRoundoff*std::max({std::abs(expected),double(std::abs(origin[axis])),double(std::abs(direction[axis]*hit.PositionDistance.W))});
                if (std::abs(actual[axis]-expected)>tolerance) { error="Lumen distance feedback position left its source ray"; return false; }
            }
            const auto& normal=hit.NormalResidual; const double square=double(normal.X)*normal.X+double(normal.Y)*normal.Y+double(normal.Z)*normal.Z;
            if (status==unsigned(LumenDistanceTraceStatus::Miss))
            {
                if (hit.Metadata.X!=LumenDistanceInvalid || hit.Metadata.Y!=LumenDistanceInvalid || square || normal.W || hit.Uncertainty.X || hit.Uncertainty.Y || hit.PositionDistance.W!=ray.DirectionMaximum.W)
                { error="Lumen distance miss retained a stale surface or identifier"; return false; }
            }
            else
            {
                if (hit.Metadata.X>=scene.UserIds.size() || hit.Metadata.Y!=scene.UserIds[hit.Metadata.X]) { error="Lumen distance feedback references an invalid instance"; return false; }
                if (upload.ShadowCastersOnly && (hit.Metadata.X>=scene.InstanceFlags.size() || !(scene.InstanceFlags[hit.Metadata.X]&LumenDistanceShadowCaster)))
                { error="Lumen shadow feedback references a non-casting instance"; return false; }
                if (status==unsigned(LumenDistanceTraceStatus::Surface))
                {
                    const bool traversalValid=result.HardwareTracing?hit.Metadata.W==0 && normal.W==0 &&
                        hit.Uncertainty.X==0 && hit.Uncertainty.Y==0 && hit.Uncertainty.W==0:hit.Metadata.W!=0 &&
                        std::abs(normal.W)<=hit.Uncertainty.Y+AbsoluteRoundoff+RelativeRoundoff*hit.Uncertainty.Y;
                    if (std::abs(square-1)>NormalTolerance || !traversalValid)
                    { error="Lumen distance surface has an invalid normal or traversal uncertainty"; return false; }
                }
                else if (square || normal.W || hit.Uncertainty.X || hit.Uncertainty.Y)
                { error="Unresolved Lumen distance feedback claimed a valid surface"; return false; }
            }
        }
        error.clear(); return true;
    }
    LumenDistanceSceneHandle NewLumenDistanceSceneHandle() noexcept
    {
        auto value=NextSceneIdentity.load(std::memory_order_relaxed);
        while (value && value!=UINT64_MAX) if (NextSceneIdentity.compare_exchange_weak(value,value+1,std::memory_order_relaxed)) return {value};
        return {};
    }
}
