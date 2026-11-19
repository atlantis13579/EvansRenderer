#include "Renderer/Nanite/NaniteVisibility.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace Evans::Renderer
{
    namespace
    {
        bool Fail(std::string& error,const char* message) { error=message; return false; }
    }
    bool ValidateNaniteVisibilityQueries(const std::vector<NaniteVisibilityQuery>& queries,std::string& error)
    {
        if (queries.size()>NaniteVisibilityMaximumQueries)
            return Fail(error,"Nanite visibility query count exceeds the native budget");
        for (const auto& q:queries)
        {
            const float bounds[]{q.Minimum.X,q.Minimum.Y,q.Minimum.Z,q.Maximum.X,q.Maximum.Y,q.Maximum.Z};
            for (auto value:bounds) if (!std::isfinite(value)) return Fail(error,"Nanite visibility bounds must be finite");
            if (q.Minimum.W!=0 || q.Maximum.W!=0 || q.Minimum.X>q.Maximum.X || q.Minimum.Y>q.Maximum.Y || q.Minimum.Z>q.Maximum.Z || q.Identity.X>1)
                return Fail(error,"Invalid Nanite visibility bounds or history flag");
            for (const auto* matrix:{&q.Current,&q.Previous}) for (auto value:matrix->Values)
                if (!std::isfinite(value)) return Fail(error,"Nanite visibility transform must be finite");
            if (q.Identity.Y!=0)
                return Fail(error,"Native Nanite query classifications must come from the GPU main pass");
        }
        error.clear(); return true;
    }
    bool BuildNaniteVisibilityQueries(const NaniteHierarchy& h,const NanitePages& pages,const NaniteGpuSelection& selection,
        const Matrix4& current,const Matrix4& previous,bool historyValid,unsigned instance,
        std::vector<NaniteVisibilityQuery>& output,std::string& error)
    {
        if (!h.Signature || pages.HierarchySource!=h.Signature || !std::isfinite(pages.MaximumQuantizationDisplacement) || pages.MaximumQuantizationDisplacement<0 ||
            selection.ActiveClusters!=selection.Clusters.size()) return Fail(error,"Nanite visibility source or selection differs");
        for (const auto* matrix:{&current,&previous}) for (auto value:matrix->Values) if (!std::isfinite(value)) return Fail(error,"Invalid Nanite object-to-clip transform");
        std::vector<NaniteVisibilityQuery> result; std::set<unsigned> seen;
        for (auto id:selection.Clusters)
        {
            if (id>=h.Clusters.size() || !seen.insert(id).second) return Fail(error,"Invalid selected Nanite cluster");
            const auto& cluster=h.Clusters[id]; NaniteVisibilityQuery q; q.Current=current; q.Previous=previous; q.Identity={historyValid?1u:0u,0,id,instance};
            const auto lower=[&](float f) { const double d=double(f)-pages.MaximumQuantizationDisplacement; float r=float(d); if (double(r)>d) r=std::nextafter(r,-INFINITY); return r; };
            const auto upper=[&](float f) { const double d=double(f)+pages.MaximumQuantizationDisplacement; float r=float(d); if (double(r)<d) r=std::nextafter(r,INFINITY); return r; };
            q.Minimum={lower(cluster.BoundsMinimum.X),lower(cluster.BoundsMinimum.Y),lower(cluster.BoundsMinimum.Z),0};
            q.Maximum={upper(cluster.BoundsMaximum.X),upper(cluster.BoundsMaximum.Y),upper(cluster.BoundsMaximum.Z),0};
            for (auto value:{q.Minimum.X,q.Minimum.Y,q.Minimum.Z,q.Maximum.X,q.Maximum.Y,q.Maximum.Z})
                if (!std::isfinite(value)) return Fail(error,"Quantized Nanite visibility bounds exceed float range");
            result.push_back(q);
        }
        output=std::move(result); error.clear(); return true;
    }
}
