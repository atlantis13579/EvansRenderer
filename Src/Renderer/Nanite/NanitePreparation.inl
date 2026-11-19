// A published scene generation leases one decoded cut per source family. All
// instances/material sections share that cut across camera, shadow and reflection
// views. Stage every family before replacing renderer-visible geometry pointers.
Renderer::NaniteGeometryGeneration::~NaniteGeometryGeneration()
{
    for (auto& family:Families) for (auto& variant:family.Variants) if (variant) Owner->DestroyGeometryResources(*variant);
    for (auto*& source:SourceReferences) Owner->ReleaseGeometry(source);
}
Renderer::NaniteGeometryFamily* Renderer::NaniteFamilyFor(const MeshResource& mesh,std::size_t& part)
{
    if (!m_NaniteGeometry) return nullptr;
    const auto& bindings=m_NaniteGeometry->Bindings;
    const auto found=std::lower_bound(bindings.begin(),bindings.end(),mesh.Mesh.Id,[](const auto& value,const auto& id){return value.Id<id;});
    if (found==bindings.end() || found->Id!=mesh.Mesh.Id) return nullptr;
    auto& family=m_NaniteGeometry->Families[found->Family];
    const auto source=std::find(family.Sources.begin(),family.Sources.end(),found->Source);
    if (source==family.Sources.end()) return nullptr;
    part=std::size_t(source-family.Sources.begin()); return &family;
}
bool Renderer::PrepareStreamedGeometry()
{
    // Restore the inactive owner's sources before publishing the new owner.
    const bool ready=m_Techniques.Nanite.Enabled ? PrepareVirtualGeometry() && PrepareNaniteGeometry() :
        PrepareNaniteGeometry() && PrepareVirtualGeometry();
    return ready && PrepareNaniteHostGeometry();
}
bool Renderer::PrepareNaniteLightmapSources(NaniteGeometryFamily& family)
{
    if (!family.LightmapSources.empty()) return true;
    std::vector<std::shared_ptr<const LightmapSourceGeometry>> snapshots;
    for (unsigned part=0;part<family.Sources.size();++part)
    {
        auto source=std::make_shared<LightmapSourceGeometry>(); source->Levels.resize(2); source->Segments.resize(2);
        std::map<std::array<std::uint32_t,NaniteVertexChannels>,unsigned> unique;
        const auto append=[&](const Vertex& vertex)
        {
            std::array<std::uint32_t,NaniteVertexChannels> key; std::memcpy(key.data(),&vertex,sizeof(vertex));
            const auto found=unique.find(key); if (found!=unique.end()) return found->second;
            const auto id=unsigned(source->Vertices.size()); source->Vertices.push_back(vertex); unique.emplace(key,id); return id;
        };
        const auto& original=*family.Sources[part];
        for (auto index:original.Indices) source->Levels[0].push_back(append(original.Vertices[index]));
        // Persistent charts cover every quantized cluster representation. The
        // selected cluster tags change without rebaking the transport source.
        for (unsigned id=0;id<family.Hierarchy.Clusters.size();++id)
        {
            if (family.Hierarchy.Clusters[id].Material!=part) continue;
            NaniteDecodedCluster cluster; if (!DecodeNaniteCluster(family.Pages,id,cluster,m_LastError)) return false;
            auto& indices=source->Levels[1];
            source->Segments[1].push_back({unsigned(indices.size()/VisibilityTriangleCorners),unsigned(cluster.Indices.size()/VisibilityTriangleCorners),id});
            for (auto index:cluster.Indices) indices.push_back(append(cluster.Vertices[index]));
        }
        snapshots.push_back(std::move(source));
    }
    family.LightmapSources=std::move(snapshots); return true;
}

bool Renderer::PrepareNaniteGeometry()
{
    m_NaniteStatistics={}; m_NanitePending=false;
    struct PreparationTimer
    {
        FrameStatistics& Statistics;
        std::chrono::steady_clock::time_point Start=std::chrono::steady_clock::now();
        ~PreparationTimer()
        {
            Statistics.NanitePreparationMicroseconds=static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-Start).count());
            const auto queue=GetNaniteAssetQueueStatistics(); Statistics.NaniteCookRequests=queue.Requests; Statistics.NaniteCookSourceBytes=queue.SourceBytes;
        }
    } timer{m_NaniteStatistics};
    const auto restore=[&](bool source)
    {
        bool changed=false;
        for (auto& entry:m_Meshes)
        {
            auto& mesh=entry.second; auto* geometry=mesh.SourceGeometry;
            if (!source)
            {
                std::size_t part=0; auto* family=NaniteFamilyFor(mesh,part);
                if (!family || family->Variants.empty()) continue;
                geometry=family->Variants[part].get();
            }
            if (mesh.Geometry==geometry) continue;
            mesh.Geometry=geometry; mesh.BoundsMinimum=geometry->BoundsMinimum; mesh.BoundsMaximum=geometry->BoundsMaximum;
            mesh.BoundsCenter=Multiply(Add(mesh.BoundsMinimum,mesh.BoundsMaximum),0.5f);
            mesh.SelectedLod=mesh.PreviousLod=0; mesh.PreviousTransformValid=false; changed=true;
        }
        if (changed) { ++m_GeometryViewSerial; ResetFrameHistory(); }
    };
    if (!m_Techniques.Nanite.Enabled)
    {
        m_PendingNaniteGeometry.reset();
        if (m_NaniteGeometry) { restore(true); m_NaniteGeometry.reset(); }
        return true;
    }
    const auto reject=[&](){restore(false); return false;};
    if (!m_RHI->GetCapabilities().NaniteRendering) { Fail("Native Nanite rendering is unavailable"); return reject(); }
    PlanarCaptureFrame planarDemand;
    if (Techniques::UsesPlanarCapture(m_Techniques.Reflections) && !BuildPlanarCaptureFrame(m_ViewProjection,m_RenderLighting.CameraPosition,
            m_RenderWidth,m_RenderHeight,m_Techniques.Reflections.Planar,planarDemand,m_LastError)) return reject();
    const auto& settings=m_Techniques.Nanite;
    m_NaniteStatistics.NaniteScenePageBudget=settings.ScenePages;
    auto candidate=std::make_unique<NaniteGeometryGeneration>(this);
    candidate->PageBudget=settings.PhysicalPages; candidate->ScenePageBudget=settings.ScenePages; candidate->SourceSerial=m_PreparedGeometry?m_PreparedGeometry->Serial:0;
    std::vector<MeshResource*> ordered; for (auto& entry:m_Meshes) if (entry.second.Mesh.Visible && !entry.second.Mesh.Vegetation && !entry.second.Mesh.DynamicVertices) ordered.push_back(&entry.second);
    std::sort(ordered.begin(),ordered.end(),[](auto* a,auto* b){return a->Mesh.Id<b->Mesh.Id;});
    struct ObjectKey
    {
        std::uint64_t Selection; MeshGeometryResource* Source; Matrix4 World;
        bool operator==(const ObjectKey& other) const
        { return Selection==other.Selection && (Selection?World.Values==other.World.Values:Source==other.Source); }
    };
    struct ObjectHash
    {
        std::size_t operator()(const ObjectKey& key) const
        {
            std::uint64_t hash=key.Selection;
            if (key.Selection) for (auto value:key.World.Values) hash=(hash^std::hash<float>{}(value))*1099511628211ull;
            else hash^=std::hash<MeshGeometryResource*>{}(key.Source);
            return static_cast<std::size_t>(hash);
        }
    };
    using Sources=std::vector<MeshGeometryResource*>;
    struct SourcesHash
    {
        std::size_t operator()(const Sources& sources) const
        {
            std::uint64_t hash=sources.size();
            for (auto* source:sources) hash=(hash^std::hash<MeshGeometryResource*>{}(source))*1099511628211ull;
            return static_cast<std::size_t>(hash);
        }
    };
    std::unordered_map<ObjectKey,std::size_t,ObjectHash> objectLookup;
    std::vector<std::vector<MeshResource*>> objects;
    for (auto* mesh:ordered)
    {
        const ObjectKey key{mesh->Mesh.SelectionGroup,mesh->SourceGeometry,mesh->Mesh.WorldTransform};
        const auto entry=objectLookup.emplace(key,objects.size());
        if (entry.second) objects.emplace_back();
        objects[entry.first->second].push_back(mesh);
    }
    std::unordered_map<Sources,std::size_t,SourcesHash> familyLookup;
    for (const auto& object:objects)
    {
        Sources sources; for (auto* member:object) sources.push_back(member->SourceGeometry);
        std::sort(sources.begin(),sources.end(),[](const auto* a,const auto* b)
        {return a!=b && NaniteAssetSectionLess({&a->Vertices,&a->Indices,a->Hash},{&b->Vertices,&b->Indices,b->Hash});});
        sources.erase(std::unique(sources.begin(),sources.end()),sources.end());
        const auto entry=familyLookup.emplace(sources,candidate->Families.size()); const auto family=entry.first->second;
        if (entry.second) { NaniteGeometryFamily value; value.Sources=std::move(sources); candidate->Families.push_back(std::move(value)); }
        for (auto* member:object) candidate->Bindings.push_back({member->Mesh.Id,member->SourceGeometry,family});
    }
    std::sort(candidate->Bindings.begin(),candidate->Bindings.end(),[](const auto& a,const auto& b){return a.Id<b.Id;});
    for (std::size_t i=0;i<candidate->Bindings.size();++i) candidate->Families[candidate->Bindings[i].Family].Bindings.push_back(i);
    const auto matches=[&](const auto& existing)
    { return existing && candidate->PageBudget==existing->PageBudget && candidate->ScenePageBudget==existing->ScenePageBudget && candidate->SourceSerial==existing->SourceSerial && candidate->Bindings==existing->Bindings; };
    const bool reuse=matches(m_NaniteGeometry),resume=!reuse && matches(m_PendingNaniteGeometry);
    if (reuse) m_PendingNaniteGeometry.reset();
    else if (!resume) m_PendingNaniteGeometry.reset();
    // A capacity change may keep rendering the previous resident cut. Changed
    // sources/bindings instead render their ordinary geometry until all new
    // families are ready, so visibility never mixes stale and current tables.
    if (m_NaniteGeometry && (candidate->SourceSerial!=m_NaniteGeometry->SourceSerial || candidate->Bindings!=m_NaniteGeometry->Bindings))
    { restore(true); m_NaniteGeometry.reset(); }
    if (m_NaniteGeometry && (m_NaniteGeometry->Budget.AssignedPages>settings.ScenePages ||
        (settings.PhysicalPages && std::any_of(m_NaniteGeometry->Families.begin(),m_NaniteGeometry->Families.end(),
            [&](const auto& family){return family.Capacity>settings.PhysicalPages;}))))
    { restore(true); m_NaniteGeometry.reset(); }
    NaniteGeometryGeneration* generation=reuse?m_NaniteGeometry.get():resume?m_PendingNaniteGeometry.get():candidate.get();
    if (!reuse && !resume)
    {
        std::set<MeshGeometryResource*,std::less<MeshGeometryResource*>> originals;
        for (auto* mesh:ordered) originals.insert(mesh->BaseGeometry);
        for (auto* source:originals) { ++source->ReferenceCount; generation->SourceReferences.push_back(source); }
        generation->Serial=++m_NaniteSerial; m_PendingNaniteGeometry=std::move(candidate);
    }
    m_NaniteStatistics.NaniteFamilyCount=static_cast<unsigned>(generation->Families.size());
    const auto retainPublished=[&]()
    {
        restore(!m_NaniteGeometry); m_NaniteStatistics.NaniteSelectedClusters=m_NaniteStatistics.NaniteResidentPages=0;
        if (m_NaniteGeometry)
        {
            m_NaniteStatistics.NaniteMeshCount=static_cast<unsigned>(m_NaniteGeometry->Bindings.size());
            m_NaniteStatistics.NaniteGeometryGeneration=m_NaniteGeometry->Serial;
            m_NaniteStatistics.NaniteAllocatedPages=m_NaniteGeometry->Budget.AssignedPages;
            m_NaniteStatistics.NaniteRootPages=m_NaniteGeometry->Budget.RootPages;
            for (const auto& family:m_NaniteGeometry->Families)
            {
                m_NaniteStatistics.NaniteSelectedClusters+=static_cast<unsigned>(family.Published.Geometry().Selection.Clusters.size());
                for (const auto& slot:family.Stream->Residency().Slots()) m_NaniteStatistics.NaniteResidentPages+=slot.Page!=NaniteInvalid;
            }
        }
        return true;
    };
    const auto frame=++m_NaniteFrame;
    if (!generation->BudgetReady)
    {
        bool assetsReady=true;
        for (std::size_t step=0;step<generation->Families.size();++step)
        {
            auto& family=generation->Families[(step+frame)%generation->Families.size()];
            if (family.AssetReady) continue;
            if (!family.Asset)
            {
                std::vector<NaniteAssetSection> sections;
                for (const auto* source:family.Sources) sections.push_back({&source->Vertices,&source->Indices,source->Hash});
                bool admitted=false;
                if (!NaniteAssetRequest::TryStart(sections,settings.PhysicalPages,family.Asset,admitted,m_LastError)) return reject();
                if (!admitted) { assetsReady=false; continue; }
            }
            bool ready=false;
            if (!family.Asset->Poll(ready,family.Hierarchy,family.Pages,m_LastError)) return reject();
            if (!ready) { assetsReady=false; continue; }
            family.Asset.reset(); family.AssetReady=true;
        }
        if (!assetsReady) { m_NanitePending=true; return retainPublished(); }
        std::vector<NaniteSceneDemand> demands;
        for (const auto& family:generation->Families) demands.push_back({family.Pages.PageCount(),static_cast<unsigned>(family.Pages.PinnedPages.size())});
        if (!PlanNaniteSceneBudget(demands,settings.PhysicalPages,settings.ScenePages,generation->Budget,m_LastError)) return reject();
        if (!generation->Budget.Feasible)
        {
            // A budget below complete root coverage cannot produce a legal cut.
            // Keep the requested technique and render ordinary source geometry
            // until the user raises the scene/per-family cap. No retry spin or
            // partially covered Nanite generation is submitted to an API.
            restore(true); m_NaniteGeometry.reset(); m_NaniteStatistics.NaniteBudgetLimited=true;
            m_NaniteStatistics.NaniteRootPages=generation->Budget.RootPages; return true;
        }
        for (unsigned i=0;i<generation->Families.size();++i)
        { generation->Families[i].Capacity=generation->Budget.Capacities[i]; generation->Families[i].Stream=std::make_unique<NaniteStream>(*m_RHI); }
        generation->BudgetReady=true;
    }
    m_NaniteStatistics.NaniteRootPages=generation->Budget.RootPages;
    m_NaniteStatistics.NaniteAllocatedPages=generation->Budget.AssignedPages;
    struct PreparedFamily
    {
        NaniteStreamSnapshot Snapshot;
        std::vector<std::unique_ptr<MeshGeometryResource>> Variants;
        std::vector<unsigned> Clusters;
        bool Changed=false;
    };
    std::vector<PreparedFamily> prepared(generation->Families.size());
    unsigned remaining=settings.UploadsPerFrame;
    // Bound both startup and moving-camera production. Rotation gives unstarted
    // families a turn while existing tasks can always be polled and retired.
    constexpr unsigned MaximumProducingFamilies=4;
    unsigned producing=unsigned(std::count_if(generation->Families.begin(),generation->Families.end(),
        [](const auto& family){return family.Stream && family.Stream->Producing();}));
    bool allReady=true;
    for (std::size_t step=0;step<generation->Families.size();++step)
    {
        const auto familyIndex=(step+frame)%generation->Families.size(); auto& family=generation->Families[familyIndex]; auto& next=prepared[familyIndex];
        struct ViewHash
        {
            std::size_t operator()(const LodView& view) const
            {
                std::uint64_t hash=view.Width^(std::uint64_t(view.Height)<<32);
                for (const auto* matrix:{&view.World,&view.ViewProjection}) for (auto value:matrix->Values)
                    hash=(hash^std::hash<float>{}(value))*1099511628211ull;
                return static_cast<std::size_t>(hash);
            }
        };
        struct ViewEqual
        {
            bool operator()(const LodView& a,const LodView& b) const
            { return a.World.Values==b.World.Values && a.ViewProjection.Values==b.ViewProjection.Values && a.Width==b.Width && a.Height==b.Height; }
        };
        std::unordered_set<LodView,ViewHash,ViewEqual> seen; seen.reserve(family.Bindings.size());
        std::vector<LodView> views; views.reserve(family.Bindings.size());
        const auto constrain=[&](const Matrix4& world,const Matrix4& projection,unsigned width,unsigned height)
        {
            LodView view; view.World=world; view.ViewProjection=projection; view.Width=width; view.Height=height;
            if (seen.insert(view).second) views.push_back(view);
        };
        const auto guarded=[&](const Matrix4& world,Matrix4 projection,unsigned width,unsigned height,bool ignoreClipDepth)
        {
            // Half XY with double viewport preserves screen-error units and
            // enlarges the visibility domain beyond any subpixel jitter, even
            // at one-pixel extents. Planar demand ignores oblique depth clipping;
            // its actual jittered clip matrix is frozen later for raster/lookup.
            for (unsigned column=0;column<4;++column)
            {
                projection.Values[column]/=2; projection.Values[4+column]/=2;
                if (ignoreClipDepth) projection.Values[8+column]=projection.Values[12+column]/2;
            }
            constrain(world,projection,width*2,height*2);
        };
        for (auto bindingIndex:family.Bindings)
        {
            const auto& binding=generation->Bindings[bindingIndex];
            const auto& mesh=m_Meshes.at(binding.Id); if (!mesh.Mesh.Visible) continue; const auto& world=mesh.Mesh.WorldTransform;
            if (Techniques::UsesTemporalReconstruction(m_Techniques.PostProcessing) || Techniques::UsesPathTracing(m_Techniques.Pipeline))
                guarded(world,m_ViewProjection,m_CreateInfo.Width,m_CreateInfo.Height,false);
            else constrain(world,m_ViewProjection,m_CreateInfo.Width,m_CreateInfo.Height);
            if (m_ShadowsEnabled && m_RenderMode!=RenderMode::Wireframe && mesh.Mesh.CastShadow && !UsesPrivateTransparency(mesh))
            {
                if (m_CascadeFrame.CascadeCount) for (unsigned cascade=0;cascade<m_CascadeFrame.CascadeCount;++cascade) constrain(world,m_CascadeFrame.Cascades[cascade].LightViewProjection,m_CreateInfo.ShadowMapSize,m_CreateInfo.ShadowMapSize);
                else constrain(world,m_LightViewProjection,m_CreateInfo.ShadowMapSize,m_CreateInfo.ShadowMapSize);
            }
            if (!(mesh.Translucent && m_Techniques.AlphaBlending) && !UsesPrivateTransparency(mesh))
            {
                if (Techniques::UsesEnvironmentCapture(m_Techniques.Reflections)) for (const auto& face:m_EnvironmentFrame.Faces) constrain(world,EnvironmentRasterProjection(face,GetBackend()),m_EnvironmentFrame.Resolution,m_EnvironmentFrame.Resolution);
                if (planarDemand.Active()) guarded(world,planarDemand.ClippedViewProjection,planarDemand.Width,planarDemand.Height,true);
            }
        }
        if (views.empty()) constrain({},m_ViewProjection,m_CreateInfo.Width,m_CreateInfo.Height);
        m_NaniteStatistics.NaniteViewCount+=static_cast<unsigned>(views.size());
        NaniteStreamingStatistics statistics;
        const bool bootstrap=!family.Stream->Ready(),wasProducing=family.Stream->Producing();
        if (bootstrap)
        {
            const auto capacity=family.Capacity;
            if (!wasProducing && producing>=MaximumProducingFamilies)
            { allReady=false; m_NanitePending=true; m_NaniteStatistics.NanitePendingPages+=unsigned(family.Pages.PinnedPages.size()); continue; }
            if (!family.Stream->InitializeAsync(family.Hierarchy,family.Pages,capacity,views,settings.MaximumErrorPixels,{},statistics,m_LastError)) return reject();
        }
        else if (reuse)
        {
            if (!wasProducing && producing>=MaximumProducingFamilies) m_NanitePending=true;
            else
            {
                if (!family.Stream->UpdateAsync(family.Hierarchy,family.Pages,views,settings.MaximumErrorPixels,{},frame,remaining,statistics,m_LastError)) return reject();
                remaining-=statistics.Native.UploadedPages;
            }
        }
        const bool nowProducing=family.Stream->Producing();
        if (nowProducing && !wasProducing) ++producing;
        else if (wasProducing && !nowProducing) --producing;
        m_NaniteStatistics.NanitePendingPages+=statistics.PendingPages;
        m_NaniteStatistics.NaniteUploadedPages+=statistics.Native.UploadedPages;
        m_NaniteStatistics.NaniteReadbackBytes+=statistics.ReadbackBytes;
        if (!family.Stream->Ready()) { allReady=false; m_NanitePending=true; continue; }
        next.Snapshot=family.Stream->Snapshot(); next.Changed=!family.Published || family.Published.Geometry().Selection.Clusters!=next.Snapshot.Geometry().Selection.Clusters;
        const auto& residency=family.Stream->Residency();
        for (const auto& slot:residency.Slots()) m_NaniteStatistics.NaniteResidentPages+=slot.Page!=NaniteInvalid;
        m_NaniteStatistics.NaniteSelectedClusters+=static_cast<unsigned>(next.Snapshot.Geometry().Selection.Clusters.size());
        const auto& feedback=family.Stream->Feedback();
        m_NaniteStatistics.NaniteBudgetBlockedGroups+=feedback.BudgetBlockedGroups;
        m_NanitePending |= statistics.PageReadPending || statistics.PageCopyPending || statistics.ProducerPending || (settings.UploadsPerFrame && (family.Stream->Pending() || feedback.MissingGroups>feedback.BudgetBlockedGroups));
        if (next.Changed)
        {
            const auto& buffers=next.Snapshot.Geometry(); next.Variants.resize(family.Sources.size());
            next.Clusters=next.Snapshot.Feedback().Clusters; std::sort(next.Clusters.begin(),next.Clusters.end());
            for (unsigned part=0;part<family.Sources.size();++part)
            {
                const auto range=std::find_if(buffers.Selection.Materials.begin(),buffers.Selection.Materials.end(),[&](const auto& value){return value.Material==part;});
                if (range==buffers.Selection.Materials.end()) { Fail("Nanite cut lost a material region"); return reject(); }
                auto variant=std::make_unique<MeshGeometryResource>();
                variant->VertexBuffer=buffers.VertexBuffer; variant->IndexBuffer=buffers.IndexBuffer; variant->FirstIndex=range->FirstIndex;
                variant->NativeIndexCount=range->IndexCount; variant->HostGeometryReady=false;
                variant->OwnsVertexBuffer=variant->OwnsIndexBuffer=false;
                variant->BoundsMinimum=family.Sources[part]->BoundsMinimum; variant->BoundsMaximum=family.Sources[part]->BoundsMaximum;
                // Every decoded endpoint is within this cooked quantization
                // bound. Expand outward without reading the vertex allocation.
                const double error=family.Pages.MaximumQuantizationDisplacement;
                constexpr float FiniteLimit=std::numeric_limits<float>::max();
                const auto lower=[&](float v){return std::nextafter(std::max(-FiniteLimit,float(double(v)-error)),-FiniteLimit);};
                const auto upper=[&](float v){return std::nextafter(std::min(FiniteLimit,float(double(v)+error)),FiniteLimit);};
                const auto lo=variant->BoundsMinimum,hi=variant->BoundsMaximum;
                variant->BoundsMinimum={lower(lo.X),lower(lo.Y),lower(lo.Z)}; variant->BoundsMaximum={upper(hi.X),upper(hi.Y),upper(hi.Z)};
                constexpr std::uint64_t HashSeed=14695981039346656037ull,HashPrime=1099511628211ull;
                auto hash=(HashSeed^family.Pages.Signature)*HashPrime; hash=(hash^part)*HashPrime;
                for (auto id:next.Clusters) if (family.Hierarchy.Clusters[id].Material==part) hash=(hash^id)*HashPrime;
                variant->Hash=hash; variant->LodPrepared=true; next.Variants[part]=std::move(variant);
            }
        }
    }
    bool changed=false;
    for (unsigned index=0;index<generation->Families.size();++index) if (prepared[index].Changed)
    {
        auto& family=generation->Families[index]; auto& next=prepared[index];
        for (auto& variant:family.Variants) if (variant) DestroyGeometryResources(*variant);
        family.Variants=std::move(next.Variants); family.Published=std::move(next.Snapshot); family.SelectedClusters=std::move(next.Clusters); changed=true;
    }
    if (!allReady)
    {
        // Private ready families keep their snapshots, but none become renderer
        // geometry until the whole replacement generation can be published.
        return retainPublished();
    }
    for (const auto& binding:generation->Bindings)
    {
        auto& mesh=m_Meshes.at(binding.Id); auto& family=generation->Families[binding.Family];
        const auto part=static_cast<std::size_t>(std::find(family.Sources.begin(),family.Sources.end(),binding.Source)-family.Sources.begin()); auto* geometry=family.Variants[part].get();
        if (mesh.Geometry!=geometry) { mesh.Geometry=geometry; mesh.PreviousTransformValid=false; }
        mesh.BoundsMinimum=geometry->BoundsMinimum; mesh.BoundsMaximum=geometry->BoundsMaximum; mesh.BoundsCenter=Multiply(Add(mesh.BoundsMinimum,mesh.BoundsMaximum),0.5f); mesh.SelectedLod=mesh.PreviousLod=0;
    }
    if (!reuse) { m_NaniteGeometry=std::move(m_PendingNaniteGeometry); m_NanitePending=true; changed=true; }
    if (changed) { ++m_GeometryViewSerial; ResetFrameHistory(); }
    m_NaniteStatistics.NaniteMeshCount=static_cast<unsigned>(generation->Bindings.size()); m_NaniteStatistics.NaniteGeometryGeneration=generation->Serial;
    return true;
}

bool Renderer::PrepareNaniteHostGeometry()
{
    if (!m_NaniteGeometry) return true;
    const auto& settings=m_Techniques;
    const bool ray=settings.AmbientOcclusion.Mode==Techniques::AmbientOcclusionMode::RayTraced ||
        settings.Reflections.Mode==Techniques::ReflectionMode::RayTraced || Techniques::UsesRayTracedGI(settings.GlobalIllumination) ||
        (m_ShadowsEnabled && settings.Shadows==Techniques::ShadowRepresentation::RayTraced);
    const bool all=m_RenderMode!=RenderMode::Solid || ray || Techniques::UsesRefraction(settings);
    std::unordered_set<const MeshGeometryResource*> outlined;
    if (!all) for (const auto& entry:m_Meshes)
        if (entry.second.Mesh.Visible && entry.second.Outline.Enabled) outlined.insert(entry.second.SourceGeometry);
    if (!all && outlined.empty()) return true;
    for (auto& family:m_NaniteGeometry->Families) for (unsigned part=0;part<family.Variants.size();++part)
    {
        auto& geometry=*family.Variants[part];
        const bool needed=all || outlined.count(family.Sources[part])!=0;
        if (!needed || geometry.HostGeometryReady) continue;
        // CPU-only consumers decode exactly the GPU-selected clusters from the
        // immutable page source. Ordinary raster/capture/shadow/velocity paths
        // use native buffers and counts and never reconstruct this host mesh.
        std::vector<Vertex> vertices; std::vector<unsigned> indices;
        for (auto id:family.Published.Geometry().Selection.Clusters)
        {
            if (family.Hierarchy.Clusters[id].Material!=part) continue;
            NaniteDecodedCluster cluster; if (!DecodeNaniteCluster(family.Pages,id,cluster,m_LastError)) return false;
            const auto base=static_cast<unsigned>(vertices.size()); vertices.insert(vertices.end(),cluster.Vertices.begin(),cluster.Vertices.end());
            for (auto index:cluster.Indices) indices.push_back(base+index);
        }
        if (indices.size()!=geometry.NativeIndexCount) return Fail("Nanite host materialization disagrees with the GPU-selected material range");
        geometry.Hash=HashMeshGeometry(vertices,indices); geometry.Vertices=std::move(vertices); geometry.Indices=std::move(indices);
        geometry.LodChain.Levels.push_back({geometry.Indices,0}); geometry.HostGeometryReady=true;
        m_NaniteStatistics.NaniteHostRebuiltVertices+=static_cast<unsigned>(geometry.Vertices.size());
    }
    return true;
}

// Two-pass culling uses the current resident cut in both camera transforms.
// Quantization and affine roundoff expand bounds before the GPU HZB test.
bool Renderer::PrepareNaniteFrameVisibility(VisibilityBufferDesc& frame)
{
    if (!frame.NaniteRaster || !m_Techniques.Nanite.Occlusion || frame.Draws.empty()) return true;
    std::vector<NaniteVisibilityQuery> queries;
    for (const auto& draw:frame.Draws)
    {
        const auto& mesh=*m_VisibilitySources[draw.Range].Mesh; std::size_t part=0;
        auto* family=NaniteFamilyFor(mesh,part);
        if (!family || !family->Published) return Fail("Nanite visibility lost its resident family");
        NaniteGpuSelection selection;
        for (auto id:family->Published.Geometry().Selection.Clusters) if (family->Hierarchy.Clusters[id].Material==part) selection.Clusters.push_back(id);
        selection.ActiveClusters=unsigned(selection.Clusters.size());
        std::vector<NaniteVisibilityQuery> bounds;
        if (!BuildNaniteVisibilityQueries(family->Hierarchy,family->Pages,selection,{}, {},false,draw.Range,bounds,m_LastError)) return false;
        const auto& range=frame.Scene->Draws[draw.Range]; unsigned first=range.FirstTriangle;
        for (const auto& bound:bounds)
        {
            NaniteVisibilityQuery query; const auto cluster=bound.Identity.Z;
            if (!BuildNaniteDrawQuery({bound.Minimum.X,bound.Minimum.Y,bound.Minimum.Z},{bound.Maximum.X,bound.Maximum.Y,bound.Maximum.Z},
                mesh.Mesh.WorldTransform,mesh.PreviousWorldTransform,frame.ViewProjection,m_NaniteHistoryProjection,
                m_NaniteDepthHistoryValid && mesh.PreviousTransformValid,cluster,draw.Range,query,m_LastError)) return false;
            const auto count=unsigned(family->Hierarchy.Clusters[cluster].Indices.size()/VisibilityTriangleCorners);
            frame.NaniteClusters.push_back({draw.Range,first,count}); queries.push_back(query); first+=count;
        }
        if (first!=range.FirstTriangle+range.TriangleCount) return Fail("Nanite cluster metadata differs from its decoded material range");
    }
    if (!m_RHI->BeginNaniteVisibility(m_NaniteDepthHistoryValid?m_NaniteDepthHistory:NaniteDepthHandle{},queries,m_NaniteMainVisibility)) return Fail(m_RHI->GetLastError().c_str());
    frame.NaniteVisibility=m_NaniteMainVisibility; return true;
}
void Renderer::CommitNaniteFrameVisibility()
{
    if (m_NaniteMainVisibility) m_RHI->DestroyNaniteVisibility(m_NaniteMainVisibility);
    if (m_NaniteDepthHistory) m_RHI->DestroyNaniteDepth(m_NaniteDepthHistory);
    m_NaniteMainVisibility={}; m_NaniteDepthHistory=m_NaniteDepthCandidate; m_NaniteDepthCandidate={};
    m_NaniteDepthHistoryValid=bool(m_NaniteDepthHistory); m_NaniteHistoryProjection=m_RasterViewProjection;
}
void Renderer::ReleaseNaniteFrameVisibility()
{
    if (m_RHI && m_RHI->IsInitialized() && m_NaniteDepthHistory) m_RHI->DestroyNaniteDepth(m_NaniteDepthHistory);
    m_NaniteDepthHistory={}; m_NaniteDepthCandidate={}; m_NaniteMainVisibility={}; m_NaniteDepthHistoryValid=false;
}
