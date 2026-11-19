// Source families are immutable; view cuts own separate native version leases.
// Material slices compact CPU vertex IDs independently, while native draws keep
// their original first-index range in the shared decoded vertex/index buffers.
Renderer::VirtualGeometryGeneration::~VirtualGeometryGeneration()
{
    for (auto& family:Families) for (auto& variant:family.Variants) if (variant) Owner->DestroyGeometryResources(*variant);
    for (auto*& source:SourceReferences) Owner->ReleaseGeometry(source);
}

Renderer::VirtualGeometryFamily* Renderer::VirtualFamilyFor(const MeshResource& mesh,std::size_t& part)
{
    if (!m_VirtualGeometry) return nullptr;
    const auto& bindings=m_VirtualGeometry->Bindings;
    const auto found=std::lower_bound(bindings.begin(),bindings.end(),mesh.Mesh.Id,[](const auto& value,const auto& id){return value.Id<id;});
    if (found==bindings.end() || found->Id!=mesh.Mesh.Id) return nullptr;
    auto& family=m_VirtualGeometry->Families[found->Family];
    const auto source=std::find(family.Sources.begin(),family.Sources.end(),found->Source);
    if (source==family.Sources.end()) return nullptr;
    part=static_cast<std::size_t>(source-family.Sources.begin()); return &family;
}

bool Renderer::PrepareVirtualLightmapSources(VirtualGeometryFamily& family)
{
    if (!family.LightmapSources.empty()) return true;
    std::vector<std::shared_ptr<const LightmapSourceGeometry>> snapshots;
    for (unsigned part=0;part<family.Sources.size();++part)
    {
        auto source=std::make_shared<LightmapSourceGeometry>(); source->Levels.resize(2); source->Segments.resize(2);
        std::vector<unsigned> remap(family.SourceVertices.size(),VirtualGeometryInvalid);
        const auto append=[&](unsigned vertex)
        {
            auto& mapped=remap[vertex];
            if (mapped==VirtualGeometryInvalid) { mapped=static_cast<unsigned>(source->Vertices.size()); source->Vertices.push_back(family.SourceVertices[vertex]); }
            return mapped;
        };
        for (auto index:family.SourceIndices[part]) source->Levels[0].push_back(append(index));
        // All cluster representations get persistent receiver UVs in one level.
        // SourceTriangleIndex preserves the partition after atlas seam splitting;
        // only selected tags are drawn. Transport still sees level zero alone.
        for (unsigned id=0;id<family.Hierarchy.Data.Clusters.size();++id)
        {
            if (family.Hierarchy.Data.Clusters[id].Material!=part) continue;
            VirtualGeometryDecodedCluster cluster;
            if (!DecodeVirtualGeometryCluster(family.Hierarchy.Data,id,cluster,m_LastError)) return false;
            auto& indices=source->Levels[1];
            source->Segments[1].push_back({static_cast<unsigned>(indices.size()/3),static_cast<unsigned>(cluster.Indices.size()/3),id});
            for (auto index:cluster.Indices) indices.push_back(append(cluster.SourceVertices[index]));
        }
        snapshots.push_back(std::move(source));
    }
    family.LightmapSources=std::move(snapshots); return true;
}

bool Renderer::PrepareVirtualGeometry()
{
    m_VirtualStatistics={}; m_VirtualPending=false;
    const auto restore=[&](bool source)
    {
        bool changed=false;
        for (auto& entry:m_Meshes)
        {
            auto& mesh=entry.second; auto* geometry=mesh.SourceGeometry;
            if (!source)
            {
                std::size_t part=0; auto* family=VirtualFamilyFor(mesh,part);
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
    if (!m_Techniques.VirtualGeometry.Enabled)
    {
        if (m_VirtualGeometry) { restore(true); m_VirtualGeometry.reset(); }
        return true;
    }
    const auto reject=[&](){restore(false); return false;};
    if (!m_RHI->GetCapabilities().GeometryVirtualization) { Fail("Native virtual geometry is unavailable"); return reject(); }
    PlanarCaptureFrame planarDemand;
    if (Techniques::UsesPlanarCapture(m_Techniques.Reflections) && !BuildPlanarCaptureFrame(m_ViewProjection,m_RenderLighting.CameraPosition,
            m_RenderWidth,m_RenderHeight,m_Techniques.Reflections.Planar,planarDemand,m_LastError)) return reject();
    const auto& settings=m_Techniques.VirtualGeometry;
    auto candidate=std::make_unique<VirtualGeometryGeneration>(this);
    candidate->PageBudget=settings.PhysicalPages; candidate->SourceSerial=m_PreparedGeometry?m_PreparedGeometry->Serial:0;
    std::vector<MeshResource*> ordered; for (auto& entry:m_Meshes) ordered.push_back(&entry.second);
    std::sort(ordered.begin(),ordered.end(),[](auto* a,auto* b){return a->Mesh.Id<b->Mesh.Id;});
    struct Object { std::uint64_t Selection; Matrix4 World; MeshGeometryResource* Standalone; std::vector<MeshResource*> Members; };
    std::vector<Object> objects;
    for (auto* mesh:ordered)
    {
        auto object=std::find_if(objects.begin(),objects.end(),[&](const auto& value)
        { return mesh->Mesh.SelectionGroup?value.Selection==mesh->Mesh.SelectionGroup && value.World.Values==mesh->Mesh.WorldTransform.Values:value.Selection==0 && value.Standalone==mesh->SourceGeometry; });
        if (object==objects.end()) { objects.push_back({mesh->Mesh.SelectionGroup,mesh->Mesh.WorldTransform,mesh->SourceGeometry,{}}); object=std::prev(objects.end()); }
        object->Members.push_back(mesh);
    }
    for (const auto& object:objects)
    {
        std::vector<MeshGeometryResource*> sources; for (auto* member:object.Members) sources.push_back(member->SourceGeometry);
        std::sort(sources.begin(),sources.end(),std::less<MeshGeometryResource*>{}); sources.erase(std::unique(sources.begin(),sources.end()),sources.end());
        const auto found=std::find_if(candidate->Families.begin(),candidate->Families.end(),[&](const auto& family){return family.Sources==sources;});
        const auto family=static_cast<std::size_t>(found-candidate->Families.begin());
        if (found==candidate->Families.end()) { VirtualGeometryFamily value; value.Sources=std::move(sources); candidate->Families.push_back(std::move(value)); }
        for (auto* member:object.Members) candidate->Bindings.push_back({member->Mesh.Id,member->SourceGeometry,family});
    }
    std::sort(candidate->Bindings.begin(),candidate->Bindings.end(),[](const auto& a,const auto& b){return a.Id<b.Id;});
    bool reuse=m_VirtualGeometry && candidate->PageBudget==m_VirtualGeometry->PageBudget && candidate->SourceSerial==m_VirtualGeometry->SourceSerial && candidate->Bindings==m_VirtualGeometry->Bindings;
    VirtualGeometryGeneration* generation=reuse?m_VirtualGeometry.get():candidate.get();
    if (!reuse)
    {
        std::set<MeshGeometryResource*,std::less<MeshGeometryResource*>> originals;
        for (auto* mesh:ordered) originals.insert(mesh->BaseGeometry);
        for (auto* source:originals) { ++source->ReferenceCount; generation->SourceReferences.push_back(source); }
        const GeometryPreparationLimits budget;
        for (auto& family:generation->Families)
        {
            std::map<std::array<std::uint32_t,VirtualGeometryVertexWords>,std::uint32_t> unique;
            std::vector<unsigned> indices,materials;
            for (unsigned part=0;part<family.Sources.size();++part)
            {
                const auto& source=*family.Sources[part]; std::vector<unsigned> slice;
                if (source.Indices.size()>budget.MaximumIndices-indices.size()) { Fail("Virtual geometry family exceeds its index budget"); return reject(); }
                for (auto index:source.Indices)
                {
                    std::array<std::uint32_t,VirtualGeometryVertexWords> key; std::memcpy(key.data(),&source.Vertices[index],sizeof(Vertex));
                    auto found=unique.find(key);
                    if (found==unique.end())
                    {
                        if (family.SourceVertices.size()>=budget.MaximumVertices) { Fail("Virtual geometry family exceeds its vertex budget"); return reject(); }
                        const auto id=static_cast<unsigned>(family.SourceVertices.size()); family.SourceVertices.push_back(source.Vertices[index]); found=unique.emplace(key,id).first;
                    }
                    slice.push_back(found->second);
                }
                indices.insert(indices.end(),slice.begin(),slice.end()); materials.insert(materials.end(),slice.size()/3,part); family.SourceIndices.push_back(std::move(slice));
            }
            if (!BuildVirtualGeometryHierarchy(family.SourceVertices,indices,materials,{},family.Hierarchy,m_LastError)) return reject();
            family.Stream=std::make_unique<VirtualGeometryStream>(*m_RHI);
        }
        generation->Serial=++m_VirtualSerial;
    }
    struct PreparedFamily
    {
        VirtualGeometryStreamSnapshot Snapshot;
        std::vector<std::unique_ptr<MeshGeometryResource>> Variants;
        std::vector<unsigned> Clusters;
        bool Changed=false;
    };
    std::vector<PreparedFamily> prepared(generation->Families.size());
    unsigned remaining=settings.UploadsPerFrame; const auto frame=++m_VirtualFrame;
    for (std::size_t step=0;step<generation->Families.size();++step)
    {
        const auto familyIndex=(step+frame)%generation->Families.size(); auto& family=generation->Families[familyIndex]; auto& next=prepared[familyIndex];
        std::vector<LodView> views;
        const auto constrain=[&](const Matrix4& world,const Matrix4& projection,unsigned width,unsigned height)
        {
            const auto duplicate=std::any_of(views.begin(),views.end(),[&](const auto& view){return view.World.Values==world.Values && view.ViewProjection.Values==projection.Values && view.Width==width && view.Height==height;});
            if (duplicate) return;
            LodView view; view.World=world; view.ViewProjection=projection; view.Width=width; view.Height=height; views.push_back(view);
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
        for (const auto& binding:generation->Bindings) if (binding.Family==familyIndex)
        {
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
        VirtualGeometryStreamingStatistics statistics; const VirtualGeometryCutSettings quality{settings.MaximumErrorPixels,settings.HysteresisFraction};
        const bool bootstrap=!family.Stream->Ready();
        if (bootstrap)
        {
            const auto pages=static_cast<unsigned>(family.Hierarchy.Data.Pages.size()); const auto capacity=settings.PhysicalPages?std::min(settings.PhysicalPages,pages):pages;
            if (!family.Stream->Initialize(family.Hierarchy,capacity,views,quality,{},statistics,m_LastError)) return reject();
            m_VirtualStatistics.VirtualBootstrapPages+=statistics.Native.UploadedPages;
        }
        else
        {
            if (!family.Stream->Update(family.Hierarchy,views,quality,{},frame,remaining,statistics,m_LastError)) return reject();
            remaining-=statistics.Native.UploadedPages;
        }
        next.Snapshot=family.Stream->Snapshot(); next.Changed=!family.Published || family.Published.Geometry().Selection.Nodes!=next.Snapshot.Geometry().Selection.Nodes;
        const auto& residency=family.Stream->Residency();
        for (const auto& slot:residency.Slots()) m_VirtualStatistics.VirtualResidentPages+=slot.Page!=VirtualGeometryInvalid;
        m_VirtualStatistics.VirtualPoolBytes+=std::uint64_t(residency.Slots().size())*VirtualGeometryPageBytes;
        m_VirtualStatistics.VirtualPendingPages+=statistics.PendingPages;
        m_VirtualStatistics.VirtualUploadedPages+=statistics.Native.UploadedPages; m_VirtualStatistics.VirtualCopiedPages+=statistics.Native.CopiedPages;
        m_VirtualStatistics.VirtualUploadedBytes+=statistics.Native.UploadedBytes; m_VirtualStatistics.VirtualCopiedBytes+=statistics.Native.CopiedBytes;
        m_VirtualStatistics.VirtualNodeCount+=static_cast<unsigned>(next.Snapshot.Geometry().Selection.Nodes.size());
        const auto& feedback=family.Stream->Feedback();
        m_VirtualPending |= settings.UploadsPerFrame && (family.Stream->Pending() || feedback.MissingDetailNodes>feedback.BudgetBlockedNodes);
        if (next.Changed)
        {
            const auto& buffers=next.Snapshot.Geometry(); next.Variants.resize(family.Sources.size());
            next.Clusters=next.Snapshot.Feedback().Clusters; std::sort(next.Clusters.begin(),next.Clusters.end());
            for (unsigned part=0;part<family.Sources.size();++part)
            {
                const auto range=std::find_if(buffers.Selection.Materials.begin(),buffers.Selection.Materials.end(),[&](const auto& value){return value.Material==part;});
                if (range==buffers.Selection.Materials.end()) { Fail("Virtual cut lost a material region"); return reject(); }
                auto variant=std::make_unique<MeshGeometryResource>(); std::vector<unsigned> remap(buffers.Readback.Vertices.size(),VirtualGeometryInvalid);
                for (unsigned at=range->FirstIndex;at<range->FirstIndex+range->IndexCount;++at)
                {
                    const auto index=buffers.Readback.Indices[at]; auto& mapped=remap[index];
                    if (mapped==VirtualGeometryInvalid) { mapped=static_cast<unsigned>(variant->Vertices.size()); variant->Vertices.push_back(buffers.Readback.Vertices[index]); }
                    variant->Indices.push_back(mapped);
                }
                variant->VertexBuffer=buffers.VertexBuffer; variant->IndexBuffer=buffers.IndexBuffer; variant->FirstIndex=range->FirstIndex;
                variant->OwnsVertexBuffer=variant->OwnsIndexBuffer=false;
                variant->BoundsMinimum=family.Sources[part]->BoundsMinimum; variant->BoundsMaximum=family.Sources[part]->BoundsMaximum;
                variant->Hash=HashMeshGeometry(variant->Vertices,variant->Indices); next.Variants[part]=std::move(variant);
                auto& published=*next.Variants[part]; published.LodChain.Levels.push_back({published.Indices,0}); published.LodPrepared=true;
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
    for (const auto& binding:generation->Bindings)
    {
        auto& mesh=m_Meshes.at(binding.Id); auto& family=generation->Families[binding.Family];
        const auto part=static_cast<std::size_t>(std::find(family.Sources.begin(),family.Sources.end(),binding.Source)-family.Sources.begin()); auto* geometry=family.Variants[part].get();
        if (mesh.Geometry!=geometry) { mesh.Geometry=geometry; mesh.PreviousTransformValid=false; }
        mesh.BoundsMinimum=geometry->BoundsMinimum; mesh.BoundsMaximum=geometry->BoundsMaximum; mesh.BoundsCenter=Multiply(Add(mesh.BoundsMinimum,mesh.BoundsMaximum),0.5f); mesh.SelectedLod=mesh.PreviousLod=0;
    }
    if (!reuse) m_VirtualGeometry=std::move(candidate);
    if (changed) { ++m_GeometryViewSerial; ResetFrameHistory(); }
    m_VirtualStatistics.VirtualMeshCount=static_cast<unsigned>(generation->Bindings.size()); m_VirtualStatistics.VirtualGeometryGeneration=generation->Serial;
    return true;
}
