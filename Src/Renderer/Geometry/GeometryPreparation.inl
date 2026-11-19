// A generation owns derived LOD allocations and native vertex/index streams
// shared by all material sections in a geometric family. Original cache references
// remain resident for restoration and to prevent source identities being recycled.
Renderer::GeometryGeneration::~GeometryGeneration()
{
    for (auto& family:Families)
    {
        for (auto& variant:family.Variants) if (variant) Owner->DestroyGeometryResources(*variant);
        if (family.VertexBuffer && Owner->m_RHI) Owner->m_RHI->DestroyBuffer(family.VertexBuffer);
        if (family.IndexBuffer && Owner->m_RHI) Owner->m_RHI->DestroyBuffer(family.IndexBuffer);
    }
    for (auto*& source:SourceReferences) Owner->ReleaseGeometry(source);
}

bool Renderer::PrepareGeometry()
{
    const auto publish=[&](GeometryGeneration* generation)
    {
        std::vector<std::pair<MeshResource*,MeshGeometryResource*>> targets;
        targets.reserve(m_Meshes.size());
        for (auto& item:m_Meshes)
        {
            auto& mesh=item.second; MeshGeometryResource* geometry=mesh.BaseGeometry;
            if (generation)
            {
                const auto binding=std::lower_bound(generation->Bindings.begin(),generation->Bindings.end(),item.first,
                    [](const GeometryBinding& value,const std::string& id){return value.Id<id;});
                if (binding==generation->Bindings.end() || binding->Id!=item.first) return false;
                auto& family=generation->Families[binding->Family];
                const auto source=std::find(family.Sources.begin(),family.Sources.end(),mesh.BaseGeometry);
                if (source==family.Sources.end()) return false;
                geometry=family.Variants[static_cast<std::size_t>(source-family.Sources.begin())].get();
            }
            targets.push_back({&mesh,geometry});
        }
        bool changed=false;
        for (const auto& target:targets)
        {
            auto& mesh=*target.first; auto* geometry=target.second;
            if (mesh.SourceGeometry==geometry) continue;
            mesh.SourceGeometry=mesh.Geometry=geometry; mesh.BoundsMinimum=geometry->BoundsMinimum; mesh.BoundsMaximum=geometry->BoundsMaximum;
            mesh.BoundsCenter=Multiply(Add(mesh.BoundsMinimum,mesh.BoundsMaximum),0.5f);
            mesh.SelectedLod=mesh.PreviousLod=0; mesh.PreviousTransformValid=false; changed=true;
        }
        if (changed) { ++m_ProbeSceneRevision; ResetFrameHistory(); }
        return true;
    };
    const bool displaces=Techniques::UsesDisplacementMapping(m_Techniques.Displacement),tessellates=Techniques::UsesTessellation(m_Techniques.Tessellation);
    if (!displaces && !tessellates)
    {
        if (!m_PreparedGeometry) return true;
        if (!publish(nullptr)) return Fail("Cannot restore original geometry sources");
        m_PreparedGeometry.reset(); return true;
    }
    if (displaces && !m_RHI->GetCapabilities().GeometryDisplacement) return Fail("Native displacement is unavailable on the active device");
    if (tessellates && !m_RHI->GetCapabilities().GeometryTessellation) return Fail("Native tessellation is unavailable on the active device");

    std::vector<MeshResource*> ordered;
    for (auto& item:m_Meshes) ordered.push_back(&item.second);
    std::sort(ordered.begin(),ordered.end(),[](const MeshResource* a,const MeshResource* b){return a->Mesh.Id<b->Mesh.Id;});
    struct Object
    {
        std::uint64_t Selection=0;
        Matrix4 World;
        MeshGeometryResource* Standalone=nullptr;
        std::vector<MeshResource*> Members;
    };
    std::vector<Object> objects;
    for (auto* mesh:ordered)
    {
        auto found=std::find_if(objects.begin(),objects.end(),[&](const Object& object)
        {
            return mesh->Mesh.SelectionGroup?object.Selection==mesh->Mesh.SelectionGroup && object.World.Values==mesh->Mesh.WorldTransform.Values:
                object.Selection==0 && object.Standalone==mesh->BaseGeometry;
        });
        if (found==objects.end())
        {
            Object object; object.Selection=mesh->Mesh.SelectionGroup; object.World=mesh->Mesh.WorldTransform;
            object.Standalone=mesh->BaseGeometry; objects.push_back(std::move(object)); found=std::prev(objects.end());
        }
        found->Members.push_back(mesh);
    }
    auto candidate=std::make_unique<GeometryGeneration>(this);
    candidate->Displacement=displaces?m_Techniques.Displacement:Techniques::DisplacementMappingSettings{};
    candidate->Tessellation=tessellates?m_Techniques.Tessellation:Techniques::TessellationSettings{};
    // PN owns density when both techniques are active. Retain the requested
    // displacement sampling in the settings, but exclude that inactive knob
    // from the generation identity so it cannot trigger redundant GPU work.
    if (tessellates && displaces) candidate->Displacement.Segments=1;
    for (const auto& object:objects)
    {
        std::vector<MeshGeometryResource*> sources;
        for (auto* member:object.Members) sources.push_back(member->BaseGeometry);
        std::sort(sources.begin(),sources.end(),std::less<MeshGeometryResource*>{}); sources.erase(std::unique(sources.begin(),sources.end()),sources.end());
        const auto found=std::find_if(candidate->Families.begin(),candidate->Families.end(),[&](const GeometryFamily& family){return family.Sources==sources;});
        const std::size_t index=static_cast<std::size_t>(found-candidate->Families.begin());
        if (found==candidate->Families.end()) { GeometryFamily family; family.Sources=std::move(sources); candidate->Families.push_back(std::move(family)); }
        for (auto* member:object.Members) candidate->Bindings.push_back({member->Mesh.Id,member->BaseGeometry,index});
    }
    std::sort(candidate->Bindings.begin(),candidate->Bindings.end(),[](const GeometryBinding& a,const GeometryBinding& b){return a.Id<b.Id;});
    if (m_PreparedGeometry && m_PreparedGeometry->Displacement==candidate->Displacement && m_PreparedGeometry->Tessellation==candidate->Tessellation && m_PreparedGeometry->Bindings==candidate->Bindings &&
        m_PreparedGeometry->Families.size()==candidate->Families.size())
    {
        bool same=true;
        for (std::size_t family=0;family<candidate->Families.size();++family) same &= m_PreparedGeometry->Families[family].Sources==candidate->Families[family].Sources;
        if (same) return publish(m_PreparedGeometry.get()); // Rigid whole-object motion does not rebuild local vertices.
    }
    std::set<MeshGeometryResource*,std::less<MeshGeometryResource*>> originals;
    for (const auto& family:candidate->Families) originals.insert(family.Sources.begin(),family.Sources.end());
    candidate->SourceReferences.reserve(originals.size());
    for (auto* source:originals) { candidate->SourceReferences.push_back(source); ++source->ReferenceCount; }

    const GeometryPreparationLimits budget;
    for (auto& family:candidate->Families)
    {
        // The adapter compacted vertices separately for material sections. Merge
        // their source context before subdivision/welding, then preserve a dense
        // triangle range for each original material. No API shader is generated.
        std::vector<Vertex> vertices; std::vector<std::uint32_t> indices; std::vector<std::size_t> begins;
        for (auto* source:family.Sources)
        {
            if (source->Vertices.size()>budget.MaximumVertices-vertices.size() || source->Indices.size()>budget.MaximumIndices-indices.size())
                return Fail("Object source exceeds the geometry preparation budget");
            const auto offset=static_cast<std::uint32_t>(vertices.size()); begins.push_back(indices.size());
            vertices.insert(vertices.end(),source->Vertices.begin(),source->Vertices.end());
            for (auto index:source->Indices) indices.push_back(index+offset);
        }
        // PN produces topology first. Height displacement consumes that exact
        // readback and retains the native PN index allocation. Displacement on
        // its own keeps its separate uniform CPU sampling prerequisite.
        const unsigned segments=tessellates?candidate->Tessellation.Segments:candidate->Displacement.Segments;
        const std::size_t growth=std::size_t(segments)*segments;
        if (tessellates)
        {
            PNTessellationGeometry prepared; TessellatedGeometryBuffers native;
            if (!PreparePNTessellation(vertices,indices,budget,prepared,m_LastError)) return false;
            if (!m_RHI->CreateTessellatedGeometry(prepared,candidate->Tessellation,native)) return Fail(m_RHI->GetLastError().c_str());
            family.VertexBuffer=native.VertexBuffer; family.IndexBuffer=native.IndexBuffer;
            candidate->TessellationDispatches+=native.DispatchCount;
            vertices=std::move(native.Vertices); indices=std::move(native.Indices);
        }
        else
        {
            SubdividedTriangles refined;
            if (!SubdivideTriangles(vertices,indices,segments,budget,refined,m_LastError)) return false;
            vertices=std::move(refined.Vertices); indices=std::move(refined.Indices);
        }
        if (displaces)
        {
            DisplacementGeometry prepared; DisplacedVertexBuffer native;
            if (!PrepareDisplacementGeometry(vertices,indices,budget,prepared,m_LastError)) return false;
            if (!m_RHI->CreateDisplacedVertexBuffer(prepared,candidate->Displacement,native)) return Fail(m_RHI->GetLastError().c_str());
            const auto previous=family.VertexBuffer; family.VertexBuffer=native.Buffer;
            if (previous) m_RHI->DestroyBuffer(previous);
            candidate->DisplacementDispatches+=native.DispatchCount; vertices=std::move(native.Vertices);
        }
        if (!family.IndexBuffer)
        {
            family.IndexBuffer=m_RHI->CreateBuffer({BufferUsage::Index,indices.size()*sizeof(std::uint32_t),false,"Prepared material topology"},indices.data());
            if (!family.IndexBuffer) return Fail(m_RHI->GetLastError().c_str());
        }
        if (vertices.size()>std::numeric_limits<std::uint32_t>::max()-candidate->VertexCount) return Fail("Prepared vertex statistics exceed their native range");
        candidate->VertexCount+=static_cast<std::uint32_t>(vertices.size()); family.Variants.resize(family.Sources.size());
        for (std::size_t part=0;part<family.Sources.size();++part)
        {
            auto variant=std::make_unique<MeshGeometryResource>(); variant->Vertices=vertices;
            const auto first=indices.begin()+begins[part]*growth;
            variant->Indices.assign(first,first+family.Sources[part]->Indices.size()*growth);
            variant->VertexBuffer=family.VertexBuffer; variant->OwnsVertexBuffer=false;
            variant->IndexBuffer=family.IndexBuffer; variant->OwnsIndexBuffer=false; variant->FirstIndex=static_cast<std::uint32_t>(begins[part]*growth);
            variant->BoundsMinimum=variant->BoundsMaximum=variant->Vertices[variant->Indices.front()].Position;
            for (auto index:variant->Indices)
            {
                const auto p=variant->Vertices[index].Position; auto& lo=variant->BoundsMinimum; auto& hi=variant->BoundsMaximum;
                lo={std::min(lo.X,p.X),std::min(lo.Y,p.Y),std::min(lo.Z,p.Z)}; hi={std::max(hi.X,p.X),std::max(hi.Y,p.Y),std::max(hi.Z,p.Z)};
            }
            variant->Hash=HashMeshGeometry(variant->Vertices,variant->Indices);

            family.Variants[part]=std::move(variant);
        }
    }
    candidate->Serial=++m_GeometrySerial;
    if (!publish(candidate.get())) return Fail("Geometry publication lost its source binding");
    m_PreparedGeometry=std::move(candidate); return true;
}
