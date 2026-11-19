struct LumenRadianceRecord
{
    std::shared_ptr<LumenDistanceRecord> TraceDistance;
    std::shared_ptr<LumenRadianceRecord> WorldCache;
    LumenRadianceInfo Info; std::shared_ptr<LumenLightingRecord> Lighting;
    ComPtr<ID3D11Buffer> Buffer,Origins; ComPtr<ID3D11ShaderResourceView> View,OriginView;
};
std::unordered_map<std::uint64_t,std::shared_ptr<LumenRadianceRecord>> LumenRadianceTraces;
LumenFrameTransaction FrameLumen;
ComPtr<ID3D11ComputeShader> LumenRadianceTrace,LumenRadianceResolve,LumenProbeTrace,LumenScreenPlacement,LumenHybridTrace,LumenScreenDepth,LumenReflectionRayShader;
ComPtr<ID3D11ComputeShader> LumenWorldLookup;
bool EnsureLumenWorldLookup()
{
    if (LumenWorldLookup) return true;
    if (!LoadTechniqueShaderSource("DX11","LumenWorldCache.hlsl",ShaderCode,LastError)) return false;
    ComPtr<ID3DBlob> binary; if (!CompileShader("CSWorldLookup","cs_5_0",binary,nullptr,true)) return false;
    const auto status=Device->CreateComputeShader(binary->GetBufferPointer(),binary->GetBufferSize(),nullptr,LumenWorldLookup.GetAddressOf());
    return SUCCEEDED(status) || Fail(HResultMessage("Create Lumen world lookup",status));
}
void DestroyLumenRadianceTraces() { LumenWorldLookup.Reset(); LumenRadianceTraces.clear(); LumenRadianceTrace.Reset(); LumenRadianceResolve.Reset(); LumenProbeTrace.Reset(); LumenScreenPlacement.Reset(); LumenHybridTrace.Reset(); LumenScreenDepth.Reset(); LumenReflectionRayShader.Reset(); }
bool EnsureLumenRadianceTrace()
{
    if (LumenRadianceTrace && LumenRadianceResolve && LumenProbeTrace && LumenScreenPlacement && LumenHybridTrace && LumenScreenDepth && LumenReflectionRayShader) return true;
    if (!LoadTechniqueShaderSource("DX11","Lumen.hlsl",ShaderCode,LastError)) return false;
    const char* stages[]{"EVANS_LUMEN_RADIANCE_TRACE_STAGE","EVANS_LUMEN_RADIANCE_RESOLVE_STAGE","EVANS_LUMEN_PROBE_TRACE_STAGE","EVANS_LUMEN_SCREEN_PROBES_STAGE","EVANS_LUMEN_SCREEN_TRACE_STAGE","EVANS_LUMEN_SCREEN_DEPTH_STAGE","EVANS_LUMEN_REFLECTION_RAYS_STAGE"}; const char* entries[]{"CSTraceDistance","CSResolveRadiance","CSTraceDistance","CSPlaceScreenProbes","CSTraceScreen","CSBuildScreenDepth","CSReflectionRays"};
    ID3D11ComputeShader** targets[]{LumenRadianceTrace.ReleaseAndGetAddressOf(),LumenRadianceResolve.ReleaseAndGetAddressOf(),LumenProbeTrace.ReleaseAndGetAddressOf(),LumenScreenPlacement.ReleaseAndGetAddressOf(),LumenHybridTrace.ReleaseAndGetAddressOf(),LumenScreenDepth.ReleaseAndGetAddressOf(),LumenReflectionRayShader.ReleaseAndGetAddressOf()};
    for (unsigned stage=0;stage<std::size(stages);++stage)
    {
        const D3D_SHADER_MACRO macros[]{{stages[stage],"1"},{nullptr,nullptr}}; ComPtr<ID3DBlob> binary;
        if (!CompileShader(entries[stage],"cs_5_0",binary,macros,true)) return false;
        const auto status=Device->CreateComputeShader(binary->GetBufferPointer(),binary->GetBufferSize(),nullptr,targets[stage]);
        if (FAILED(status)) return Fail(HResultMessage("Create Lumen radiance stage",status));
    }
    return true;
}
bool CreateLumenRadianceBuffer(std::size_t bytes,unsigned stride,const void* source,ComPtr<ID3D11Buffer>& buffer,
    ComPtr<ID3D11ShaderResourceView>& view,ComPtr<ID3D11UnorderedAccessView>* target=nullptr)
{
    D3D11_BUFFER_DESC desc{}; desc.ByteWidth=UINT(bytes); desc.Usage=source?D3D11_USAGE_IMMUTABLE:D3D11_USAGE_DEFAULT;
    desc.BindFlags=D3D11_BIND_SHADER_RESOURCE|(target?D3D11_BIND_UNORDERED_ACCESS:0); desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; desc.StructureByteStride=stride;
    D3D11_SUBRESOURCE_DATA initial{}; initial.pSysMem=source; auto status=Device->CreateBuffer(&desc,source?&initial:nullptr,buffer.GetAddressOf());
    if (FAILED(status)) return Fail(HResultMessage("Create Lumen radiance buffer",status));
    D3D11_SHADER_RESOURCE_VIEW_DESC srv{}; srv.ViewDimension=D3D11_SRV_DIMENSION_BUFFER; srv.Buffer.NumElements=UINT(bytes/stride);
    status=Device->CreateShaderResourceView(buffer.Get(),&srv,view.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("View Lumen radiance buffer",status));
    if (target)
    {
        D3D11_UNORDERED_ACCESS_VIEW_DESC uav{}; uav.ViewDimension=D3D11_UAV_DIMENSION_BUFFER; uav.Buffer.NumElements=UINT(bytes/stride);
        status=Device->CreateUnorderedAccessView(buffer.Get(),&uav,target->GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Write Lumen radiance buffer",status));
    }
    return true;
}
bool ReadLumenOriginRecord(const LumenRadianceRecord& record,std::vector<LumenProbeOrigin>& output)
{
    if (!LumenRadianceProbeCount(record.Info)) return Fail("This Lumen trace contains explicit reflection rays, not probe origins");
    const auto bytes=std::size_t(LumenRadianceProbeCount(record.Info))*sizeof(LumenProbeOrigin); D3D11_BUFFER_DESC desc{};
    desc.ByteWidth=UINT(bytes); desc.Usage=D3D11_USAGE_STAGING; desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ; ComPtr<ID3D11Buffer> staging;
    auto status=Device->CreateBuffer(&desc,nullptr,staging.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Lumen radiance staging",status));
    Context->CopyResource(staging.Get(),record.Origins.Get()); D3D11_MAPPED_SUBRESOURCE mapped{}; status=Context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped);
    if (FAILED(status)) return Fail(HResultMessage("Read Lumen ray radiance",status));
    std::vector<LumenProbeOrigin> samples(LumenRadianceProbeCount(record.Info)); std::memcpy(samples.data(),mapped.pData,bytes); Context->Unmap(staging.Get(),0);
    if (!ValidateLumenRadianceOrigins(record.Info,samples,LastError)) return false;
    output=std::move(samples); return true;
}
bool ReadLumenRadianceRecord(const LumenRadianceRecord& record,std::vector<LumenRadianceSample>& output)
{
    const auto bytes=std::size_t(record.Info.RayCount)*sizeof(LumenRadianceSample); D3D11_BUFFER_DESC desc{};
    desc.ByteWidth=UINT(bytes); desc.Usage=D3D11_USAGE_STAGING; desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ; ComPtr<ID3D11Buffer> staging;
    auto status=Device->CreateBuffer(&desc,nullptr,staging.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Lumen radiance staging",status));
    Context->CopyResource(staging.Get(),record.Buffer.Get()); D3D11_MAPPED_SUBRESOURCE mapped{}; status=Context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped);
    if (FAILED(status)) return Fail(HResultMessage("Read Lumen ray radiance",status));
    std::vector<LumenRadianceSample> samples(record.Info.RayCount); std::memcpy(samples.data(),mapped.pData,bytes); Context->Unmap(staging.Get(),0);
    if (!ValidateLumenRadianceSamples(record.Info,record.Lighting->Distance->Info,record.Lighting->CardInstances,samples,LastError)) return false;
    output=std::move(samples); return true;
}
bool TraceLumenRadiance(LumenLightingHandle lighting,const LumenRadianceTraceUpload& upload,LumenRadianceTraceResult& output)
{
    if (upload.Hybrid.Enabled && (!FrameOpen || (!upload.Reflections.Enabled && (GI.Resolved || FrameLumen.HasGathered())) || !HdrSceneActive || !SceneColorWritten || !GBufferWritten ||
        (FrameSampleCount>1 && (!SamplesOpaqueResolved || SamplesColorRestored || SamplesTransparentWritten)) ||
        (DeferredActive && !DeferredResolved) || (AmbientOcclusionActive && !AmbientOcclusionApplied) ||
        ReflectionApplied || VolumetricFog.Resolved || LightShafts.Resolved || TemporalAA.Resolved || MotionBlurResolved || DepthOfFieldResolved ||
        RefractionResolved || !Hair.Complete() || !SubsurfaceComplete()))
        return Fail("Lumen hybrid tracing requires resolved opaque HDR before secondary composition");
    if (upload.Hybrid.Enabled && LumenHasTransparentMainDraws(PendingFrame)) return Fail("Lumen hybrid tracing must precede transparent draws");
    if (upload.Screen.Enabled && (!FrameOpen || !GBufferWritten || (FrameSampleCount>1 && !SamplesOpaqueResolved) ||
        upload.Screen.Width!=unsigned(CreateInfo.Width) || upload.Screen.Height!=unsigned(CreateInfo.Height)))
        return Fail("Lumen screen placement needs this frame's resolved G-buffer and matching viewport");
    const auto found=LumenLightingCaches.find(lighting.Value); if (found==LumenLightingCaches.end()) return Fail("Unknown Lumen radiance source lighting");
    const auto source=found->second; LumenRadianceInput input;
    auto distance=source->Distance;
    if (upload.GlobalDistance)
    {
        const auto global=LumenDistanceScenes.find(upload.GlobalDistance.Value);
        if (global==LumenDistanceScenes.end() || global->second->Base!=source->Distance) return Fail("Global distance scene has a different or expired mesh source");
        distance=global->second;
    }
    if (!BuildLumenRadianceInput(source->Info,distance->Info,source->Surface->Captures,source->CardInstances,upload,input,LastError) || (FrameOpen && !FrameLumen.CanTrace(input.Resolve.Counts.X,LastError)) || !EnsureLumenRadianceTrace()) return false;
    std::shared_ptr<LumenRadianceRecord> worldCache; LumenWorldLookupFrame worldFrame{};
    if (upload.WorldCache.Radiance)
    {
        const auto cache=LumenRadianceTraces.find(upload.WorldCache.Radiance.Value);
        if (cache==LumenRadianceTraces.end() || cache->second->Lighting!=source) return Fail("World radiance cache has a different or expired lighting source");
        worldCache=cache->second;
        if (!BuildLumenWorldLookupFrame(worldCache->Info,upload,input,worldFrame,LastError) || !EnsureLumenWorldLookup()) return false;
    }
    auto record=std::make_shared<LumenRadianceRecord>(); record->Info=MakeLumenRadianceInfo(source->Info.Identity,upload,input,worldCache?unsigned(worldFrame.Query.W):0); record->Lighting=source; record->WorldCache=worldCache; record->TraceDistance=distance;
    const auto count=std::size_t(input.Resolve.Counts.X); const auto sources=LumenRadianceSources(upload,input),traceFrame=LumenRadianceFrameData(input); ComPtr<ID3D11Buffer> rays,mapping,hits; ComPtr<ID3D11ShaderResourceView> rayView,mapView,hitView;
    ComPtr<ID3D11UnorderedAccessView> hitTarget,resultTarget,originTarget;
    if (!CreateLumenRadianceBuffer(sources.Bytes,sizeof(LumenDistanceRay),sources.Data,rays,rayView,upload.Screen.Enabled?std::addressof(originTarget):nullptr) ||
        !CreateLumenRadianceBuffer(input.Words.size()*sizeof(std::uint32_t),sizeof(ShaderUInt4),input.Words.data(),mapping,mapView) ||
        !CreateLumenRadianceBuffer(count*sizeof(LumenDistanceHit),sizeof(LumenDistanceHit),nullptr,hits,hitView,&hitTarget) ||
        !CreateLumenRadianceBuffer(count*sizeof(LumenRadianceSample),sizeof(LumenRadianceSample),nullptr,record->Buffer,record->View,&resultTarget)) return false;
    record->Origins=rays; record->OriginView=rayView;
    const auto placement=LumenRadiancePlacementData(input);
    ComPtr<ID3D11Buffer> uniforms[4]; const void* frames[]{traceFrame.Data,&input.Resolve,placement.Data,&input.Hybrid}; const unsigned sizes[]{unsigned(traceFrame.Bytes),sizeof(input.Resolve),unsigned(placement.Bytes),sizeof(input.Hybrid)};
    for (unsigned stage=0;stage<(upload.Hybrid.Enabled?4u:upload.Screen.Enabled?3u:2u);++stage)
    {
        D3D11_BUFFER_DESC desc{}; desc.ByteWidth=sizes[stage]; desc.Usage=D3D11_USAGE_IMMUTABLE; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA data{}; data.pSysMem=frames[stage]; const auto status=Device->CreateBuffer(&desc,&data,uniforms[stage].GetAddressOf());
        if (FAILED(status)) return Fail(HResultMessage("Create Lumen radiance frame",status));
    }
    ComPtr<ID3D11Buffer> pyramid; ComPtr<ID3D11ShaderResourceView> pyramidView; ComPtr<ID3D11UnorderedAccessView> pyramidTarget;
    std::vector<ComPtr<ID3D11Buffer>> depthFrames(input.Depth.Levels.size());
    if (input.Depth.Bytes && !CreateLumenRadianceBuffer(input.Depth.Bytes,sizeof(std::uint32_t),nullptr,pyramid,pyramidView,&pyramidTarget)) return false;
    for (unsigned level=0;level<depthFrames.size();++level)
    {
        D3D11_BUFFER_DESC desc{}; desc.ByteWidth=sizeof(LumenScreenDepthFrame); desc.Usage=D3D11_USAGE_IMMUTABLE; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA data{}; data.pSysMem=&input.Depth.Levels[level];
        const auto status=Device->CreateBuffer(&desc,&data,depthFrames[level].GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Lumen screen depth frame",status));
    }
    ComPtr<ID3D11Buffer> worldUniform;
    if (worldCache)
    {
        D3D11_BUFFER_DESC desc{}; desc.ByteWidth=sizeof(worldFrame); desc.Usage=D3D11_USAGE_IMMUTABLE; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA data{}; data.pSysMem=&worldFrame;
        const auto status=Device->CreateBuffer(&desc,&data,worldUniform.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Lumen world frame",status));
    }
    ComPtr<ID3D11Query> completion;
    if (!upload.Readback && !FrameOpen)
    { const D3D11_QUERY_DESC desc{D3D11_QUERY_EVENT,0}; const auto status=Device->CreateQuery(&desc,completion.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Lumen radiance completion",status)); }
    const auto groups=unsigned((count+LumenRadianceThreads-1)/LumenRadianceThreads),x=std::min(groups,LumenRadianceColumns),y=(groups+LumenRadianceColumns-1)/LumenRadianceColumns;
    if (upload.Screen.Enabled)
    {
        // Logical EndPass does not detach D3D11 attachments. Preserve OM state
        // explicitly while compute samples the just-written depth and normals.
        ID3D11RenderTargetView* attachments[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT]{}; ID3D11DepthStencilView* depth=nullptr;
        Context->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT,attachments,&depth); Context->OMSetRenderTargets(0,nullptr,nullptr);
        auto* uniform=uniforms[2].Get(); auto* target=originTarget.Get(); ID3D11ShaderResourceView* views[]{DepthShaderView.Get(),GBufferSurfaces[1].View.Get()};
        Context->CSSetShader((upload.Reflections.Enabled?LumenReflectionRayShader:LumenScreenPlacement).Get(),nullptr,0); Context->CSSetConstantBuffers(0,1,&uniform);
        Context->CSSetShaderResources(0,2,views); Context->CSSetUnorderedAccessViews(0,1,&target,nullptr);
        Context->Dispatch((input.Screen.Grid.Y+LumenRadianceThreads-1)/LumenRadianceThreads,1,1);
        target=nullptr; ID3D11ShaderResourceView* empty[2]{}; Context->CSSetUnorderedAccessViews(0,1,&target,nullptr); Context->CSSetShaderResources(0,2,empty);
        if (upload.Hybrid.Enabled)
        {
            for (unsigned level=0;level<depthFrames.size();++level)
            {
                auto* depthFrame=depthFrames[level].Get(); auto* depthInput=DepthShaderView.Get(); auto* depthOutput=pyramidTarget.Get();
                Context->CSSetShader(LumenScreenDepth.Get(),nullptr,0); Context->CSSetConstantBuffers(0,1,&depthFrame);
                Context->CSSetShaderResources(0,1,&depthInput); Context->CSSetUnorderedAccessViews(0,1,&depthOutput,nullptr);
                const auto& extent=input.Depth.Levels[level].Destination;
                const auto levelGroups=(extent.X*extent.Y+LumenRadianceThreads-1)/LumenRadianceThreads;
                Context->Dispatch(std::min(levelGroups,LumenRadianceColumns),(levelGroups+LumenRadianceColumns-1)/LumenRadianceColumns,1);
                depthOutput=nullptr; depthInput=nullptr; Context->CSSetUnorderedAccessViews(0,1,&depthOutput,nullptr); Context->CSSetShaderResources(0,1,&depthInput);
            }
            uniform=uniforms[3].Get(); ID3D11ShaderResourceView* inputs[]{DepthShaderView.Get(),GBufferSurfaces[1].View.Get(),HdrSurfaces[HdrScene].View.Get(),rayView.Get(),pyramidView?pyramidView.Get():rayView.Get()};
            ID3D11UnorderedAccessView* outputs[]{hitTarget.Get(),resultTarget.Get()};
            Context->CSSetShader(LumenHybridTrace.Get(),nullptr,0); Context->CSSetConstantBuffers(0,1,&uniform);
            Context->CSSetShaderResources(0,5,inputs); Context->CSSetUnorderedAccessViews(0,2,outputs,nullptr); Context->Dispatch(x,y,1);
            ID3D11ShaderResourceView* nullViews[5]{}; ID3D11UnorderedAccessView* nullTargets[2]{};
            Context->CSSetShaderResources(0,5,nullViews); Context->CSSetUnorderedAccessViews(0,2,nullTargets,nullptr);
        }
        Context->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT,attachments,depth);
        for (auto* attachment:attachments) if (attachment) attachment->Release(); if (depth) depth->Release();
    }
    if (worldCache)
    {
        auto* frame=worldUniform.Get(); ID3D11ShaderResourceView* views[]{worldCache->View.Get(),rayView.Get()};
        ID3D11UnorderedAccessView* targets[]{hitTarget.Get(),resultTarget.Get()};
        Context->CSSetShader(LumenWorldLookup.Get(),nullptr,0); Context->CSSetConstantBuffers(0,1,&frame);
        Context->CSSetShaderResources(0,2,views); Context->CSSetUnorderedAccessViews(0,2,targets,nullptr); Context->Dispatch(x,y,1);
        ID3D11ShaderResourceView* noViews[2]{}; ID3D11UnorderedAccessView* noTargets[2]{};
        Context->CSSetShaderResources(0,2,noViews); Context->CSSetUnorderedAccessViews(0,2,noTargets,nullptr);
    }
    auto* uniform=uniforms[0].Get(); auto* target=hitTarget.Get(); ID3D11ShaderResourceView* traceViews[]{distance->View.Get(),rayView.Get()};
    Context->CSSetShader((input.ProbeTrace.Sampling.X?LumenProbeTrace:LumenRadianceTrace).Get(),nullptr,0); Context->CSSetConstantBuffers(0,1,&uniform); Context->CSSetShaderResources(0,2,traceViews); Context->CSSetUnorderedAccessViews(0,1,&target,nullptr); Context->Dispatch(x,y,1);
    ID3D11ShaderResourceView* empty[4]{}; target=nullptr; Context->CSSetUnorderedAccessViews(0,1,&target,nullptr); Context->CSSetShaderResources(0,4,empty);
    uniform=uniforms[1].Get(); target=resultTarget.Get(); ID3D11ShaderResourceView* resolveViews[]{source->Surface->Views[1].Get(),hitView.Get(),source->View.Get(),mapView.Get()};
    Context->CSSetShader(LumenRadianceResolve.Get(),nullptr,0); Context->CSSetConstantBuffers(0,1,&uniform); Context->CSSetShaderResources(0,4,resolveViews); Context->CSSetUnorderedAccessViews(0,1,&target,nullptr); Context->Dispatch(x,y,1);
    uniform=nullptr; target=nullptr; Context->CSSetShaderResources(0,4,empty); Context->CSSetUnorderedAccessViews(0,1,&target,nullptr); Context->CSSetConstantBuffers(0,1,&uniform); Context->CSSetShader(nullptr,nullptr,0);
    LumenRadianceTraceResult result; result.Info=record->Info;
    if (upload.Readback) { if (!ReadLumenRadianceRecord(*record,result.Samples)) return false; }
    else if (!FrameOpen)
    {
        Context->End(completion.Get()); Context->Flush(); constexpr ULONGLONG TimeoutMilliseconds=10000; const auto deadline=GetTickCount64()+TimeoutMilliseconds; HRESULT status;
        while ((status=Context->GetData(completion.Get(),nullptr,0,0))==S_FALSE)
        { if (GetTickCount64()>=deadline) return Fail("Lumen radiance tracing timed out"); std::this_thread::yield(); }
        if (FAILED(status)) return Fail(HResultMessage("Complete Lumen radiance trace",status));
    }
    result.Handle=NewLumenRadianceHandle(); if (!result.Handle) return Fail("Lumen radiance handle space exhausted");
    LumenRadianceTraces.emplace(result.Handle.Value,std::move(record));
    if (FrameOpen) FrameLumen.Record(result.Handle,result.Info.RayCount);
    output=std::move(result); return true;
}
bool AssembleLumenRadiance(const LumenRadianceAssembly& assembly,LumenRadianceTraceResult& output)
{
    if (!ValidateLumenRadianceAssembly(assembly,LastError)) return false;
    std::vector<std::shared_ptr<LumenRadianceRecord>> sources;
    for (const auto& copy:assembly.Copies)
    {
        const auto found=LumenRadianceTraces.find(copy.Source.Value);
        if (found==LumenRadianceTraces.end()) return Fail("Unknown world radiance assembly source");
        if (!ValidateLumenRadianceCopy(found->second->Info,assembly,copy,LastError)) return false;
        if (!sources.empty() && found->second->Lighting!=sources.front()->Lighting) return Fail("World radiance assembly mixes lighting revisions");
        sources.push_back(found->second);
    }
    auto record=std::make_shared<LumenRadianceRecord>(); record->Lighting=sources.front()->Lighting;
    record->Info=sources.front()->Info; record->Info.Probes=assembly.Probes;
    record->Info.RayCount=unsigned(assembly.Probes.Origins.size())*assembly.Probes.RaysPerProbe*assembly.Probes.DepthLayers;
    const auto probeBytes=std::size_t(assembly.Probes.RaysPerProbe)*assembly.Probes.DepthLayers*sizeof(LumenRadianceSample);
    if (!CreateLumenRadianceBuffer(assembly.Probes.Origins.size()*sizeof(LumenProbeOrigin),sizeof(LumenProbeOrigin),assembly.Probes.Origins.data(),record->Origins,record->OriginView) ||
        !CreateLumenRadianceBuffer(std::size_t(record->Info.RayCount)*sizeof(LumenRadianceSample),sizeof(LumenRadianceSample),nullptr,record->Buffer,record->View)) return false;
    D3D11_QUERY_DESC desc{}; desc.Query=D3D11_QUERY_EVENT; ComPtr<ID3D11Query> completion;
    auto status=Device->CreateQuery(&desc,completion.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create world assembly event",status));
    for (unsigned i=0;i<assembly.Copies.size();++i)
    {
        const auto& copy=assembly.Copies[i]; D3D11_BOX box{UINT(copy.SourceProbe*probeBytes),0,0,UINT((copy.SourceProbe+copy.ProbeCount)*probeBytes),1,1};
        Context->CopySubresourceRegion(record->Buffer.Get(),0,UINT(copy.DestinationProbe*probeBytes),0,0,sources[i]->Buffer.Get(),0,&box);
    }
    Context->End(completion.Get()); Context->Flush(); constexpr ULONGLONG TimeoutMilliseconds=10000; const auto deadline=GetTickCount64()+TimeoutMilliseconds;
    while ((status=Context->GetData(completion.Get(),nullptr,0,0))==S_FALSE)
    { if (GetTickCount64()>=deadline) return Fail("World radiance assembly timed out"); std::this_thread::yield(); }
    if (FAILED(status)) return Fail(HResultMessage("Complete world radiance assembly",status));
    LumenRadianceTraceResult result; result.Info=record->Info; result.Handle=NewLumenRadianceHandle();
    if (!result.Handle) return Fail("Lumen radiance handle space exhausted");
    LumenRadianceTraces.emplace(result.Handle.Value,std::move(record)); output=std::move(result); return true;
}
bool ReadLumenProbeOrigins(LumenRadianceHandle handle,std::vector<LumenProbeOrigin>& output)
{
    const auto found=LumenRadianceTraces.find(handle.Value); if (found==LumenRadianceTraces.end()) return Fail("Unknown Lumen radiance trace");
    if (!LumenRadianceProbeCount(found->second->Info)) return Fail("Lumen trace has no probe origins");
    return ReadLumenOriginRecord(*found->second,output);
}
bool ReadLumenRadiance(LumenRadianceHandle handle,std::vector<LumenRadianceSample>& output)
{
    const auto found=LumenRadianceTraces.find(handle.Value); if (found==LumenRadianceTraces.end()) return Fail("Unknown Lumen radiance trace");
    return ReadLumenRadianceRecord(*found->second,output);
}
bool DestroyLumenRadiance(LumenRadianceHandle handle) { return LumenRadianceTraces.erase(handle.Value)!=0 || Fail("Unknown Lumen radiance trace"); }
