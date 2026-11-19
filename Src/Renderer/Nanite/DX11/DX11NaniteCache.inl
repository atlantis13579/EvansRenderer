// Immutable cache versions share unchanged pools or build a separate buffer from
// resident GPU copy runs plus changed CPU pages. A tiny staging read completes
// the ordered copy/update stream before the candidate handle becomes visible.
struct NaniteCachePool
{
    ComPtr<ID3D11Buffer> Buffer;
    ComPtr<ID3D11ShaderResourceView> View;
};
struct NaniteNativeCache { NaniteCacheInfo Info; std::shared_ptr<NaniteCachePool> Pool; };
std::unordered_map<std::uint64_t,NaniteNativeCache> NaniteCaches;
void DestroyNaniteCaches() { NaniteCaches.clear(); }
bool DestroyNaniteCache(NaniteCacheHandle handle)
{
    if (!handle || !NaniteCaches.erase(handle.Value)) return Fail("D3D11 Nanite cache handle is invalid or belongs to another RHI");
    return true;
}
bool PrepareNaniteCache(const NaniteGpuUpload& upload,NaniteCacheHandle previous,NaniteCacheHandle& output,NaniteCacheStatistics& statistics)
{
    const NaniteNativeCache* old=nullptr;
    if (previous)
    { const auto found=NaniteCaches.find(previous.Value); if (found==NaniteCaches.end()) return Fail("D3D11 previous Nanite cache handle is invalid"); old=&found->second; }
    NaniteCachePlan plan; if (!BuildNaniteCachePlan(old?&old->Info:nullptr,upload,plan,LastError)) return false;
    auto pool=plan.ReusePool?old->Pool:std::make_shared<NaniteCachePool>();
    if (!plan.ReusePool)
    {
        D3D11_BUFFER_DESC desc{}; desc.ByteWidth=static_cast<UINT>(plan.Statistics.AllocatedBytes); desc.Usage=D3D11_USAGE_DEFAULT;
        desc.BindFlags=D3D11_BIND_SHADER_RESOURCE; desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        auto result=Device->CreateBuffer(&desc,nullptr,pool->Buffer.GetAddressOf()); if (FAILED(result)) return Fail(HResultMessage("Create Nanite cache pool",result));
        D3D11_SHADER_RESOURCE_VIEW_DESC view{}; view.Format=DXGI_FORMAT_R32_TYPELESS; view.ViewDimension=D3D11_SRV_DIMENSION_BUFFEREX;
        view.BufferEx.NumElements=desc.ByteWidth/sizeof(std::uint32_t); view.BufferEx.Flags=D3D11_BUFFEREX_SRV_FLAG_RAW;
        result=Device->CreateShaderResourceView(pool->Buffer.Get(),&view,pool->View.GetAddressOf()); if (FAILED(result)) return Fail(HResultMessage("Create Nanite cache view",result));
        for (const auto& run:plan.Copies)
        {
            D3D11_BOX region{run.FirstSlot*NaniteGpuPageBytes,0,0,(run.FirstSlot+run.SlotCount)*NaniteGpuPageBytes,1,1};
            Context->CopySubresourceRegion(pool->Buffer.Get(),0,region.left,0,0,old->Pool->Buffer.Get(),0,&region);
        }
        for (auto slot:plan.UploadSlots)
        {
            D3D11_BOX region{slot*NaniteGpuPageBytes,0,0,(slot+1)*NaniteGpuPageBytes,1,1};
            Context->UpdateSubresource(pool->Buffer.Get(),0,&region,&upload.PhysicalPages[slot],0,0);
        }
        ComPtr<ID3D11Buffer> complete; D3D11_BUFFER_DESC staging{}; staging.ByteWidth=sizeof(std::uint32_t); staging.Usage=D3D11_USAGE_STAGING; staging.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
        result=Device->CreateBuffer(&staging,nullptr,complete.GetAddressOf()); if (FAILED(result)) return Fail(HResultMessage("Create Nanite cache completion read",result));
        const D3D11_BOX word{0,0,0,sizeof(std::uint32_t),1,1}; Context->CopySubresourceRegion(complete.Get(),0,0,0,0,pool->Buffer.Get(),0,&word);
        D3D11_MAPPED_SUBRESOURCE mapped{}; result=Context->Map(complete.Get(),0,D3D11_MAP_READ,0,&mapped);
        if (FAILED(result)) return Fail(HResultMessage("Complete Nanite cache upload",result)); Context->Unmap(complete.Get(),0);
    }
    const auto handle=NewNaniteCacheHandle(); if (!handle) return Fail("Nanite cache handle space is exhausted");
    NaniteCaches.emplace(handle.Value,NaniteNativeCache{std::move(plan.Next),std::move(pool)});
    output=handle; statistics=plan.Statistics; return true;
}
