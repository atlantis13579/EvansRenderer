#include "Renderer/Nanite/NaniteAsset.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#endif

namespace Evans::Renderer
{
    namespace
    {
        // Independent little-endian archive, not an Unreal asset format. Bump
        // CookerVersion when hierarchy construction or quantization changes.
        constexpr std::uint64_t ArchiveMagic=0x4554494e414e5645ull; // "EVNANITE"
        constexpr std::uint64_t CookerVersion=2,HashSeed=14695981039346656037ull,HashPrime=1099511628211ull;
        constexpr std::size_t HeaderWords=8,HeaderBytes=HeaderWords*sizeof(std::uint64_t);
        constexpr unsigned DiskReadBatchPages=8;
        constexpr std::uint64_t MaximumMetadataBytes=512ull*1024*1024,MaximumArchivePages=16*1024;
        std::atomic<std::uint64_t> NextTemporaryFile{1};
        std::mutex AdmissionMutex;
        NaniteAssetQueueStatistics Admission;
        struct SourceReservation
        {
            std::uint64_t Bytes=0;
            bool Active=false;
            ~SourceReservation()
            {
                if (!Active) return;
                std::lock_guard<std::mutex> lock(AdmissionMutex);
                --Admission.Requests; Admission.SourceBytes-=Bytes;
            }
        };

        struct Hash
        {
            std::uint64_t Value=HashSeed;
            void Bytes(const void* data,std::size_t size)
            {
                const auto* bytes=static_cast<const unsigned char*>(data);
                for (std::size_t i=0;i<size;++i) { Value^=bytes[i]; Value*=HashPrime; }
            }
            template<class T> void Scalar(const T& value) { Bytes(&value,sizeof(value)); }
            template<class T> void Vector(const std::vector<T>& values)
            { Scalar(std::uint64_t(values.size())); Bytes(values.data(),values.size()*sizeof(T)); }
        };
        bool LittleEndian() { const unsigned word=1; return *reinterpret_cast<const unsigned char*>(&word)==1; }
        struct Writer
        {
            std::vector<unsigned char> Bytes;
            template<class T> void Scalar(const T& value)
            {
                static_assert(std::is_trivially_copyable_v<T>);
                const auto* first=reinterpret_cast<const unsigned char*>(&value); Bytes.insert(Bytes.end(),first,first+sizeof(T));
            }
            template<class T> void Vector(const std::vector<T>& values)
            {
                Scalar(std::uint64_t(values.size()));
                if (!values.empty())
                {
                    const auto* first=reinterpret_cast<const unsigned char*>(values.data());
                    Bytes.insert(Bytes.end(),first,first+values.size()*sizeof(T));
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
            std::size_t Count(std::size_t minimumBytes)
            {
                std::uint64_t count=0; Scalar(count);
                if (!Valid || count>(Bytes.size()-At)/minimumBytes) { Valid=false; return 0; }
                return static_cast<std::size_t>(count);
            }
            template<class T> void Vector(std::vector<T>& values)
            {
                const auto count=Count(sizeof(T)); if (!Valid) return;
                values.resize(count);
                if (count) std::memcpy(values.data(),Bytes.data()+At,count*sizeof(T));
                At+=count*sizeof(T);
            }
        };
        template<class IO,class H> void Hierarchy(IO& io,H& h)
        {
            io.Vector(h.Vertices); io.Vector(h.SourceIndices); io.Vector(h.SourceMaterials); io.Vector(h.Roots);
            io.Scalar(h.ClusterTriangleLimit); io.Scalar(h.ClusterVertexLimit); io.Scalar(h.GroupClusterLimit); io.Scalar(h.Signature);
        }
        void WriteMetadata(Writer& w,const NaniteHierarchy& hierarchy,const NanitePages& p,const std::vector<std::uint64_t>& hashes)
        {
            const auto& h=hierarchy; Hierarchy(w,h);
            w.Scalar(std::uint64_t(h.Clusters.size()));
            for (const auto& c:h.Clusters)
            {
                w.Vector(c.Indices); w.Vector(c.SourceTriangles); w.Scalar(c.Material); w.Scalar(c.Level);
                w.Scalar(c.GeneratingGroup); w.Scalar(c.ParentGroup); w.Scalar(c.BoundsMinimum); w.Scalar(c.BoundsMaximum);
            }
            w.Scalar(std::uint64_t(h.Groups.size()));
            for (const auto& g:h.Groups)
            {
                w.Vector(g.Children); w.Vector(g.Parents); w.Scalar(g.Level); w.Scalar(g.Material);
                w.Scalar(g.BoundsMinimum); w.Scalar(g.BoundsMaximum); w.Scalar(g.LocalDisplacement); w.Scalar(g.MaximumDisplacement);
            }
            w.Scalar(p.PositionExponent); w.Scalar(p.MaximumQuantizationDisplacement); w.Scalar(p.HierarchySource); w.Scalar(p.Signature);
            w.Scalar(p.PayloadBytes); w.Scalar(p.UncompressedBytes); w.Vector(p.Clusters); w.Vector(p.Groups);
            w.Vector(p.PinnedPages); w.Vector(p.SourceVertices); w.Vector(hashes);
        }
        bool ReadMetadata(const std::vector<unsigned char>& bytes,NaniteHierarchy& h,NanitePages& p,std::vector<std::uint64_t>& hashes)
        {
            Reader r{bytes}; Hierarchy(r,h);
            constexpr std::size_t MinimumClusterBytes=2*sizeof(std::uint64_t)+4*sizeof(std::uint32_t)+2*sizeof(Float3);
            constexpr std::size_t MinimumGroupBytes=2*sizeof(std::uint64_t)+2*sizeof(std::uint32_t)+2*sizeof(Float3)+2*sizeof(double);
            h.Clusters.resize(r.Count(MinimumClusterBytes));
            for (auto& c:h.Clusters)
            {
                r.Vector(c.Indices); r.Vector(c.SourceTriangles); r.Scalar(c.Material); r.Scalar(c.Level);
                r.Scalar(c.GeneratingGroup); r.Scalar(c.ParentGroup); r.Scalar(c.BoundsMinimum); r.Scalar(c.BoundsMaximum);
                if (!r.Valid) return false;
            }
            h.Groups.resize(r.Count(MinimumGroupBytes));
            for (auto& g:h.Groups)
            {
                r.Vector(g.Children); r.Vector(g.Parents); r.Scalar(g.Level); r.Scalar(g.Material);
                r.Scalar(g.BoundsMinimum); r.Scalar(g.BoundsMaximum); r.Scalar(g.LocalDisplacement); r.Scalar(g.MaximumDisplacement);
                if (!r.Valid) return false;
            }
            r.Scalar(p.PositionExponent); r.Scalar(p.MaximumQuantizationDisplacement); r.Scalar(p.HierarchySource); r.Scalar(p.Signature);
            r.Scalar(p.PayloadBytes); r.Scalar(p.UncompressedBytes); r.Vector(p.Clusters); r.Vector(p.Groups);
            r.Vector(p.PinnedPages); r.Vector(p.SourceVertices); r.Vector(hashes);
            return r.Valid && r.At==bytes.size();
        }
        std::uint64_t Manifest(const NanitePages& p)
        {
            Hash hash; hash.Scalar(p.PositionExponent); hash.Scalar(p.MaximumQuantizationDisplacement);
            hash.Scalar(p.HierarchySource); hash.Scalar(p.Signature); hash.Scalar(p.PayloadBytes); hash.Scalar(p.UncompressedBytes);
            hash.Vector(p.Clusters); hash.Vector(p.Groups); hash.Vector(p.PinnedPages); hash.Vector(p.SourceVertices); return hash.Value;
        }
        bool ReadBytes(std::ifstream& input,std::uint64_t offset,void* data,std::size_t bytes)
        {
            input.clear(); input.seekg(static_cast<std::streamoff>(offset));
            return bool(input.read(static_cast<char*>(data),static_cast<std::streamsize>(bytes)));
        }

        // One fair disk queue serves all families. Thousands of scene objects
        // must not create thousands of threads; short batches let other sources
        // make progress while a large family is loading its next resident cut.
        class AssetWorker
        {
            std::mutex Mutex;
            std::condition_variable Wake;
            std::deque<std::function<void()>> Queue;
            bool Stopping=false;
            std::thread Thread;
        public:
            AssetWorker():Thread([this]
            {
                for (;;)
                {
                    std::function<void()> job;
                    {
                        std::unique_lock<std::mutex> lock(Mutex);
                        Wake.wait(lock,[&]{return Stopping || !Queue.empty();});
                        if (Stopping) return;
                        job=std::move(Queue.front()); Queue.pop_front();
                    }
                    job();
                }
            }) {}
            ~AssetWorker()
            {
                { std::lock_guard<std::mutex> lock(Mutex); Stopping=true; Queue.clear(); }
                Wake.notify_one(); Thread.join();
            }
            void Submit(std::function<void()> job)
            {
                { std::lock_guard<std::mutex> lock(Mutex); if (Stopping) return; Queue.push_back(std::move(job)); }
                Wake.notify_one();
            }
        };
        AssetWorker& PageWorker() { static AssetWorker worker; return worker; }
        AssetWorker& CookWorker() { static AssetWorker worker; return worker; }

        class DiskPages final:public NanitePageSource,public std::enable_shared_from_this<DiskPages>
        {
            struct Entry { std::shared_ptr<const NanitePage> Page; std::uint64_t Used=0; };
            std::filesystem::path File;
            std::uint64_t Offset,HierarchySignature,MetadataSignature;
            std::vector<std::uint64_t> Hashes;
            std::size_t Capacity;
            mutable std::mutex Mutex;
            mutable std::unordered_map<unsigned,Entry> Cache;
            std::unordered_set<unsigned> Pending;
            std::deque<unsigned> Queue;
            mutable std::uint64_t Clock=0;
            bool Scheduled=false;
            std::string Failure;
            std::ifstream Input;

            bool ReadPage(std::ifstream& input,unsigned id,NanitePage& page,std::string& error) const
            {
                if (id>=Hashes.size() || !ReadBytes(input,Offset+std::uint64_t(id)*NanitePageBytes,&page,sizeof(page)))
                { error="Nanite asset page read failed or was truncated"; return false; }
                Hash hash; hash.Bytes(&page,sizeof(page));
                if (hash.Value!=Hashes[id]) { error="Nanite asset page checksum mismatch"; return false; }
                error.clear(); return true;
            }
            void Schedule()
            {
                auto weak=weak_from_this();
                PageWorker().Submit([weak]{if (const auto source=weak.lock()) source->Run();});
            }
            void Run()
            {
                try
                {
                    if (!Input.is_open()) Input.open(File,std::ios::binary);
                    for (unsigned batch=0;batch<DiskReadBatchPages;++batch)
                    {
                        unsigned id=0;
                        {
                            std::lock_guard<std::mutex> lock(Mutex);
                            if (Queue.empty()) { Scheduled=false; return; }
                            id=Queue.front(); Queue.pop_front();
                        }
                        auto page=std::make_shared<NanitePage>(); std::string error;
                        const bool loaded=ReadPage(Input,id,*page,error);
                        std::lock_guard<std::mutex> lock(Mutex); Pending.erase(id);
                        if (!loaded) { Failure=std::move(error); Queue.clear(); Pending.clear(); Scheduled=false; return; }
                        if (Cache.size()>=Capacity)
                        {
                            const auto oldest=std::min_element(Cache.begin(),Cache.end(),[](const auto& a,const auto& b){return a.second.Used<b.second.Used;});
                            if (oldest!=Cache.end()) Cache.erase(oldest);
                        }
                        Cache[id]={std::move(page),++Clock};
                    }
                    Schedule();
                }
                catch (const std::exception& exception)
                { std::lock_guard<std::mutex> lock(Mutex); Failure=exception.what(); Queue.clear(); Pending.clear(); Scheduled=false; }
            }
        public:
            DiskPages(std::filesystem::path file,std::uint64_t offset,std::uint64_t hierarchy,std::uint64_t metadata,
                std::vector<std::uint64_t> hashes,unsigned capacity):File(std::move(file)),Offset(offset),HierarchySignature(hierarchy),
                MetadataSignature(metadata),Hashes(std::move(hashes)),Capacity(std::max(std::size_t(1),std::min(std::size_t(capacity),Hashes.size()))) {}
            unsigned Count() const noexcept override { return static_cast<unsigned>(Hashes.size()); }
            bool Read(unsigned id,NanitePage& output,std::string& error) const override
            {
                {
                    std::lock_guard<std::mutex> lock(Mutex); const auto found=Cache.find(id);
                    if (found!=Cache.end()) { output=*found->second.Page; found->second.Used=++Clock; error.clear(); return true; }
                }
                // Synchronous callers (offline validation and lightmap charts)
                // use a separate file cursor and do not perturb the worker queue.
                std::ifstream input(File,std::ios::binary); return ReadPage(input,id,output,error);
            }
            bool Prefetch(const std::vector<unsigned>& ids,bool& ready,std::string& error) override
            {
                std::lock_guard<std::mutex> lock(Mutex); ready=false;
                if (!Failure.empty()) { error=Failure; return false; }
                if (ids.size()>Capacity) { error="Nanite disk request exceeds the host page cache budget"; return false; }
                for (auto id:ids) if (id>=Hashes.size()) { error="Nanite disk request has an invalid page ID"; return false; }
                ready=true;
                for (auto id:ids)
                {
                    const auto found=Cache.find(id);
                    if (found!=Cache.end()) { found->second.Used=++Clock; continue; }
                    ready=false;
                    if (Pending.size()<Capacity && Pending.insert(id).second) Queue.push_back(id);
                }
                if (!Scheduled && !Queue.empty()) { Scheduled=true; Schedule(); }
                error.clear(); return true;
            }
            bool Matches(const NaniteHierarchy& h,const NanitePages& p,std::string& error) const override
            {
                // A loaded hierarchy is immutable for the lifetime of its stream.
                // The cooker/load boundary validates its DAG; frame preparation
                // checks the resident manifest without decoding every disk page.
                if (h.Signature!=HierarchySignature || p.HierarchySource!=HierarchySignature || !p.Pages.empty() || Manifest(p)!=MetadataSignature)
                { error="Nanite cooked metadata changed after publication"; return false; }
                error.clear(); return true;
            }
        };
    }

    std::uint32_t NanitePages::PageCount() const noexcept { return Source?Source->Count():static_cast<std::uint32_t>(Pages.size()); }
    bool NaniteAssetSectionLess(const NaniteAssetSection& a,const NaniteAssetSection& b)
    {
        if (a.GeometryHash!=b.GeometryHash) return a.GeometryHash<b.GeometryHash;
        if (a.Vertices->size()!=b.Vertices->size()) return a.Vertices->size()<b.Vertices->size();
        const int order=a.Vertices->empty()?0:std::memcmp(a.Vertices->data(),b.Vertices->data(),a.Vertices->size()*sizeof(Vertex));
        return order?order<0:*a.Indices<*b.Indices;
    }
    bool AssembleNaniteAsset(const std::vector<NaniteAssetSection>& sections,NaniteAssetInput& output,std::string& error)
    {
        static_assert(sizeof(Vertex)==NaniteVertexChannels*sizeof(std::uint32_t),"Nanite source identity requires packed vertex channels");
        const GeometryPreparationLimits budget; NaniteAssetInput input;
        std::map<std::array<std::uint32_t,NaniteVertexChannels>,unsigned> unique;
        for (unsigned part=0;part<sections.size();++part)
        {
            const auto& section=sections[part];
            if (!section.Vertices || !section.Indices) { error="Nanite section has no source streams"; return false; }
            if (!ValidateTriangleGeometry(*section.Vertices,*section.Indices,error)) return false;
            if (section.Indices->size()>budget.MaximumIndices-input.Indices.size()) { error="Nanite family exceeds its index budget"; return false; }
            for (auto index:*section.Indices)
            {
                std::array<std::uint32_t,NaniteVertexChannels> key; std::memcpy(key.data(),&(*section.Vertices)[index],sizeof(Vertex));
                auto found=unique.find(key);
                if (found==unique.end())
                {
                    if (input.Vertices.size()>=budget.MaximumVertices) { error="Nanite family exceeds its vertex budget"; return false; }
                    const auto id=static_cast<unsigned>(input.Vertices.size()); input.Vertices.push_back((*section.Vertices)[index]); found=unique.emplace(key,id).first;
                }
                input.Indices.push_back(found->second);
            }
            input.Materials.insert(input.Materials.end(),section.Indices->size()/3,part);
        }
        if (input.Indices.empty()) { error="Nanite asset has no triangle sections"; return false; }
        output=std::move(input); error.clear(); return true;
    }
    bool ReadNanitePage(const NanitePages& pages,unsigned id,NanitePage& output,std::string& error)
    {
        if (pages.Source) return pages.Source->Read(id,output,error);
        if (id>=pages.Pages.size()) { error="Nanite page ID is out of bounds"; return false; }
        output=pages.Pages[id]; error.clear(); return true;
    }
    bool PrefetchNanitePages(const NanitePages& pages,const std::vector<unsigned>& ids,bool& ready,std::string& error)
    {
        if (pages.Source) return pages.Source->Prefetch(ids,ready,error);
        for (auto id:ids) if (id>=pages.Pages.size()) { error="Nanite prefetch page ID is out of bounds"; return false; }
        ready=true; error.clear(); return true;
    }
    std::uint64_t NaniteAssetKey(const std::vector<Vertex>& vertices,const std::vector<unsigned>& indices,const std::vector<unsigned>& materials)
    { Hash hash; hash.Scalar(CookerVersion); hash.Vector(vertices); hash.Vector(indices); hash.Vector(materials); return hash.Value; }

    bool WriteNaniteAsset(const std::filesystem::path& file,std::uint64_t key,const NaniteHierarchy& h,const NanitePages& p,std::string& error)
    {
        if (!LittleEndian()) { error="Nanite archives require a little-endian host"; return false; }
        if (!ValidateNanitePages(h,p,error)) return false;
        std::vector<std::uint64_t> hashes(p.PageCount()); NanitePage page;
        for (unsigned id=0;id<hashes.size();++id)
        {
            if (!ReadNanitePage(p,id,page,error)) return false;
            Hash hash; hash.Bytes(&page,sizeof(page)); hashes[id]=hash.Value;
        }
        Writer metadata; WriteMetadata(metadata,h,p,hashes);
        if (metadata.Bytes.size()>MaximumMetadataBytes || hashes.size()>MaximumArchivePages)
        { error="Nanite archive exceeds its metadata or page budget"; return false; }
        Hash hash; hash.Bytes(metadata.Bytes.data(),metadata.Bytes.size());
        const std::array<std::uint64_t,HeaderWords> header{ArchiveMagic,CookerVersion,key,metadata.Bytes.size(),hash.Value,hashes.size(),p.Signature,NanitePageBytes};
        std::error_code ec; if (!file.parent_path().empty()) std::filesystem::create_directories(file.parent_path(),ec);
        if (ec) { error="Cannot create Nanite archive directory: "+ec.message(); return false; }
        const auto timestamp=std::chrono::steady_clock::now().time_since_epoch().count();
        auto temporary=file; temporary+=std::filesystem::path(".writing-"+std::to_string(timestamp)+"-"+std::to_string(NextTemporaryFile.fetch_add(1)));
        const auto cleanup=[&]{std::error_code ignored; std::filesystem::remove(temporary,ignored);};
        std::ofstream output(temporary,std::ios::binary|std::ios::trunc);
        if (!output) { error="Cannot create Nanite archive"; return false; }
        output.write(reinterpret_cast<const char*>(header.data()),sizeof(header));
        output.write(reinterpret_cast<const char*>(metadata.Bytes.data()),static_cast<std::streamsize>(metadata.Bytes.size()));
        for (unsigned id=0;id<hashes.size();++id)
        {
            if (!ReadNanitePage(p,id,page,error)) { output.close(); cleanup(); return false; }
            output.write(reinterpret_cast<const char*>(&page),sizeof(page));
        }
        output.flush(); const bool written=bool(output); output.close();
        if (!written) { cleanup(); error="Nanite archive write failed"; return false; }
#if defined(_WIN32)
        if (!MoveFileExW(temporary.c_str(),file.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))
        { cleanup(); error="Cannot publish Nanite archive"; return false; }
#else
        std::filesystem::rename(temporary,file,ec);
        if (ec) { cleanup(); error="Cannot publish Nanite archive: "+ec.message(); return false; }
#endif
        error.clear(); return true;
    }
    bool OpenNaniteAsset(const std::filesystem::path& file,std::uint64_t key,unsigned hostPages,NaniteHierarchy& hierarchy,NanitePages& pages,std::string& error)
    {
        const auto fail=[&](const char* message){error=message; return false;};
        if (!LittleEndian() || !hostPages) return fail("Nanite archive host format or page budget is invalid");
        std::ifstream input(file,std::ios::binary); std::array<std::uint64_t,HeaderWords> header{};
        if (!ReadBytes(input,0,header.data(),sizeof(header))) return fail("Cannot read Nanite archive header");
        if (header[0]!=ArchiveMagic || header[1]!=CookerVersion || header[2]!=key || !header[3] || header[3]>MaximumMetadataBytes ||
            !header[5] || header[5]>MaximumArchivePages || header[7]!=NanitePageBytes) return fail("Nanite archive version, source or dimensions are invalid");
        input.seekg(0,std::ios::end);
        if (input.tellg()!=static_cast<std::streamoff>(HeaderBytes+header[3]+header[5]*NanitePageBytes)) return fail("Nanite archive length is invalid");
        std::vector<unsigned char> metadata(static_cast<std::size_t>(header[3]));
        if (!ReadBytes(input,HeaderBytes,metadata.data(),metadata.size())) return fail("Cannot read Nanite archive metadata");
        Hash hash; hash.Bytes(metadata.data(),metadata.size());
        if (hash.Value!=header[4]) return fail("Nanite archive metadata checksum mismatch");
        NaniteHierarchy h; NanitePages p; std::vector<std::uint64_t> hashes;
        if (!ReadMetadata(metadata,h,p,hashes) || hashes.size()!=header[5] || p.Signature!=header[6] || p.HierarchySource!=h.Signature ||
            p.Clusters.size()!=h.Clusters.size() || p.Groups.size()!=h.Groups.size() || p.PinnedPages.empty() || p.PositionExponent<-126 || p.PositionExponent>127 ||
            !std::isfinite(p.MaximumQuantizationDisplacement) || p.MaximumQuantizationDisplacement<0) return fail("Nanite archive metadata is inconsistent");
        if (!ValidateNaniteHierarchy(h,error)) return false;
        for (const auto& location:p.Clusters)
            if (location.Page>=hashes.size() || location.Word<NanitePageHeaderWords || location.Word>NanitePageWords-NaniteClusterHeaderWords ||
                !location.VertexCount || std::uint64_t(location.FirstSourceVertex)+location.VertexCount>p.SourceVertices.size()) return fail("Nanite archive cluster location is invalid");
        for (const auto& range:p.Groups)
            if (!range.Count || std::uint64_t(range.First)+range.Count>hashes.size()) return fail("Nanite archive group range is invalid");
        std::unordered_set<unsigned> pinned;
        for (auto id:p.PinnedPages) if (id>=hashes.size() || !pinned.insert(id).second) return fail("Nanite archive pinned page is invalid");
        p.Source=std::make_shared<DiskPages>(file,HeaderBytes+header[3],h.Signature,Manifest(p),std::move(hashes),hostPages);
        hierarchy=std::move(h); pages=std::move(p); error.clear(); return true;
    }
    bool LoadOrBuildNaniteAsset(const std::vector<Vertex>& vertices,const std::vector<unsigned>& indices,const std::vector<unsigned>& materials,
        unsigned physicalPages,NaniteHierarchy& hierarchy,NanitePages& pages,std::string& error)
    {
        const auto key=NaniteAssetKey(vertices,indices,materials); std::ostringstream name; name<<std::hex<<key<<".enanite";
#if defined(EVANS_NANITE_CACHE_DIRECTORY)
        const auto file=std::filesystem::path(EVANS_NANITE_CACHE_DIRECTORY)/name.str();
#else
        const auto file=std::filesystem::temp_directory_path()/"EvansRenderer"/"Nanite"/name.str();
#endif
        const auto hostPages=physicalPages?std::min<std::uint64_t>(2ull*physicalPages,MaximumArchivePages):MaximumArchivePages;
        if (OpenNaniteAsset(file,key,static_cast<unsigned>(hostPages),hierarchy,pages,error)) return true;
        NaniteHierarchy h; NanitePages p;
        if (!BuildNaniteHierarchy(vertices,indices,materials,{},h,error) || !BuildNanitePages(h,{},p,error) || !WriteNaniteAsset(file,key,h,p,error)) return false;
        return OpenNaniteAsset(file,key,static_cast<unsigned>(hostPages),hierarchy,pages,error);
    }

    struct NaniteAssetRequest::State
    {
        struct Source { std::vector<Vertex> Vertices; std::vector<unsigned> Indices; };
        std::mutex Mutex;
        std::atomic<bool> Cancelled{false};
        bool Complete=false,Consumed=false;
        std::unique_ptr<SourceReservation> Reservation;
        std::vector<Source> Sources;
        unsigned PhysicalPages=0;
        NaniteHierarchy Hierarchy;
        NanitePages Pages;
        std::string Error;
    };
    NaniteAssetQueueStatistics GetNaniteAssetQueueStatistics()
    { std::lock_guard<std::mutex> lock(AdmissionMutex); return Admission; }
    bool NaniteAssetRequest::TryStart(const std::vector<NaniteAssetSection>& sections,unsigned physicalPages,
        std::unique_ptr<NaniteAssetRequest>& output,bool& admitted,std::string& error)
    {
        admitted=false;
        if (output || sections.empty()) { error="Nanite asset admission requires source sections and an empty request"; return false; }
        std::uint64_t bytes=0;
        for (const auto& section:sections)
        {
            if (!section.Vertices || !section.Indices || section.Vertices->empty() || section.Indices->empty() || section.Indices->size()%3)
            { error="Nanite asset admission has an invalid source section"; return false; }
            const auto charge=std::uint64_t(section.Vertices->size())*sizeof(Vertex)+
                std::uint64_t(section.Indices->size()+section.Indices->size()/3)*sizeof(unsigned);
            if (charge>NaniteAssetMaximumQueuedSourceBytes-bytes)
            { error="Nanite asset source exceeds the queued-input memory budget"; return false; }
            bytes+=charge;
        }
        auto reservation=std::make_unique<SourceReservation>();
        {
            std::lock_guard<std::mutex> lock(AdmissionMutex);
            if (Admission.Requests>=NaniteAssetMaximumRequests || bytes>NaniteAssetMaximumQueuedSourceBytes-Admission.SourceBytes)
            { error.clear(); return true; }
            ++Admission.Requests; Admission.SourceBytes+=bytes; reservation->Bytes=bytes; reservation->Active=true;
        }
        auto state=std::make_shared<State>(); state->Reservation=std::move(reservation); state->PhysicalPages=physicalPages;
        for (const auto& section:sections) state->Sources.push_back({*section.Vertices,*section.Indices});
        const std::weak_ptr<State> weak=state;
        auto request=std::unique_ptr<NaniteAssetRequest>(new NaniteAssetRequest(std::move(state)));
        CookWorker().Submit([weak]
        {
            const auto state=weak.lock(); if (!state || state->Cancelled.load(std::memory_order_relaxed)) return;
            NaniteHierarchy hierarchy; NanitePages pages; NaniteAssetInput input; std::string error;
            try
            {
                std::vector<NaniteAssetSection> sections;
                for (const auto& source:state->Sources) sections.push_back({&source.Vertices,&source.Indices});
                if (AssembleNaniteAsset(sections,input,error) && !state->Cancelled.load(std::memory_order_relaxed) &&
                    !LoadOrBuildNaniteAsset(input.Vertices,input.Indices,input.Materials,state->PhysicalPages,hierarchy,pages,error) && error.empty())
                    error="Nanite asset cooking failed";
            }
            catch (const std::exception& exception) { error=exception.what(); }
            input={}; std::vector<State::Source>().swap(state->Sources);
            std::lock_guard<std::mutex> lock(state->Mutex);
            state->Reservation.reset();
            if (!state->Cancelled.load(std::memory_order_relaxed))
            { state->Hierarchy=std::move(hierarchy); state->Pages=std::move(pages); state->Error=std::move(error); state->Complete=true; }
        });
        output=std::move(request); admitted=true; error.clear(); return true;
    }
    NaniteAssetRequest::~NaniteAssetRequest() { m_State->Cancelled.store(true,std::memory_order_relaxed); }
    bool NaniteAssetRequest::Poll(bool& ready,NaniteHierarchy& hierarchy,NanitePages& pages,std::string& error)
    {
        std::lock_guard<std::mutex> lock(m_State->Mutex); ready=false;
        if (!m_State->Complete) { error.clear(); return true; }
        if (!m_State->Error.empty()) { error=m_State->Error; return false; }
        if (m_State->Consumed) { error="Nanite asset result was already consumed"; return false; }
        hierarchy=std::move(m_State->Hierarchy); pages=std::move(m_State->Pages); m_State->Consumed=true; ready=true; error.clear(); return true;
    }
}
