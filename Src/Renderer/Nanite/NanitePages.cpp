#include "Renderer/Nanite/NanitePages.h"
#include "Renderer/Nanite/NaniteAsset.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <numeric>
#include <set>
#include <tuple>

namespace Evans::Renderer
{
    namespace
    {
        constexpr unsigned Corners=3,BitsPerWord=32,MaximumTriangles=256,MaximumVertices=MaximumTriangles*Corners;
        constexpr int MinimumExponent=-126,MaximumExponent=127;
        constexpr std::uint64_t HashSeed=14695981039346656037ull,HashPrime=1099511628211ull;
        using Channels=std::array<std::uint32_t,NaniteVertexChannels>;
        static_assert(sizeof(Vertex)==sizeof(Channels),"Nanite vertex channels must match the canonical vertex layout");
        static_assert(offsetof(Vertex,Position)==0 && offsetof(Vertex,Normal)==3*sizeof(float) &&
            offsetof(Vertex,Color)==6*sizeof(float) && offsetof(Vertex,TexCoord)==10*sizeof(float),"Nanite channel order must match the native vertex ABI");
        static_assert(sizeof(NanitePage)==NanitePageBytes,"Nanite physical pages require a fixed stride");
        static_assert(NanitePageHeaderWords+NaniteClusterHeaderWords+MaximumVertices*NaniteVertexChannels+
            MaximumTriangles*Corners<NanitePageWords,"A worst-case cluster must fit in one page");
        unsigned Word(NanitePageWord value) { return unsigned(value); }
        unsigned Word(NaniteClusterWord value) { return unsigned(value); }
        std::uint32_t SignedBits(std::int32_t value) { std::uint32_t bits; std::memcpy(&bits,&value,sizeof(bits)); return bits; }
        std::int32_t SignedValue(std::uint32_t bits) { std::int32_t value; std::memcpy(&value,&bits,sizeof(value)); return value; }
        std::uint32_t Mask(unsigned width) { return width==BitsPerWord?~std::uint32_t(0):(std::uint32_t(1)<<width)-1; }
        unsigned Width(std::uint32_t value) { unsigned width=0; for (;value;value>>=1) ++width; return width; }
        std::size_t WordsForBits(std::size_t bits) { return (bits+BitsPerWord-1)/BitsPerWord; }
        void Put(std::uint32_t* words,std::size_t bit,unsigned width,std::uint32_t value)
        {
            if (!width) return;
            const auto at=bit/BitsPerWord,shift=bit%BitsPerWord;
            words[at]|=value<<shift;
            if (shift+width>BitsPerWord) words[at+1]|=value>>(BitsPerWord-shift);
        }
        std::uint32_t Get(const std::uint32_t* words,std::size_t bit,unsigned width)
        {
            if (!width) return 0;
            const auto at=bit/BitsPerWord,shift=bit%BitsPerWord;
            std::uint64_t value=words[at];
            if (shift+width>BitsPerWord) value|=std::uint64_t(words[at+1])<<BitsPerWord;
            return std::uint32_t(value>>shift)&Mask(width);
        }
        bool Quantize(const Vertex& vertex,int exponent,Channels& values,Vertex& decoded,double& displacement)
        {
            std::memcpy(values.data(),&vertex,sizeof(vertex)); decoded=vertex;
            const float position[]{vertex.Position.X,vertex.Position.Y,vertex.Position.Z}; float reconstructed[Corners]{};
            for (unsigned axis=0;axis<Corners;++axis)
            {
                const double coordinate=std::round(std::ldexp(double(position[axis]),-exponent));
                if (!std::isfinite(coordinate) || coordinate<std::numeric_limits<std::int32_t>::min() || coordinate>std::numeric_limits<std::int32_t>::max()) return false;
                const auto integer=std::int32_t(coordinate);
                values[axis]=SignedBits(integer); reconstructed[axis]=std::ldexp(float(integer),exponent);
                if (!std::isfinite(reconstructed[axis])) return false;
            }
            decoded.Position={reconstructed[0],reconstructed[1],reconstructed[2]};
            displacement=std::hypot(double(position[0])-reconstructed[0],double(position[1])-reconstructed[1],double(position[2])-reconstructed[2]);
            if (displacement>0) displacement=std::nextafter(displacement,std::numeric_limits<double>::infinity());
            return true;
        }
        std::uint64_t Signature(const NanitePages& data)
        {
            std::uint64_t hash=HashSeed; const auto mix=[&](std::uint64_t value) { hash^=value; hash*=HashPrime; };
            mix(data.HierarchySource); mix(SignedBits(data.PositionExponent)); std::uint64_t bits=0;
            std::memcpy(&bits,&data.MaximumQuantizationDisplacement,sizeof(bits)); mix(bits); mix(data.PayloadBytes); mix(data.UncompressedBytes);
            mix(data.Pages.size()); for (const auto& page:data.Pages) for (auto value:page.Words) mix(value);
            mix(data.Clusters.size()); for (const auto& c:data.Clusters) { mix(c.Page); mix(c.Word); mix(c.FirstSourceVertex); mix(c.VertexCount); }
            mix(data.Groups.size()); for (const auto& g:data.Groups) { mix(g.First); mix(g.Count); }
            mix(data.PinnedPages.size()); for (auto page:data.PinnedPages) mix(page);
            mix(data.SourceVertices.size()); for (auto vertex:data.SourceVertices) mix(vertex);
            return hash;
        }
        bool Header(const NanitePages& data,unsigned id,const NanitePage& page,std::string& error)
        {
            if (id>=data.PageCount() || data.PositionExponent<MinimumExponent || data.PositionExponent>MaximumExponent)
            { error="Nanite page ID or position exponent is invalid"; return false; }
            const auto& p=page.Words;
            if (p[Word(NanitePageWord::Magic)]!=NanitePageMagic || p[Word(NanitePageWord::Version)]!=NanitePageVersion ||
                p[Word(NanitePageWord::UsedWords)]<NanitePageHeaderWords+NaniteClusterHeaderWords || p[Word(NanitePageWord::UsedWords)]>NanitePageWords ||
                !p[Word(NanitePageWord::ClusterCount)] || (p[Word(NanitePageWord::Flags)]&~NanitePagePinned) ||
                p[Word(NanitePageWord::PrecisionExponent)]!=SignedBits(data.PositionExponent) || p[Word(NanitePageWord::Page)]!=id || p[Word(NanitePageWord::Reserved)])
            { error="Nanite page header is malformed"; return false; }
            return true;
        }
        bool Encode(const NaniteHierarchy& hierarchy,unsigned id,int exponent,std::vector<std::uint32_t>& words,
            std::vector<std::uint32_t>& sources,double& errorBound,std::string& error)
        {
            const auto& cluster=hierarchy.Clusters[id]; std::vector<unsigned> indices; std::map<unsigned,unsigned> local;
            for (auto index:cluster.Indices)
            {
                const auto inserted=local.emplace(index,unsigned(sources.size()));
                if (inserted.second) sources.push_back(index);
                indices.push_back(inserted.first->second);
            }
            std::vector<Channels> vertices(sources.size());
            Channels bases{},widths{},offsets{};
            for (unsigned at=0;at<sources.size();++at)
            {
                Vertex decoded; double displacement=0;
                if (!Quantize(hierarchy.Vertices[sources[at]],exponent,vertices[at],decoded,displacement))
                { error="Nanite position quantization exceeds the finite coordinate range"; return false; }
                errorBound=std::max(errorBound,displacement);
                if (!at) bases=vertices[at];
                for (unsigned axis=0;axis<Corners;++axis)
                    if (SignedValue(vertices[at][axis])<SignedValue(bases[axis])) bases[axis]=vertices[at][axis];
            }
            for (auto& vertex:vertices) for (unsigned channel=0;channel<NaniteVertexChannels;++channel)
            {
                vertex[channel]=channel<Corners?vertex[channel]-bases[channel]:vertex[channel]^bases[channel];
                widths[channel]=std::max(widths[channel],Width(vertex[channel]));
            }
            unsigned bits=0;
            for (unsigned channel=0;channel<NaniteVertexChannels;++channel)
            { offsets[channel]=bits; bits+=unsigned(sources.size())*widths[channel]; }
            const auto indexOffset=bits,indexWidth=Width(unsigned(sources.size()-1)); bits+=unsigned(indices.size())*indexWidth;
            words.assign(NaniteClusterHeaderWords+WordsForBits(bits),0);
            if (words.size()+NanitePageHeaderWords>NanitePageWords) { error="Nanite cluster exceeds a physical page"; return false; }
            words[Word(NaniteClusterWord::Cluster)]=id; words[Word(NaniteClusterWord::VertexCount)]=unsigned(sources.size());
            words[Word(NaniteClusterWord::IndexCount)]=unsigned(indices.size()); words[Word(NaniteClusterWord::Material)]=cluster.Material;
            words[Word(NaniteClusterWord::GeneratingGroup)]=cluster.GeneratingGroup; words[Word(NaniteClusterWord::ParentGroup)]=cluster.ParentGroup;
            words[Word(NaniteClusterWord::WordCount)]=unsigned(words.size()); words[Word(NaniteClusterWord::IndexBitOffset)]=indexOffset;
            std::copy(bases.begin(),bases.end(),words.begin()+Word(NaniteClusterWord::Bases));
            std::copy(widths.begin(),widths.end(),words.begin()+Word(NaniteClusterWord::Widths));
            std::copy(offsets.begin(),offsets.end(),words.begin()+Word(NaniteClusterWord::Offsets));
            auto* payload=words.data()+NaniteClusterHeaderWords;
            for (unsigned channel=0;channel<NaniteVertexChannels;++channel) for (unsigned vertex=0;vertex<vertices.size();++vertex)
                Put(payload,offsets[channel]+vertex*widths[channel],widths[channel],vertices[vertex][channel]);
            for (unsigned at=0;at<indices.size();++at) Put(payload,indexOffset+at*indexWidth,indexWidth,indices[at]);
            return true;
        }
    }

    bool DecodeNaniteCluster(const NanitePages& data,std::uint32_t cluster,NaniteDecodedCluster& output,std::string& error)
    {
        const auto fail=[&](const char* message) { error=message; return false; };
        if (cluster>=data.Clusters.size()) return fail("Nanite cluster ID is invalid");
        const auto& location=data.Clusters[cluster];
        NanitePage storage;
        if (!ReadNanitePage(data,location.Page,storage,error) || !Header(data,location.Page,storage,error)) return false;
        const auto& page=storage.Words; const auto used=page[Word(NanitePageWord::UsedWords)];
        if (location.Word<NanitePageHeaderWords || location.Word>used-NaniteClusterHeaderWords) return fail("Nanite cluster header is out of bounds");
        const auto* header=page.data()+location.Word; const auto vertices=header[Word(NaniteClusterWord::VertexCount)],indices=header[Word(NaniteClusterWord::IndexCount)];
        const auto words=header[Word(NaniteClusterWord::WordCount)];
        if (header[Word(NaniteClusterWord::Cluster)]!=cluster || !vertices || vertices>MaximumVertices || !indices || indices%Corners || indices>MaximumTriangles*Corners ||
            words<NaniteClusterHeaderWords || words>used-location.Word || vertices!=location.VertexCount ||
            std::uint64_t(location.FirstSourceVertex)+vertices>data.SourceVertices.size()) return fail("Nanite cluster dimensions are invalid");
        const auto* bases=header+Word(NaniteClusterWord::Bases); const auto* widths=header+Word(NaniteClusterWord::Widths);
        const auto* offsets=header+Word(NaniteClusterWord::Offsets); unsigned bits=0;
        for (unsigned channel=0;channel<NaniteVertexChannels;++channel)
        {
            if (widths[channel]>BitsPerWord || offsets[channel]!=bits) return fail("Nanite vertex bit fields overlap or exceed their width");
            bits+=vertices*widths[channel];
        }
        if (header[Word(NaniteClusterWord::IndexBitOffset)]!=bits) return fail("Nanite index bit offset is invalid");
        const auto indexOffset=bits,indexWidth=Width(vertices-1); bits+=indices*indexWidth;
        if (NaniteClusterHeaderWords+WordsForBits(bits)!=words) return fail("Nanite packed payload length is invalid");
        const auto* payload=header+NaniteClusterHeaderWords;
        if (bits%BitsPerWord && (payload[WordsForBits(bits)-1]&~Mask(bits%BitsPerWord))) return fail("Nanite packed tail is not canonical");
        NaniteDecodedCluster decoded; decoded.Material=header[Word(NaniteClusterWord::Material)]; decoded.Vertices.resize(vertices); decoded.Indices.resize(indices);
        decoded.SourceVertices.assign(data.SourceVertices.begin()+location.FirstSourceVertex,data.SourceVertices.begin()+location.FirstSourceVertex+vertices);
        for (unsigned vertex=0;vertex<vertices;++vertex)
        {
            Channels values{};
            for (unsigned channel=0;channel<NaniteVertexChannels;++channel)
            {
                const auto value=Get(payload,offsets[channel]+vertex*widths[channel],widths[channel]);
                if (channel<Corners)
                {
                    const auto coordinate=std::int64_t(SignedValue(bases[channel]))+value;
                    if (coordinate>std::numeric_limits<std::int32_t>::max()) return fail("Nanite quantized position overflows its signed coordinate");
                    const float position=std::ldexp(float(coordinate),data.PositionExponent); std::memcpy(&values[channel],&position,sizeof(position));
                }
                else values[channel]=bases[channel]^value;
                float number; std::memcpy(&number,&values[channel],sizeof(number));
                if (!std::isfinite(number)) return fail("Nanite decoded vertex is not finite");
            }
            std::memcpy(&decoded.Vertices[vertex],values.data(),sizeof(Vertex));
        }
        for (unsigned at=0;at<indices;++at)
        {
            decoded.Indices[at]=Get(payload,indexOffset+at*indexWidth,indexWidth);
            if (decoded.Indices[at]>=vertices) return fail("Nanite packed triangle references a missing local vertex");
        }
        output=std::move(decoded); error.clear(); return true;
    }

    bool BuildNanitePages(const NaniteHierarchy& hierarchy,const NanitePageSettings& settings,NanitePages& output,std::string& error)
    {
        if (!ValidateNaniteHierarchy(hierarchy,error)) return false;
        constexpr unsigned MinimumPrecisionBits=8,MaximumPrecisionBits=30;
        if (settings.PositionPrecisionBits<MinimumPrecisionBits || settings.PositionPrecisionBits>MaximumPrecisionBits ||
            !settings.MaximumPages || settings.MaximumPages>std::numeric_limits<unsigned>::max()/NanitePageWords)
        { error="Nanite page or position precision budget is invalid"; return false; }
        NanitePages data; data.HierarchySource=hierarchy.Signature; data.Clusters.resize(hierarchy.Clusters.size()); data.Groups.resize(hierarchy.Groups.size());
        double magnitude=0;
        for (const auto& vertex:hierarchy.Vertices) for (auto value:{vertex.Position.X,vertex.Position.Y,vertex.Position.Z}) magnitude=std::max(magnitude,std::abs(double(value)));
        int exponent=0; const double fraction=std::frexp(magnitude,&exponent); if (fraction==0.5) --exponent;
        data.PositionExponent=std::max(MinimumExponent,exponent-int(settings.PositionPrecisionBits));
        const auto append=[&](unsigned id,bool pinned)
        {
            std::vector<unsigned> words,sources;
            if (!Encode(hierarchy,id,data.PositionExponent,words,sources,data.MaximumQuantizationDisplacement,error)) return false;
            const auto flags=pinned?NanitePagePinned:0;
            if (data.Pages.empty() || data.Pages.back().Words[Word(NanitePageWord::Flags)]!=flags ||
                data.Pages.back().Words[Word(NanitePageWord::UsedWords)]+words.size()>NanitePageWords)
            {
                if (data.Pages.size()>=settings.MaximumPages) { error="Nanite compressed pages exceed their storage budget"; return false; }
                const auto pageId=unsigned(data.Pages.size()); data.Pages.emplace_back(); auto& page=data.Pages.back().Words;
                page[Word(NanitePageWord::Magic)]=NanitePageMagic; page[Word(NanitePageWord::Version)]=NanitePageVersion;
                page[Word(NanitePageWord::UsedWords)]=NanitePageHeaderWords; page[Word(NanitePageWord::Flags)]=flags;
                page[Word(NanitePageWord::PrecisionExponent)]=SignedBits(data.PositionExponent); page[Word(NanitePageWord::Page)]=pageId;
                if (pinned) data.PinnedPages.push_back(pageId);
            }
            if (data.SourceVertices.size()+sources.size()>NaniteInvalid) { error="Nanite source key addressing overflow"; return false; }
            auto& page=data.Pages.back().Words; auto& used=page[Word(NanitePageWord::UsedWords)];
            data.Clusters[id]={unsigned(data.Pages.size()-1),used,unsigned(data.SourceVertices.size()),unsigned(sources.size())};
            std::copy(words.begin(),words.end(),page.begin()+used); used+=unsigned(words.size()); ++page[Word(NanitePageWord::ClusterCount)];
            data.SourceVertices.insert(data.SourceVertices.end(),sources.begin(),sources.end()); data.PayloadBytes+=words.size()*sizeof(unsigned);
            data.UncompressedBytes+=(NaniteClusterHeaderWords+hierarchy.Clusters[id].Indices.size())*sizeof(unsigned)+sources.size()*sizeof(Vertex);
            return true;
        };
        for (auto root:hierarchy.Roots) if (!append(root,true)) return false;
        std::vector<unsigned> groups(hierarchy.Groups.size()); std::iota(groups.begin(),groups.end(),0);
        const auto key=[&](unsigned id) { const auto& g=hierarchy.Groups[id]; return std::make_tuple(g.Level,g.BoundsMinimum.X,g.BoundsMinimum.Y,g.BoundsMinimum.Z,id); };
        std::sort(groups.begin(),groups.end(),[&](auto a,auto b) { return key(a)>key(b); });
        for (auto id:groups)
        {
            const auto& group=hierarchy.Groups[id];
            for (auto child:group.Children) if (!append(child,false)) return false;
            const auto first=data.Clusters[group.Children.front()].Page,last=data.Clusters[group.Children.back()].Page;
            data.Groups[id]={first,last-first+1};
        }
        data.Signature=Signature(data);
        if (!ValidateNanitePages(hierarchy,data,error)) return false;
        output=std::move(data); error.clear(); return true;
    }

    bool ValidateNanitePages(const NaniteHierarchy& hierarchy,const NanitePages& data,std::string& error)
    {
        if (data.Source) return data.Source->Matches(hierarchy,data,error);
        const auto fail=[&](const char* message) { error=message; return false; };
        if (!ValidateNaniteHierarchy(hierarchy,error)) return false;
        if (data.HierarchySource!=hierarchy.Signature || data.Pages.empty() || data.Pages.size()>std::numeric_limits<unsigned>::max()/NanitePageWords ||
            data.Clusters.size()!=hierarchy.Clusters.size() || data.Groups.size()!=hierarchy.Groups.size() ||
            !std::isfinite(data.MaximumQuantizationDisplacement) || data.MaximumQuantizationDisplacement<0)
            return fail("Nanite page dataset has incompatible source or dimensions");
        std::vector<bool> seen(data.Clusters.size()); std::vector<unsigned> ordinal(data.Clusters.size()),pinned;
        std::size_t sourceEnd=0,clusterCount=0; std::uint64_t payloadBytes=0,rawBytes=0; double displacementBound=0;
        for (unsigned id=0;id<data.Pages.size();++id)
        {
            if (!Header(data,id,data.Pages[id],error)) return false;
            const auto& page=data.Pages[id].Words; const auto used=page[Word(NanitePageWord::UsedWords)];
            const bool isPinned=(page[Word(NanitePageWord::Flags)]&NanitePagePinned)!=0;
            if (isPinned) pinned.push_back(id);
            unsigned at=NanitePageHeaderWords,count=0;
            while (at<used)
            {
                if (used-at<NaniteClusterHeaderWords) return fail("Nanite page ends inside a cluster header");
                const auto* header=page.data()+at; const auto cluster=header[Word(NaniteClusterWord::Cluster)];
                if (cluster>=data.Clusters.size() || seen[cluster]) return fail("Nanite pages repeat or invent a cluster");
                const auto& location=data.Clusters[cluster]; const auto& source=hierarchy.Clusters[cluster];
                if (location.Page!=id || location.Word!=at || location.FirstSourceVertex!=sourceEnd ||
                    (source.ParentGroup==NaniteInvalid)!=isPinned || header[Word(NaniteClusterWord::GeneratingGroup)]!=source.GeneratingGroup ||
                    header[Word(NaniteClusterWord::ParentGroup)]!=source.ParentGroup) return fail("Nanite cluster directory or group identity is inconsistent");
                NaniteDecodedCluster decoded; if (!DecodeNaniteCluster(data,cluster,decoded,error)) return false;
                if (decoded.Material!=source.Material || decoded.Indices.size()!=source.Indices.size() ||
                    std::set<unsigned>(decoded.SourceVertices.begin(),decoded.SourceVertices.end()).size()!=decoded.SourceVertices.size())
                    return fail("Nanite decoded cluster changed its topology, material or vertex identity");
                for (unsigned vertex=0;vertex<decoded.Vertices.size();++vertex)
                {
                    const auto key=decoded.SourceVertices[vertex]; if (key>=hierarchy.Vertices.size()) return fail("Nanite source vertex is out of bounds");
                    Channels values; Vertex expected; double displacement=0;
                    if (!Quantize(hierarchy.Vertices[key],data.PositionExponent,values,expected,displacement) ||
                        std::memcmp(&expected,&decoded.Vertices[vertex],sizeof(Vertex))) return fail("Nanite payload changed a quantized position or lossless attribute");
                    displacementBound=std::max(displacementBound,displacement);
                }
                for (unsigned index=0;index<decoded.Indices.size();++index)
                    if (decoded.SourceVertices[decoded.Indices[index]]!=source.Indices[index]) return fail("Nanite payload changed triangle order or winding");
                const auto words=header[Word(NaniteClusterWord::WordCount)]; payloadBytes+=words*sizeof(unsigned);
                rawBytes+=(NaniteClusterHeaderWords+source.Indices.size())*sizeof(unsigned)+decoded.Vertices.size()*sizeof(Vertex);
                sourceEnd+=decoded.Vertices.size(); seen[cluster]=true; ordinal[cluster]=unsigned(clusterCount++); ++count; at+=words;
            }
            if (at!=used || count!=page[Word(NanitePageWord::ClusterCount)] ||
                std::any_of(page.begin()+used,page.end(),[](unsigned value) { return value!=0; })) return fail("Nanite page extent, cluster count or padding is invalid");
        }
        if (clusterCount!=data.Clusters.size() || sourceEnd!=data.SourceVertices.size() || pinned!=data.PinnedPages || pinned.empty() ||
            data.MaximumQuantizationDisplacement!=displacementBound || payloadBytes!=data.PayloadBytes || rawBytes!=data.UncompressedBytes)
            return fail("Nanite page coverage, pinning, precision bound or byte counts are inconsistent");
        for (unsigned id=0;id<data.Groups.size();++id)
        {
            const auto& group=hierarchy.Groups[id]; const auto& range=data.Groups[id]; unsigned first=NaniteInvalid,last=0,minOrdinal=NaniteInvalid,maxOrdinal=0;
            for (auto child:group.Children)
            { first=std::min(first,data.Clusters[child].Page); last=std::max(last,data.Clusters[child].Page); minOrdinal=std::min(minOrdinal,ordinal[child]); maxOrdinal=std::max(maxOrdinal,ordinal[child]); }
            if (range.First!=first || range.Count!=last-first+1 || maxOrdinal-minOrdinal+1!=group.Children.size())
                return fail("Nanite group parts are not one contiguous page bundle");
        }
        if (data.Signature!=Signature(data)) return fail("Nanite page content signature is stale");
        error.clear(); return true;
    }
}
