#include "Renderer/ShaderCache.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

namespace Evans::Renderer::ShaderCache
{
    namespace
    {
        constexpr std::uint32_t Constants[64]={
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        std::uint32_t Rotate(std::uint32_t value,unsigned count) { return value>>count | value<<(32-count); }
        constexpr char Magic[8]={'E','V','C','A','C','H','E','1'};
        constexpr std::size_t HeaderSize=80,MaximumArtifactSize=256*1024*1024;
        std::filesystem::path Path(std::string_view kind,const Digest& key)
        { return Directory()/std::string(kind)/(Hex(key)+".bin"); }
    }

    Digest Hash(const void* data,std::size_t size)
    {
        std::uint32_t state[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
        const auto* bytes=static_cast<const std::uint8_t*>(data);
        const auto padded=(size+9+63)/64*64;
        for(std::size_t block=0;block<padded;block+=64)
        {
            std::uint32_t words[64]{};
            for(unsigned i=0;i<64;++i)
            {
                const auto position=block+i;
                const auto byte=position<size?bytes[position]:position==size?0x80u:position>=padded-8?
                    unsigned((std::uint64_t(size)*8)>>((padded-1-position)*8))&255u:0u;
                words[i/4]|=byte<<((3-i%4)*8);
            }
            for(unsigned i=16;i<64;++i)
            {
                const auto a=words[i-15],b=words[i-2];
                words[i]=words[i-16]+(Rotate(a,7)^Rotate(a,18)^(a>>3))+words[i-7]+(Rotate(b,17)^Rotate(b,19)^(b>>10));
            }
            auto a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
            for(unsigned i=0;i<64;++i)
            {
                const auto t1=h+(Rotate(e,6)^Rotate(e,11)^Rotate(e,25))+((e&f)^(~e&g))+Constants[i]+words[i];
                const auto t2=(Rotate(a,2)^Rotate(a,13)^Rotate(a,22))+((a&b)^(a&c)^(b&c));
                h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
            }
            state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
        }
        Digest digest{};
        for(unsigned i=0;i<32;++i)digest[i]=std::uint8_t(state[i/4]>>((3-i%4)*8));
        return digest;
    }
    std::string Hex(const Digest& digest)
    {
        constexpr char digits[]="0123456789abcdef";std::string result;result.reserve(64);
        for(auto byte:digest){result+=digits[byte>>4];result+=digits[byte&15];}return result;
    }
    std::string FileIdentity(const std::filesystem::path& path)
    {
        std::ifstream input(path,std::ios::binary);
        if(!input)return {};
        const std::vector<char> data{std::istreambuf_iterator<char>(input),{}};
        return input.bad()?std::string{}:Hex(Hash(data.data(),data.size()));
    }
    Key& Key::Add(const void* bytes,std::size_t size)
    {
        for(unsigned i=0;i<8;++i)Bytes.push_back(std::uint8_t(std::uint64_t(size)>>(i*8)));
        if(size){const auto* first=static_cast<const std::uint8_t*>(bytes);Bytes.insert(Bytes.end(),first,first+size);}
        return *this;
    }
    std::filesystem::path Directory()
    {
        if(const auto* value=std::getenv("EVANS_SHADER_CACHE_DIR");value && *value)return std::filesystem::u8path(value);
        return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path()/"Intermediate/ShaderCache";
    }
    bool Enabled() noexcept
    { const auto* value=std::getenv("EVANS_SHADER_CACHE");return !value || std::strcmp(value,"0")!=0; }
    bool Read(std::string_view kind,const Digest& key,std::vector<std::uint8_t>& bytes) noexcept
    {
        bytes.clear();
        try
        {
            if(Enabled())
            {
                std::ifstream input(Path(kind,key),std::ios::binary|std::ios::ate);
                if(input)
                {
                    const auto fileSize=input.tellg();input.seekg(0);
                    std::array<std::uint8_t,HeaderSize> header{};
                    if(fileSize>=std::streamoff(HeaderSize) && fileSize<=std::streamoff(HeaderSize+MaximumArtifactSize) &&
                        input.read(reinterpret_cast<char*>(header.data()),header.size()))
                    {
                        std::uint64_t length=0;for(unsigned i=0;i<8;++i)length|=std::uint64_t(header[8+i])<<(8*i);
                        if(!std::memcmp(header.data(),Magic,8) && !std::memcmp(header.data()+16,key.data(),32) &&
                            length==std::uint64_t(fileSize)-HeaderSize && length>0)
                        {
                            bytes.resize(std::size_t(length));input.read(reinterpret_cast<char*>(bytes.data()),std::streamsize(length));
                            const auto checksum=Hash(bytes.data(),bytes.size());
                            if(input && !std::memcmp(header.data()+48,checksum.data(),32))
                                return true;
                        }
                    }
                }
            }
        }
        catch(...) {}
        bytes.clear();return false;
    }
    void Write(std::string_view kind,const Digest& key,const void* bytes,std::size_t size) noexcept
    {
        std::filesystem::path temporary;
        try
        {
            if(!Enabled() || !size || size>MaximumArtifactSize)return;
            const auto path=Path(kind,key);std::filesystem::create_directories(path.parent_path());
            static std::atomic<std::uint64_t> serial{0};
            temporary=path;temporary+="."+std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count())+
                "."+std::to_string(++serial)+".tmp";
            std::array<std::uint8_t,HeaderSize> header{};std::memcpy(header.data(),Magic,8);
            for(unsigned i=0;i<8;++i)header[8+i]=std::uint8_t(std::uint64_t(size)>>(8*i));
            std::memcpy(header.data()+16,key.data(),32);const auto checksum=Hash(bytes,size);
            std::memcpy(header.data()+48,checksum.data(),32);
            std::ofstream output(temporary,std::ios::binary|std::ios::trunc);
            output.write(reinterpret_cast<const char*>(header.data()),header.size());
            output.write(static_cast<const char*>(bytes),std::streamsize(size));output.close();
            if(!output)throw std::runtime_error("Cache write failed");
#if defined(_WIN32)
            if(!MoveFileExW(temporary.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING))throw std::runtime_error("Cache publication failed");
#else
            std::filesystem::rename(temporary,path);
#endif
        }
        catch(...){if(!temporary.empty()){std::error_code error;std::filesystem::remove(temporary,error);}}
    }
}
