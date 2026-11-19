#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace Evans::Renderer::ShaderCache
{
    using Digest=std::array<std::uint8_t,32>;
    Digest Hash(const void* data,std::size_t size);
    std::string Hex(const Digest& digest);
    std::string FileIdentity(const std::filesystem::path& path);

    // Length-delimited inputs prevent ambiguous keys. Never hash native pointers
    // or structure padding: serialize semantic fields and pointed-to contents.
    class Key
    {
    public:
        explicit Key(std::string_view version) { Add(version); }
        Key& Add(const void* bytes,std::size_t size);
        Key& Add(std::string_view text) { return Add(text.data(),text.size()); }
        template<class T> Key& Value(T value)
        {
            static_assert(std::is_arithmetic_v<T> || std::is_enum_v<T>);
            return Add(&value,sizeof(value));
        }
        Digest Finish() const { return Hash(Bytes.data(),Bytes.size()); }
    private:
        std::vector<std::uint8_t> Bytes;
    };

    // Cache failures are optional optimization failures, never render failures.
    // EVANS_SHADER_CACHE_DIR overrides the ignored Intermediate/ShaderCache root.
    // EVANS_SHADER_CACHE=0 disables reads and writes for comparison or diagnosis.
    std::filesystem::path Directory();
    bool Enabled() noexcept;
    bool Read(std::string_view kind,const Digest& key,std::vector<std::uint8_t>& bytes) noexcept;
    void Write(std::string_view kind,const Digest& key,const void* bytes,std::size_t size) noexcept;
}
