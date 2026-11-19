#include "Editor/Resources.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <tuple>
#include <unordered_map>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <objbase.h>
#include <wincodec.h>
#include <wrl/client.h>
#undef LoadImage
#elif defined(__APPLE__)
#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>
#endif

namespace Evans::Editor
{
namespace
{

[[nodiscard]] std::string_view Trim(std::string_view value) noexcept
{
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos)
    {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

[[nodiscard]] std::string_view StripComment(std::string_view value) noexcept
{
    const auto comment = value.find('#');
    return Trim(value.substr(0, comment));
}

[[nodiscard]] std::pair<std::string_view, std::string_view> SplitCommand(std::string_view line) noexcept
{
    line = StripComment(line);
    const auto separator = line.find_first_of(" \t");
    if (separator == std::string_view::npos)
    {
        return {line, {}};
    }
    return {line.substr(0, separator), Trim(line.substr(separator + 1))};
}

[[nodiscard]] float ParseFloat(std::string_view value, std::size_t line)
{
    const std::string owned(value);
    char* end = nullptr;
    const float result = std::strtof(owned.c_str(), &end);
    if (end == owned.c_str() || *end != '\0' || !std::isfinite(result))
    {
        throw ResourceParseError(line, "Invalid floating-point value: " + owned);
    }
    return result;
}

[[nodiscard]] int ParseInteger(std::string_view value, std::size_t line)
{
    int result = 0;
    const char* begin = value.data();
    const char* end = begin + value.size();
    const auto parsed = std::from_chars(begin, end, result);
    if (parsed.ec != std::errc() || parsed.ptr != end)
    {
        throw ResourceParseError(line, "Invalid integer value: " + std::string(value));
    }
    return result;
}

[[nodiscard]] std::vector<std::string_view> Words(std::string_view value)
{
    std::vector<std::string_view> words;
    while (!value.empty())
    {
        const auto first = value.find_first_not_of(" \t");
        if (first == std::string_view::npos)
        {
            break;
        }
        value.remove_prefix(first);
        const auto end = value.find_first_of(" \t");
        words.push_back(value.substr(0, end));
        if (end == std::string_view::npos)
        {
            break;
        }
        value.remove_prefix(end);
    }
    return words;
}

[[nodiscard]] Float3 ParseFloat3(std::string_view arguments, std::size_t line)
{
    const auto words = Words(arguments);
    if (words.size() < 3)
    {
        throw ResourceParseError(line, "Expected three floating-point components");
    }
    return {ParseFloat(words[0], line), ParseFloat(words[1], line), ParseFloat(words[2], line)};
}

[[nodiscard]] Float2 ParseFloat2(std::string_view arguments, std::size_t line)
{
    const auto words = Words(arguments);
    if (words.size() < 2)
    {
        throw ResourceParseError(line, "Expected two floating-point components");
    }
    return {ParseFloat(words[0], line), ParseFloat(words[1], line)};
}

[[nodiscard]] float ParseScalar(std::string_view arguments, std::size_t line)
{
    const auto words = Words(arguments);
    if (words.empty())
    {
        throw ResourceParseError(line, "Expected a floating-point value");
    }
    return ParseFloat(words[0], line);
}

[[nodiscard]] Float3 Subtract(const Float3& left, const Float3& right) noexcept
{
    return {left.X - right.X, left.Y - right.Y, left.Z - right.Z};
}

[[nodiscard]] Float3 Cross(const Float3& left, const Float3& right) noexcept
{
    return {
        left.Y * right.Z - left.Z * right.Y,
        left.Z * right.X - left.X * right.Z,
        left.X * right.Y - left.Y * right.X};
}

[[nodiscard]] Float3 Normalize(const Float3& value) noexcept
{
    const float lengthSquared = value.X * value.X + value.Y * value.Y + value.Z * value.Z;
    if (lengthSquared <= 1.0e-20f)
    {
        return {0.0f, 1.0f, 0.0f};
    }
    const float inverseLength = 1.0f / std::sqrt(lengthSquared);
    return {value.X * inverseLength, value.Y * inverseLength, value.Z * inverseLength};
}

struct FaceCorner
{
    std::size_t Position = 0;
    std::size_t TexCoord = 0;
    std::size_t Normal = 0;
    bool HasTexCoord = false;
    bool HasNormal = false;
};

// Preserve explicit OBJ corner identity for topology-aware LOD. Equal positions
// alone do not imply a shared vertex: UV/normal seams and separate source shells
// must remain distinct. Missing normals keep the original per-triangle shading.
using CornerKey = std::tuple<std::size_t, std::size_t, std::size_t>;
using CornerCache = std::map<CornerKey, std::uint32_t>;

[[nodiscard]] std::size_t ResolveIndex(
    int parsed,
    std::size_t elementCount,
    std::size_t line,
    std::string_view semantic)
{
    if (parsed == 0)
    {
        throw ResourceParseError(line, std::string(semantic) + " index cannot be zero");
    }

    const std::int64_t index = parsed > 0
                                   ? static_cast<std::int64_t>(parsed - 1)
                                   : static_cast<std::int64_t>(elementCount) + parsed;
    if (index < 0 || index >= static_cast<std::int64_t>(elementCount))
    {
        throw ResourceParseError(line, std::string(semantic) + " index is out of range");
    }
    return static_cast<std::size_t>(index);
}

[[nodiscard]] FaceCorner ParseFaceCorner(
    std::string_view value,
    std::size_t positionCount,
    std::size_t texCoordCount,
    std::size_t normalCount,
    std::size_t line)
{
    FaceCorner result;
    const auto firstSlash = value.find('/');
    const auto secondSlash = firstSlash == std::string_view::npos
                                 ? std::string_view::npos
                                 : value.find('/', firstSlash + 1);

    const std::string_view position = value.substr(0, firstSlash);
    if (position.empty())
    {
        throw ResourceParseError(line, "A face corner is missing its position index");
    }
    result.Position = ResolveIndex(ParseInteger(position, line), positionCount, line, "Position");

    if (firstSlash != std::string_view::npos)
    {
        const std::string_view texCoord = value.substr(
            firstSlash + 1,
            secondSlash == std::string_view::npos ? secondSlash : secondSlash - firstSlash - 1);
        if (!texCoord.empty())
        {
            result.TexCoord = ResolveIndex(
                ParseInteger(texCoord, line), texCoordCount, line, "Texture coordinate");
            result.HasTexCoord = true;
        }
    }

    if (secondSlash != std::string_view::npos)
    {
        const std::string_view normal = value.substr(secondSlash + 1);
        if (!normal.empty())
        {
            result.Normal = ResolveIndex(ParseInteger(normal, line), normalCount, line, "Normal");
            result.HasNormal = true;
        }
    }
    return result;
}

void EnsureSection(MeshAsset& mesh, const std::string& materialName)
{
    if (!mesh.Sections.empty() && mesh.Sections.back().MaterialName == materialName &&
        mesh.Sections.back().FirstIndex + mesh.Sections.back().IndexCount == mesh.Indices.size())
    {
        return;
    }
    mesh.Sections.push_back(
        {materialName, static_cast<std::uint32_t>(mesh.Indices.size()), 0});
}

void AppendTriangle(
    MeshAsset& mesh,
    CornerCache& cornerCache,
    const FaceCorner& corner0,
    const FaceCorner& corner1,
    const FaceCorner& corner2,
    const std::vector<Float3>& positions,
    const std::vector<Float2>& texCoords,
    const std::vector<Float3>& normals,
    const std::string& materialName)
{
    EnsureSection(mesh, materialName);
    const Float3 flatNormal = Normalize(Cross(
        Subtract(positions[corner1.Position], positions[corner0.Position]),
        Subtract(positions[corner2.Position], positions[corner0.Position])));
    const FaceCorner corners[] = {corner0, corner1, corner2};
    for (const FaceCorner& corner : corners)
    {
        constexpr std::size_t MissingTexCoord = std::numeric_limits<std::size_t>::max();
        const CornerKey key{corner.Position, corner.HasTexCoord ? corner.TexCoord : MissingTexCoord, corner.Normal};
        if (corner.HasNormal)
        {
            const auto found = cornerCache.find(key);
            if (found != cornerCache.end()) { mesh.Indices.push_back(found->second); continue; }
        }
        Vertex vertex;
        vertex.Position = positions[corner.Position];
        vertex.TexCoord = corner.HasTexCoord ? texCoords[corner.TexCoord] : Float2{};
        vertex.Normal = corner.HasNormal ? normals[corner.Normal] : flatNormal;
        const auto index = static_cast<std::uint32_t>(mesh.Vertices.size());
        mesh.Indices.push_back(index);
        mesh.Vertices.push_back(vertex);
        if (corner.HasNormal) cornerCache.emplace(key, index);
    }
    mesh.Sections.back().IndexCount += 3;
}

[[nodiscard]] Color ToColor(const Float3& value, float alpha = 1.0f) noexcept
{
    return {value.X, value.Y, value.Z, alpha};
}

[[nodiscard]] std::string LastWord(std::string_view arguments)
{
    const auto words = Words(arguments);
    return words.empty() ? std::string() : std::string(words.back());
}

[[nodiscard]] std::vector<std::uint8_t> ReadBinaryFile(
    const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("Failed to open image resource: " + path.string());
    }
    stream.seekg(0, std::ios::end);
    const std::streamoff size = stream.tellg();
    if (size < 0)
    {
        throw std::runtime_error("Failed to query image resource size: " + path.string());
    }
    stream.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> result(static_cast<std::size_t>(size));
    if (size > 0)
    {
        stream.read(
            reinterpret_cast<char*>(result.data()),
            static_cast<std::streamsize>(result.size()));
    }
    if (!stream && size > 0)
    {
        throw std::runtime_error("Failed to read image resource: " + path.string());
    }
    return result;
}

[[nodiscard]] std::shared_ptr<const ImageAsset> LoadTga(
    const std::filesystem::path& path)
{
    const std::vector<std::uint8_t> source = ReadBinaryFile(path);
    if (source.size() < 18U)
    {
        throw std::runtime_error("TGA header is truncated: " + path.string());
    }
    const std::uint8_t idLength = source[0];
    const std::uint8_t colorMapType = source[1];
    const std::uint8_t imageType = source[2];
    const std::uint32_t width =
        static_cast<std::uint32_t>(source[12]) |
        (static_cast<std::uint32_t>(source[13]) << 8U);
    const std::uint32_t height =
        static_cast<std::uint32_t>(source[14]) |
        (static_cast<std::uint32_t>(source[15]) << 8U);
    const std::uint8_t bitsPerPixel = source[16];
    const std::uint8_t descriptor = source[17];
    if (colorMapType != 0 || imageType != 2 ||
        (bitsPerPixel != 24 && bitsPerPixel != 32) ||
        width == 0 || height == 0)
    {
        throw std::runtime_error(
            "Only uncompressed 24-bit and 32-bit true-color TGA images are supported: " +
            path.string());
    }
    if (width > std::numeric_limits<std::size_t>::max() / height)
    {
        throw std::runtime_error("TGA dimensions overflow host memory: " + path.string());
    }
    const std::size_t bytesPerPixel = bitsPerPixel / 8U;
    const std::size_t pixelCount = static_cast<std::size_t>(width) * height;
    if (pixelCount > std::numeric_limits<std::size_t>::max() / bytesPerPixel)
    {
        throw std::runtime_error("TGA dimensions overflow host memory: " + path.string());
    }
    const std::size_t offset = 18U + idLength;
    const std::size_t sourceBytes = pixelCount * bytesPerPixel;
    if (offset > source.size() || sourceBytes > source.size() - offset)
    {
        throw std::runtime_error("TGA pixel data is truncated: " + path.string());
    }

    auto result = std::make_shared<ImageAsset>();
    result->Width = width;
    result->Height = height;
    result->PixelsRGBA8.resize(pixelCount * 4U);
    const bool topOrigin = (descriptor & 0x20U) != 0;
    const bool rightOrigin = (descriptor & 0x10U) != 0;
    for (std::uint32_t sourceY = 0; sourceY < height; ++sourceY)
    {
        const std::uint32_t targetY = topOrigin ? sourceY : height - 1U - sourceY;
        for (std::uint32_t sourceX = 0; sourceX < width; ++sourceX)
        {
            const std::uint32_t targetX = rightOrigin ? width - 1U - sourceX : sourceX;
            const std::size_t sourcePixel =
                offset + (static_cast<std::size_t>(sourceY) * width + sourceX) * bytesPerPixel;
            const std::size_t targetPixel =
                (static_cast<std::size_t>(targetY) * width + targetX) * 4U;
            result->PixelsRGBA8[targetPixel] = source[sourcePixel + 2U];
            result->PixelsRGBA8[targetPixel + 1U] = source[sourcePixel + 1U];
            result->PixelsRGBA8[targetPixel + 2U] = source[sourcePixel];
            result->PixelsRGBA8[targetPixel + 3U] =
                bytesPerPixel == 4U ? source[sourcePixel + 3U] : 255U;
        }
    }
    return result;
}

#if defined(_WIN32)
[[nodiscard]] std::shared_ptr<const ImageAsset> LoadWindowsImage(
    const std::filesystem::path& path)
{
    using Microsoft::WRL::ComPtr;
    const HRESULT apartmentResult = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool uninitialize = SUCCEEDED(apartmentResult);
    if (FAILED(apartmentResult) && apartmentResult != RPC_E_CHANGED_MODE)
    {
        throw std::runtime_error("Failed to initialize COM for image decoding");
    }

    ComPtr<IWICImagingFactory> factory;
    HRESULT result = CoCreateInstance(
        CLSID_WICImagingFactory,
        nullptr,
        CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(factory.GetAddressOf()));
    ComPtr<IWICBitmapDecoder> decoder;
    ComPtr<IWICBitmapFrameDecode> frame;
    ComPtr<IWICFormatConverter> converter;
    if (SUCCEEDED(result))
    {
        result = factory->CreateDecoderFromFilename(
            path.c_str(),
            nullptr,
            GENERIC_READ,
            WICDecodeMetadataCacheOnLoad,
            decoder.GetAddressOf());
    }
    if (SUCCEEDED(result)) result = decoder->GetFrame(0, frame.GetAddressOf());
    if (SUCCEEDED(result)) result = factory->CreateFormatConverter(converter.GetAddressOf());
    if (SUCCEEDED(result))
    {
        result = converter->Initialize(
            frame.Get(),
            GUID_WICPixelFormat32bppRGBA,
            WICBitmapDitherTypeNone,
            nullptr,
            0.0,
            WICBitmapPaletteTypeCustom);
    }

    UINT width = 0;
    UINT height = 0;
    if (SUCCEEDED(result)) result = converter->GetSize(&width, &height);
    auto image = std::make_shared<ImageAsset>();
    const std::uint64_t byteCount =
        static_cast<std::uint64_t>(width) * height * 4U;
    if (SUCCEEDED(result) && width > 0 && height > 0 &&
        width <= std::numeric_limits<UINT>::max() / 4U &&
        byteCount <= std::numeric_limits<std::size_t>::max() &&
        byteCount <= std::numeric_limits<UINT>::max())
    {
        image->Width = width;
        image->Height = height;
        image->PixelsRGBA8.resize(static_cast<std::size_t>(byteCount));
        result = converter->CopyPixels(
            nullptr,
            width * 4U,
            static_cast<UINT>(byteCount),
            image->PixelsRGBA8.data());
    }
    else if (SUCCEEDED(result))
    {
        result = E_OUTOFMEMORY;
    }
    // WIC objects must be released while their COM apartment is still active.
    converter.Reset();
    frame.Reset();
    decoder.Reset();
    factory.Reset();
    if (uninitialize) CoUninitialize();
    if (FAILED(result) || !image->IsValid())
    {
        throw std::runtime_error("Failed to decode image resource: " + path.string());
    }
    return image;
}
#elif defined(__APPLE__)
[[nodiscard]] std::shared_ptr<const ImageAsset> LoadAppleImage(
    const std::filesystem::path& path)
{
    const std::string utf8 = path.string();
    CFStringRef pathString = CFStringCreateWithBytes(
        nullptr,
        reinterpret_cast<const UInt8*>(utf8.data()),
        static_cast<CFIndex>(utf8.size()),
        kCFStringEncodingUTF8,
        false);
    CFURLRef url = pathString
        ? CFURLCreateWithFileSystemPath(nullptr, pathString, kCFURLPOSIXPathStyle, false)
        : nullptr;
    CGImageSourceRef source = url ? CGImageSourceCreateWithURL(url, nullptr) : nullptr;
    CGImageRef sourceImage = source
        ? CGImageSourceCreateImageAtIndex(source, 0, nullptr)
        : nullptr;
    if (pathString) CFRelease(pathString);
    if (url) CFRelease(url);
    if (source) CFRelease(source);
    if (!sourceImage)
    {
        throw std::runtime_error("Failed to decode image resource: " + path.string());
    }

    auto image = std::make_shared<ImageAsset>();
    image->Width = static_cast<std::uint32_t>(CGImageGetWidth(sourceImage));
    image->Height = static_cast<std::uint32_t>(CGImageGetHeight(sourceImage));
    image->PixelsRGBA8.resize(
        static_cast<std::size_t>(image->Width) * image->Height * 4U);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(
        image->PixelsRGBA8.data(),
        image->Width,
        image->Height,
        8,
        image->Width * 4U,
        colorSpace,
        kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    if (context)
    {
        CGContextTranslateCTM(context, 0.0, image->Height);
        CGContextScaleCTM(context, 1.0, -1.0);
        CGContextDrawImage(
            context,
            CGRectMake(0.0, 0.0, image->Width, image->Height),
            sourceImage);
    }
    if (context) CGContextRelease(context);
    if (colorSpace) CGColorSpaceRelease(colorSpace);
    CGImageRelease(sourceImage);
    if (!context || !image->IsValid())
    {
        throw std::runtime_error("Failed to convert image resource: " + path.string());
    }
    return image;
}
#endif

} // namespace

ResourceParseError::ResourceParseError(std::size_t line, const std::string& message)
    : std::runtime_error("Resource parse error at line " + std::to_string(line) + ": " + message),
      Line_(line)
{
}

ObjDocument ResourceParser::ParseObj(std::string_view source)
{
    ObjDocument document;
    std::vector<Float3> positions;
    std::vector<Float2> texCoords;
    std::vector<Float3> normals;
    document.Meshes.push_back({"Default"});
    MeshAsset* currentMesh = &document.Meshes.back();
    std::string currentMaterial;
    CornerCache cornerCache;

    std::istringstream stream{std::string(source)};
    std::string ownedLine;
    std::size_t lineNumber = 0;
    while (std::getline(stream, ownedLine))
    {
        ++lineNumber;
        const auto [command, arguments] = SplitCommand(ownedLine);
        if (command.empty())
        {
            continue;
        }

        if (command == "v")
        {
            positions.push_back(ParseFloat3(arguments, lineNumber));
        }
        else if (command == "vt")
        {
            texCoords.push_back(ParseFloat2(arguments, lineNumber));
        }
        else if (command == "vn")
        {
            normals.push_back(Normalize(ParseFloat3(arguments, lineNumber)));
        }
        else if (command == "o" || command == "g")
        {
            cornerCache.clear();
            const std::string name = arguments.empty() ? "Unnamed" : std::string(arguments);
            if (currentMesh->Indices.empty())
            {
                currentMesh->Name = name;
            }
            else
            {
                document.Meshes.push_back({name});
                currentMesh = &document.Meshes.back();
            }
        }
        else if (command == "usemtl")
        {
            if (currentMaterial != arguments) cornerCache.clear();
            currentMaterial = std::string(arguments);
        }
        else if (command == "mtllib")
        {
            for (const std::string_view library : Words(arguments))
            {
                document.MaterialLibraries.emplace_back(library);
            }
        }
        else if (command == "f")
        {
            const auto faceWords = Words(arguments);
            if (faceWords.size() < 3)
            {
                throw ResourceParseError(lineNumber, "A face must contain at least three corners");
            }

            std::vector<FaceCorner> corners;
            corners.reserve(faceWords.size());
            for (const std::string_view faceWord : faceWords)
            {
                corners.push_back(ParseFaceCorner(
                    faceWord,
                    positions.size(),
                    texCoords.size(),
                    normals.size(),
                    lineNumber));
            }

            for (std::size_t index = 1; index + 1 < corners.size(); ++index)
            {
                AppendTriangle(
                    *currentMesh,
                    cornerCache,
                    corners[0],
                    corners[index],
                    corners[index + 1],
                    positions,
                    texCoords,
                    normals,
                    currentMaterial);
            }
        }
        else if (command != "s" && command != "l" && command != "p")
        {
            document.Warnings.push_back(
                "Line " + std::to_string(lineNumber) + ": ignored OBJ command " + std::string(command));
        }
    }

    document.Meshes.erase(
        std::remove_if(
            document.Meshes.begin(),
            document.Meshes.end(),
            [](const MeshAsset& mesh) { return mesh.Indices.empty(); }),
        document.Meshes.end());
    return document;
}

std::vector<Material> ResourceParser::ParseMtl(std::string_view source)
{
    std::vector<Material> materials;
    Material* current = nullptr;
    std::istringstream stream{std::string(source)};
    std::string ownedLine;
    std::size_t lineNumber = 0;
    while (std::getline(stream, ownedLine))
    {
        ++lineNumber;
        const auto [command, arguments] = SplitCommand(ownedLine);
        if (command.empty())
        {
            continue;
        }
        if (command == "newmtl")
        {
            if (arguments.empty())
            {
                throw ResourceParseError(lineNumber, "A material must have a name");
            }
            materials.push_back({});
            current = &materials.back();
            current->Name = std::string(arguments);
            continue;
        }
        if (current == nullptr)
        {
            continue;
        }

        if (command == "Ka")
        {
            current->Ambient = ToColor(ParseFloat3(arguments, lineNumber));
        }
        else if (command == "Kd")
        {
            current->Diffuse = ToColor(ParseFloat3(arguments, lineNumber), current->Opacity);
        }
        else if (command == "Ks")
        {
            current->Specular = ToColor(ParseFloat3(arguments, lineNumber));
        }
        else if (command == "Ke")
        {
            current->Emissive = ToColor(ParseFloat3(arguments, lineNumber));
        }
        else if (command == "Ns")
        {
            current->Shininess = ParseScalar(arguments, lineNumber);
        }
        else if (command == "d")
        {
            current->Opacity = ParseScalar(arguments, lineNumber);
            current->Diffuse.A = current->Opacity;
        }
        else if (command == "Tr")
        {
            current->Opacity = 1.0f - ParseScalar(arguments, lineNumber);
            current->Diffuse.A = current->Opacity;
        }
        else if (command == "Pr") current->Roughness = ParseScalar(arguments, lineNumber);
        else if (command == "Pm") current->Metallic = ParseScalar(arguments, lineNumber);
        else if (command == "map_MetallicRoughness") current->MetallicRoughnessTexture = LastWord(arguments);
        else if (command == "map_Kd")
        {
            current->DiffuseTexture = LastWord(arguments);
        }
        else if (command == "map_Bump" || command == "bump" ||
                 command == "norm" || command == "map_Disp")
        {
            current->NormalTexture = LastWord(arguments);
        }
        else if (command == "map_d")
        {
            current->OpacityTexture = LastWord(arguments);
        }
    }
    return materials;
}

std::string FileResourceLoader::ReadTextFile(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("Failed to open resource: " + path.string());
    }
    stream.seekg(0, std::ios::end);
    const std::streamoff size = stream.tellg();
    if (size < 0)
    {
        throw std::runtime_error("Failed to query resource size: " + path.string());
    }
    stream.seekg(0, std::ios::beg);
    std::string result(static_cast<std::size_t>(size), '\0');
    if (size > 0)
    {
        stream.read(result.data(), size);
    }
    if (!stream && size > 0)
    {
        throw std::runtime_error("Failed to read resource: " + path.string());
    }
    return result;
}

std::shared_ptr<const ImageAsset> FileResourceLoader::LoadImage(
    const std::filesystem::path& path)
{
    std::string extension = path.extension().string();
    std::transform(
        extension.begin(),
        extension.end(),
        extension.begin(),
        [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    if (extension == ".tga")
    {
        return LoadTga(path);
    }
#if defined(_WIN32)
    return LoadWindowsImage(path);
#elif defined(__APPLE__)
    return LoadAppleImage(path);
#else
    throw std::runtime_error(
        "This platform has no decoder for image format " + extension + ": " + path.string());
#endif
}

std::vector<Material> FileResourceLoader::LoadMaterialLibrary(const std::filesystem::path& path)
{
    auto materials = ResourceParser::ParseMtl(ReadTextFile(path));
    std::unordered_map<std::string, std::shared_ptr<const ImageAsset>> cache;
    const auto load = [&](std::string& name, std::shared_ptr<const ImageAsset>& image)
    {
        if (name.empty()) return;
        name = (path.parent_path() / name).lexically_normal().string();
        auto& cached = cache[name];
        if (!cached) cached = LoadImage(name);
        image = cached;
    };
    for (auto& material : materials)
    {
        load(material.DiffuseTexture, material.DiffuseImage);
        load(material.NormalTexture, material.NormalImage);
        load(material.OpacityTexture, material.OpacityImage);
        load(material.MetallicRoughnessTexture, material.MetallicRoughnessImage);
    }
    return materials;
}

LoadedModel FileResourceLoader::LoadObj(const std::filesystem::path& path)
{
    LoadedModel model;
    model.Geometry = ResourceParser::ParseObj(ReadTextFile(path));
    std::unordered_map<std::string, std::size_t> materialIndices;
    for (const std::string& library : model.Geometry.MaterialLibraries)
    {
        auto loadedMaterials = LoadMaterialLibrary(path.parent_path() / library);
        for (Material& material : loadedMaterials)
        {
            const auto existing = materialIndices.find(material.Name);
            if (existing == materialIndices.end())
            {
                materialIndices.emplace(material.Name, model.Materials.size());
                model.Materials.push_back(std::move(material));
            }
            else
            {
                model.Materials[existing->second] = std::move(material);
            }
        }
    }
    return model;
}

} // namespace Evans::Editor
