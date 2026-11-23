#include "Renderer/Neural/GaussianScene.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace Evans::Neural
{
namespace
{
constexpr float ShZero=0.2820947917738781f;
constexpr const char* Extension="KHR_gaussian_splatting";

float Sigmoid(float value) { return 1.0f/(1.0f+std::exp(-std::clamp(value,-30.0f,30.0f))); }
float Logit(float value)
{
    value=std::clamp(value,1.0e-6f,1.0f-1.0e-6f);
    return std::log(value/(1.0f-value));
}
float LinearSrgb(float value)
{
    return value<=0.04045f?value/12.92f:std::pow((value+0.055f)/1.055f,2.4f);
}
void Append32(std::vector<std::uint8_t>& bytes,std::uint32_t value)
{
    for(int shift=0;shift<32;shift+=8) bytes.push_back(std::uint8_t(value>>shift));
}
void AppendFloat(std::vector<std::uint8_t>& bytes,float value)
{
    if(!std::isfinite(value)) throw std::runtime_error("Non-finite Gaussian glTF attribute");
    std::uint32_t bits=0;std::memcpy(&bits,&value,sizeof(bits));Append32(bytes,bits);
}
std::uint32_t Read32(const std::vector<std::uint8_t>& bytes,std::size_t offset)
{
    if(offset>bytes.size() || bytes.size()-offset<4) throw std::runtime_error("Truncated Gaussian GLB");
    return std::uint32_t(bytes[offset])|(std::uint32_t(bytes[offset+1])<<8)|
        (std::uint32_t(bytes[offset+2])<<16)|(std::uint32_t(bytes[offset+3])<<24);
}
float ReadFloat(const std::vector<std::uint8_t>& bytes,std::size_t offset)
{
    const auto bits=Read32(bytes,offset);float value=0;std::memcpy(&value,&bits,sizeof(value));
    if(!std::isfinite(value)) throw std::runtime_error("Non-finite Gaussian glTF attribute");
    return value;
}
std::size_t Index(const Json& value,std::size_t limit,const char* description)
{
    const int index=value.integer();
    if(index<0 || std::size_t(index)>=limit) throw std::runtime_error(std::string("Invalid glTF ")+description);
    return std::size_t(index);
}
std::size_t Size(const Json& object,const char* name,std::size_t fallback=0)
{
    if(!object.contains(name)) return fallback;
    const double value=object.at(name).number();
    if(!std::isfinite(value) || value<0 || value>double(std::numeric_limits<std::uint32_t>::max()) || std::floor(value)!=value)
        throw std::runtime_error(std::string("Invalid glTF ")+name);
    return std::size_t(value);
}
bool IdentityNode(const Json& node)
{
    const auto matches=[](const Json& values,const std::vector<double>& identity) {
        if(values.array().size()!=identity.size()) return false;
        for(std::size_t i=0;i<identity.size();++i)
            if(std::abs(values.at(i).number()-identity[i])>1.0e-6) return false;
        return true;
    };
    if(node.contains("matrix") && !matches(node.at("matrix"),{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1})) return false;
    if(node.contains("translation") && !matches(node.at("translation"),{0,0,0})) return false;
    if(node.contains("rotation") && !matches(node.at("rotation"),{0,0,0,1})) return false;
    if(node.contains("scale") && !matches(node.at("scale"),{1,1,1})) return false;
    return true;
}
struct BufferBuilder
{
    std::vector<std::uint8_t> Bytes;
    Json::Array Views,Accessors;
    int Add(const std::vector<float>& values,int components,const char* type,const Json::Array& minimum={},const Json::Array& maximum={})
    {
        if(values.empty() || values.size()%std::size_t(components)!=0 || values.size()/std::size_t(components)>10000000)
            throw std::runtime_error("Invalid Gaussian glTF attribute size");
        const auto offset=Bytes.size();
        for(float value:values) AppendFloat(Bytes,value);
        Views.emplace_back(Json::Object{{"buffer",0},{"byteOffset",int(offset)},{"byteLength",int(Bytes.size()-offset)},{"target",34962}});
        Json::Object accessor{{"bufferView",int(Views.size()-1)},{"componentType",5126},
            {"count",int(values.size()/std::size_t(components))},{"type",type}};
        if(!minimum.empty()) accessor["min"]=minimum;
        if(!maximum.empty()) accessor["max"]=maximum;
        Accessors.emplace_back(std::move(accessor));
        return int(Accessors.size()-1);
    }
};

std::vector<float> Attribute(const Json& root,const std::vector<std::uint8_t>& bytes,std::size_t binOffset,
    std::size_t binLength,const Json& attributes,const char* name,const char* type,int components,std::size_t expectedCount)
{
    const auto& accessors=root.at("accessors").array();
    const auto& views=root.at("bufferViews").array();
    const auto& accessor=accessors.at(Index(attributes.at(name),accessors.size(),"accessor"));
    if(accessor.contains("sparse") || !accessor.contains("bufferView") || accessor.at("componentType").integer()!=5126 ||
        accessor.at("type").string()!=type || accessor.bool_or("normalized",false))
        throw std::runtime_error(std::string("Unsupported Gaussian glTF attribute: ")+name);
    const auto count=Size(accessor,"count");
    if(count<1 || count>10000000 || (expectedCount && count!=expectedCount))
        throw std::runtime_error(std::string("Gaussian glTF attribute count mismatch: ")+name);
    const auto& view=views.at(Index(accessor.at("bufferView"),views.size(),"buffer view"));
    if(view.at("buffer").integer()!=0) throw std::runtime_error("Gaussian GLB needs one binary buffer");
    const auto viewOffset=Size(view,"byteOffset"),viewLength=Size(view,"byteLength");
    const auto accessorOffset=Size(accessor,"byteOffset"),elementBytes=std::size_t(components)*sizeof(float);
    const auto stride=Size(view,"byteStride",elementBytes);
    if(stride<elementBytes || stride%4 || viewOffset>binLength || viewLength>binLength-viewOffset ||
        accessorOffset>viewLength || viewLength-accessorOffset<elementBytes ||
        (count-1)>(viewLength-accessorOffset-elementBytes)/stride)
        throw std::runtime_error(std::string("Gaussian glTF attribute exceeds buffer: ")+name);
    std::vector<float> result;result.reserve(count*std::size_t(components));
    for(std::size_t i=0;i<count;++i)
        for(int c=0;c<components;++c)
            result.push_back(ReadFloat(bytes,binOffset+viewOffset+accessorOffset+i*stride+std::size_t(c)*4));
    return result;
}
}

void SaveGlb(const std::filesystem::path& path,const GaussianScene& scene)
{
    if(scene.Points.empty() || scene.Points.size()>10000000 || path.extension()!=".glb")
        throw std::runtime_error("Expected a non-empty Gaussian .glb asset");
    std::vector<float> positions,scales,rotations,opacities,shZero,fallback;
    const auto count=scene.Points.size();
    positions.reserve(count*3);scales.reserve(count*3);rotations.reserve(count*4);
    opacities.reserve(count);shZero.reserve(count*3);fallback.reserve(count*4);
    std::array<float,3> low{std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max()};
    std::array<float,3> high{-low[0],-low[1],-low[2]};
    for(const auto& point:scene.Points)
    {
        const auto& p=point.Values;
        for(int c=0;c<3;++c) { positions.push_back(p[c]);low[c]=std::min(low[c],p[c]);high[c]=std::max(high[c],p[c]); }
        for(int c=3;c<6;++c) scales.push_back(std::exp(std::clamp(p[c],-9.0f,1.0f)));
        const float length=std::sqrt(p[6]*p[6]+p[7]*p[7]+p[8]*p[8]+p[9]*p[9]);
        if(!std::isfinite(length) || length<1.0e-12f) throw std::runtime_error("Invalid Gaussian rotation");
        for(int c=7;c<10;++c) rotations.push_back(p[c]/length);
        rotations.push_back(p[6]/length);
        const float opacity=Sigmoid(p[10]);opacities.push_back(opacity);
        for(int c=11;c<14;++c)
        {
            const float color=Sigmoid(p[c]);
            shZero.push_back((color-0.5f)/ShZero);
            fallback.push_back(LinearSrgb(color));
        }
        fallback.push_back(opacity);
    }
    BufferBuilder buffer;
    const Json::Array minimum{double(low[0]),double(low[1]),double(low[2])};
    const Json::Array maximum{double(high[0]),double(high[1]),double(high[2])};
    Json::Object attributes{
        {"POSITION",buffer.Add(positions,3,"VEC3",minimum,maximum)},
        {"KHR_gaussian_splatting:SCALE",buffer.Add(scales,3,"VEC3")},
        {"KHR_gaussian_splatting:ROTATION",buffer.Add(rotations,4,"VEC4")},
        {"KHR_gaussian_splatting:OPACITY",buffer.Add(opacities,1,"SCALAR")},
        {"KHR_gaussian_splatting:SH_DEGREE_0_COEF_0",buffer.Add(shZero,3,"VEC3")},
        {"COLOR_0",buffer.Add(fallback,4,"VEC4")}};
    const Json extension(Json::Object{{"kernel","ellipse"},{"colorSpace","srgb_rec709_display"},
        {"projection","perspective"},{"sortingMethod","cameraDistance"}});
    const Json root(Json::Object{
        {"asset",Json::Object{{"version","2.0"},{"generator","EvansRenderer Neural 3DGS"}}},
        {"extensionsUsed",Json::Array{Extension}},
        {"scene",0},{"scenes",Json::Array{Json::Object{{"nodes",Json::Array{0}}}}},
        {"nodes",Json::Array{Json::Object{{"mesh",0},{"name","Gaussian scene"}}}},
        {"meshes",Json::Array{Json::Object{{"primitives",Json::Array{Json::Object{
            {"attributes",std::move(attributes)},{"mode",0},
            {"extensions",Json::Object{{Extension,extension}}}}}}}}},
        {"buffers",Json::Array{Json::Object{{"byteLength",int(buffer.Bytes.size())}}}},
        {"bufferViews",std::move(buffer.Views)},{"accessors",std::move(buffer.Accessors)}});
    std::string json=root.dump();
    json.append((4-json.size()%4)%4,' ');
    buffer.Bytes.insert(buffer.Bytes.end(),(4-buffer.Bytes.size()%4)%4,0);
    const std::uint64_t total=20ull+json.size()+8ull+buffer.Bytes.size();
    if(total>std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error("Gaussian GLB exceeds 4 GiB");
    std::vector<std::uint8_t> header;
    Append32(header,0x46546C67);Append32(header,2);Append32(header,std::uint32_t(total));
    Append32(header,std::uint32_t(json.size()));Append32(header,0x4E4F534A);
    std::filesystem::create_directories(path.parent_path());
    auto staged=path;staged+=".tmp";
    {
        std::ofstream output(staged,std::ios::binary|std::ios::trunc);
        output.write(reinterpret_cast<const char*>(header.data()),std::streamsize(header.size()));
        output.write(json.data(),std::streamsize(json.size()));
        header.clear();Append32(header,std::uint32_t(buffer.Bytes.size()));Append32(header,0x004E4942);
        output.write(reinterpret_cast<const char*>(header.data()),std::streamsize(header.size()));
        output.write(reinterpret_cast<const char*>(buffer.Bytes.data()),std::streamsize(buffer.Bytes.size()));
        if(!output) throw std::runtime_error("Cannot write Gaussian GLB: "+staged.u8string());
    }
    // Read the staged asset before replacing a model that may be in use.
    if(LoadGlb(staged).Points.size()!=count) throw std::runtime_error("Gaussian GLB round trip failed");
    std::filesystem::rename(staged,path);
}

GaussianScene LoadGlb(const std::filesystem::path& path)
{
    std::ifstream input(path,std::ios::binary);
    if(!input) throw std::runtime_error("Cannot open Gaussian GLB: "+path.u8string());
    const std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>()};
    if(bytes.size()<28 || bytes.size()>1024ull*1024*1024 || Read32(bytes,0)!=0x46546C67 ||
        Read32(bytes,4)!=2 || Read32(bytes,8)!=bytes.size())
        throw std::runtime_error("Invalid Gaussian GLB header");
    const auto jsonLength=std::size_t(Read32(bytes,12));
    if(Read32(bytes,16)!=0x4E4F534A || jsonLength%4 || jsonLength>bytes.size()-28)
        throw std::runtime_error("Invalid Gaussian GLB JSON chunk");
    const auto binHeader=20+jsonLength,binLength=std::size_t(Read32(bytes,binHeader));
    if(Read32(bytes,binHeader+4)!=0x004E4942 || binLength%4 || binLength!=bytes.size()-binHeader-8)
        throw std::runtime_error("Invalid Gaussian GLB binary chunk");
    const auto root=Json::parse(std::string(reinterpret_cast<const char*>(bytes.data()+20),jsonLength));
    if(root.at("asset").at("version").string()!="2.0" || root.at("buffers").array().size()!=1 ||
        root.at("buffers").at(0).contains("uri") || Size(root.at("buffers").at(0),"byteLength")>binLength)
        throw std::runtime_error("Unsupported Gaussian GLB buffer");
    const auto bufferLength=Size(root.at("buffers").at(0),"byteLength");
    const auto& nodes=root.at("nodes").array();
    const auto& meshes=root.at("meshes").array();
    const auto& sceneNodes=root.at("scenes").at(Index(Json(root.int_or("scene",0)),root.at("scenes").array().size(),"scene")).at("nodes").array();
    GaussianScene result;result.SortByCameraDistance=true;result.SupportCutoff=9;
    for(const auto& nodeId:sceneNodes)
    {
        const auto& node=nodes.at(Index(nodeId,nodes.size(),"node"));
        if(!IdentityNode(node)) throw std::runtime_error("Gaussian GLB node transforms are not supported");
        if(!node.contains("mesh")) continue;
        const auto& primitiveList=meshes.at(Index(node.at("mesh"),meshes.size(),"mesh")).at("primitives").array();
        for(const auto& primitive:primitiveList)
        {
            if(!primitive.contains("extensions") || !primitive.at("extensions").contains(Extension)) continue;
            const auto& extension=primitive.at("extensions").at(Extension);
            if(primitive.int_or("mode",4)!=0 || extension.at("kernel").string()!="ellipse" ||
                extension.at("colorSpace").string()!="srgb_rec709_display" ||
                extension.string_or("projection","perspective")!="perspective" ||
                extension.string_or("sortingMethod","cameraDistance")!="cameraDistance")
                throw std::runtime_error("Unsupported Gaussian glTF primitive");
            const auto& attributes=primitive.at("attributes");
            for(const auto& entry:attributes.object())
                if(entry.first.rfind("KHR_gaussian_splatting:SH_DEGREE_",0)==0 &&
                    entry.first!="KHR_gaussian_splatting:SH_DEGREE_0_COEF_0")
                    throw std::runtime_error("Gaussian glTF higher SH degrees are not supported");
            const auto offset=binHeader+8;
            const auto position=Attribute(root,bytes,offset,bufferLength,attributes,"POSITION","VEC3",3,0);
            const auto count=position.size()/3;
            const auto scale=Attribute(root,bytes,offset,bufferLength,attributes,"KHR_gaussian_splatting:SCALE","VEC3",3,count);
            const auto rotation=Attribute(root,bytes,offset,bufferLength,attributes,"KHR_gaussian_splatting:ROTATION","VEC4",4,count);
            const auto opacity=Attribute(root,bytes,offset,bufferLength,attributes,"KHR_gaussian_splatting:OPACITY","SCALAR",1,count);
            const auto sh=Attribute(root,bytes,offset,bufferLength,attributes,"KHR_gaussian_splatting:SH_DEGREE_0_COEF_0","VEC3",3,count);
            if(result.Points.size()+count>10000000) throw std::runtime_error("Gaussian GLB has too many splats");
            for(std::size_t i=0;i<count;++i)
            {
                Gaussian point;auto& p=point.Values;
                for(int c=0;c<3;++c) {
                    const float radius=scale[i*3+c];
                    if(radius<=0) throw std::runtime_error("Gaussian glTF scale must be positive");
                    p[c]=position[i*3+c];p[3+c]=std::log(radius);
                }
                const float qx=rotation[i*4],qy=rotation[i*4+1],qz=rotation[i*4+2],qw=rotation[i*4+3];
                const float length=std::sqrt(qx*qx+qy*qy+qz*qz+qw*qw);
                if(length<0.999f || length>1.001f) throw std::runtime_error("Gaussian glTF rotation must be a unit quaternion");
                p[6]=qw/length;p[7]=qx/length;p[8]=qy/length;p[9]=qz/length;
                if(opacity[i]<0 || opacity[i]>1) throw std::runtime_error("Gaussian glTF opacity is outside [0,1]");
                p[10]=Logit(opacity[i]);
                for(int c=0;c<3;++c) p[11+c]=Logit(std::clamp(0.5f+ShZero*sh[i*3+c],0.0f,1.0f));
                result.Points.push_back(point);
            }
        }
    }
    if(result.Points.empty()) throw std::runtime_error("Gaussian GLB has no supported splat primitive");
    return result;
}
}
