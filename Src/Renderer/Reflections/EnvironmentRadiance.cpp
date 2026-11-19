#include "Renderer/Reflections/EnvironmentRadiance.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace Evans::Renderer
{
    namespace
    {
        // Ward, "Real Pixels", Graphics Gems II (1991). RGBE stores three
        // mantissas with one shared exponent. This importer accepts the modern
        // per-channel RLE and flat RGBE encodings, with either scan direction.
        // https://radsite.lbl.gov/radiance/refer/filefmts.pdf (Picture File Format)
        constexpr int RGBEExponentBias=128,RGBEMantissaBits=8;
        constexpr unsigned RGBEChannels=4;
        constexpr unsigned RunMarker=128,MaximumImageSide=16384,MaximumHeaderLines=128;
        constexpr unsigned MinimumCubeSide=16,MaximumCubeSide=1024;
        constexpr float MaximumHalfRadiance=65504;
        constexpr double Pi=3.14159265358979323846;

        unsigned Byte(std::istream& input)
        {
            const int value=input.get();
            if (value==std::char_traits<char>::eof()) throw std::runtime_error("Truncated RGBE environment");
            return static_cast<unsigned char>(value);
        }

        std::string Line(std::istream& input)
        {
            constexpr std::size_t MaximumLineBytes=4096;
            std::string line;
            for (std::size_t count=0;count<MaximumLineBytes;++count)
            {
                const char value=static_cast<char>(Byte(input));
                if (value=='\n') return line;
                if (value!='\r') line.push_back(value);
            }
            throw std::runtime_error("RGBE header line is too long");
        }

        struct Panorama
        {
            unsigned Width=0,Height=0;
            std::vector<Float3> Pixels;
        };

        Panorama ReadRadiance(const std::filesystem::path& path)
        {
            std::ifstream input(path,std::ios::binary);
            if (!input) throw std::runtime_error("Cannot open HDR environment: "+path.string());
            const auto signature=Line(input);
            if (signature!="#?RADIANCE" && signature!="#?RGBE") throw std::runtime_error("Expected a Radiance RGBE environment");
            bool format=false,complete=false;
            for (unsigned count=0;count<MaximumHeaderLines;++count)
            {
                const auto line=Line(input);
                if (line.empty()) { complete=true; break; }
                if (line=="FORMAT=32-bit_rle_rgbe") format=true;
                else if (line.rfind("FORMAT=",0)==0) throw std::runtime_error("Unsupported Radiance color format");
            }
            if (!format || !complete) throw std::runtime_error("Incomplete RGBE environment header");
            Panorama result;
            std::string rows,columns,extra;
            std::istringstream dimensions(Line(input));
            if (!(dimensions>>rows>>result.Height>>columns>>result.Width) || dimensions>>extra ||
                (rows!="-Y" && rows!="+Y") || (columns!="+X" && columns!="-X") ||
                !result.Width || !result.Height || result.Width>MaximumImageSide || result.Height>MaximumImageSide ||
                result.Width!=result.Height*2)
                throw std::runtime_error("Environment HDR must be a 2:1 Y-major latitude-longitude image");
            result.Pixels.resize(std::size_t(result.Width)*result.Height);
            std::vector<unsigned char> scanline(std::size_t(result.Width)*RGBEChannels);
            for (unsigned row=0;row<result.Height;++row)
            {
                std::array<unsigned,RGBEChannels> first{};
                for (auto& value:first) value=Byte(input);
                const bool channelRuns=first[0]==2 && first[1]==2 && !(first[2]&RunMarker);
                if (channelRuns)
                {
                    if ((first[2]<<RGBEMantissaBits|first[3])!=result.Width)
                        throw std::runtime_error("RGBE scanline width disagrees with its header");
                    for (unsigned channel=0;channel<RGBEChannels;++channel)
                    {
                        unsigned at=0;
                        while (at<result.Width)
                        {
                            const auto code=Byte(input),length=code>RunMarker?code-RunMarker:code;
                            if (!length || length>result.Width-at) throw std::runtime_error("Invalid RGBE scanline run");
                            if (code>RunMarker)
                            {
                                const auto value=Byte(input);
                                for (unsigned n=0;n<length;++n) scanline[(at+n)*RGBEChannels+channel]=static_cast<unsigned char>(value);
                            }
                            else for (unsigned n=0;n<length;++n) scanline[(at+n)*RGBEChannels+channel]=static_cast<unsigned char>(Byte(input));
                            at+=length;
                        }
                    }
                }
                else
                {
                    for (unsigned c=0;c<RGBEChannels;++c) scanline[c]=static_cast<unsigned char>(first[c]);
                    for (std::size_t at=RGBEChannels;at<scanline.size();++at) scanline[at]=static_cast<unsigned char>(Byte(input));
                }
                for (unsigned column=0;column<result.Width;++column)
                {
                    const auto* pixel=scanline.data()+column*RGBEChannels;
                    // Old whole-pixel repeat records are deliberately rejected.
                    if (!channelRuns && pixel[0]==1 && pixel[1]==1 && pixel[2]==1)
                        throw std::runtime_error("Legacy RGBE repeat encoding is unsupported");
                    const float scale=pixel[3]?std::ldexp(1.0f,int(pixel[3])-RGBEExponentBias-RGBEMantissaBits):0;
                    const Float3 value{pixel[0]*scale,pixel[1]*scale,pixel[2]*scale};
                    if (!std::isfinite(value.X) || !std::isfinite(value.Y) || !std::isfinite(value.Z) ||
                        std::max({value.X,value.Y,value.Z})>MaximumHalfRadiance)
                        throw std::runtime_error("Environment radiance exceeds finite RGBA16F storage");
                    const auto x=columns=="+X"?column:result.Width-1-column,y=rows=="-Y"?row:result.Height-1-row;
                    result.Pixels[std::size_t(y)*result.Width+x]=value;
                }
            }
            return result;
        }

        std::uint16_t Half(float value)
        {
            // Nonnegative finite radiance only. Nearbyint implements unbiased
            // ties-to-even rounding, including subnormals and exponent carry.
            constexpr unsigned FractionBits=10,ExponentBias=15;
            constexpr int MinimumExponent=-14,SubnormalScale=24;
            constexpr unsigned FractionScale=1u<<FractionBits;
            if (value==0) return 0;
            int exponent=0; const double fraction=std::frexp(double(value),&exponent)*2;
            --exponent;
            if (exponent<MinimumExponent) return static_cast<std::uint16_t>(std::nearbyint(std::ldexp(double(value),SubnormalScale)));
            const auto mantissa=static_cast<unsigned>(std::nearbyint((fraction-1)*FractionScale));
            return static_cast<std::uint16_t>(((exponent+ExponentBias)<<FractionBits)+mantissa);
        }

        Float3 Sample(const Panorama& source,Float3 direction)
        {
            const double u=std::atan2(direction.Z,direction.X)/(2*Pi)+0.5;
            const double v=std::acos(std::clamp(double(direction.Y),-1.0,1.0))/Pi;
            const double x=u*source.Width-0.5,y=v*source.Height-0.5;
            const int left=int(std::floor(x)),top=int(std::floor(y));
            const float tx=float(x-left),ty=float(y-top);
            auto at=[&](int column,int row) {
                column=(column%int(source.Width)+int(source.Width))%int(source.Width);
                row=std::clamp(row,0,int(source.Height)-1);
                return source.Pixels[std::size_t(row)*source.Width+column];
            };
            auto lerp=[](Float3 a,Float3 b,float t) { return Float3{a.X+(b.X-a.X)*t,a.Y+(b.Y-a.Y)*t,a.Z+(b.Z-a.Z)*t}; };
            return lerp(lerp(at(left,top),at(left+1,top),tx),lerp(at(left,top+1),at(left+1,top+1),tx),ty);
        }
    }

    Float3 EnvironmentRadiance::Texel(EnvironmentFace face,unsigned x,unsigned y) const noexcept
    {
        const auto& pixels=FacesRGBA16[static_cast<unsigned>(face)];
        const auto offset=(std::size_t(y)*Resolution+x)*4;
        const auto decode=[](std::uint16_t value)
        {
            const unsigned exponent=(value>>10)&31,fraction=value&1023;
            return exponent?std::ldexp(1.0f+float(fraction)/1024,int(exponent)-15):
                std::ldexp(float(fraction),-24);
        };
        return {decode(pixels[offset]),decode(pixels[offset+1]),decode(pixels[offset+2])};
    }

    bool ValidEnvironmentSource(const EnvironmentCaptureFrame& frame) noexcept
    {
        if (!frame.Source) return true;
        const auto side=frame.Source->Resolution;
        if (side<MinimumCubeSide || side>MaximumCubeSide || (side & (side-1)) || side!=frame.Resolution) return false;
        const auto channels=static_cast<std::size_t>(side)*side*RGBEChannels;
        for (const auto& face:frame.Source->FacesRGBA16) if (face.size()!=channels) return false;
        return true;
    }

    std::shared_ptr<const EnvironmentRadiance> LoadEnvironmentRadiance(
        const std::filesystem::path& path,std::uint32_t faceResolution,std::string& error)
    {
        try
        {
            if (faceResolution<MinimumCubeSide || faceResolution>MaximumCubeSide || (faceResolution&(faceResolution-1)))
                throw std::runtime_error("Environment cube side must be a power of two from 16 to 1024");
            const auto panorama=ReadRadiance(path);
            auto result=std::make_shared<EnvironmentRadiance>(); result->Resolution=faceResolution;
            for (std::size_t face=0;face<EnvironmentFaceCount;++face)
            {
                auto& pixels=result->FacesRGBA16[face]; pixels.reserve(std::size_t(faceResolution)*faceResolution*RGBEChannels);
                for (unsigned y=0;y<faceResolution;++y) for (unsigned x=0;x<faceResolution;++x)
                {
                    Float3 direction;
                    if (!EnvironmentFaceDirection(static_cast<EnvironmentFace>(face),{(x+0.5f)/faceResolution,(y+0.5f)/faceResolution},direction))
                        throw std::runtime_error("Invalid environment cube direction");
                    const auto value=Sample(panorama,direction);
                    pixels.insert(pixels.end(),{Half(value.X),Half(value.Y),Half(value.Z),Half(1)});
                }
            }
            error.clear(); return result;
        }
        catch (const std::exception& exception) { error=exception.what(); return {}; }
    }
}
