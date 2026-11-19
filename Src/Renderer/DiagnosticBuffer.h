#pragma once
#include "Renderer/Types.h"
#include <algorithm>
#include <cmath>
namespace Evans::Renderer
{
    inline float DiagnosticHalf(std::uint16_t bits)
    {
        const int exponent=(bits>>10)&31,fraction=bits&1023;
        const float value=exponent==0?std::ldexp(float(fraction),-24):exponent==31?0.0f:std::ldexp(float(fraction+1024),exponent-25);
        return bits&32768?-value:value;
    }
    inline void DecodeDiagnosticRows(const void* data,std::size_t pitch,unsigned width,unsigned height,std::vector<Float4>& output)
    {
        output.resize(std::size_t(width)*height);
        for (unsigned y=0;y<height;++y)
        {
            const auto* row=reinterpret_cast<const std::uint16_t*>(static_cast<const std::uint8_t*>(data)+y*pitch);
            for (unsigned x=0;x<width;++x) output[std::size_t(y)*width+x]={DiagnosticHalf(row[x*4]),DiagnosticHalf(row[x*4+1]),DiagnosticHalf(row[x*4+2]),DiagnosticHalf(row[x*4+3])};
        }
    }
    inline void ApplyDiagnosticCapture(FrameCapture& frame,Techniques::RenderBuffer buffer)
    {
        using B=Techniques::RenderBuffer;
        if (buffer==B::Color) return;
        const std::vector<float>* depth=nullptr;unsigned width=frame.Width,height=frame.Height;
        if (buffer==B::Depth) depth=&frame.Depth32;
        if (buffer==B::Shadow) {depth=&frame.ShadowDepth32;width=frame.ShadowWidth;height=frame.ShadowHeight;}
        if (buffer>=B::Cascade0 && buffer<=B::Cascade3)
        { const auto i=unsigned(buffer)-unsigned(B::Cascade0);if (i<frame.CascadeDepth32.size()) {depth=&frame.CascadeDepth32[i];width=height=unsigned(std::sqrt(double(depth->size())));} }
        frame.ColorRGBA8.assign(std::size_t(frame.Width)*frame.Height*4,0);
        for (unsigned y=0;y<frame.Height;++y) for (unsigned x=0;x<frame.Width;++x)
        {
            const auto p=std::size_t(y)*frame.Width+x;Float4 value{0,0,0,1};
            if (depth && width && height && depth->size()==std::size_t(width)*height)
            {
                float d=(*depth)[std::size_t(y*height/frame.Height)*width+x*width/frame.Width];
                d=buffer==B::Depth?std::pow(std::max(0.0f,1-d),0.2f):d;
                value={d,d,d,1};
            }
            else if (buffer==B::Motion && p<frame.MotionVectors.size())
            { const auto m=frame.MotionVectors[p];value={m.X/32+0.5f,m.Y/32+0.5f,m.W,1}; }
            else if (!frame.DiagnosticPixels.empty())
            {
                const bool renderSized=frame.RenderWidth && frame.RenderHeight && frame.DiagnosticPixels.size()==std::size_t(frame.RenderWidth)*frame.RenderHeight;
                const unsigned sourceWidth=renderSized?frame.RenderWidth:frame.Width,sourceHeight=renderSized?frame.RenderHeight:frame.Height;
                const auto source=std::size_t(y*sourceHeight/frame.Height)*sourceWidth+x*sourceWidth/frame.Width;
                if (source>=frame.DiagnosticPixels.size()) continue;
                value=frame.DiagnosticPixels[source];
                if (buffer==B::Normal) value={value.X*0.5f+0.5f,value.Y*0.5f+0.5f,value.Z*0.5f+0.5f,1};
                else if (buffer==B::Roughness || buffer==B::Metallic) value={value.W,value.W,value.W,1};
                else if (buffer==B::AmbientOcclusion) value={value.X,value.X,value.X,1};
                else {value.X=std::pow(std::max(0.0f,value.X),1/2.2f);value.Y=std::pow(std::max(0.0f,value.Y),1/2.2f);value.Z=std::pow(std::max(0.0f,value.Z),1/2.2f);}
            }
            frame.ColorRGBA8[p*4]=std::uint8_t(std::clamp(value.X,0.0f,1.0f)*255);
            frame.ColorRGBA8[p*4+1]=std::uint8_t(std::clamp(value.Y,0.0f,1.0f)*255);
            frame.ColorRGBA8[p*4+2]=std::uint8_t(std::clamp(value.Z,0.0f,1.0f)*255);frame.ColorRGBA8[p*4+3]=255;
        }
    }
}
