#pragma once
#include <cstdint>
namespace Evans::Techniques
{
    enum class RenderBuffer : std::uint32_t { Color,Depth,Shadow,Cascade0,Cascade1,Cascade2,Cascade3,Albedo,Normal,Roughness,Metallic,Emission,AmbientOcclusion,Motion };
    inline int MaterialBufferIndex(RenderBuffer value)
    {
        switch (value) {case RenderBuffer::Albedo:return 0;case RenderBuffer::Normal:case RenderBuffer::Roughness:return 1;
        case RenderBuffer::Metallic:return 3;case RenderBuffer::Emission:return 4;default:return -1;}
    }
}
