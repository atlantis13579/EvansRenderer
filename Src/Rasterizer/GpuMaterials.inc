EVANS_GPU_CODE(
struct RasterTextureGradients { float2 X; float2 Y; };

R_FN float GpuDecodeSRGB(float channel)
{
    return channel <= 0.04045f ? channel / 12.92f : R_POW((channel + 0.055f) / 1.055f, 2.4f);
}

R_FN int GpuTextureWrap(int coordinate, uint extent)
{
    return coordinate < 0 ? int(extent) - 1 : coordinate >= int(extent) ? 0 : coordinate;
}

R_FN float4 GpuTextureTexel(R_TEX descriptor, int x, int y, uint format, R_READ(uint) texels)
{
    uint value = texels[descriptor.OffsetTexels + uint(GpuTextureWrap(y, descriptor.Height)) * descriptor.Width +
        uint(GpuTextureWrap(x, descriptor.Width))];
    float4 rgba = R_FLOAT4(float(value & 255u), float((value >> 8u) & 255u),
        float((value >> 16u) & 255u), float(value >> 24u)) / 255.0f;
    if (format == 1)
        return R_FLOAT4(GpuDecodeSRGB(rgba.x), GpuDecodeSRGB(rgba.y), GpuDecodeSRGB(rgba.z), rgba.w);
    if (format == 2)
        return rgba * (value >> 24u ? R_EXP2(float(value >> 24u) - 128.0f) : 0.0f);
    return rgba;
}

R_FN float4 GpuTextureLevel(R_TEX descriptor, float2 uv, uint format, R_READ(uint) texels)
{
    float x = uv.x;
    float y = format == 2 ? uv.y : 1.0f - uv.y;
    x = (x - R_FLOOR(x)) * descriptor.Width - 0.5f;
    y = (y - R_FLOOR(y)) * descriptor.Height - 0.5f;
    int ix = int(R_FLOOR(x)), iy = int(R_FLOOR(y));
    float blendX = x - float(ix), blendY = y - float(iy);
    float4 top = GpuTextureTexel(descriptor, ix, iy, format, texels) * (1.0f - blendX) +
        GpuTextureTexel(descriptor, ix + 1, iy, format, texels) * blendX;
    float4 bottom = GpuTextureTexel(descriptor, ix, iy + 1, format, texels) * (1.0f - blendX) +
        GpuTextureTexel(descriptor, ix + 1, iy + 1, format, texels) * blendX;
    return top * (1.0f - blendY) + bottom * blendY;
}

R_FN float4 GpuSampleTexture(uint index, float2 uv, uint format, RasterTextureGradients gradients,
    uint textureCount, bool anisotropic, R_READ(R_TEX) descriptors, R_READ(uint) texels)
{
    if (index >= textureCount || descriptors[index].Width == 0) return R_FLOAT4(1, 1, 1, 1);
    R_TEX base = descriptors[index];
    if (!anisotropic || base.FirstMip == 0)
        return GpuTextureLevel(base, uv, format, texels);
    float2 dx = R_FLOAT2(gradients.X.x * base.Width, gradients.X.y * base.Height);
    float2 dy = R_FLOAT2(gradients.Y.x * base.Width, gradients.Y.y * base.Height);
    float lengthX = dx.x * dx.x + dx.y * dx.y, lengthY = dy.x * dy.x + dy.y * dy.y;
    float major = R_SQRT(R_MAX(lengthX, lengthY));
    if (!(major > 1.0f) || !R_FINITE(major)) return GpuTextureLevel(base, uv, format, texels);
    float minor = R_MAX(R_MAX(1.0f, major / 8.0f), R_ABS(dx.x * dy.y - dx.y * dy.x) / major);
    uint taps = uint(R_MIN(8.0f, R_CEIL(major / minor)));
    float2 axis = lengthX >= lengthY ? gradients.X : gradients.Y;
    float lod = R_LOG2(minor);
    R_TEX current = base;
    uint nextIndex = base.FirstMip;
    for (uint level = 0; level < uint(lod) && (current.Width > 1 || current.Height > 1); ++level)
        current = descriptors[nextIndex++];
    R_TEX next = current;
    if (current.Width > 1 || current.Height > 1) next = descriptors[nextIndex];
    float4 total = R_FLOAT4(0, 0, 0, 0);
    float fraction = lod - R_FLOOR(lod);
    for (uint tap = 0; tap < taps; ++tap)
    {
        float distance = (float(tap) + 0.5f) / float(taps) - 0.5f;
        float2 sampleUV = R_FLOAT2(uv.x + distance * axis.x, uv.y + distance * axis.y);
        float4 color = GpuTextureLevel(current, sampleUV, format, texels);
        if (fraction > 0.0f && next.OffsetTexels != current.OffsetTexels)
            color = color * (1.0f - fraction) + GpuTextureLevel(next, sampleUV, format, texels) * fraction;
        total = total + color;
    }
    return total / float(taps);
}

R_FN RasterTextureGradients GpuTextureGradients(R_TRI primitive, float2 uv, float denominator, uint width, uint height)
{
    RasterTextureGradients result;
    result.X = R_FLOAT2(0, 0);
    result.Y = R_FLOAT2(0, 0);
    float4 a = primitive.Vertices[0].Clip, b = primitive.Vertices[1].Clip, c = primitive.Vertices[2].Clip;
    if (R_ABS(a.w) < 1.0e-6f || R_ABS(b.w) < 1.0e-6f || R_ABS(c.w) < 1.0e-6f) return result;
    float2 screen[3];
    screen[0] = R_FLOAT2(0.5f * width * a.x / a.w, -0.5f * height * a.y / a.w);
    screen[1] = R_FLOAT2(0.5f * width * b.x / b.w, -0.5f * height * b.y / b.w);
    screen[2] = R_FLOAT2(0.5f * width * c.x / c.w, -0.5f * height * c.y / c.w);
    float area = (screen[1].x - screen[0].x) * (screen[2].y - screen[0].y) -
        (screen[1].y - screen[0].y) * (screen[2].x - screen[0].x);
    if (R_ABS(area * denominator) < 1.0e-6f) return result;
    for (uint i = 0; i < 3; ++i)
    {
        float2 next = screen[(i + 1) % 3], last = screen[(i + 2) % 3];
        float scale = 1.0f / (primitive.Vertices[i].Clip.w * area * denominator);
        float dx = (next.y - last.y) * scale, dy = (last.x - next.x) * scale;
        float4 tex = primitive.Vertices[i].TexCoord;
        result.X.x += (tex.x - uv.x) * dx; result.X.y += (tex.y - uv.y) * dx;
        result.Y.x += (tex.x - uv.x) * dy; result.Y.y += (tex.y - uv.y) * dy;
    }
    return result;
}

R_FN float3 GpuMappedNormal(R_TRI primitive, float3 normal, float4 normalTexel)
{
    float4 a = primitive.Vertices[0].TexCoord, b = primitive.Vertices[1].TexCoord, c = primitive.Vertices[2].TexCoord;
    float2 uv1 = R_FLOAT2(b.x - a.x, a.y - b.y), uv2 = R_FLOAT2(c.x - a.x, a.y - c.y);
    float determinant = uv1.x * uv2.y - uv1.y * uv2.x;
    if (R_ABS(determinant) <= 1.0e-6f) return normal;
    float3 edge1 = R_XYZ(primitive.Vertices[1].Position) - R_XYZ(primitive.Vertices[0].Position);
    float3 edge2 = R_XYZ(primitive.Vertices[2].Position) - R_XYZ(primitive.Vertices[0].Position);
    float3 tangent = (edge1 * uv2.y - edge2 * uv1.y) / determinant;
    float3 referenceBitangent = (edge2 * uv1.x - edge1 * uv2.x) / determinant;
    tangent = R_NORMALIZE_OR(tangent - normal * R_DOT(tangent, normal), R_FLOAT3(0, 0, 0));
    float3 bitangent = R_CROSS(normal, tangent);
    if (R_DOT(bitangent, referenceBitangent) < 0) bitangent = bitangent * -1.0f;
    return R_NORMALIZE_OR(tangent * (2.0f * normalTexel.x - 1.0f) + bitangent * (2.0f * normalTexel.y - 1.0f) +
        normal * (2.0f * normalTexel.z - 1.0f), normal);
}

R_FN float GpuAcesChannel(float c, float exposure)
{
    c = R_MAX(c * exposure, 0.0f);
    return R_MIN(1.0f, c * (2.51f * c + 0.03f) / (c * (2.43f * c + 0.59f) + 0.14f));
}

R_FN float3 GpuDisplayColor(float3 color, uint aces, float4 display)
{
    if (aces == 0)
    {
        float scale = R_MAX(1.0f, R_MAX(color.x, R_MAX(color.y, color.z)));
        return R_FLOAT3(R_MAX(color.x, 0.0f), R_MAX(color.y, 0.0f), R_MAX(color.z, 0.0f)) / scale;
    }
    float grey = color.x * 0.2126f + color.y * 0.7152f + color.z * 0.0722f;
    float bias = (grey * (1.0f - display.z) - 0.18f) * display.y + 0.18f;
    float3 grade = color * (display.y * display.z) + R_FLOAT3(bias, bias, bias);
    return R_FLOAT3(GpuAcesChannel(grade.x, display.x), GpuAcesChannel(grade.y, display.x),
        GpuAcesChannel(grade.z, display.x));
}

R_FN float3 GpuSkyRay(float x, float y, uint width, uint height,
    float3 right, float3 up, float3 back, float scaleX, float scaleY)
{
    return right * ((2.0f * x / width - 1.0f) / scaleX) +
        up * ((1.0f - 2.0f * y / height) / scaleY) - back;
}

R_FN float3 GpuSampleSky(float3 direction, uint index, uint textureCount,
    R_READ(R_TEX) descriptors, R_READ(uint) texels)
{
    float3 normal = R_NORMALIZE_OR(direction, R_FLOAT3(0, 1, 0));
    const float pi = 3.14159265f;
    float2 uv = R_FLOAT2(R_ATAN2(normal.z, normal.x) / (2.0f * pi) + 0.5f,
        R_ACOS(R_MAX(-1.0f, R_MIN(1.0f, normal.y))) / pi);
    RasterTextureGradients gradients;
    gradients.X = R_FLOAT2(0, 0);
    gradients.Y = gradients.X;
    return R_XYZ(GpuSampleTexture(index, uv, 2, gradients, textureCount, false, descriptors, texels));
}

R_FN float GpuShadowBias(float constantBias, float slopeBias, float3 receiverNormal, float3 directionToLight)
{
    if (slopeBias <= 0.0f) return constantBias;
    const float cosine = R_MAX(0.0f, R_MIN(1.0f, R_DOT(receiverNormal,
        R_NORMALIZE_OR(directionToLight, receiverNormal))));
    const float slope = R_SQRT(R_MAX(0.0f, 1.0f - cosine * cosine)) / R_MAX(0.15f, cosine);
    return R_MIN(0.0045f, constantBias + slopeBias * slope);
}
)
