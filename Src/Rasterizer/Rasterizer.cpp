#include "Rasterizer/Rasterizer.hpp"

#include "Maths/ColorSpace.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace Evans::Rasterizer
{
namespace
{

[[nodiscard]] Float3 Subtract(const Float3& left, const Float3& right) noexcept
{
    return {left.X - right.X, left.Y - right.Y, left.Z - right.Z};
}

[[nodiscard]] float Dot(const Float3& left, const Float3& right) noexcept
{
    return left.X * right.X + left.Y * right.Y + left.Z * right.Z;
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
    const float lengthSquared = Dot(value, value);
    if (lengthSquared <= 1.0e-12f)
    {
        return {};
    }
    const float inverseLength = 1.0f / std::sqrt(lengthSquared);
    return {value.X * inverseLength, value.Y * inverseLength, value.Z * inverseLength};
}

} // namespace

Matrix4 Matrix4::operator*(const Matrix4& right) const noexcept
{
    Matrix4 result;
    result.Values.fill(0.0f);
    for (std::size_t row = 0; row < 4; ++row)
    {
        for (std::size_t column = 0; column < 4; ++column)
        {
            for (std::size_t inner = 0; inner < 4; ++inner)
            {
                result(row, column) += (*this)(row, inner) * right(inner, column);
            }
        }
    }
    return result;
}

Float4 Matrix4::operator*(const Float4& vector) const noexcept
{
    return {
        (*this)(0, 0) * vector.X + (*this)(0, 1) * vector.Y +
            (*this)(0, 2) * vector.Z + (*this)(0, 3) * vector.W,
        (*this)(1, 0) * vector.X + (*this)(1, 1) * vector.Y +
            (*this)(1, 2) * vector.Z + (*this)(1, 3) * vector.W,
        (*this)(2, 0) * vector.X + (*this)(2, 1) * vector.Y +
            (*this)(2, 2) * vector.Z + (*this)(2, 3) * vector.W,
        (*this)(3, 0) * vector.X + (*this)(3, 1) * vector.Y +
            (*this)(3, 2) * vector.Z + (*this)(3, 3) * vector.W};
}

Matrix4 Matrix4::Translation(const Float3& translation) noexcept
{
    Matrix4 result;
    result(0, 3) = translation.X;
    result(1, 3) = translation.Y;
    result(2, 3) = translation.Z;
    return result;
}

Matrix4 Matrix4::Scale(const Float3& scale) noexcept
{
    Matrix4 result;
    result(0, 0) = scale.X;
    result(1, 1) = scale.Y;
    result(2, 2) = scale.Z;
    return result;
}

Matrix4 Matrix4::RotationY(float radians) noexcept
{
    Matrix4 result;
    const float sine = std::sin(radians);
    const float cosine = std::cos(radians);
    result(0, 0) = cosine;
    result(0, 2) = sine;
    result(2, 0) = -sine;
    result(2, 2) = cosine;
    return result;
}

Matrix4 Matrix4::PerspectiveRH(
    float verticalFieldOfViewRadians,
    float aspectRatio,
    float nearPlane,
    float farPlane)
{
    if (verticalFieldOfViewRadians <= 0.0f || aspectRatio <= 0.0f ||
        nearPlane <= 0.0f || farPlane <= nearPlane)
    {
        throw std::invalid_argument("Invalid perspective projection parameters");
    }
    Matrix4 result;
    result.Values.fill(0.0f);
    const float cotangent = 1.0f / std::tan(verticalFieldOfViewRadians * 0.5f);
    result(0, 0) = cotangent / aspectRatio;
    result(1, 1) = cotangent;
    result(2, 2) = farPlane / (nearPlane - farPlane);
    result(2, 3) = farPlane * nearPlane / (nearPlane - farPlane);
    result(3, 2) = -1.0f;
    return result;
}

Matrix4 Matrix4::LookAtRH(
    const Float3& eye,
    const Float3& target,
    const Float3& up)
{
    const Float3 forward = Normalize(Subtract(eye, target));
    const Float3 right = Normalize(Cross(up, forward));
    const Float3 correctedUp = Cross(forward, right);
    if (Dot(forward, forward) <= 1.0e-12f || Dot(right, right) <= 1.0e-12f)
    {
        throw std::invalid_argument("LookAt requires non-degenerate eye, target, and up vectors");
    }
    Matrix4 result;
    result.Values = {
        right.X, right.Y, right.Z, -Dot(right, eye),
        correctedUp.X, correctedUp.Y, correctedUp.Z, -Dot(correctedUp, eye),
        forward.X, forward.Y, forward.Z, -Dot(forward, eye),
        0.0f, 0.0f, 0.0f, 1.0f};
    return result;
}

std::vector<std::uint32_t> RenderTarget::ToRGBA8() const
{
    std::vector<std::uint32_t> result;
    result.reserve(Colors_.size());
    for (const Color& source : Colors_)
    {
        const auto red = static_cast<std::uint32_t>(Maths::LinearToSRGB8(source.R));
        const auto green = static_cast<std::uint32_t>(Maths::LinearToSRGB8(source.G));
        const auto blue = static_cast<std::uint32_t>(Maths::LinearToSRGB8(source.B));
        const auto alpha = static_cast<std::uint32_t>(Maths::LinearToUNorm8(source.A));
        result.push_back(red | (green << 8U) | (blue << 16U) | (alpha << 24U));
    }
    return result;
}

SoftwareRasterizer::SoftwareRasterizer(
    RasterBackend backend,
    const RasterRHICreateInfo& createInfo)
    : RHI_(CreateRasterRHI(backend, createInfo).release())
{
}

SoftwareRasterizer::~SoftwareRasterizer()
{
    std::shared_ptr<IRasterRHI> rhi;
    {
        std::lock_guard<std::mutex> lock(Mutex_);
        rhi = std::move(RHI_);
    }
    if (rhi)
    {
        rhi->Cancel();
        rhi->WaitIdle();
        rhi->Shutdown();
    }
}

void SoftwareRasterizer::SelectBackend(
    RasterBackend backend,
    const RasterRHICreateInfo& createInfo)
{
    std::shared_ptr<IRasterRHI> replacement(CreateRasterRHI(backend, createInfo).release());
    std::shared_ptr<IRasterRHI> previous;
    {
        std::lock_guard<std::mutex> lock(Mutex_);
        if (RHI_ && RHI_->Backend() == backend)
        {
            return;
        }
        previous = std::exchange(RHI_, std::move(replacement));
    }
    if (previous)
    {
        previous->Cancel();
        previous->WaitIdle();
        previous->Shutdown();
    }
}

RasterBackend SoftwareRasterizer::SelectedBackend() const noexcept
{
    std::lock_guard<std::mutex> lock(Mutex_);
    return RHI_->Backend();
}

std::string_view SoftwareRasterizer::BackendName() const noexcept
{
    std::lock_guard<std::mutex> lock(Mutex_);
    return RHI_->Name();
}

RasterBackendCapabilities SoftwareRasterizer::Capabilities() const
{
    std::shared_ptr<IRasterRHI> rhi;
    {
        std::lock_guard<std::mutex> lock(Mutex_);
        rhi = RHI_;
    }
    return rhi->Capabilities();
}

RasterStatus SoftwareRasterizer::Render(
    const RasterScene& scene,
    const RasterSettings& settings,
    RenderTarget& target,
    const RasterCancellationToken& cancellation)
{
    std::shared_ptr<IRasterRHI> rhi;
    {
        std::lock_guard<std::mutex> lock(Mutex_);
        rhi = RHI_;
    }
    return rhi->Render(scene, settings, target, cancellation);
}

void SoftwareRasterizer::Cancel() noexcept
{
    std::shared_ptr<IRasterRHI> rhi;
    {
        std::lock_guard<std::mutex> lock(Mutex_);
        rhi = RHI_;
    }
    if (rhi)
    {
        rhi->Cancel();
    }
}

void SoftwareRasterizer::WaitIdle() noexcept
{
    std::shared_ptr<IRasterRHI> rhi;
    {
        std::lock_guard<std::mutex> lock(Mutex_);
        rhi = RHI_;
    }
    if (rhi)
    {
        rhi->WaitIdle();
    }
}

RasterStatistics SoftwareRasterizer::LastStatistics() const
{
    std::shared_ptr<IRasterRHI> rhi;
    {
        std::lock_guard<std::mutex> lock(Mutex_);
        rhi = RHI_;
    }
    return rhi->LastStatistics();
}

} // namespace Evans::Rasterizer
