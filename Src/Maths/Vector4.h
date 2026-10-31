#pragma once

#include "Vector3.h"

#include <cmath>
#include <cstddef>

namespace Evans::Maths
{
	struct Vector4
	{
		float x = 0.0f;
		float y = 0.0f;
		float z = 0.0f;
		float w = 0.0f;

		constexpr Vector4() = default;
		constexpr explicit Vector4(float value) : x(value), y(value), z(value), w(value) {}
		constexpr Vector4(float xValue, float yValue, float zValue, float wValue)
			: x(xValue), y(yValue), z(zValue), w(wValue) {}
		constexpr Vector4(const Vector3& xyz, float wValue) : x(xyz.x), y(xyz.y), z(xyz.z), w(wValue) {}

		constexpr Vector3 XYZ() const { return { x, y, z }; }
		constexpr float Dot(const Vector4& rhs) const { return x * rhs.x + y * rhs.y + z * rhs.z + w * rhs.w; }
		constexpr float LengthSquared() const { return Dot(*this); }
		float Length() const { return std::sqrt(LengthSquared()); }
		Vector4 Normalized(float epsilon = 1.0e-8f) const
		{
			const float length = Length();
			return length > epsilon ? *this / length : Zero();
		}
		bool NearlyEquals(const Vector4& rhs, float epsilon = 1.0e-5f) const
		{
			return std::fabs(x - rhs.x) <= epsilon && std::fabs(y - rhs.y) <= epsilon
				&& std::fabs(z - rhs.z) <= epsilon && std::fabs(w - rhs.w) <= epsilon;
		}

		float& operator[](std::size_t index)
		{
			return index == 0 ? x : (index == 1 ? y : (index == 2 ? z : w));
		}
		const float& operator[](std::size_t index) const
		{
			return index == 0 ? x : (index == 1 ? y : (index == 2 ? z : w));
		}
		constexpr Vector4 operator-() const { return { -x, -y, -z, -w }; }
		constexpr Vector4 operator+(const Vector4& rhs) const { return { x + rhs.x, y + rhs.y, z + rhs.z, w + rhs.w }; }
		constexpr Vector4 operator-(const Vector4& rhs) const { return { x - rhs.x, y - rhs.y, z - rhs.z, w - rhs.w }; }
		constexpr Vector4 operator*(float scalar) const { return { x * scalar, y * scalar, z * scalar, w * scalar }; }
		constexpr Vector4 operator/(float scalar) const { return { x / scalar, y / scalar, z / scalar, w / scalar }; }
		Vector4& operator+=(const Vector4& rhs) { x += rhs.x; y += rhs.y; z += rhs.z; w += rhs.w; return *this; }
		Vector4& operator-=(const Vector4& rhs) { x -= rhs.x; y -= rhs.y; z -= rhs.z; w -= rhs.w; return *this; }
		Vector4& operator*=(float scalar) { x *= scalar; y *= scalar; z *= scalar; w *= scalar; return *this; }
		Vector4& operator/=(float scalar) { x /= scalar; y /= scalar; z /= scalar; w /= scalar; return *this; }

		static constexpr Vector4 Zero() { return {}; }
		static constexpr Vector4 One() { return { 1.0f, 1.0f, 1.0f, 1.0f }; }
	};

	constexpr Vector4 operator*(float scalar, const Vector4& vector) { return vector * scalar; }
	constexpr bool operator==(const Vector4& a, const Vector4& b)
	{
		return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
	}
	constexpr bool operator!=(const Vector4& a, const Vector4& b) { return !(a == b); }
}
