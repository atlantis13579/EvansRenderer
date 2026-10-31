#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace Evans::Maths
{
	struct Vector3
	{
		float x = 0.0f;
		float y = 0.0f;
		float z = 0.0f;

		constexpr Vector3() = default;
		constexpr explicit Vector3(float value) : x(value), y(value), z(value) {}
		constexpr Vector3(float xValue, float yValue, float zValue) : x(xValue), y(yValue), z(zValue) {}

		constexpr float Dot(const Vector3& rhs) const { return x * rhs.x + y * rhs.y + z * rhs.z; }
		constexpr Vector3 Cross(const Vector3& rhs) const
		{
			return { y * rhs.z - z * rhs.y, z * rhs.x - x * rhs.z, x * rhs.y - y * rhs.x };
		}
		constexpr float LengthSquared() const { return Dot(*this); }
		float Length() const { return std::sqrt(LengthSquared()); }
		Vector3 Normalized(float epsilon = 1.0e-8f) const
		{
			const float length = Length();
			return length > epsilon ? *this / length : Zero();
		}
		float Normalize(float epsilon = 1.0e-8f)
		{
			const float length = Length();
			*this = length > epsilon ? *this / length : Zero();
			return length;
		}
		bool NearlyEquals(const Vector3& rhs, float epsilon = 1.0e-5f) const
		{
			return std::fabs(x - rhs.x) <= epsilon
				&& std::fabs(y - rhs.y) <= epsilon
				&& std::fabs(z - rhs.z) <= epsilon;
		}
		int LargestAxis() const
		{
			return x >= y ? (x >= z ? 0 : 2) : (y >= z ? 1 : 2);
		}

		float& operator[](std::size_t index) { return index == 0 ? x : (index == 1 ? y : z); }
		const float& operator[](std::size_t index) const { return index == 0 ? x : (index == 1 ? y : z); }
		constexpr Vector3 operator-() const { return { -x, -y, -z }; }
		constexpr Vector3 operator+(const Vector3& rhs) const { return { x + rhs.x, y + rhs.y, z + rhs.z }; }
		constexpr Vector3 operator-(const Vector3& rhs) const { return { x - rhs.x, y - rhs.y, z - rhs.z }; }
		constexpr Vector3 operator*(const Vector3& rhs) const { return { x * rhs.x, y * rhs.y, z * rhs.z }; }
		constexpr Vector3 operator/(const Vector3& rhs) const { return { x / rhs.x, y / rhs.y, z / rhs.z }; }
		constexpr Vector3 operator*(float scalar) const { return { x * scalar, y * scalar, z * scalar }; }
		constexpr Vector3 operator/(float scalar) const { return { x / scalar, y / scalar, z / scalar }; }
		Vector3& operator+=(const Vector3& rhs) { x += rhs.x; y += rhs.y; z += rhs.z; return *this; }
		Vector3& operator-=(const Vector3& rhs) { x -= rhs.x; y -= rhs.y; z -= rhs.z; return *this; }
		Vector3& operator*=(const Vector3& rhs) { x *= rhs.x; y *= rhs.y; z *= rhs.z; return *this; }
		Vector3& operator/=(const Vector3& rhs) { x /= rhs.x; y /= rhs.y; z /= rhs.z; return *this; }
		Vector3& operator*=(float scalar) { x *= scalar; y *= scalar; z *= scalar; return *this; }
		Vector3& operator/=(float scalar) { x /= scalar; y /= scalar; z /= scalar; return *this; }

		static constexpr Vector3 Zero() { return {}; }
		static constexpr Vector3 One() { return { 1.0f, 1.0f, 1.0f }; }
		static constexpr Vector3 UnitX() { return { 1.0f, 0.0f, 0.0f }; }
		static constexpr Vector3 UnitY() { return { 0.0f, 1.0f, 0.0f }; }
		static constexpr Vector3 UnitZ() { return { 0.0f, 0.0f, 1.0f }; }
		static constexpr Vector3 Min(const Vector3& a, const Vector3& b)
		{
			return {
				a.x < b.x ? a.x : b.x,
				a.y < b.y ? a.y : b.y,
				a.z < b.z ? a.z : b.z
			};
		}
		static constexpr Vector3 Max(const Vector3& a, const Vector3& b)
		{
			return {
				a.x > b.x ? a.x : b.x,
				a.y > b.y ? a.y : b.y,
				a.z > b.z ? a.z : b.z
			};
		}
	};

	constexpr Vector3 operator*(float scalar, const Vector3& vector) { return vector * scalar; }
	constexpr bool operator==(const Vector3& a, const Vector3& b)
	{
		return a.x == b.x && a.y == b.y && a.z == b.z;
	}
	constexpr bool operator!=(const Vector3& a, const Vector3& b) { return !(a == b); }
}
