#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace Evans::Maths
{
	struct Vector2
	{
		float x = 0.0f;
		float y = 0.0f;

		constexpr Vector2() = default;
		constexpr explicit Vector2(float value) : x(value), y(value) {}
		constexpr Vector2(float xValue, float yValue) : x(xValue), y(yValue) {}

		constexpr float Dot(const Vector2& rhs) const { return x * rhs.x + y * rhs.y; }
		constexpr float Cross(const Vector2& rhs) const { return x * rhs.y - y * rhs.x; }
		constexpr float LengthSquared() const { return Dot(*this); }
		float Length() const { return std::sqrt(LengthSquared()); }
		Vector2 Normalized(float epsilon = 1.0e-8f) const
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
		bool NearlyEquals(const Vector2& rhs, float epsilon = 1.0e-5f) const
		{
			return std::fabs(x - rhs.x) <= epsilon && std::fabs(y - rhs.y) <= epsilon;
		}

		float& operator[](std::size_t index) { return index == 0 ? x : y; }
		const float& operator[](std::size_t index) const { return index == 0 ? x : y; }
		constexpr Vector2 operator-() const { return { -x, -y }; }
		constexpr Vector2 operator+(const Vector2& rhs) const { return { x + rhs.x, y + rhs.y }; }
		constexpr Vector2 operator-(const Vector2& rhs) const { return { x - rhs.x, y - rhs.y }; }
		constexpr Vector2 operator*(const Vector2& rhs) const { return { x * rhs.x, y * rhs.y }; }
		constexpr Vector2 operator/(const Vector2& rhs) const { return { x / rhs.x, y / rhs.y }; }
		constexpr Vector2 operator*(float scalar) const { return { x * scalar, y * scalar }; }
		constexpr Vector2 operator/(float scalar) const { return { x / scalar, y / scalar }; }
		Vector2& operator+=(const Vector2& rhs) { x += rhs.x; y += rhs.y; return *this; }
		Vector2& operator-=(const Vector2& rhs) { x -= rhs.x; y -= rhs.y; return *this; }
		Vector2& operator*=(float scalar) { x *= scalar; y *= scalar; return *this; }
		Vector2& operator/=(float scalar) { x /= scalar; y /= scalar; return *this; }

		static constexpr Vector2 Zero() { return {}; }
		static constexpr Vector2 One() { return { 1.0f, 1.0f }; }
		static constexpr Vector2 UnitX() { return { 1.0f, 0.0f }; }
		static constexpr Vector2 UnitY() { return { 0.0f, 1.0f }; }
		static constexpr Vector2 Min(const Vector2& a, const Vector2& b)
		{
			return { a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y };
		}
		static constexpr Vector2 Max(const Vector2& a, const Vector2& b)
		{
			return { a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y };
		}
	};

	constexpr Vector2 operator*(float scalar, const Vector2& vector) { return vector * scalar; }
	constexpr bool operator==(const Vector2& a, const Vector2& b) { return a.x == b.x && a.y == b.y; }
	constexpr bool operator!=(const Vector2& a, const Vector2& b) { return !(a == b); }
}
