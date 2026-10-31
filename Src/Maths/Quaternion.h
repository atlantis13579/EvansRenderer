#pragma once

#include "Vector3.h"

#include <algorithm>
#include <cmath>

namespace Evans::Maths
{
	struct Quaternion
	{
		float x = 0.0f;
		float y = 0.0f;
		float z = 0.0f;
		float w = 1.0f;

		constexpr Quaternion() = default;
		constexpr Quaternion(float xValue, float yValue, float zValue, float wValue)
			: x(xValue), y(yValue), z(zValue), w(wValue) {}

		constexpr float Dot(const Quaternion& rhs) const
		{
			return x * rhs.x + y * rhs.y + z * rhs.z + w * rhs.w;
		}
		constexpr float LengthSquared() const { return Dot(*this); }
		float Length() const { return std::sqrt(LengthSquared()); }
		Quaternion Normalized(float epsilon = 1.0e-8f) const
		{
			const float length = Length();
			return length > epsilon ? *this / length : Identity();
		}
		float Normalize(float epsilon = 1.0e-8f)
		{
			const float length = Length();
			*this = length > epsilon ? *this / length : Identity();
			return length;
		}
		constexpr Quaternion Conjugated() const { return { -x, -y, -z, w }; }
		Quaternion Inversed(float epsilon = 1.0e-8f) const
		{
			const float lengthSquared = LengthSquared();
			return lengthSquared > epsilon ? Conjugated() / lengthSquared : Identity();
		}
		Vector3 Rotate(const Vector3& vector) const
		{
			const Vector3 imaginary(x, y, z);
			const Vector3 twiceCross = 2.0f * imaginary.Cross(vector);
			return vector + w * twiceCross + imaginary.Cross(twiceCross);
		}
		bool NearlyEquals(const Quaternion& rhs, float epsilon = 1.0e-5f) const
		{
			const float direct = std::fabs(x - rhs.x) + std::fabs(y - rhs.y)
				+ std::fabs(z - rhs.z) + std::fabs(w - rhs.w);
			const float negated = std::fabs(x + rhs.x) + std::fabs(y + rhs.y)
				+ std::fabs(z + rhs.z) + std::fabs(w + rhs.w);
			return std::min(direct, negated) <= epsilon * 4.0f;
		}

		constexpr Quaternion operator-() const { return { -x, -y, -z, -w }; }
		constexpr Quaternion operator+(const Quaternion& rhs) const
		{
			return { x + rhs.x, y + rhs.y, z + rhs.z, w + rhs.w };
		}
		constexpr Quaternion operator-(const Quaternion& rhs) const
		{
			return { x - rhs.x, y - rhs.y, z - rhs.z, w - rhs.w };
		}
		constexpr Quaternion operator*(float scalar) const { return { x * scalar, y * scalar, z * scalar, w * scalar }; }
		constexpr Quaternion operator/(float scalar) const { return { x / scalar, y / scalar, z / scalar, w / scalar }; }
		constexpr Quaternion operator*(const Quaternion& rhs) const
		{
			return {
				w * rhs.x + x * rhs.w + y * rhs.z - z * rhs.y,
				w * rhs.y - x * rhs.z + y * rhs.w + z * rhs.x,
				w * rhs.z + x * rhs.y - y * rhs.x + z * rhs.w,
				w * rhs.w - x * rhs.x - y * rhs.y - z * rhs.z
			};
		}
		Quaternion& operator*=(const Quaternion& rhs) { *this = *this * rhs; return *this; }
		Quaternion& operator*=(float scalar) { x *= scalar; y *= scalar; z *= scalar; w *= scalar; return *this; }

		static constexpr Quaternion Identity() { return {}; }
		static Quaternion FromAxisAngle(const Vector3& axis, float radians)
		{
			const Vector3 unitAxis = axis.Normalized();
			if (unitAxis.LengthSquared() == 0.0f)
			{
				return Identity();
			}
			const float halfAngle = radians * 0.5f;
			const float sine = std::sin(halfAngle);
			return { unitAxis.x * sine, unitAxis.y * sine, unitAxis.z * sine, std::cos(halfAngle) };
		}
		static Quaternion FromEulerXYZ(const Vector3& radians)
		{
			const Quaternion xRotation = FromAxisAngle(Vector3::UnitX(), radians.x);
			const Quaternion yRotation = FromAxisAngle(Vector3::UnitY(), radians.y);
			const Quaternion zRotation = FromAxisAngle(Vector3::UnitZ(), radians.z);
			return (zRotation * yRotation * xRotation).Normalized();
		}
		static Quaternion Slerp(const Quaternion& from, const Quaternion& to, float alpha)
		{
			Quaternion end = to;
			float cosine = from.Dot(end);
			if (cosine < 0.0f)
			{
				end = -end;
				cosine = -cosine;
			}
			cosine = std::clamp(cosine, -1.0f, 1.0f);
			if (cosine > 0.9995f)
			{
				return (from + (end - from) * alpha).Normalized();
			}
			const float angle = std::acos(cosine);
			const float sine = std::sin(angle);
			const float fromWeight = std::sin((1.0f - alpha) * angle) / sine;
			const float toWeight = std::sin(alpha * angle) / sine;
			return (from * fromWeight + end * toWeight).Normalized();
		}
	};

	constexpr Quaternion operator*(float scalar, const Quaternion& quaternion) { return quaternion * scalar; }
	constexpr bool operator==(const Quaternion& a, const Quaternion& b)
	{
		return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
	}
	constexpr bool operator!=(const Quaternion& a, const Quaternion& b) { return !(a == b); }
}
