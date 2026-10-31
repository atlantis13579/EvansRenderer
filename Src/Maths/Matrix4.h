#pragma once

#include "Quaternion.h"
#include "Vector4.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace Evans::Maths
{
	// Matrix4 is row-major and multiplies column vectors on the right.
	class Matrix4
	{
	public:
		constexpr Matrix4()
			: m_Values {
				1.0f, 0.0f, 0.0f, 0.0f,
				0.0f, 1.0f, 0.0f, 0.0f,
				0.0f, 0.0f, 1.0f, 0.0f,
				0.0f, 0.0f, 0.0f, 1.0f }
		{
		}

		constexpr explicit Matrix4(const std::array<float, 16>& values) : m_Values(values) {}

		float& operator()(std::size_t row, std::size_t column) { return m_Values[row * 4 + column]; }
		const float& operator()(std::size_t row, std::size_t column) const { return m_Values[row * 4 + column]; }
		float* Data() { return m_Values.data(); }
		const float* Data() const { return m_Values.data(); }

		Matrix4 operator*(const Matrix4& rhs) const
		{
			Matrix4 result = Zero();
			for (std::size_t row = 0; row < 4; ++row)
			{
				for (std::size_t column = 0; column < 4; ++column)
				{
					for (std::size_t inner = 0; inner < 4; ++inner)
					{
						result(row, column) += (*this)(row, inner) * rhs(inner, column);
					}
				}
			}
			return result;
		}
		Vector4 operator*(const Vector4& vector) const
		{
			return {
				(*this)(0, 0) * vector.x + (*this)(0, 1) * vector.y + (*this)(0, 2) * vector.z + (*this)(0, 3) * vector.w,
				(*this)(1, 0) * vector.x + (*this)(1, 1) * vector.y + (*this)(1, 2) * vector.z + (*this)(1, 3) * vector.w,
				(*this)(2, 0) * vector.x + (*this)(2, 1) * vector.y + (*this)(2, 2) * vector.z + (*this)(2, 3) * vector.w,
				(*this)(3, 0) * vector.x + (*this)(3, 1) * vector.y + (*this)(3, 2) * vector.z + (*this)(3, 3) * vector.w
			};
		}
		Matrix4& operator*=(const Matrix4& rhs) { *this = *this * rhs; return *this; }

		Vector3 TransformPoint(const Vector3& point) const
		{
			const Vector4 transformed = *this * Vector4(point, 1.0f);
			if (std::fabs(transformed.w) > 1.0e-8f && std::fabs(transformed.w - 1.0f) > 1.0e-8f)
			{
				return transformed.XYZ() / transformed.w;
			}
			return transformed.XYZ();
		}
		Vector3 TransformVector(const Vector3& vector) const
		{
			return (*this * Vector4(vector, 0.0f)).XYZ();
		}
		Matrix4 Transposed() const
		{
			Matrix4 result = Zero();
			for (std::size_t row = 0; row < 4; ++row)
			{
				for (std::size_t column = 0; column < 4; ++column)
				{
					result(row, column) = (*this)(column, row);
				}
			}
			return result;
		}
		bool Inverse(Matrix4& result, float epsilon = 1.0e-8f) const
		{
			float augmented[4][8] = {};
			for (std::size_t row = 0; row < 4; ++row)
			{
				for (std::size_t column = 0; column < 4; ++column)
				{
					augmented[row][column] = (*this)(row, column);
				}
				augmented[row][row + 4] = 1.0f;
			}

			for (std::size_t pivotColumn = 0; pivotColumn < 4; ++pivotColumn)
			{
				std::size_t pivotRow = pivotColumn;
				for (std::size_t row = pivotColumn + 1; row < 4; ++row)
				{
					if (std::fabs(augmented[row][pivotColumn]) > std::fabs(augmented[pivotRow][pivotColumn]))
					{
						pivotRow = row;
					}
				}
				if (std::fabs(augmented[pivotRow][pivotColumn]) <= epsilon)
				{
					return false;
				}
				if (pivotRow != pivotColumn)
				{
					for (std::size_t column = 0; column < 8; ++column)
					{
						std::swap(augmented[pivotRow][column], augmented[pivotColumn][column]);
					}
				}
				const float inversePivot = 1.0f / augmented[pivotColumn][pivotColumn];
				for (float& value : augmented[pivotColumn])
				{
					value *= inversePivot;
				}
				for (std::size_t row = 0; row < 4; ++row)
				{
					if (row == pivotColumn)
					{
						continue;
					}
					const float factor = augmented[row][pivotColumn];
					for (std::size_t column = 0; column < 8; ++column)
					{
						augmented[row][column] -= factor * augmented[pivotColumn][column];
					}
				}
			}

			result = Zero();
			for (std::size_t row = 0; row < 4; ++row)
			{
				for (std::size_t column = 0; column < 4; ++column)
				{
					result(row, column) = augmented[row][column + 4];
				}
			}
			return true;
		}
		Matrix4 Inversed(float epsilon = 1.0e-8f) const
		{
			Matrix4 result;
			return Inverse(result, epsilon) ? result : Identity();
		}
		bool NearlyEquals(const Matrix4& rhs, float epsilon = 1.0e-5f) const
		{
			for (std::size_t index = 0; index < 16; ++index)
			{
				if (std::fabs(m_Values[index] - rhs.m_Values[index]) > epsilon)
				{
					return false;
				}
			}
			return true;
		}

		static constexpr Matrix4 Identity() { return {}; }
		static constexpr Matrix4 Zero()
		{
			return Matrix4({
				0.0f, 0.0f, 0.0f, 0.0f,
				0.0f, 0.0f, 0.0f, 0.0f,
				0.0f, 0.0f, 0.0f, 0.0f,
				0.0f, 0.0f, 0.0f, 0.0f });
		}
		static Matrix4 Translation(const Vector3& translation)
		{
			Matrix4 result;
			result(0, 3) = translation.x;
			result(1, 3) = translation.y;
			result(2, 3) = translation.z;
			return result;
		}
		static Matrix4 Scale(const Vector3& scale)
		{
			Matrix4 result;
			result(0, 0) = scale.x;
			result(1, 1) = scale.y;
			result(2, 2) = scale.z;
			return result;
		}
		static Matrix4 Rotation(const Quaternion& rotation)
		{
			const Quaternion q = rotation.Normalized();
			const float xx = q.x * q.x;
			const float yy = q.y * q.y;
			const float zz = q.z * q.z;
			const float xy = q.x * q.y;
			const float xz = q.x * q.z;
			const float yz = q.y * q.z;
			const float wx = q.w * q.x;
			const float wy = q.w * q.y;
			const float wz = q.w * q.z;
			return Matrix4({
				1.0f - 2.0f * (yy + zz), 2.0f * (xy - wz), 2.0f * (xz + wy), 0.0f,
				2.0f * (xy + wz), 1.0f - 2.0f * (xx + zz), 2.0f * (yz - wx), 0.0f,
				2.0f * (xz - wy), 2.0f * (yz + wx), 1.0f - 2.0f * (xx + yy), 0.0f,
				0.0f, 0.0f, 0.0f, 1.0f });
		}
		static Matrix4 TRS(const Vector3& translation, const Quaternion& rotation, const Vector3& scale)
		{
			return Translation(translation) * Rotation(rotation) * Scale(scale);
		}
		static Matrix4 PerspectiveRH(float verticalFovRadians, float aspect, float nearPlane, float farPlane)
		{
			if (verticalFovRadians <= 0.0f || aspect <= 0.0f || nearPlane <= 0.0f || farPlane <= nearPlane)
			{
				return Identity();
			}
			const float cotangent = 1.0f / std::tan(verticalFovRadians * 0.5f);
			Matrix4 result = Zero();
			result(0, 0) = cotangent / aspect;
			result(1, 1) = cotangent;
			result(2, 2) = farPlane / (nearPlane - farPlane);
			result(2, 3) = (farPlane * nearPlane) / (nearPlane - farPlane);
			result(3, 2) = -1.0f;
			return result;
		}
		static Matrix4 LookAtRH(const Vector3& eye, const Vector3& target, const Vector3& up)
		{
			const Vector3 forward = (eye - target).Normalized();
			const Vector3 right = up.Cross(forward).Normalized();
			const Vector3 correctedUp = forward.Cross(right);
			if (forward.LengthSquared() == 0.0f || right.LengthSquared() == 0.0f)
			{
				return Identity();
			}
			return Matrix4({
				right.x, right.y, right.z, -right.Dot(eye),
				correctedUp.x, correctedUp.y, correctedUp.z, -correctedUp.Dot(eye),
				forward.x, forward.y, forward.z, -forward.Dot(eye),
				0.0f, 0.0f, 0.0f, 1.0f });
		}

	private:
		std::array<float, 16> m_Values;
	};
}
