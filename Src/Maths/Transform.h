#pragma once

#include "Matrix4.h"

#include <cmath>

namespace Evans::Maths
{
	struct Transform
	{
		Vector3 Translation = Vector3::Zero();
		Quaternion Rotation = Quaternion::Identity();
		Vector3 Scale = Vector3::One();

		Transform() = default;
		explicit Transform(const Vector3& translation) : Translation(translation) {}
		Transform(const Vector3& translation, const Quaternion& rotation, const Vector3& scale = Vector3::One())
			: Translation(translation), Rotation(rotation.Normalized()), Scale(scale) {}

		Vector3 TransformPoint(const Vector3& point) const
		{
			return Translation + Rotation.Rotate(Scale * point);
		}
		Vector3 TransformVector(const Vector3& vector) const
		{
			return Rotation.Rotate(Scale * vector);
		}
		Vector3 TransformDirection(const Vector3& direction) const
		{
			return Rotation.Rotate(direction);
		}
		Vector3 InverseTransformPoint(const Vector3& point, float epsilon = 1.0e-8f) const
		{
			return SafeReciprocal(Scale, epsilon) * Rotation.Inversed().Rotate(point - Translation);
		}
		Vector3 InverseTransformVector(const Vector3& vector, float epsilon = 1.0e-8f) const
		{
			return SafeReciprocal(Scale, epsilon) * Rotation.Inversed().Rotate(vector);
		}
		Vector3 InverseTransformDirection(const Vector3& direction) const
		{
			return Rotation.Inversed().Rotate(direction);
		}
		Matrix4 ToMatrix() const { return Matrix4::TRS(Translation, Rotation, Scale); }
		Matrix4 ToInverseMatrix() const { return ToMatrix().Inversed(); }

		// Composition is exact when the parent scale is uniform. Non-uniform scale
		// followed by rotation can introduce shear and should use Matrix4 directly.
		Transform operator*(const Transform& child) const
		{
			return {
				TransformPoint(child.Translation),
				(Rotation * child.Rotation).Normalized(),
				Scale * child.Scale
			};
		}

		static Transform Identity() { return {}; }

	private:
		static Vector3 SafeReciprocal(const Vector3& value, float epsilon)
		{
			return {
				std::fabs(value.x) > epsilon ? 1.0f / value.x : 0.0f,
				std::fabs(value.y) > epsilon ? 1.0f / value.y : 0.0f,
				std::fabs(value.z) > epsilon ? 1.0f / value.z : 0.0f
			};
		}
	};
}
