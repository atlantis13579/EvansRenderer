#pragma once

#include "Vector3.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace Evans::Maths
{
	struct Ray
	{
		Vector3 Origin = Vector3::Zero();
		Vector3 Direction = Vector3::UnitZ();

		Ray() = default;
		Ray(const Vector3& origin, const Vector3& direction, bool normalizeDirection = true)
			: Origin(origin)
			, Direction(normalizeDirection ? direction.Normalized() : direction)
		{
		}

		Vector3 PointAt(float distance) const { return Origin + Direction * distance; }
		bool IsFinite() const
		{
			return std::isfinite(Origin.x) && std::isfinite(Origin.y) && std::isfinite(Origin.z)
				&& std::isfinite(Direction.x) && std::isfinite(Direction.y) && std::isfinite(Direction.z);
		}
		bool TryGetDistanceRay(Ray& result, float epsilon = 1.0e-8f) const
		{
			if (!IsFinite())
			{
				return false;
			}
			const float lengthSquared = Direction.LengthSquared();
			if (!std::isfinite(lengthSquared) || lengthSquared <= epsilon * epsilon)
			{
				return false;
			}
			// Intersection distances are measured in world units regardless of the input ray scale.
			result.Origin = Origin;
			result.Direction = std::fabs(lengthSquared - 1.0f) <= 1.0e-6f
				? Direction
				: Direction / std::sqrt(lengthSquared);
			return true;
		}
		static Ray FromPoints(const Vector3& origin, const Vector3& destination)
		{
			return { origin, destination - origin, true };
		}
	};

	struct AABB
	{
		Vector3 Min;
		Vector3 Max;

		AABB()
			: Min(std::numeric_limits<float>::infinity())
			, Max(-std::numeric_limits<float>::infinity())
		{
		}
		AABB(const Vector3& minimum, const Vector3& maximum) : Min(minimum), Max(maximum) {}

		bool IsValid() const { return Min.x <= Max.x && Min.y <= Max.y && Min.z <= Max.z; }
		Vector3 Center() const { return IsValid() ? (Min + Max) * 0.5f : Vector3::Zero(); }
		Vector3 Extents() const { return IsValid() ? (Max - Min) * 0.5f : Vector3::Zero(); }
		float SurfaceArea() const
		{
			if (!IsValid()) return 0.0f;
			const Vector3 size = Max - Min;
			return 2.0f * (size.x * size.y + size.y * size.z + size.z * size.x);
		}
		int LongestAxis() const { return (Max - Min).LargestAxis(); }
		void Expand(const Vector3& point)
		{
			Min = Vector3::Min(Min, point);
			Max = Vector3::Max(Max, point);
		}
		void Expand(const AABB& bounds)
		{
			if (!bounds.IsValid()) return;
			Expand(bounds.Min);
			Expand(bounds.Max);
		}
		bool Contains(const Vector3& point) const
		{
			return IsValid() && point.x >= Min.x && point.y >= Min.y && point.z >= Min.z
				&& point.x <= Max.x && point.y <= Max.y && point.z <= Max.z;
		}
		bool Intersects(const AABB& other) const
		{
			return IsValid() && other.IsValid()
				&& Min.x <= other.Max.x && Max.x >= other.Min.x
				&& Min.y <= other.Max.y && Max.y >= other.Min.y
				&& Min.z <= other.Max.z && Max.z >= other.Min.z;
		}
		bool IntersectRay(const Ray& ray, float& enterDistance, float& exitDistance,
			float maximumDistance = std::numeric_limits<float>::infinity()) const
		{
			Ray distanceRay;
			if (!IsValid() || std::isnan(maximumDistance) || maximumDistance < 0.0f
				|| !ray.TryGetDistanceRay(distanceRay))
			{
				return false;
			}
			float enter = 0.0f;
			float exit = maximumDistance;
			for (std::size_t axis = 0; axis < 3; ++axis)
			{
				const float direction = distanceRay.Direction[axis];
				if (std::fabs(direction) <= 1.0e-8f)
				{
					if (distanceRay.Origin[axis] < Min[axis] || distanceRay.Origin[axis] > Max[axis])
					{
						return false;
					}
					continue;
				}
				const float inverseDirection = 1.0f / direction;
				float nearDistance = (Min[axis] - distanceRay.Origin[axis]) * inverseDirection;
				float farDistance = (Max[axis] - distanceRay.Origin[axis]) * inverseDirection;
				if (nearDistance > farDistance)
				{
					std::swap(nearDistance, farDistance);
				}
				enter = std::max(enter, nearDistance);
				exit = std::min(exit, farDistance);
				if (enter > exit)
				{
					return false;
				}
			}
			enterDistance = enter;
			exitDistance = exit;
			return exit >= 0.0f;
		}

		static AABB FromTriangle(const Vector3& a, const Vector3& b, const Vector3& c)
		{
			AABB result;
			result.Expand(a);
			result.Expand(b);
			result.Expand(c);
			return result;
		}
		static AABB Empty() { return {}; }
	};

	struct RayTriangleHit
	{
		float Distance = std::numeric_limits<float>::infinity();
		Vector3 Position = Vector3::Zero();
		Vector3 Normal = Vector3::Zero();
		Vector3 Barycentric = Vector3::Zero();
		bool FrontFace = true;
	};

	inline bool IntersectRayTriangle(const Ray& ray, const Vector3& a, const Vector3& b, const Vector3& c,
		RayTriangleHit& hit, bool cullBackFaces = false, float minimumDistance = 1.0e-6f,
		float maximumDistance = std::numeric_limits<float>::infinity())
	{
		constexpr float DeterminantEpsilon = 1.0e-8f;
		Ray distanceRay;
		if (!std::isfinite(minimumDistance) || minimumDistance < 0.0f
			|| std::isnan(maximumDistance) || maximumDistance < minimumDistance
			|| !ray.TryGetDistanceRay(distanceRay))
		{
			return false;
		}
		const Vector3 edgeAB = b - a;
		const Vector3 edgeAC = c - a;
		const float edgeABLengthSquared = edgeAB.LengthSquared();
		const float edgeACLengthSquared = edgeAC.LengthSquared();
		const Vector3 geometricNormal = edgeAB.Cross(edgeAC);
		const float normalLengthSquared = geometricNormal.LengthSquared();
		if (!std::isfinite(edgeABLengthSquared) || !std::isfinite(edgeACLengthSquared)
			|| !std::isfinite(normalLengthSquared) || normalLengthSquared <=
				1.0e-16f * edgeABLengthSquared * edgeACLengthSquared)
		{
			return false;
		}
		const Vector3 determinantVector = distanceRay.Direction.Cross(edgeAC);
		const float determinant = edgeAB.Dot(determinantVector);
		if (!std::isfinite(determinant)
			|| (cullBackFaces && determinant <= DeterminantEpsilon)
			|| (!cullBackFaces && std::fabs(determinant) <= DeterminantEpsilon))
		{
			return false;
		}

		// Ray-aligned edge functions evaluate a shared edge from the same two
		// transformed endpoints in either triangle. Independent Moller-Trumbore
		// barycentric divisions can round both adjacent triangles to "outside".
		// Use double intermediates throughout this CPU variant; retain the
		// existing degeneracy, facing and distance-interval contracts above.
		// Woop, Benthin and Wald, "Watertight Ray/Triangle Intersection" (2013):
		// https://jcgt.org/published/0002/01/05/
		const auto& direction = distanceRay.Direction;
		const int depthAxis = Vector3(std::abs(direction.x),std::abs(direction.y),std::abs(direction.z)).LargestAxis();
		const int horizontalAxis = (depthAxis+1)%3, verticalAxis = (horizontalAxis+1)%3;
		const double shearX = double(direction[horizontalAxis])/direction[depthAxis];
		const double shearY = double(direction[verticalAxis])/direction[depthAxis];
		const double inverseDepth = 1.0/direction[depthAxis];
		double projectedX[3], projectedY[3], projectedDepth[3];
		const Vector3 corners[] = {a,b,c};
		for (int i = 0; i < 3; ++i)
		{
			const double depth = double(corners[i][depthAxis])-distanceRay.Origin[depthAxis];
			projectedX[i] = double(corners[i][horizontalAxis])-distanceRay.Origin[horizontalAxis]-shearX*depth;
			projectedY[i] = double(corners[i][verticalAxis])-distanceRay.Origin[verticalAxis]-shearY*depth;
			projectedDepth[i] = depth*inverseDepth;
		}
		const double edgeA = projectedX[1]*projectedY[2]-projectedY[1]*projectedX[2];
		const double edgeB = projectedX[2]*projectedY[0]-projectedY[2]*projectedX[0];
		const double edgeC = projectedX[0]*projectedY[1]-projectedY[0]*projectedX[1];
		if ((edgeA < 0 || edgeB < 0 || edgeC < 0) && (edgeA > 0 || edgeB > 0 || edgeC > 0)) return false;
		const double area = edgeA+edgeB+edgeC;
		if (!std::isfinite(area) || area == 0) return false;
		const double parameter = (edgeA*projectedDepth[0]+edgeB*projectedDepth[1]+edgeC*projectedDepth[2])/area;
		if (!std::isfinite(parameter) || parameter < minimumDistance || parameter > maximumDistance) return false;
		const float distance = static_cast<float>(parameter);
		if (!std::isfinite(distance)) return false;

		hit.Distance = distance;
		hit.Position = distanceRay.PointAt(distance);
		hit.Normal = geometricNormal.Normalized();
		hit.Barycentric = { static_cast<float>(edgeA/area), static_cast<float>(edgeB/area), static_cast<float>(edgeC/area) };
		hit.FrontFace = determinant > 0.0f;
		return true;
	}
}
