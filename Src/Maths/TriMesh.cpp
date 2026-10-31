#include "Maths/TriMesh.h"

#include <cmath>
#include <limits>
#include <utility>

namespace Evans::Maths
{
	TriMesh::TriMesh(std::vector<Vector3> vertices, std::vector<std::uint32_t> indices)
	{
		if (SetGeometry(std::move(vertices), std::move(indices)))
		{
			BuildAcceleration();
		}
	}

	bool TriMesh::SetGeometry(std::vector<Vector3> vertices, std::vector<std::uint32_t> indices,
		std::string* error)
	{
		auto fail = [error](const char* message) {
			if (error != nullptr)
			{
				*error = message;
			}
			return false;
		};

		if (vertices.empty() || indices.empty() || indices.size() % 3 != 0)
		{
			return fail("A triangle mesh requires vertices and triangle-list indices");
		}
		constexpr std::size_t MaximumTriangleCount =
			static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max() / 2u) + 1u;
		if (vertices.size() > std::numeric_limits<std::uint32_t>::max()
			|| indices.size() / 3 > MaximumTriangleCount)
		{
			return fail("Triangle mesh exceeds the 32-bit acceleration structure limits");
		}
		for (const Vector3& vertex : vertices)
		{
			if (!std::isfinite(vertex.x) || !std::isfinite(vertex.y) || !std::isfinite(vertex.z))
			{
				return fail("Triangle mesh vertices must be finite");
			}
		}
		for (const std::uint32_t index : indices)
		{
			if (index >= vertices.size())
			{
				return fail("Triangle mesh index is outside the vertex array");
			}
		}

		AABB bounds;
		for (const Vector3& vertex : vertices)
		{
			bounds.Expand(vertex);
		}
		m_Vertices = std::move(vertices);
		m_Indices = std::move(indices);
		m_Bounds = bounds;
		m_Acceleration.Clear();
		if (error != nullptr)
		{
			error->clear();
		}
		return true;
	}

	void TriMesh::Clear()
	{
		m_Vertices.clear();
		m_Indices.clear();
		m_Bounds = AABB::Empty();
		m_Acceleration.Clear();
	}

	bool TriMesh::BuildAcceleration(std::size_t maximumLeafSize)
	{
		return m_Acceleration.Build(*this, maximumLeafSize);
	}

	bool TriMesh::RayCast(const Ray& ray, TriMeshHit& hit, const TriMeshHitOptions& options) const
	{
		if (m_Acceleration.IsBuilt())
		{
			return m_Acceleration.RayCast(*this, ray, options, hit);
		}
		return RayCastBruteForce(ray, hit, options);
	}

	bool TriMesh::RayCastBruteForce(const Ray& ray, TriMeshHit& hit, const TriMeshHitOptions& options) const
	{
		hit = {};
		Ray distanceRay;
		if (!IsValid() || !ray.TryGetDistanceRay(distanceRay)
			|| !std::isfinite(options.MinimumDistance) || options.MinimumDistance < 0.0f
			|| std::isnan(options.MaximumDistance) || options.MaximumDistance < options.MinimumDistance)
		{
			return false;
		}

		bool found = false;
		float closestDistance = options.MaximumDistance;
		for (std::uint32_t triangleIndex = 0; triangleIndex < GetTriangleCount(); ++triangleIndex)
		{
			Vector3 a;
			Vector3 b;
			Vector3 c;
			GetTriangle(triangleIndex, a, b, c);
			++hit.TriangleTestCount;
			RayTriangleHit triangleHit;
			if (!IntersectRayTriangle(distanceRay, a, b, c, triangleHit, options.CullBackFaces,
				options.MinimumDistance, closestDistance))
			{
				continue;
			}
			static_cast<RayTriangleHit&>(hit) = triangleHit;
			hit.TriangleIndex = triangleIndex;
			closestDistance = triangleHit.Distance;
			found = true;
		}
		return found;
	}

	bool TriMesh::GetTriangle(std::uint32_t triangleIndex, Vector3& a, Vector3& b, Vector3& c) const
	{
		if (triangleIndex >= GetTriangleCount())
		{
			return false;
		}
		const std::size_t firstIndex = static_cast<std::size_t>(triangleIndex) * 3;
		a = m_Vertices[m_Indices[firstIndex]];
		b = m_Vertices[m_Indices[firstIndex + 1]];
		c = m_Vertices[m_Indices[firstIndex + 2]];
		return true;
	}
}
