#pragma once

#include "StaticAABBTree.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace Evans::Maths
{
	struct TriMeshHitOptions
	{
		bool CullBackFaces = false;
		float MinimumDistance = 1.0e-5f;
		float MaximumDistance = std::numeric_limits<float>::infinity();
		// Rejecting a candidate must not shorten the search interval. This lets
		// material-aware callers trace through cutouts, including coplanar layers.
		std::function<bool(std::uint32_t, const RayTriangleHit&)> AcceptHit;
	};

	struct TriMeshHit : RayTriangleHit
	{
		std::uint32_t TriangleIndex = std::numeric_limits<std::uint32_t>::max();
		std::uint32_t AABBTestCount = 0;
		std::uint32_t TriangleTestCount = 0;
	};

	class TriMesh
	{
	public:
		TriMesh() = default;
		TriMesh(std::vector<Vector3> vertices, std::vector<std::uint32_t> indices);

		bool SetGeometry(std::vector<Vector3> vertices, std::vector<std::uint32_t> indices,
			std::string* error = nullptr);
		void Clear();
		bool BuildAcceleration(std::size_t maximumLeafSize = 4);
		bool RayCast(const Ray& ray, TriMeshHit& hit, const TriMeshHitOptions& options = {}) const;
		bool RayCastBruteForce(const Ray& ray, TriMeshHit& hit, const TriMeshHitOptions& options = {}) const;

		bool IsValid() const { return !m_Vertices.empty() && !m_Indices.empty(); }
		std::size_t GetVertexCount() const { return m_Vertices.size(); }
		std::size_t GetTriangleCount() const { return m_Indices.size() / 3; }
		const std::vector<Vector3>& GetVertices() const { return m_Vertices; }
		const std::vector<std::uint32_t>& GetIndices() const { return m_Indices; }
		const AABB& GetBounds() const { return m_Bounds; }
		const StaticAABBTree& GetAcceleration() const { return m_Acceleration; }
		bool GetTriangle(std::uint32_t triangleIndex, Vector3& a, Vector3& b, Vector3& c) const;

	private:
		friend class StaticAABBTree;

		std::vector<Vector3> m_Vertices;
		std::vector<std::uint32_t> m_Indices;
		AABB m_Bounds;
		StaticAABBTree m_Acceleration;
	};
}
