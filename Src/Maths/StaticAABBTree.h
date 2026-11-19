#pragma once

#include "Geometry.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace Evans::Maths
{
	class TriMesh;
	struct TriMeshHit;
	struct TriMeshHitOptions;

	class StaticAABBTree
	{
	public:
		static constexpr std::uint32_t InvalidIndex = std::numeric_limits<std::uint32_t>::max();

		struct Node
		{
			AABB Bounds;
			std::uint32_t Left = InvalidIndex;
			std::uint32_t Right = InvalidIndex;
			std::uint32_t FirstPrimitive = 0;
			std::uint32_t PrimitiveCount = 0;

			bool IsLeaf() const { return PrimitiveCount != 0; }
		};

		bool Build(const TriMesh& mesh, std::size_t maximumLeafSize = 4);
		void Clear();
		bool IsBuilt() const { return !m_Nodes.empty(); }
		std::size_t GetNodeCount() const { return m_Nodes.size(); }
		std::size_t GetLeafCount() const;
		const AABB& GetBounds() const;
		const std::vector<Node>& GetNodes() const { return m_Nodes; }
		const std::vector<std::uint32_t>& GetPrimitiveIndices() const { return m_PrimitiveIndices; }
		bool RayCast(const TriMesh& mesh, const Ray& ray, const TriMeshHitOptions& options, TriMeshHit& hit) const;

	private:
		std::uint32_t BuildRecursive(const TriMesh& mesh, std::size_t begin, std::size_t end,
			std::size_t maximumLeafSize, std::size_t depth);
		AABB CalculatePrimitiveBounds(const TriMesh& mesh, std::size_t begin, std::size_t end) const;
		AABB CalculateCentroidBounds(const TriMesh& mesh, std::size_t begin, std::size_t end) const;

		std::vector<Node> m_Nodes;
		std::vector<std::uint32_t> m_PrimitiveIndices;
	};
}
