#include "Maths/StaticAABBTree.h"

#include "Maths/TriMesh.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace Evans::Maths
{
	bool StaticAABBTree::Build(const TriMesh& mesh, std::size_t maximumLeafSize)
	{
		Clear();
		if (!mesh.IsValid() || maximumLeafSize == 0)
		{
			return false;
		}

		m_PrimitiveIndices.resize(mesh.GetTriangleCount());
		std::iota(m_PrimitiveIndices.begin(), m_PrimitiveIndices.end(), 0u);
		m_Nodes.reserve(mesh.GetTriangleCount() * 2);
		BuildRecursive(mesh, 0, m_PrimitiveIndices.size(), maximumLeafSize, 0);
		return !m_Nodes.empty();
	}

	void StaticAABBTree::Clear()
	{
		m_Nodes.clear();
		m_PrimitiveIndices.clear();
	}

	std::size_t StaticAABBTree::GetLeafCount() const
	{
		return static_cast<std::size_t>(std::count_if(m_Nodes.begin(), m_Nodes.end(), [](const Node& node) {
			return node.IsLeaf();
		}));
	}

	const AABB& StaticAABBTree::GetBounds() const
	{
		static const AABB EmptyBounds;
		return m_Nodes.empty() ? EmptyBounds : m_Nodes.front().Bounds;
	}

	bool StaticAABBTree::RayCast(const TriMesh& mesh, const Ray& ray,
		const TriMeshHitOptions& options, TriMeshHit& hit) const
	{
		hit = {};
		Ray distanceRay;
		if (!IsBuilt() || !mesh.IsValid() || !ray.TryGetDistanceRay(distanceRay)
			|| !std::isfinite(options.MinimumDistance) || options.MinimumDistance < 0.0f
			|| std::isnan(options.MaximumDistance) || options.MaximumDistance < options.MinimumDistance)
		{
			return false;
		}

		struct StackEntry
		{
			std::uint32_t NodeIndex;
			float EnterDistance;
		};

		float rootEnter = 0.0f;
		float rootExit = 0.0f;
		++hit.AABBTestCount;
		if (!m_Nodes.front().Bounds.IntersectRay(distanceRay, rootEnter, rootExit, options.MaximumDistance))
		{
			return false;
		}

		std::vector<StackEntry> stack;
		stack.reserve(64);
		stack.push_back({ 0, rootEnter });
		float closestDistance = options.MaximumDistance;
		bool found = false;
		while (!stack.empty())
		{
			const StackEntry entry = stack.back();
			stack.pop_back();
			if (entry.EnterDistance > closestDistance)
			{
				continue;
			}

			const Node& node = m_Nodes[entry.NodeIndex];
			if (node.IsLeaf())
			{
				for (std::uint32_t offset = 0; offset < node.PrimitiveCount; ++offset)
				{
					const std::uint32_t triangleIndex = m_PrimitiveIndices[node.FirstPrimitive + offset];
					Vector3 a;
					Vector3 b;
					Vector3 c;
					mesh.GetTriangle(triangleIndex, a, b, c);
					++hit.TriangleTestCount;
					RayTriangleHit triangleHit;
					if (!IntersectRayTriangle(distanceRay, a, b, c, triangleHit, options.CullBackFaces,
						options.MinimumDistance, closestDistance))
					{
						continue;
					}
					if (options.AcceptHit && !options.AcceptHit(triangleIndex, triangleHit)) continue;
					static_cast<RayTriangleHit&>(hit) = triangleHit;
					hit.TriangleIndex = triangleIndex;
					closestDistance = triangleHit.Distance;
					found = true;
				}
				continue;
			}

			float leftEnter = 0.0f;
			float leftExit = 0.0f;
			float rightEnter = 0.0f;
			float rightExit = 0.0f;
			++hit.AABBTestCount;
			const bool hitLeft = m_Nodes[node.Left].Bounds.IntersectRay(
				distanceRay, leftEnter, leftExit, closestDistance);
			++hit.AABBTestCount;
			const bool hitRight = m_Nodes[node.Right].Bounds.IntersectRay(
				distanceRay, rightEnter, rightExit, closestDistance);
			if (hitLeft && hitRight)
			{
				if (leftEnter <= rightEnter)
				{
					stack.push_back({ node.Right, rightEnter });
					stack.push_back({ node.Left, leftEnter });
				}
				else
				{
					stack.push_back({ node.Left, leftEnter });
					stack.push_back({ node.Right, rightEnter });
				}
			}
			else if (hitLeft)
			{
				stack.push_back({ node.Left, leftEnter });
			}
			else if (hitRight)
			{
				stack.push_back({ node.Right, rightEnter });
			}
		}
		return found;
	}

	std::uint32_t StaticAABBTree::BuildRecursive(const TriMesh& mesh, std::size_t begin, std::size_t end,
		std::size_t maximumLeafSize, std::size_t depth)
	{
		const std::uint32_t nodeIndex = static_cast<std::uint32_t>(m_Nodes.size());
		m_Nodes.emplace_back();
		m_Nodes[nodeIndex].Bounds = CalculatePrimitiveBounds(mesh, begin, end);
		const std::size_t primitiveCount = end - begin;
		if (primitiveCount <= maximumLeafSize || depth >= 64)
		{
			m_Nodes[nodeIndex].FirstPrimitive = static_cast<std::uint32_t>(begin);
			m_Nodes[nodeIndex].PrimitiveCount = static_cast<std::uint32_t>(primitiveCount);
			return nodeIndex;
		}

		const AABB centroidBounds = CalculateCentroidBounds(mesh, begin, end);
		const int splitAxis = centroidBounds.LongestAxis();
		const std::size_t middle = begin + primitiveCount / 2;
		std::nth_element(m_PrimitiveIndices.begin() + static_cast<std::ptrdiff_t>(begin),
			m_PrimitiveIndices.begin() + static_cast<std::ptrdiff_t>(middle),
			m_PrimitiveIndices.begin() + static_cast<std::ptrdiff_t>(end),
			[&mesh, splitAxis](std::uint32_t leftTriangle, std::uint32_t rightTriangle) {
				Vector3 la;
				Vector3 lb;
				Vector3 lc;
				Vector3 ra;
				Vector3 rb;
				Vector3 rc;
				mesh.GetTriangle(leftTriangle, la, lb, lc);
				mesh.GetTriangle(rightTriangle, ra, rb, rc);
				return ((la + lb + lc) / 3.0f)[splitAxis] < ((ra + rb + rc) / 3.0f)[splitAxis];
			});

		const std::uint32_t left = BuildRecursive(mesh, begin, middle, maximumLeafSize, depth + 1);
		const std::uint32_t right = BuildRecursive(mesh, middle, end, maximumLeafSize, depth + 1);
		m_Nodes[nodeIndex].Left = left;
		m_Nodes[nodeIndex].Right = right;
		return nodeIndex;
	}

	AABB StaticAABBTree::CalculatePrimitiveBounds(const TriMesh& mesh, std::size_t begin, std::size_t end) const
	{
		AABB bounds;
		for (std::size_t index = begin; index < end; ++index)
		{
			Vector3 a;
			Vector3 b;
			Vector3 c;
			mesh.GetTriangle(m_PrimitiveIndices[index], a, b, c);
			bounds.Expand(AABB::FromTriangle(a, b, c));
		}
		return bounds;
	}

	AABB StaticAABBTree::CalculateCentroidBounds(const TriMesh& mesh, std::size_t begin, std::size_t end) const
	{
		AABB bounds;
		for (std::size_t index = begin; index < end; ++index)
		{
			Vector3 a;
			Vector3 b;
			Vector3 c;
			mesh.GetTriangle(m_PrimitiveIndices[index], a, b, c);
			bounds.Expand((a + b + c) / 3.0f);
		}
		return bounds;
	}
}
