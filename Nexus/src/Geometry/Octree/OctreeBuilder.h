#pragma once
#include <vector>
#include "Geometry/Triangle.h"
#include "Geometry/Octree/Octree.h"

class OctreeBuilder
{
public:
	OctreeBuilder(const std::vector<Triangle>& tri);
	Octree Build();

private:
	// We include splitSize to avoid precision errors
	void Subdivide(const uint32_t nodeIdx, const uint32_t splitSize);

	void UpdateRootBounds();
	uint32_t GetNodeIndex(const OctreeNode& node, const float3& position);
	float NodeCost(const uint32_t nodeIdx);

private:
	std::vector<Triangle> m_Triangles;
	std::vector<uint32_t> m_TriangleIdx;
	std::vector<OctreeNode> m_Nodes;
};