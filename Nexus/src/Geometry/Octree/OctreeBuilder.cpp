#include "OctreeBuilder.h"
#include <numeric>
#include <map>
#include <set>
#include "Utils/Utils.h"

OctreeBuilder::OctreeBuilder(const std::vector<Triangle>& tri)
	:m_Triangles(tri) 
{
	m_TriangleIdx = std::vector<uint32_t>(tri.size());
	std::iota(m_TriangleIdx.begin(), m_TriangleIdx.end(), 0);
}

Octree OctreeBuilder::Build()
{
	OctreeNode root;
	root.triBaseIdx = 0;
	root.childBaseIdx = 1;
	root.triCount = m_Triangles.size();
	m_Nodes.push_back(root);
	UpdateRootBounds();
	Subdivide(0, root.splitSize);
	std::cout << "Tri Count: " << m_Triangles.size() << std::endl;
	std::cout << "Node Count: " << m_Nodes.size() << std::endl;
	Octree octree;
	octree.nodes = m_Nodes.data();
	octree.triangleIdx = m_TriangleIdx.data();
	octree.triangles = m_Triangles.data();
	return octree;
}

float OctreeBuilder::NodeCost(const uint32_t nodeIdx)
{
	const OctreeNode& node = m_Nodes[nodeIdx];
	return node.Area() * node.triCount;
}

void OctreeBuilder::Subdivide(const uint32_t nodeIdx, const uint32_t splitSize)
{
	OctreeNode node = m_Nodes[nodeIdx];

	if (node.triCount <= 2)
	{
		m_Nodes[nodeIdx].childBaseIdx = 0;
		return;
	}

	if (node.size <= m_Nodes[0].size / m_Triangles.size())
	{
		if (node.triCount > 100)
			std::cout << "Yes: " << node.triCount << std::endl;
		return;
	}

	uint32_t childCount = node.splitSize * node.splitSize * node.splitSize;

	float halfSize = m_Nodes[0].size / splitSize;
	for (size_t i = 0; i < childCount; i++)
	{
		OctreeNode child;
		child.triBaseIdx = m_Nodes[nodeIdx].triBaseIdx;

		// We take the size of the root node for better accuracy
		child.size = halfSize;
		//child.size = node.size / node.splitSize;
		child.inverseSize = 1.0f / child.size;
		child.parentIdx = nodeIdx;

		m_Nodes.push_back(child);
	}
	node = m_Nodes[nodeIdx];

	uint32_t assignedTriCount = 0;
	for (size_t i = node.triBaseIdx; i < node.triBaseIdx + node.triCount; i++)
	{
		std::set<uint32_t> assignedChildIdx;
		const Triangle& triangle = m_Triangles[m_TriangleIdx[i]];

		for (int j = 0; j < 3; j++)
		{
			float3 pos = j == 0 ? triangle.pos0 : (j == 1 ? triangle.pos1 : triangle.pos2);
			uint32_t relativeChildIdx = GetNodeIndex(node, pos);
			uint32_t childIdx = node.childBaseIdx + relativeChildIdx;
			OctreeNode& child = m_Nodes[childIdx];

			// If no node contains triangle yet
			if (assignedChildIdx.size() == 0)
			{
				child.triCount++;

				for (int k = childIdx + 1; k < node.childBaseIdx + childCount; k++)
				{
					Utils::Swap(m_TriangleIdx[i], m_TriangleIdx[m_Nodes[k - 1].triBaseIdx + m_Nodes[k - 1].triCount - 1]);
					m_Nodes[k].triBaseIdx++;
				}
				assignedTriCount++;
			}
			// If the specific node doesn't contain triangle yet
			else if (assignedChildIdx.find(childIdx) == assignedChildIdx.end())
			{
				child.triCount++;
				m_TriangleIdx.push_back(m_TriangleIdx[i]);

				// Maximum of 7 loops since node.firstChildIdx represents the last 8 nodes
				for (int k = childIdx + 1; k < m_Nodes.size(); k++)
				{
					Utils::Swap(m_TriangleIdx[m_TriangleIdx.size() - 1], m_TriangleIdx[m_Nodes[k - 1].triBaseIdx + m_Nodes[k - 1].triCount - 1]);
					m_Nodes[k].triBaseIdx++;
				}

				assignedTriCount++;
			}
			assignedChildIdx.insert(childIdx);
		}
	}

	// Split cost evaluation, assuming node.splitSize children are intersected in average
	//float splitCost = static_cast<float>(assignedTriCount) / node.splitSize * (node.splitSize * halfSize * halfSize);

	//if (splitCost >= NodeCost(nodeIdx))
	//{
	//	std::cout << "End of building, triCount: " << node.triCount << std::endl;
	//	return;
	//}

	for (size_t i = 0; i < childCount; i++)
	{
		OctreeNode& child = m_Nodes[node.childBaseIdx + i];
		child.childBaseIdx = m_Nodes.size();
		int z = i / (node.splitSize * node.splitSize);
		int x = i % node.splitSize;
		int y = (i - z * node.splitSize * node.splitSize) / node.splitSize;
		child.position = node.position + make_float3(x, y, z) * child.size;
		child.splitSize = 2;
		Subdivide(node.childBaseIdx + i, splitSize * child.splitSize);
	}
}

void OctreeBuilder::UpdateRootBounds()
{
	OctreeNode& root = m_Nodes[0];
	root.position = make_float3(1.0e30f);
	root.size = -1e30f;

	for (size_t i = 0; i < m_Triangles.size(); i++)
	{
		const Triangle& triangle = m_Triangles[i];
		root.position = fminf(root.position, triangle.pos0);
		root.position = fminf(root.position, triangle.pos1);
		root.position = fminf(root.position, triangle.pos2);
		root.size = fmax(root.size, fmaxf(triangle.pos0 - root.position));
		root.size = fmax(root.size, fmaxf(triangle.pos1 - root.position));
		root.size = fmax(root.size, fmaxf(triangle.pos2 - root.position));
	}
	root.inverseSize = 1.0f / root.size;
}

uint32_t OctreeBuilder::GetNodeIndex(const OctreeNode& node, const float3& position)
{
	int x = max(0, min((position.x - node.position.x) * node.inverseSize * node.splitSize, node.splitSize - 1));
	int y = max(0, min((position.y - node.position.y) * node.inverseSize * node.splitSize, node.splitSize - 1));
	int z = max(0, min((position.z - node.position.z) * node.inverseSize * node.splitSize, node.splitSize - 1));
	return z * node.splitSize * node.splitSize + y * node.splitSize + x;
}

