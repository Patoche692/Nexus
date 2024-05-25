#pragma once

#include <cuda_runtime_api.h>
#include "Geometry/BVH/BVH8.h"

#define WARP_SIZE 32	// Same size for all NVIDIA GPUs

// If the ratio of active threads in a warp is less than POSTPONE_RATIO_THRESHOLD, postpone triangle intersection
#define POSTPONE_RATIO_THRESHOLD 0.2

// Compressed stack entry (we don't use it in the traversal algorithm)
struct StackEntry
{
	struct Internal {
		// Base node index in this entry
		uint32_t childBaseIdx;

		// Field indicating for each node if it has not already been traversed (1 if not traversed)
		byte hits;

		// Dummy
		byte pad;

		// imask of the parent node
		byte imask;
	};
	struct Triangle {
		// Base triangle index
		uint32_t triangleBaseIdx;

		// Dummy
		unsigned pad : 8;

		// Field indicating for each triangle if it has not already been traversed (1 if not traversed)
		unsigned triangleHits : 24;
	};
	union {
		Internal internal;
		Triangle triangle;
	};
};

inline __device__ void IntersectChildren(const BVH8Node& bvh8node, Ray& ray, uint2& internalEntry, uint2 triangleEntry)
{
	float3 transformedDirection = make_float3(
		exp2f(bvh8node.e[0]) / ray.direction.x,
		exp2f(bvh8node.e[1]) / ray.direction.y,
		exp2f(bvh8node.e[2]) / ray.direction.z
	);

	float3 transformedOrigin = (bvh8node.p - ray.origin) / ray.direction;


	float3 sth = sth * transformedDirection + transformedOrigin;
	float3 sth = sth * transformedDirection + transformedOrigin;
}

inline __device__ void IntersectBVH8(const BVH8& bvh8, Ray& ray, const uint32_t instanceIdx)
{
	uint2 stack[32];
	int stackPtr = 0;

	uint2 nodeEntry = make_uint2(0);
	uint2 triangleEntry = make_uint2(0);

	// We set the hit bit of the root node to 1
	nodeEntry.y |= 0x80000000;


	while (true)
	{
		// If the hit field is different from 0, it is an internal node entry
		if (nodeEntry.y & 0xff000000)
		{
			int n = GetClosestNode(bvh8, nodeEntry, ray);

			// Set the hit bit of the selected node to 0
			nodeEntry.y &= ~(1 << (31 - n));

			if (nodeEntry.y & 0xff000000)
			{
				stack[stackPtr++] = nodeEntry;
			};

			const BVH8Node& node = bvh8.nodes[nodeEntry.x + n];

			IntersectChildren(node, ray, nodeEntry, triangleEntry);
		}
		else
		{
			triangleEntry = nodeEntry;
			nodeEntry = make_uint2(0);
		}

		while (triangleEntry.y)
		{
			float ratio = __popc(__activemask()) / (float)WARP_SIZE;

			// If the ratio of active threads in the warp is less than the threshold, postpone triangle intersection
			if (ratio < POSTPONE_RATIO_THRESHOLD)
			{
				stack[stackPtr++] = triangleEntry;
				break;
			}

			int triangleIdx = GetNextTriangle(triangleEntry);
			triangleEntry.y &= ~(1 << (23 - triangleIdx));
			bvh8.triangles[triangleIdx].Hit(ray, instanceIdx, triangleIdx);
		}

		// If the node entry is empty (hit field equals 0), pop from the stack
		if ((nodeEntry.y & 0xff000000) == 0)
		{
			if (stackPtr == 0)
				break;

			nodeEntry = stack[--stackPtr];
		}
	}

}
