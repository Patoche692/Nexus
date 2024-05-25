#pragma once

#include "Geometry/BVH/BVH8.h"

struct StackEntry
{
	struct Internal {
		uint32_t childBaseIdx;
		byte hits;
		byte pad;
		byte imask;
	};
	struct Triangle {
		uint32_t triangleBaseIdx;
		unsigned pad : 8;
		unsigned triangleHits : 24;
	};
	union {
		Internal internal;
		Triangle triangle;
	};
};

inline __device__ void IntersectNode(const BVH8Node& bvh8node, Ray& ray, const uint32_t instanceIdx)
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

	uint2 currentEntry = make_uint2(0, 0);
	currentEntry.y |= 0x80000000;
	uint2 internalEntry;
	uint2 triangleEntry;

	while (1) {
		if (currentEntry.y & 0xff000000) {

			int n = GetClosestNode(bvh8, currentEntry, ray);
			currentEntry.y &= ~(1 << (n + 24));

			const BVH8Node& node = bvh8.nodes[currentEntry.x + n];

			if (currentEntry.y & 0xff000000) {
				stack[stackPtr++] = currentEntry;
			};

			IntersectNode(node, ray, instanceIdx);	// TODO internalEntry and 

		}
		else {
			triangleEntry = currentEntry;
			currentEntry = make_uint2(0);
		}
		while (triangleEntry.y) {
			
			float ratio = __popc(__activemask()) / 32.0f;
			if (ratio < 0.2) {
				stack[stackPtr++] = triangleEntry;
				break;
			}

			int triangleIdx = GetNextTriangle(triangleEntry);
			triangleEntry.y &= ~(1 << (triangleIdx));
			bvh8.triangles[triangleIdx].Hit(ray, instanceIdx, triangleIdx);
		}
		if ((currentEntry.y & 0xff000000) == 0) {
			if (stackPtr == 0)
				break;
			currentEntry = stack[--stackPtr];
		}
	}

}
