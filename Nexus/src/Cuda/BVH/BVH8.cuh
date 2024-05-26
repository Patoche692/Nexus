#pragma once

#include <cuda_runtime_api.h>
#include <cudart_platform.h>
#include <device_launch_parameters.h>
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

__device__ __inline__ uint SignExtendS8x4(uint i) { uint v; asm("prmt.b32 %0, %1, 0x0, 0x0000BA98;" : "=r"(v) : "r"(i)); return v; }

__device__ float vMaxMax(float a, float b, float c) {
	int result;

	asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(result) : "r"(__float_as_int(a)), "r"(__float_as_int(b)), "r"(__float_as_int(c)));

	return __int_as_float(result);
}

__device__ float vMinMin(float a, float b, float c) {
	int result;

	asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(result) : "r"(__float_as_int(a)), "r"(__float_as_int(b)), "r"(__float_as_int(c)));

	return __int_as_float(result);
}

__device__ unsigned ExtractByte(unsigned x, unsigned i) {
	return (x >> (i * 8)) & 0xff;
}

inline __device__ void IntersectChildren(const BVH8Node& bvh8node, Ray& ray, const uint32_t invOctant, uint2& internalEntry, uint2& triangleEntry)
{
	// We can easily compute 2^ei by shifting ei to the 8 exponent bits and interpreting the result as a float
	float3 transformedDirection = make_float3(
		__uint_as_float(bvh8node.e[0] << 23) * ray.invDirection.x,
		__uint_as_float(bvh8node.e[1] << 23) * ray.invDirection.y,
		__uint_as_float(bvh8node.e[2] << 23) * ray.invDirection.z
	);

	uint32_t hitMask = 0;

	float3 transformedOrigin = (bvh8node.p - ray.origin) * ray.invDirection;

	const uint32_t invOctant4 = invOctant * 0x01010101;

	for (int i = 0; i < 2; i++)
	{
		uint32_t meta4 = (i == 0) ? bvh8node.meta[0] << 24 | bvh8node.meta[1] << 16 | bvh8node.meta[2] << 8 | bvh8node.meta[3] :
			bvh8node.meta[4] << 24 | bvh8node.meta[5] << 16 | bvh8node.meta[6] << 8 | bvh8node.meta[7];
		uint32_t isInner4 = (meta4 & (meta4 << 1)) & 0x01010101;
		uint32_t innerMask4 = SignExtendS8x4(isInner4 << 3);
		uint32_t bitIndex4 = (meta4 ^ (invOctant4 & innerMask4)) & 0x1f1f1f1f;

		uint32_t childBits4 = (meta4 >> 5) & 0x07070707;

		const uint32_t qlox = (i == 0) ? bvh8node.qlox[0] << 24 | bvh8node.qlox[1] << 16 | bvh8node.qlox[2] << 8 | bvh8node.qlox[3] :
			bvh8node.qlox[0] << 24 | bvh8node.qlox[1] << 16 | bvh8node.qlox[2] << 8 | bvh8node.qlox[3];
		const uint32_t qhix = (i == 0) ? bvh8node.qhix[0] << 24 | bvh8node.qhix[1] << 16 | bvh8node.qhix[2] << 8 | bvh8node.qhix[3] :
			bvh8node.qhix[5] << 24 | bvh8node.qhix[6] << 16 | bvh8node.qhix[7] << 8 | bvh8node.qhix[8];

		const uint32_t qloy = (i == 0) ? bvh8node.qloy[0] << 24 | bvh8node.qloy[1] << 16 | bvh8node.qloy[2] << 8 | bvh8node.qloy[3] :
			bvh8node.qloy[0] << 24 | bvh8node.qloy[1] << 16 | bvh8node.qloy[2] << 8 | bvh8node.qloy[3];
		const uint32_t qhiy = (i == 0) ? bvh8node.qhiy[0] << 24 | bvh8node.qhiy[1] << 16 | bvh8node.qhiy[2] << 8 | bvh8node.qhiy[3] :
			bvh8node.qhiy[5] << 24 | bvh8node.qhiy[6] << 16 | bvh8node.qhiy[7] << 8 | bvh8node.qhiy[8];

		const uint32_t qloz = (i == 0) ? bvh8node.qloz[0] << 24 | bvh8node.qloz[1] << 16 | bvh8node.qloz[2] << 8 | bvh8node.qloz[3] :
			bvh8node.qloz[0] << 24 | bvh8node.qloz[1] << 16 | bvh8node.qloz[2] << 8 | bvh8node.qloz[3];
		const uint32_t qhiz = (i == 0) ? bvh8node.qhiz[0] << 24 | bvh8node.qhiz[1] << 16 | bvh8node.qhiz[2] << 8 | bvh8node.qhiz[3] :
			bvh8node.qhiz[5] << 24 | bvh8node.qhiz[6] << 16 | bvh8node.qhiz[7] << 8 | bvh8node.qhiz[8];

		uint32_t xMin = ray.direction.x < 0.0f ? qhix : qlox;
		uint32_t xMax = ray.direction.x < 0.0f ? qlox : qhix;

		uint32_t yMin = ray.direction.y < 0.0f ? qhiy : qloy;
		uint32_t yMax = ray.direction.y < 0.0f ? qloy : qhiy;

		uint32_t zMin = ray.direction.z < 0.0f ? qhiz : qloz;
		uint32_t zMax = ray.direction.z < 0.0f ? qloz : qhiz;

		for (int j = 0; j < 4; j++) {
			// Extract j-th byte
			float3 tmin3 = make_float3(float(ExtractByte(xMin, j)), float(ExtractByte(yMin, j)), float(ExtractByte(zMin, j)));
			float3 tmax3 = make_float3(float(ExtractByte(xMax, j)), float(ExtractByte(yMax, j)), float(ExtractByte(zMax, j)));

			// Account for grid origin and scale
			tmin3 = tmin3 * transformedDirection + transformedOrigin;
			tmax3 = tmax3 * transformedDirection + transformedOrigin;

			float tmin = vMaxMax(tmin3.x, tmin3.y, fmaxf(tmin3.z, 0.0f));
			float tmax = vMinMin(tmax3.x, tmax3.y, fminf(tmax3.z, ray.hit.t));

			bool intersected = tmin < tmax;
			if (intersected) {
				unsigned child_bits = ExtractByte(childBits4, j);
				unsigned bit_index  = ExtractByte(bitIndex4,  j);

				hitMask |= child_bits << bit_index;
			}
		}
	}
	internalEntry.x = bvh8node.childBaseIdx;
	internalEntry.y = (hitMask & 0xff000000) | (uint32_t)bvh8node.imask;

	triangleEntry.x = bvh8node.triangleBaseIdx;
	triangleEntry.y = (hitMask & 0x00ffffff);
}

inline __device__ uint32_t Octant(const float3& a)
{
	return ((a.x < 0 ? 1 : 0) << 2) | ((a.y < 0 ? 1 : 0) << 1) | ((a.z < 0 ? 1 : 0));
}


inline __device__ void IntersectBVH8(const BVH8& bvh8, Ray& ray, const uint32_t instanceIdx)
{
	uint2 stack[32];
	int stackPtr = 0;

	uint2 nodeEntry = make_uint2(0);
	uint2 triangleEntry = make_uint2(0);
	const uint32_t invOctant = 7 - Octant(ray.direction);

	// We set the hits bit of the root node to 1
	nodeEntry.y |= 0x80000000;

	while (true)
	{
		// If the hits field is different from 0, it is an internal node entry
		if (nodeEntry.y & 0xff000000)
		{
			// Position of the first non zero bit
			const int nodeOffset = 31 - __clz(nodeEntry.y);

			// Set the hits bit of the selected node to 0
			nodeEntry.y &= ~(1 << nodeOffset);

			// If some nodes are remaining in the hits field
			if (nodeEntry.y & 0xff000000)
			{
				stack[stackPtr++] = nodeEntry;
			};

			// Slot in (0 .. 7) referring to the octant order in which the node should be traversed
			const int nodeSlot = (nodeOffset - 24) ^ invOctant;

			// We need to account for the number of internal nodes in the parent node. The relative
			// index is thus the number of neighboring nodes stored in the lower child slots
			const int relativeNodeIdx = __popc(nodeEntry.y & ~(0xffffffff << nodeSlot));

			const BVH8Node& node = bvh8.nodes[nodeEntry.x + relativeNodeIdx];

			IntersectChildren(node, ray, invOctant, nodeEntry, triangleEntry);
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

			int triangleIdx = 31 - __clz(triangleEntry.x);
			triangleEntry.y &= ~(1 << (triangleIdx));

			bvh8.triangles[triangleIdx].Hit(ray, instanceIdx, triangleIdx);
		}

		// If the node entry is empty (hits field equals 0), pop from the stack
		if ((nodeEntry.y & 0xff000000) == 0)
		{
			if (stackPtr == 0)
				break;

			nodeEntry = stack[--stackPtr];
		}
	}

}
