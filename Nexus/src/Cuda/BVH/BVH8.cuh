#pragma once

#include <stdint.h>
#include "Cuda/Geometry/Triangle.cuh"

using byte = unsigned char;

// Compressed stack entry (we don't use it in the traversal algorithm, instead we use a uint2 for performance)
struct D_StackEntry
{
	struct D_Internal
	{
		// Base node index in this entry
		uint32_t childBaseIdx;

		// Field indicating for each node if it has not already been traversed (1 if not traversed)
		byte hits;

		// Dummy
		byte pad;

		// imask of the parent node
		byte imask;
	};

	struct D_Triangle
	{
		// Base triangle index
		uint32_t triangleBaseIdx;

		// Dummy
		unsigned pad : 8;

		// Field indicating for each triangle if it has not already been traversed (1 if not traversed)
		unsigned triangleHits : 24;
	};

	union
	{
		D_Internal internal;
		D_Triangle triangle;
	};
};
;

// Compressed BVH8 node
struct D_BVH8Node
{
	// P (12 bytes), e (3 bytes), imask (1 byte)
	float4 p_e_imask;

	// Child base index (4 bytes), triangle base index (4 bytes), meta (8 bytes)
	float4 childidx_tridx_meta;

	// qlox (8 bytes), qloy (8 bytes)
	float4 qlox_qloy;

	// qloz (8 bytes), qlix (8 bytes)
	float4 qloz_qhix;

	// qliy (8 bytes), qliz (8 bytes)
	float4 qhiy_qhiz;
};

struct D_BVH8
{
	D_Triangle* triangles;
	uint32_t* triangleIdx;
	uint32_t nodesUsed, triCount;
	D_BVH8Node* nodes;
};
