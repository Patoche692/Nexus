#pragma once
#include "Utils/cuda_math.h"
#include "BVH.h"


// Compressed wide BVH based on the paper "Efficient
// Incoherent Ray Traversal on GPUs Through Compressed Wide BVHs"
// See https://research.nvidia.com/sites/default/files/publications/ylitie2017hpg-paper.pdf


typedef unsigned char byte;

#define C_PRIM 0.3f // Cost of a ray-primitive intersection
#define C_NODE 1.0f // Cost of a ray-node intersection
#define P_MAX  3	// Maximum allowed leaf size
#define N_Q 8		// Number of bits used to store the childs' AABB coordinates


struct BVH8Node
{
	// Origin point of the local grid
	float3 p;

	// Scale of the grid
	byte e[3];

	// 8-bit mask to indicate which of the children are internal nodes
	byte imask = 0;

	// Index of the first child
	uint32_t childBaseIdx = 0;

	// Index of the first triangle
	uint32_t triangleBaseIdx = 0;

	// Field encoding the indexing information of every child
	byte meta[8];

	// Quantized origin of the childs' AABBs
	byte qlox[8], qloy[8], qloz[8];

	// Quantized end point of the childs' AABBs
	byte qhix[8], qhiy[8], qhiz[8];
};

struct BVH8
{
	BVH8() = default;
	BVH8(BVH* bvh2);
	~BVH8();

	Triangle* triangles = nullptr;
	uint32_t* triangleIdx = nullptr;
	uint32_t nodesUsed, triCount;

	BVH8Node* nodes = nullptr;
};
