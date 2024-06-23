#pragma once

#include <vector>
#include "Utils/cuda_math.h"
#include "Utils/Utils.h"
#include "Geometry/Triangle.h"

// My own implementation of a hybrid tree, mixing octree and bvh

#define EPSILON 1.0e-6f

inline __device__ unsigned ExtractByte(unsigned x, unsigned i) {
	return (x >> (i * 8)) & 0xff;
}

using byte = unsigned char;

inline __device__ uint32_t Octant(const float3& a)
{
	return ((a.x < 0 ? 1 : 0) << 2) | ((a.y < 0 ? 1 : 0) << 1) | ((a.z < 0 ? 1 : 0));
}

struct BVH8Node
{
	float3 position;
	byte e[3];
	byte imask;
	uint32_t childBaseIdx;
};

struct AWBVHNode
{

	union {
		struct Internal
		{
			float3 position;
			float3 bMax;
			uint32_t childBaseIdx;
			byte childMask;
			uint16_t splitCount_splitPos;
		} internal;

		struct Leaf
		{
			uint32_t triBaseIdx;
			uint32_t triCount;
		} leaf;
	};

	inline __host__ __device__ bool IsLeaf() { return internal.childMask == 0; }

	inline __host__ __device__ byte GetChildIdx(const float3& pos)
	{
		const float xPlane = internal.position.x + (internal.bMax.x - internal.position.x) * exp2f(-3) * ((internal.splitCount_splitPos >> 6) & 0xfff7 + 1);
		const float yPlane = internal.position.y + (internal.bMax.y - internal.position.y) * exp2f(-3) * ((internal.splitCount_splitPos >> 3) & 0xfff7 + 1);
		const float zPlane = internal.position.z + (internal.bMax.z - internal.position.z) * exp2f(-3) * (internal.splitCount_splitPos & 0xfff7 + 1);
		return ((pos.x > xPlane) & (internal.splitCount_splitPos >> 11)) << 2 |
			   ((pos.y > yPlane) & (internal.splitCount_splitPos >> 10)) << 1 |
			   ((pos.z > zPlane) & (internal.splitCount_splitPos >> 9));
	}

	inline __host__ __device__ void Intersect(Ray& ray, uint32_t instanceIdx)
	{
		OctreeNode
	}

	inline __host__ __device__ uint32_t GetNextNode(const Ray& ray, char currentChildIdx)
	{
		byte octant = Octant(ray.direction);
		const float xPlane = currentChildIdx & 0x1 ? 
	}
};

struct OctreeNode
{
	float3 position = make_float3(0.0f);
	float3 size = make_float3(0.0f);

	uint32_t childBaseIdx = 0;
	uint32_t triBaseIdx = 0;
	uint32_t parentIdx = 0;
	uint32_t triCount = 0;
	unsigned char splitSize = 2;

	inline float Cost() const {
		return Area() * triCount;
	}
	inline float Area() const {
		return 6 * size * size;
	}

	inline __host__ __device__ bool IsLeaf() { return childBaseIdx = 0; }

	inline __host__ __device__ uint3 GetChildIdx(const float3& pos)
	{
		const int x = max(0, min((int)((pos.x - position.x) * inverseSize * splitSize), (int)(splitSize - 1)));
		const int y = max(0, min((int)((pos.y - position.y) * inverseSize * splitSize), (int)(splitSize - 1)));
		const int z = max(0, min((int)((pos.z - position.z) * inverseSize * splitSize), (int)(splitSize - 1)));
		return make_uint3(x, y, z);
	}
};

struct Octree
{
	OctreeNode* nodes;
	Triangle* triangles;
	uint32_t* triangleIdx;

	// Finds the next node intersected by the ray assuming we just intersected with the child of relative index currentChildIdx in node
	inline __host__ __device__ uint32_t FindNextNode(const Ray& ray, uint3& currentChildIdx, OctreeNode& node)
	{
		uint32_t childIdx[3] = { currentChildIdx.x, currentChildIdx.y, currentChildIdx.z };
		while (true)
		{
			const float scale = node.size / node.splitSize;

			// Compute intersection planes depending on the ray direction
			float planeX = node.position.x + currentChildIdx.x * scale;
			if (ray.direction.x > 0.0f)
				planeX += scale;
			float planeY = node.position.y + currentChildIdx.y * scale;
			if (ray.direction.y > 0.0f)
				planeY += scale;
			float planeZ = node.position.z + currentChildIdx.z * scale;
			if (ray.direction.z > 0.0f)
				planeZ += scale;

			// Get the intersection point for each plane
			const float t[3] = {
				(planeX - ray.origin.x) * ray.invDirection.x,
				(planeY - ray.origin.y) * ray.invDirection.y,
				(planeZ - ray.origin.z) * ray.invDirection.z,
			};
			unsigned char minAxis = t[0] < t[1] ? 0 : 1;
			minAxis = t[minAxis] < t[2] ? minAxis : 2;

			const int newIdx = childIdx[minAxis] + Utils::SgnE((int)*((float*)&ray.direction + minAxis));
			if (newIdx >= 0 && newIdx < node.splitSize)
			{
				childIdx[minAxis] = newIdx;
				currentChildIdx = make_uint3(childIdx[0], childIdx[1], childIdx[2]);
				return childIdx[0] + childIdx[1] * node.splitSize + childIdx[2] * node.splitSize * node.splitSize;
			}

			// Index is invalid, we need to backtrace to the parent
			OctreeNode parent = nodes[node.parentIdx];
			currentChildIdx = parent.GetChildIdx(node.position + node.size * 0.5);
		}
	}

	inline __host__ __device__ void Intersect(Ray& ray, uint32_t instanceIdx)
	{
		OctreeNode currentNode = nodes[0];
		OctreeNode currentChild = currentNode;
		uint3 currentChildIdx = make_uint3(0);
		uint32_t returnNode = 0;
		float t = 1e30f;

		while (true)
		{
			if (currentChild.IsLeaf())
			{
				for (int i = 0; i < currentChild.triCount; i++)
					triangles[triangleIdx[i]].Hit(ray, instanceIdx, i);

				// TODO: test if moving this if statement in the triangle intersection loop is faster
				if (ray.hit.t < 1.0e30f)
					break;

				currentChild = nodes[FindNextNode(ray, currentChildIdx, currentNode)];
			}


			// AABB intersection
			const float3 bMax = currentChild.position + currentChild.size;
			const float tx1 = (currentChild.position.x - ray.origin.x) * ray.invDirection.x, tx2 = (bMax.x - ray.origin.x) * ray.invDirection.x;
			float tmin = fmin(tx1, tx2), tmax = fmax(tx1, tx2);
			const float ty1 = (currentChild.position.y - ray.origin.y) * ray.invDirection.y, ty2 = (bMax.y - ray.origin.y) * ray.invDirection.y;
			tmin = fmax(tmin, fmin(ty1, ty2)), tmax = fmin(tmax, fmax(ty1, ty2));
			const float tz1 = (currentChild.position.z - ray.origin.z) * ray.invDirection.z, tz2 = (bMax.z - ray.origin.z) * ray.invDirection.z;
			tmin = fmax(tmin, fmin(tz1, tz2)), tmax = fmin(tmax, fmax(tz1, tz2));

			if (tmax >= tmin && tmin < ray.hit.t && tmax > 0)
			{
				// Get the index of the child intersected by the ray
				currentChildIdx = currentChild.GetChildIdx(ray.origin + tmin * ray.direction);
				currentNode = currentChild;
				currentChild = nodes[currentChildIdx.z * currentChild.splitSize * currentChild.splitSize + currentChildIdx.y * currentChild.splitSize + currentChildIdx.x];
			}
		}
	}
};