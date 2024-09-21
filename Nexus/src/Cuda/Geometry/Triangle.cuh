#pragma once

#include <cudart_platform.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include "Ray.cuh"


struct D_Triangle_SOA
{
	float3* pos0;
	float3* pos1;
	float3* pos2;

	float3* normal0;
	float3* normal1;
	float3* normal2;

	float3* texCoord0;
	float3* texCoord1;
	float3* texCoord2;
};


struct D_Triangle
{
	// Positions
	float3 pos0;
	float3 pos1;
	float3 pos2;

	// Normals
	float3 normal0;
	float3 normal1;
	float3 normal2;

	// Texture coordinates
	float2 texCoord0;
	float2 texCoord1;
	float2 texCoord2;

	__device__ D_Triangle() = default;

	__device__ D_Triangle(
		float3 p0, float3 p1, float3 p2,
		float3 n0 = make_float3(0.0f), float3 n1 = make_float3(0.0f), float3 n2 = make_float3(0.0f),
		float2 t0 = make_float2(0.0f), float2 t1 = make_float2(0.0f), float2 t2 = make_float2(0.0f)
	): pos0(p0), pos1(p1), pos2(p2), normal0(n0), normal1(n1),
		normal2(n2), texCoord0(t0), texCoord1(t1), texCoord2(t2) { }


	// M�ller-Trumbore intersection algorithm. See https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
	inline __device__ void Trace(D_Ray& r, D_Intersection& intersection, const uint32_t instIdx, const uint32_t primIdx)
	{
		const float3 edge0 = pos1 - pos0;
		const float3 edge1 = pos2 - pos0;

		const float3 rayCrossEdge1 = cross(r.direction, edge1);
		const float det = dot(edge0, rayCrossEdge1);

		const float invDet = 1.0f / det;

		const float3 s = r.origin - pos0;
		
		const float u = invDet * dot(s, rayCrossEdge1);

		if (u < 0.0f || u > 1.0f)
			return;

		const float3 sCrossEdge0 = cross(s, edge0);
		const float v = invDet * dot(r.direction, sCrossEdge0);

		if (v < 0.0f || u + v > 1.0f)
			return;

		const float t = invDet * dot(edge1, sCrossEdge0);

		if (t > 0.0f && t < intersection.hitDistance)
		{
			intersection.hitDistance = t;
			intersection.u = u;
			intersection.v = v;
			intersection.instanceIdx = instIdx;
			intersection.triIdx = primIdx;
		}
	}

	// true if any hit, else false
	inline __device__ bool ShadowTrace(D_Ray& r, float hitDistance)
	{
		const float3 edge0 = pos1 - pos0;
		const float3 edge1 = pos2 - pos0;

		const float3 rayCrossEdge1 = cross(r.direction, edge1);
		const float det = dot(edge0, rayCrossEdge1);

		const float invDet = 1.0f / det;

		const float3 s = r.origin - pos0;
		
		const float u = invDet * dot(s, rayCrossEdge1);

		if (u < 0.0f || u > 1.0f)
			return false;

		const float3 sCrossEdge0 = cross(s, edge0);
		const float v = invDet * dot(r.direction, sCrossEdge0);

		if (v < 0.0f || u + v > 1.0f)
			return false;

		const float t = invDet * dot(edge1, sCrossEdge0);

		if (t > 0.0f && t < hitDistance)
			return true;

		return false;
	}

	// Normal (not normalized)
	inline __device__ float3 Normal() const
	{
		const float3 edge0 = pos1 - pos0;
		const float3 edge1 = pos2 - pos0;

		return cross(edge0, edge1);
	}

	// See https://community.khronos.org/t/how-can-i-find-the-area-of-a-3d-triangle/49777/2
	inline __device__ float Area() const
	{
		const float3 edge0 = pos1 - pos0;
		const float3 edge1 = pos2 - pos0;

		const float3 normal = cross(edge0, edge1);

		return 0.5f * length(normal);
	}
};
