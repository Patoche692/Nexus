#pragma once
#include "../Utils/cuda_math.h"
#include "Material.h"

struct Triangle
{
	float3 pos0;
	float3 pos1;
	float3 pos2;

	float3 normal0;
	float3 normal1;
	float3 normal2;

	float3 texCoord0;
	float3 texCoord1;
	float3 texCoord2;

	__host__ __device__ Triangle() = default;

	__host__ __device__ Triangle(float3 p0, float3 p1, float3 p2)
		:pos0(p0), pos1(p1), pos2(p2), normal0(make_float3(0.0f)), normal1(make_float3(0.0f)),
		normal2(make_float3(0.0f)), texCoord0(make_float3(0.0f)), texCoord1(make_float3(0.0f)), texCoord2(make_float3(0.0f)) { }

	__host__ __device__ Triangle(float3 p0, float3 p1, float3 p2, float3 n0, float3 n1, float3 n2)
		:pos0(p0), pos1(p1), pos2(p2), normal0(n0), normal1(n1),
		normal2(n2), texCoord0(make_float3(0.0f)), texCoord1(make_float3(0.0f)), texCoord2(make_float3(0.0f)) { }

	__host__ __device__ Triangle(float3 p0, float3 p1, float3 p2, float3 n0, float3 n1, float3 n2, float3 t0, float3 t1, float3 t2)
		:pos0(p0), pos1(p1), pos2(p2), normal0(n0), normal1(n1),
		normal2(n2), texCoord0(t0), texCoord1(t1), texCoord2(t2) { }

	inline __host__ __device__ bool Hit(const Ray& r, float& t)
	{
		float3 edge0 = pos1 - pos0;
		float3 edge1 = pos2 - pos0;

		float3 n = cross(edge0, edge1);

		if (dot(n, r.direction) < 1.0e-6)
			return false;

		float d = dot(-n, pos0);

		t = -(dot(n, r.origin) + d);

		if (t < 0)
			return false;

		float3 p = r.origin + t * r.direction;

		float3 vp0 = p - pos0;
		float3 c = cross(edge0, vp0);

		if (dot(n, c) < 0.0f)
			return false;

		edge1 = pos2 - pos1;
		float3 vp1 = p - pos1;
		c = cross(edge1, vp1);

		if (dot(n, c) < 0.0f)
			return false;

		float3 edge2 = pos0 - pos2;
		float3 vp2 = p - pos2;
		c = cross(edge2, vp2);

		if (dot(n, c) < 0.0f)
			return false;

		return true;
	}
};
