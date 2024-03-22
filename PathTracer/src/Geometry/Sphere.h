#pragma once
#include "Material.h"
#include "cuda/cuda_math.h"
#include "Hitable.h"

#define MAX_SPHERES 50

struct Sphere
{
	__host__ __device__ Sphere() { };
	__host__ __device__ Sphere(float radius, float3 position, Material material)
		:radius(radius), position(position), material(material) { }

	inline __host__ __device__ bool Hit(const Ray& r, float& t)
	{
		float3 origin = r.origin - position;

		float a = dot(r.direction, r.direction);
		float b = dot(origin, r.direction);
		float c = dot(origin, origin) - radius * radius;

		float discriminant = b * b - a * c;

		if (discriminant > 0.0f)
		{
			float temp = (-b - sqrt(discriminant)) / a;
			if (temp > 0.0f)
			{
				t = temp;
				return true;
			}

			temp = (-b + sqrt(discriminant)) / a;
			if (temp > 0)
			{
				t = temp;
				return true;
			}
		}
		return false;
	}

	float radius;
	float3 position;
	Material material;
};



