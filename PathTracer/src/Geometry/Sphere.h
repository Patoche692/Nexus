#pragma once
#include "Material.h"
#include "cuda/cuda_math.h"
#include "Hitable.h"

#define MAX_SPHERES 50

struct Sphere
{
	__host__ __device__ Sphere() { };
	__host__ __device__ Sphere(float radius, float3 position, Material* material);
	__host__ __device__ bool Hit(const Ray& r, HitResult& hitResult) const;

	float radius;
	float3 position;
	Material* material;
};

__host__ __device__ Sphere::Sphere(float radius, float3 position, Material* material)
	:radius(radius), position(position), material(material) { }


__host__ __device__ bool Sphere::Hit(const Ray& r, HitResult& hitResult) const
{
	float3 origin = r.origin - position;

	float a = dot(r.direction, r.direction);
	float b = dot(origin, r.direction);
	float c = dot(origin, origin) - radius * radius;

	float discriminant = b * b - a * c;

	if (discriminant > 0)
	{
		float temp = (-b - sqrt(discriminant)) / a;
		if (temp > 0)
		{
			hitResult.t = temp;
			hitResult.p = r.PointAtParameter(temp);
			hitResult.material = material;
			hitResult.normal = (hitResult.p - position) / radius;
			return true;
		}

		temp = (-b + sqrt(discriminant)) / a;
		if (temp > 0)
		{
			hitResult.t = temp;
			hitResult.p = r.PointAtParameter(temp);
			hitResult.material = material;
			hitResult.normal = (hitResult.p - position) / radius;
			return true;
		}
	}
	return false;
}
