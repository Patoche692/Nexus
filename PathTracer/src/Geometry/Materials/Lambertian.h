#pragma once
#include "Material.h"

struct Lambertian : public Material
{

	inline __host__ __device__ bool Scatter() override {

	}
};
