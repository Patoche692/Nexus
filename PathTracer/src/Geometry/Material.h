#pragma once
#include "cuda/cuda_math.h"

struct Material {
	Material() {};
	Material(float3 c) : color(c) {};
	float3 color;
};

