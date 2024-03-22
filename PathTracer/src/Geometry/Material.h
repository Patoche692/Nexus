#pragma once
#include "../Utils/cuda_math.h"

struct Material {
	Material() {};
	Material(float3 c) : color(c) {};
	float3 color;
};

