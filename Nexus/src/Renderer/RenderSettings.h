#pragma once
#include "Utils/cuda_math.h"

struct RenderSettings
{
	bool useMIS = true;
	unsigned char pathLength = 10;

	float3 backgroundColor = make_float3(1.0f);
	float backgroundIntensity = 0.0f;
};