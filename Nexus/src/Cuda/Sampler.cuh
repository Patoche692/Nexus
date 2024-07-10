#pragma once
#include <cuda_runtime_api.h>

class Sampler
{
public:

	// Balance heuristic for multiple importance sampling with two
	// samples from two different sampling strategies. (We use BSDF and direct light sampling)
	// See Eric Veach's thesis
	inline __device__ static float BalanceHeuristic(const float pdf1, const float pdf2)
	{
		return pdf1 / (pdf1 + pdf2);
	}

	// Power heuristic with beta = 2 for multiple importance sampling with two
	// samples from two different sampling strategies. (We use BSDF and direct light sampling)
	// See Eric Veach's thesis
	inline __device__ static float PowerHeuristic(const float pdf1, const float pdf2)
	{
		return pdf1 * pdf1 / (pdf1 * pdf1 + pdf2 * pdf2);
	}
};