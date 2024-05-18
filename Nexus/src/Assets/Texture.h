#pragma once
#include <iostream>
#include <cuda_runtime_api.h>
#include <Utils/cuda_math.h>

struct Texture
{
	enum struct Type {
		DIFFUSE,
		ROUGHNESS,
		METALLIC,
		EMISSIVE
	};

	Texture() = default;
	Texture(uint32_t w, uint32_t h, uint32_t c, float3* d);
	inline __host__ __device__ float3 GetPixel(float x, float y) const
	{
		int iu = (int)(x * width) % width;
		int iv = (int)(y * height) % height;
		int index = iv * width + iu;

		return pixels[index];
	}

	uint32_t width;
	uint32_t height;
	uint32_t channels;

	float3* pixels;
};