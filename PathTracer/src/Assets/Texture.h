#pragma once
#include <iostream>
#include <cuda_runtime_api.h>
#include <Utils/cuda_math.h>

struct Texture
{
	Texture() = default;
	Texture(uint32_t w, uint32_t h, uint32_t c, unsigned char* d);
	__host__ __device__ float3 GetPixel(float x, float y) const
	{
		//x = clamp(x, 0, width - 1);
		//y = clamp(y, 0, height - 1);

		int iu = (int)(x * width) % width;
		int iv = (int)(y * height) % height;
		int index = (iv * width + iu) * channels;

		unsigned char r = data[index];
		unsigned char g = data[index + 1];
		unsigned char b = data[index + 2];

		return make_float3(r / 255.0f, g / 255.0f, b / 255.0f);
	}

	uint32_t width;
	uint32_t height;
	uint32_t channels;

	unsigned char* data;
};