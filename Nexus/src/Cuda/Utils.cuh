#pragma once

#include <cuda_runtime_api.h>
#include <cudart_platform.h>
#include <device_launch_parameters.h>
#include <stdint.h>

// Sign extend function. See https://github.com/AlanIWBFT/CWBVH/blob/master/src/TraversalKernelCWBVH.cu
__device__ __inline__ uint32_t SignExtendS8x4(uint32_t i) { uint32_t v; asm("prmt.b32 %0, %1, 0x0, 0x0000BA98;" : "=r"(v) : "r"(i)); return v; }

// Returns max(a, max(b, c))
__device__ float vMaxMax(float a, float b, float c) {
	int result;

	asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(result) : "r"(__float_as_int(a)), "r"(__float_as_int(b)), "r"(__float_as_int(c)));

	return __int_as_float(result);
}

// Returns min(a, min(b, c))
__device__ float vMinMin(float a, float b, float c) {
	int result;

	asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(result) : "r"(__float_as_int(a)), "r"(__float_as_int(b)), "r"(__float_as_int(c)));

	return __int_as_float(result);
}

// Extract the i-th byte of x
inline __device__ uint32_t ExtractByte(uint32_t x, uint32_t i) {
	return (x >> (i * 8)) & 0xff;
}
