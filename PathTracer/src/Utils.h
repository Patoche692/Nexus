#pragma once
#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime_api.h>
#include <iostream>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);

#endif
