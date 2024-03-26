#pragma once

#include <cuda_runtime_api.h>
#include "../Geometry/Materials/Material.h"


void newDeviceMaterial(Material& m, uint32_t size);
void changeDeviceMaterial(Material& m, uint32_t id);
inline __device__ bool diffuseScatter(Material& material, float3& p, float3& attenuation, float3& normal, Ray& scattered, uint32_t& rngState);
