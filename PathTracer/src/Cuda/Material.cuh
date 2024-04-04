#pragma once

#include <cuda_runtime_api.h>
#include "../Geometry/Materials/Material.h"


void newDeviceMaterial(Material& m, uint32_t size);
void changeDeviceMaterial(Material& m, uint32_t id);
