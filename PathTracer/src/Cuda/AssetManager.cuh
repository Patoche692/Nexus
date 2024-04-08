#pragma once

#include <cuda_runtime_api.h>
#include "../Geometry/Material.h"
#include "../Geometry/Mesh.h"


void newDeviceMesh(Mesh& mesh, uint32_t size);
void newDeviceMaterial(Material& m, uint32_t size);
void changeDeviceMaterial(Material& m, uint32_t id);
Material** getMaterialSymbolAddress();
Mesh** getMeshSymbolAddress();
