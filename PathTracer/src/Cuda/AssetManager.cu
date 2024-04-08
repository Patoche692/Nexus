#include "AssetManager.cuh"
#include <vector>

#include "MemoryHelper.cuh"
#include "../Utils/Utils.h"
#include "Random.cuh"
#include "../Geometry/Ray.h"
#include "../Geometry/Mesh.h"

__constant__ __device__ Material* materials;
__constant__ __device__ Mesh* meshes;

void newDeviceMesh(Mesh& mesh, uint32_t size)
{
	Mesh** meshesSymbolAddress;

	// Retreive the address of materials
	checkCudaErrors(cudaGetSymbolAddress((void**)&meshesSymbolAddress, meshes));

	MemoryHelper::ResizeDeviceArray(meshesSymbolAddress, size);

	MemoryHelper::SetToIndex(meshesSymbolAddress, size - 1, mesh);
}

void newDeviceMaterial(Material& material, uint32_t size)
{
	Material** materialsSymbolAddress;

	// Retreive the address of materials
	checkCudaErrors(cudaGetSymbolAddress((void**)&materialsSymbolAddress, materials));

	MemoryHelper::ResizeDeviceArray(materialsSymbolAddress, size);

	MemoryHelper::SetToIndex(materialsSymbolAddress, size - 1, material);
}

void changeDeviceMaterial(Material& m, uint32_t id)
{
	Material** materialsSymbolAddress;

	// Retreive the address of materials
	checkCudaErrors(cudaGetSymbolAddress((void**)&materialsSymbolAddress, materials));

	MemoryHelper::SetToIndex(materialsSymbolAddress, id, m);
}


