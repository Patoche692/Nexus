#include "Material.cuh"
#include <vector>
#include "../Utils/Utils.h"
#include "Random.cuh"
#include "../Geometry/Ray.h"

__constant__ __device__ Material* materials;


void addMaterialsToDevice(std::vector<Material>& m)
{
	Material* materialsPtr;
	checkCudaErrors(cudaGetSymbolAddress((void**)&materialsPtr, materials));
	checkCudaErrors(cudaMalloc((void**)&materialsPtr, sizeof(Material) * m.size()));
	checkCudaErrors(cudaMemcpy(materialsPtr, &m[0], sizeof(Material) * m.size(), cudaMemcpyHostToDevice));
}

void newDeviceMaterial(Material& m, uint32_t size)
{
	Material** materialsSymbolAddress;
	Material* materialsPtr;
	Material* materialsCpy = new Material[size];

	// Retreive the address of materials
	checkCudaErrors(cudaGetSymbolAddress((void**)&materialsSymbolAddress, materials));

	// Retrieve the address pointed to by materials
	checkCudaErrors(cudaMemcpy(&materialsPtr, materialsSymbolAddress, sizeof(Material*), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(materialsCpy, materialsPtr, sizeof(Material) * (size - 1), cudaMemcpyDeviceToHost));
	materialsCpy[size - 1] = m;

	if (size > 1)
		checkCudaErrors(cudaFree(materialsPtr));

	Material* temp;
	checkCudaErrors(cudaMalloc((void**)&temp, sizeof(Material) * size));

	checkCudaErrors(cudaMemcpy(temp, materialsCpy, sizeof(Material) * size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(materials, &temp, sizeof(Material*)));

	delete[] materialsCpy;
}

void changeDeviceMaterial(Material& m, uint32_t id)
{
	checkCudaErrors(cudaMemcpyToSymbol(materials, &m, sizeof(Material)));
}


