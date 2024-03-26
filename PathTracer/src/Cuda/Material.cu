#include "Material.cuh"
#include <vector>
#include "../Utils/Utils.h"

__constant__ __device__ CMaterialType* materialTypes;
__constant__ __device__ CMaterial* materials;


void addMaterialsToDevice(std::vector<CMaterial>& m)
{
	CMaterial* materialsPtr;
	checkCudaErrors(cudaGetSymbolAddress((void**)&materialsPtr, materials));
	checkCudaErrors(cudaMalloc((void**)&materialsPtr, sizeof(CMaterial) * m.size()));
	checkCudaErrors(cudaMemcpy(materialsPtr, &m[0], sizeof(CMaterial) * m.size(), cudaMemcpyHostToDevice));
}

void addMaterialToDevice(CMaterial& m, CMaterialType mType, uint32_t size)
{
	CMaterial* materialsPtr;
	CMaterial* materialsCpy = new CMaterial[size];

	checkCudaErrors(cudaGetSymbolAddress((void**)&materialsPtr, materials));

	checkCudaErrors(cudaMemcpyFromSymbol(materialsCpy, materials, size - 1));
	materialsCpy[size - 1] = m;

	if (size > 0)
		checkCudaErrors(cudaFree(materialsPtr));

	checkCudaErrors(cudaMalloc((void**)&materialsPtr, size));

	checkCudaErrors(cudaMemcpyToSymbol(materials, materialsCpy, size));

	delete[] materialsCpy;

	CMaterialType* materialTypesPtr;
	CMaterialType* materialTypesCpy = new CMaterialType[size];

	checkCudaErrors(cudaGetSymbolAddress((void**)&materialTypesPtr, materialTypes));

	checkCudaErrors(cudaMemcpyFromSymbol(materialTypesCpy, materialTypes, size - 1));
	materialTypesCpy[size - 1] = mType;

	if (size > 0)
		checkCudaErrors(cudaFree(materialTypesPtr));

	checkCudaErrors(cudaMalloc((void**)&materialTypesPtr, size));

	checkCudaErrors(cudaMemcpyToSymbol(materialTypes, materialTypesCpy, size));

	delete[] materialTypesCpy;
}
