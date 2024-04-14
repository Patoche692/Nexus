#include "AssetManager.cuh"
#include <vector>

#include "CudaMemory.cuh"
#include "../Utils/Utils.h"
#include "Random.cuh"
#include "Geometry/Ray.h"

__constant__ __device__ Material* materials;
__constant__ __device__ Mesh* meshes;


void newDeviceMesh(Mesh& mesh, uint32_t size)
{
	Mesh** meshesSymbolAddress;

	// Retreive the address of meshes
	checkCudaErrors(cudaGetSymbolAddress((void**)&meshesSymbolAddress, meshes));

	Triangle* triangles = CudaMemory::Allocate<Triangle>(mesh.nTriangles);
	CudaMemory::MemCpy(triangles, mesh.triangles, mesh.nTriangles, cudaMemcpyHostToDevice);

	Mesh newMesh = mesh;
	newMesh.triangles = triangles;

	CudaMemory::ResizeDeviceArray(meshesSymbolAddress, size);

	CudaMemory::SetToIndex(meshesSymbolAddress, size - 1, newMesh);
}

void newDeviceMaterial(Material& material, uint32_t size)
{
	Material** materialsSymbolAddress;

	// Retreive the address of materials
	checkCudaErrors(cudaGetSymbolAddress((void**)&materialsSymbolAddress, materials));

	CudaMemory::ResizeDeviceArray(materialsSymbolAddress, size);

	CudaMemory::SetToIndex(materialsSymbolAddress, size - 1, material);
}

__global__ void freeMeshesKernel(int meshesCount)
{
	for (int i = 0; i < meshesCount; i++)
	{
		free(meshes[i].triangles);
	}
	free(meshes);
	printf("%d", sizeof(Triangle));
}

void freeDeviceMeshes(int meshesCount)
{
	freeMeshesKernel<<<1, 1>>>(meshesCount);
	checkCudaErrors(cudaDeviceSynchronize());
}

void freeDeviceMaterials()
{
	CudaMemory::Free(materials);
}

void changeDeviceMaterial(Material& m, uint32_t id)
{
	Material** materialsSymbolAddress;

	// Retreive the address of materials
	checkCudaErrors(cudaGetSymbolAddress((void**)&materialsSymbolAddress, materials));

	CudaMemory::SetToIndex(materialsSymbolAddress, id, m);
}

Material** getMaterialSymbolAddress()
{
	Material** materialSymbolAddress;
	checkCudaErrors(cudaGetSymbolAddress((void**)&materialSymbolAddress, materials));
	return materialSymbolAddress;
}

Mesh** getMeshSymbolAddress()
{
	Mesh** meshSymbolAddress;
	checkCudaErrors(cudaGetSymbolAddress((void**)&meshSymbolAddress, meshes));
	return meshSymbolAddress;
}


