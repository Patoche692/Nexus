#include "AssetManager.cuh"
#include <vector>
#include <map>

#include "CudaMemory.cuh"
#include "../Utils/Utils.h"
#include "Random.cuh"
#include "Geometry/Ray.h"

__constant__ __device__ Material* materials;
__constant__ __device__ Mesh* meshes;
__constant__ __device__ TLAS tlas;


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

void cpyMaterialToDevice(Material& m, uint32_t id)
{
	Material** materialsSymbolAddress;

	// Retreive the address of materials
	checkCudaErrors(cudaGetSymbolAddress((void**)&materialsSymbolAddress, materials));

	CudaMemory::SetToIndex(materialsSymbolAddress, id, m);
}

BVH* newDeviceBVH(BVH& bvh)
{
	Triangle* triangles = CudaMemory::Allocate<Triangle>(bvh.triCount);
	BVHNode* nodes = CudaMemory::Allocate<BVHNode>(bvh.triCount * 2);
	uint32_t* triangleIdx = CudaMemory::Allocate<uint32_t>(bvh.triCount);

	CudaMemory::MemCpy(triangles, bvh.triangles, bvh.triCount, cudaMemcpyHostToDevice);
	CudaMemory::MemCpy(nodes, bvh.nodes, bvh.triCount * 2, cudaMemcpyHostToDevice);
	CudaMemory::MemCpy(triangleIdx, bvh.triangleIdx, bvh.triCount, cudaMemcpyHostToDevice);

	BVH newBvh;
	newBvh.triangles = triangles;
	newBvh.nodes = nodes;
	newBvh.triangleIdx = triangleIdx;
	newBvh.triCount = bvh.triCount;

	newBvh.nodesUsed = bvh.nodesUsed;

	BVH* bvhPtr = CudaMemory::Allocate<BVH>(1);
	CudaMemory::MemCpy(bvhPtr, &newBvh, 1, cudaMemcpyHostToDevice);

	return bvhPtr;
}

void newDeviceTLAS(TLAS& tl)
{
	TLASNode* tlasNodes = CudaMemory::Allocate<TLASNode>(tl.blasCount * 2);
	uint32_t *nodesIdx = CudaMemory::Allocate<uint32_t>(tl.blasCount);
	BVHInstance* instances = CudaMemory::Allocate<BVHInstance>(tl.blasCount);

	CudaMemory::MemCpy(tlasNodes, tl.nodes, tl.blasCount * 2, cudaMemcpyHostToDevice);
	CudaMemory::MemCpy(nodesIdx, tl.nodesIdx, tl.blasCount, cudaMemcpyHostToDevice);

	// Map from cpu memory to device memory
	std::map<BVH*, BVH*> bvhMap;
	for (int i = 0; i < tl.blasCount; i++)
	{
		if (!bvhMap.count(tl.blas[i].bvh))
		{
			bvhMap[tl.blas[i].bvh] = newDeviceBVH(*tl.blas[i].bvh);
		}
		BVHInstance instance = tl.blas[i];
		instance.bvh = bvhMap[tl.blas[i].bvh];
		CudaMemory::MemCpy(instances + i, &instance, 1, cudaMemcpyHostToDevice);
	}

	TLAS newTlas;
	newTlas.blasCount = tl.blasCount;
	newTlas.nodesUsed = tl.nodesUsed;
	newTlas.nodes = tlasNodes;
	newTlas.blas = instances;
	newTlas.nodesIdx = nodesIdx;

	checkCudaErrors(cudaMemcpyToSymbol(tlas, &newTlas, sizeof(TLAS)));
}

void updateDeviceTLAS(TLAS& tl)
{
	TLAS tlasCpy;
	BVHInstance* instancesCpy = new BVHInstance[tl.blasCount];
	checkCudaErrors(cudaMemcpyFromSymbol(&tlasCpy, tlas, sizeof(TLAS)));

	// Update the nodes
	CudaMemory::MemCpy(tlasCpy.nodes, tl.nodes, tl.blasCount * 2, cudaMemcpyHostToDevice);

	// TODO: handle the case when tl.blasCount has changed (new instance or deleted instance)
	CudaMemory::MemCpy(instancesCpy, tlasCpy.blas, tl.blasCount, cudaMemcpyDeviceToHost);

	for (int i = 0; i < tl.blasCount; i++)
	{
		BVH* bvhBackup = instancesCpy[i].bvh;
		instancesCpy[i].bvh = tl.blas[i].bvh;
		instancesCpy[i].SetTransform(tl.blas[i].transform);
		instancesCpy[i].materialId = tl.blas[i].materialId;
		instancesCpy[i].bvh = bvhBackup;
	}
	
	// Copy the instances back to the GPU
	CudaMemory::MemCpy(tlasCpy.blas, instancesCpy, tl.blasCount, cudaMemcpyHostToDevice);

	delete[] instancesCpy;
}

__global__ void freeDeviceTLASKernel()
{
	for (int i = 0; i < tlas.blasCount; i++)
	{
		BVH* bvh = tlas.blas[i].bvh;
		free(bvh->nodes);
		free(bvh->triangles);
		free(bvh->triangleIdx);
	}
	free(tlas.blas);
	free(tlas.nodes);
	free(tlas.nodesIdx);
}

void freeDeviceTLAS()
{
	freeDeviceTLASKernel<<<1, 1>>>();
	checkCudaErrors(cudaDeviceSynchronize());
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


