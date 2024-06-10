#include "AssetManager.cuh"
#include <vector>
#include <map>
#include "CudaMemory.cuh"
#include "../Utils/Utils.h"
#include "Random.cuh"
#include "Geometry/Ray.h"
#include "Geometry/Triangle.h"

//__constant__ __device__ Material* materials;
//__constant__ __device__ cudaTextureObject_t* diffuseMaps;
//__constant__ __device__ cudaTextureObject_t* emissiveMaps;
//__constant__ __device__ BVH8* bvhs;
//__constant__ __device__ TLAS tlas;
//
//
//void newDeviceMesh(Mesh& mesh, uint32_t size)
//{
//	//Mesh** meshesSymbolAddress;
//
//	//// Retreive the address of meshes
//	//checkCudaErrors(cudaGetSymbolAddress((void**)&meshesSymbolAddress, bvhs));
//
//	//Triangle* triangles = CudaMemory::Allocate<Triangle>(mesh.nTriangles);
//	//CudaMemory::MemCpy(triangles, mesh.triangles, mesh.nTriangles, cudaMemcpyHostToDevice);
//
//	//Mesh newMesh = mesh;
//	//newMesh.triangles = triangles;
//
//	//CudaMemory::ResizeDeviceArray(meshesSymbolAddress, size);
//
//	//CudaMemory::SetToIndex(meshesSymbolAddress, size - 1, newMesh);
//}
//
//void newDeviceMaterial(Material& material, uint32_t size)
//{
//	Material** materialsSymbolAddress;
//
//	// Retreive the address of materials
//	checkCudaErrors(cudaGetSymbolAddress((void**)&materialsSymbolAddress, materials));
//
//	CudaMemory::ResizeDeviceArray(materialsSymbolAddress, size);
//
//	CudaMemory::SetToIndex(materialsSymbolAddress, size - 1, material);
//}
//
//void newDeviceTexture(Texture& texture, uint32_t size) {
//	
//	cudaTextureObject_t** texturesSymbolAddress;
//
//	if (texture.type == Texture::Type::DIFFUSE) {
//		checkCudaErrors(cudaGetSymbolAddress((void**)&texturesSymbolAddress, diffuseMaps));
//	}
//	else if (texture.type == Texture::Type::EMISSIVE) {
//		checkCudaErrors(cudaGetSymbolAddress((void**)&texturesSymbolAddress, emissiveMaps));
//	}
//		
//
//	CudaMemory::ResizeDeviceArray(texturesSymbolAddress, size);
//
//	// Channel descriptor for 4 Channels (RGBA)
//	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
//	cudaArray_t cuArray;
//	checkCudaErrors(cudaMallocArray(&cuArray, &channelDesc, texture.width, texture.height));
//
//	const size_t spitch = texture.width * 4 * sizeof(float);
//	checkCudaErrors(cudaMemcpy2DToArray(cuArray, 0, 0, texture.pixels, spitch, texture.width * 4 * sizeof(float), texture.height, cudaMemcpyHostToDevice));
//
//	cudaResourceDesc resDesc;
//	memset(&resDesc, 0, sizeof(resDesc));
//	resDesc.resType = cudaResourceTypeArray;
//	resDesc.res.array.array = cuArray;
//
//	cudaTextureDesc texDesc;
//	memset(&texDesc, 0, sizeof(texDesc));
//	texDesc.addressMode[0] = cudaAddressModeWrap;
//	texDesc.addressMode[1] = cudaAddressModeWrap;
//	texDesc.filterMode = cudaFilterModeLinear;
//	texDesc.readMode = cudaReadModeElementType;
//	texDesc.normalizedCoords = 1;
//
//	cudaTextureObject_t texObject = 0;
//	checkCudaErrors(cudaCreateTextureObject(&texObject, &resDesc, &texDesc, NULL));
//
//	CudaMemory::SetToIndex(texturesSymbolAddress, size - 1, texObject);
//}
//
//__global__ void freeMeshesKernel(int meshesCount)
//{
//	for (int i = 0; i < meshesCount; i++)
//	{
//		free(bvhs[i].triangles);
//	}
//	free(bvhs);
//}
//
//void freeDeviceMeshes(int meshesCount)
//{
//	freeMeshesKernel<<<1, 1>>>(meshesCount);
//}
//
//__global__ void freeMaterialsKernel()
//{
//	free(materials);
//}
//
//void freeDeviceMaterials()
//{
//	freeMaterialsKernel<<<1, 1>>>();
//}
//
//__global__ void freeTexturesKernel(int texturesCount)
//{
//	//for (int i = 0; i < texturesCount; i++)
//	//	free(textures[i].pixels);
//
//	//free(textures);
//}
//
//void freeDeviceTextures(int texturesCount)
//{
//	freeTexturesKernel<<<1, 1>>>(texturesCount);
//}
//
//void cpyMaterialToDevice(Material& m, uint32_t id)
//{
//	Material** materialsSymbolAddress;
//
//	// Retreive the address of materials
//	checkCudaErrors(cudaGetSymbolAddress((void**)&materialsSymbolAddress, materials));
//
//	CudaMemory::SetToIndex(materialsSymbolAddress, id, m);
//}
//
//BVH2* newDeviceBVH(BVH2& bvh)
//{
//	Triangle* triangles = CudaMemory::Allocate<Triangle>(bvh.triCount);
//	BVH2Node* nodes = CudaMemory::Allocate<BVH2Node>(bvh.triCount * 2);
//	uint32_t* triangleIdx = CudaMemory::Allocate<uint32_t>(bvh.triCount);
//
//	CudaMemory::MemCpy(triangles, bvh.triangles, bvh.triCount, cudaMemcpyHostToDevice);
//	CudaMemory::MemCpy(nodes, bvh.nodes, bvh.triCount * 2, cudaMemcpyHostToDevice);
//	CudaMemory::MemCpy(triangleIdx, bvh.triangleIdx, bvh.triCount, cudaMemcpyHostToDevice);
//
//	BVH2 newBvh;
//	newBvh.triangles = triangles;
//	newBvh.nodes = nodes;
//	newBvh.triangleIdx = triangleIdx;
//	newBvh.triCount = bvh.triCount;
//
//	newBvh.nodesUsed = bvh.nodesUsed;
//
//	BVH2* bvhPtr = CudaMemory::Allocate<BVH2>(1);
//	CudaMemory::MemCpy(bvhPtr, &newBvh, 1, cudaMemcpyHostToDevice);
//
//	// TODO: Move all structures to the GPU. For now, avoid calling delete on a device ptr
//	newBvh.triangles = nullptr;
//	newBvh.nodes = nullptr;
//	newBvh.triangleIdx = nullptr;
//
//	return bvhPtr;
//}
//
//BVH8* newDeviceBVH8(BVH8& bvh)
//{
//	Triangle* triangles = CudaMemory::Allocate<Triangle>(bvh.triCount);
//	BVH8Node* nodes = CudaMemory::Allocate<BVH8Node>(bvh.triCount * 2);
//	uint32_t* triangleIdx = CudaMemory::Allocate<uint32_t>(bvh.triCount);
//
//	CudaMemory::MemCpy(triangles, bvh.triangles, bvh.triCount, cudaMemcpyHostToDevice);
//	CudaMemory::MemCpy(nodes, bvh.nodes, bvh.triCount * 2, cudaMemcpyHostToDevice);
//	CudaMemory::MemCpy(triangleIdx, bvh.triangleIdx, bvh.triCount, cudaMemcpyHostToDevice);
//
//	BVH8 newBvh;
//	newBvh.triangles = triangles;
//	newBvh.nodes = nodes;
//	newBvh.triangleIdx = triangleIdx;
//	newBvh.triCount = bvh.triCount;
//
//	newBvh.nodesUsed = bvh.nodesUsed;
//
//	BVH8* bvhPtr = CudaMemory::Allocate<BVH8>(1);
//	CudaMemory::MemCpy(bvhPtr, &newBvh, 1, cudaMemcpyHostToDevice);
//
//	// TODO: Move all structures to the GPU. For now, avoid calling delete on a device ptr
//	newBvh.triangles = nullptr;
//	newBvh.nodes = nullptr;
//	newBvh.triangleIdx = nullptr;
//
//	return bvhPtr;
//}
//
//void newDeviceTLAS(TLAS& tl)
//{
//	TLASNode* tlasNodes = CudaMemory::Allocate<TLASNode>(tl.blasCount * 2);
//	uint32_t *nodesIdx = CudaMemory::Allocate<uint32_t>(tl.blasCount);
//	BVHInstance* instances = CudaMemory::Allocate<BVHInstance>(tl.blasCount);
//
//	CudaMemory::MemCpy(tlasNodes, tl.nodes, tl.blasCount * 2, cudaMemcpyHostToDevice);
//	CudaMemory::MemCpy(nodesIdx, tl.nodesIdx, tl.blasCount, cudaMemcpyHostToDevice);
//
//	// Map from cpu memory to device memory
//	std::map<BVH8*, BVH8*> bvhMap;
//	for (int i = 0; i < tl.blasCount; i++)
//	{
//		if (!bvhMap.count(tl.blas[i].bvh))
//		{
//			bvhMap[tl.blas[i].bvh] = newDeviceBVH8(*tl.blas[i].bvh);
//		}
//		BVHInstance instance = tl.blas[i];
//		instance.bvh = bvhMap[tl.blas[i].bvh];
//		CudaMemory::MemCpy(instances + i, &instance, 1, cudaMemcpyHostToDevice);
//	}
//
//	TLAS newTlas;
//	newTlas.blasCount = tl.blasCount;
//	newTlas.nodesUsed = tl.nodesUsed;
//	newTlas.nodes = tlasNodes;
//	newTlas.blas = instances;
//	newTlas.nodesIdx = nodesIdx;
//
//	checkCudaErrors(cudaMemcpyToSymbol(tlas, &newTlas, sizeof(TLAS)));
//}
//
//void updateDeviceTLAS(TLAS& tl)
//{
//	TLAS tlasCpy;
//	BVHInstance* instancesCpy = new BVHInstance[tl.blasCount];
//	checkCudaErrors(cudaMemcpyFromSymbol(&tlasCpy, tlas, sizeof(TLAS)));
//
//	// Update the nodes
//	CudaMemory::MemCpy(tlasCpy.nodes, tl.nodes, tl.blasCount * 2, cudaMemcpyHostToDevice);
//
//	// TODO: handle the case when tl.blasCount has changed (new instance or deleted instance)
//	CudaMemory::MemCpy(instancesCpy, tlasCpy.blas, tl.blasCount, cudaMemcpyDeviceToHost);
//
//	for (int i = 0; i < tl.blasCount; i++)
//	{
//		BVH8* bvhBackup = instancesCpy[i].bvh;
//		instancesCpy[i].bvh = tl.blas[i].bvh;
//		instancesCpy[i].SetTransform(tl.blas[i].transform);
//		instancesCpy[i].materialId = tl.blas[i].materialId;
//		instancesCpy[i].bvh = bvhBackup;
//	}
//	
//	// Copy the instances back to the GPU
//	CudaMemory::MemCpy(tlasCpy.blas, instancesCpy, tl.blasCount, cudaMemcpyHostToDevice);
//
//	delete[] instancesCpy;
//}
//
//__global__ void freeDeviceTLASKernel()
//{
//	for (int i = 0; i < tlas.blasCount; i++)
//	{
//		BVH8* bvh = tlas.blas[i].bvh;
//		free(bvh->nodes);
//		free(bvh->triangles);
//		free(bvh->triangleIdx);
//		free(bvh);
//	}
//	free(tlas.blas);
//	free(tlas.nodes);
//	free(tlas.nodesIdx);
//}
//
//void freeDeviceTLAS()
//{
//	freeDeviceTLASKernel<<<1, 1>>>();
//}
//
//Material** getMaterialSymbolAddress()
//{
//	Material** materialSymbolAddress;
//	checkCudaErrors(cudaGetSymbolAddress((void**)&materialSymbolAddress, materials));
//	return materialSymbolAddress;
//}
//
//Mesh** getMeshSymbolAddress()
//{
//	Mesh** meshSymbolAddress;
//	checkCudaErrors(cudaGetSymbolAddress((void**)&meshSymbolAddress, bvhs));
//	return meshSymbolAddress;
//}
//
//
