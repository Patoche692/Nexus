#include "Texture.h"
#include "Utils/Utils.h"
#include <cuda_runtime_api.h>


Texture::Texture(uint32_t w, uint32_t h, uint32_t c, float3* d) : width(w), height(h), channels(c), pixels(d)
{
}

cudaTextureObject_t Texture::ToDevice(const Texture& texture)
{
	// Channel descriptor for 4 Channels (RGBA)
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	cudaArray_t cuArray;

	checkCudaErrors(cudaMallocArray(&cuArray, &channelDesc, texture.width, texture.height));

	const size_t spitch = texture.width * 4 * sizeof(float);
	checkCudaErrors(cudaMemcpy2DToArray(cuArray, 0, 0, texture.pixels, spitch, texture.width * 4 * sizeof(float), texture.height, cudaMemcpyHostToDevice));

	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 1;

	cudaTextureObject_t texObject = 0;
	checkCudaErrors(cudaCreateTextureObject(&texObject, &resDesc, &texDesc, NULL));

	return texObject;
}

void Texture::DestructFromDevice(const cudaTextureObject_t& texture)
{
	cudaResourceDesc resDesc;
	checkCudaErrors(cudaGetTextureObjectResourceDesc(&resDesc, texture));
	checkCudaErrors(cudaDestroyTextureObject(texture));
	checkCudaErrors(cudaFreeArray(resDesc.res.array.array));
}

