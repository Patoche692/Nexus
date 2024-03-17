#include "Renderer.h"
#include "Renderer.cuh"
#include "Utils.h"

void Renderer::Render()
{ 

	std::shared_ptr<PixelBuffer> pixelBuffer = m_TextureRenderer->GetPixelBuffer();
	checkCudaErrors(cudaGraphicsMapResources(1, &pixelBuffer->GetCudaResource()));
	size_t size = 0;
	void* device_ptr = 0;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer(&device_ptr, &size, pixelBuffer->GetCudaResource()));

	cudaRender(device_ptr, m_ImageWidth, m_ImageHeight);

	checkCudaErrors(cudaGraphicsUnmapResources(1, &pixelBuffer->GetCudaResource(), 0));

	m_TextureRenderer->Render();
}
