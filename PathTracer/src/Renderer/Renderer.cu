#include "Renderer.cuh"
#include "../Utils.h"

__global__ void traceRay(void *device_ptr, uint32_t imageWidth, uint32_t imageHeight)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	float x = (float)i / (float)imageWidth * 2.0f - 1.0f;
	float y = (float)j / (float)imageHeight * 2.0f - 1.0f;

	if (i >= imageWidth || j >= imageHeight)
		return;

	uint32_t* imagePtr = (uint32_t*)device_ptr;

	glm::vec3 rayOrigin(0, 0, 2.0f);
	glm::vec3 rayDirection(x, y, -1.0f);

	float radius = 0.5f;

	float a = glm::dot(rayDirection, rayDirection);
	float b = 2.0f * glm::dot(rayOrigin, rayDirection);
	float c = glm::dot(rayOrigin, rayOrigin) - radius * radius;


	float discriminant = b * b - 4.0f * a * c;

	if (discriminant < 0.0f)
	{
		imagePtr[j * imageWidth + i] = 0xff000000;
		return;
	}

	float t0 = (- b + glm::sqrt(discriminant)) / 2.0f * a;
	float t1 = (- b - glm::sqrt(discriminant)) / 2.0f * a;

	glm::vec3 hitPoint = rayOrigin + rayDirection * t1;
	glm::vec3 normal = glm::normalize(hitPoint);

	glm::vec3 lightDir = glm::normalize(glm::vec3(-1.0f, -1.0f, -1.0f));

	float d = glm::max(glm::dot(normal, -lightDir), 0.0f);

	glm::vec3 sphereColor(1.0f, 0.0f, 1.0f);
	sphereColor = sphereColor * d;

	glm::vec4 color = glm::clamp(glm::vec4(sphereColor, 1.0f), glm::vec4(0.0f), glm::vec4(1.0f));
	uint8_t red = (uint8_t)(color.r * 255.0f);
	uint8_t green = (uint8_t)(color.g * 255.0f);
	uint8_t blue = (uint8_t)(color.b * 255.0f);
	uint8_t alpha = (uint8_t)(color.a * 255.0f);
	 
	imagePtr[j * imageWidth + i] = alpha << 24 | blue << 16 | green << 8 | red;

}

void RenderViewport(std::shared_ptr<PixelBuffer> pixelBuffer)
{
	checkCudaErrors(cudaGraphicsMapResources(1, &pixelBuffer->GetCudaResource()));
	size_t size = 0;
	void* devicePtr = 0;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer(&devicePtr, &size, pixelBuffer->GetCudaResource()));

	uint32_t tx = 8, ty = 8;
	dim3 blocks(pixelBuffer->GetWidth() / tx + 1, pixelBuffer->GetHeight() / ty + 1);
	dim3 threads(tx, ty);

	traceRay<<<blocks, threads>>>(devicePtr, pixelBuffer->GetWidth(), pixelBuffer->GetHeight());

	checkCudaErrors(cudaGraphicsUnmapResources(1, &pixelBuffer->GetCudaResource(), 0));
}

