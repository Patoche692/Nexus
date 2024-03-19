#include "Renderer.cuh"
#include "cuda/cuda_math.h"
#include "../Utils.h"

__device__ __constant__ CameraData cameraData;


__global__ void traceRay(void *bufferDevicePtr)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	float x = (float)i / (float)cameraData.viewportWidth * 2.0f - 1.0f;
	float y = (float)j / (float)cameraData.viewportHeight * 2.0f - 1.0f;

	if (i >= cameraData.viewportWidth || j >= cameraData.viewportHeight)
		return;

	uint32_t* imagePtr = (uint32_t*)bufferDevicePtr;

	float3 rayOrigin = cameraData.position;
	float3 up = cameraData.upDirection;
	float aspectRatio = cameraData.viewportWidth / (float)cameraData.viewportHeight;
	float3 rayDirection = normalize(cameraData.forwardDirection + x * aspectRatio * cameraData.rightDirection * cameraData.imagePlaneHalfHeight + y * up * cameraData.imagePlaneHalfHeight);

	float radius = 0.5f;

	float a = dot(rayDirection, rayDirection);
	float b = 2.0f * dot(rayOrigin, rayDirection);
	float c = dot(rayOrigin, rayOrigin) - radius * radius;


	float discriminant = b * b - 4.0f * a * c;

	float t0 = (- b + sqrt(discriminant)) / 2.0f * a;
	float t1 = (- b - sqrt(discriminant)) / 2.0f * a;

	if (discriminant < 0.0f || t1 < 0.0f)
	{
		imagePtr[j * cameraData.viewportWidth + i] = 0xff000000;
		return;
	}

	float3 hitPoint = rayOrigin + rayDirection * t1;
	float3 normal = normalize(hitPoint);

	float3 lightDir = normalize(make_float3(-1.0f, -1.0f, -1.0f));

	float d = max(dot(normal, -lightDir), 0.0f);

	float3 sphereColor = make_float3(1.0f, 0.0f, 1.0f);
	sphereColor = sphereColor * d;

	float4 color = clamp(make_float4(sphereColor, 1.0f), make_float4(0.0f), make_float4(1.0f));
	uint8_t red = (uint8_t)(color.x * 255.0f);
	uint8_t green = (uint8_t)(color.y * 255.0f);
	uint8_t blue = (uint8_t)(color.z * 255.0f);
	uint8_t alpha = (uint8_t)(color.w * 255.0f);
	 
	imagePtr[j * cameraData.viewportWidth + i] = alpha << 24 | blue << 16 | green << 8 | red;

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

	traceRay<<<blocks, threads>>>(devicePtr);

	checkCudaErrors(cudaGraphicsUnmapResources(1, &pixelBuffer->GetCudaResource(), 0));
}

void SendCameraDataToDevice(Camera* camera)
{

	glm::vec3 position = camera->GetPosition();
	glm::vec3 forwardDirection = camera->GetForwardDirection();
	glm::vec3 rightDirection = camera->GetRightDirection();
	glm::vec3 upDirection = glm::cross(rightDirection, forwardDirection);
	CameraData data = {
		camera->GetVerticalFOV(),
		tanf(camera->GetVerticalFOV() / 2.0f * M_PI / 180.0f),
		camera->GetViewportWidth(),
		camera->GetViewportHeight(),
		make_float3(position.x, position.y, position.z),
		make_float3(forwardDirection.x, forwardDirection.y, forwardDirection.z),
		make_float3(rightDirection.x, rightDirection.y, rightDirection.z),
		make_float3(upDirection.x, upDirection.y, upDirection.z)
	};
	checkCudaErrors(cudaMemcpyToSymbol(cameraData, &data, sizeof(CameraData)));
}

