#include "Renderer.cuh"
#include "../Utils/cuda_math.h"
#include "../Utils/Utils.h"
#include "../Camera.h"

__device__ __constant__ CameraData cameraData;
__device__ __constant__ SceneData sceneData;


inline __device__ uint32_t toColorUInt(float3& color)
{
	float4 clamped = clamp(make_float4(color, 1.0f), make_float4(0.0f), make_float4(1.0f));
	uint8_t red = (uint8_t)(clamped.x * 255.0f);
	uint8_t green = (uint8_t)(clamped.y * 255.0f);
	uint8_t blue = (uint8_t)(clamped.z * 255.0f);
	uint8_t alpha = (uint8_t)(clamped.w * 255.0f);
	 
	return alpha << 24 | blue << 16 | green << 8 | red;
}


__global__ void traceRay(uint32_t *outBufferPtr)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	// Avoid using modulo, it significantly impacts performance
	float x = i / (float)cameraData.viewportWidth;
	float y = j / (float)cameraData.viewportHeight;

	if (i >= cameraData.viewportWidth || j >= cameraData.viewportHeight)
		return;

	Ray ray(
		cameraData.position,
		cameraData.lowerLeftCorner + x * cameraData.horizontal + y * cameraData.vertical - cameraData.position
	);

	Sphere* closestSphere = nullptr;
	float hitDistance = FLT_MAX;
	float t;

	for (int i = 0; i < sceneData.nSpheres; i++)
	{
		if (sceneData.spheres[i].Hit(ray, t) && t < hitDistance)
		{
			hitDistance = t;
			closestSphere = &sceneData.spheres[i];
		}
	}

	if (closestSphere == nullptr)
	{
		outBufferPtr[j * cameraData.viewportWidth + i] = 0xff000000;
		return;
	}

	float3 hitPoint = ray.origin + ray.direction * hitDistance;
	float3 normal = (hitPoint - closestSphere->position) / closestSphere->radius;

	float3 lightDir = normalize(make_float3(-1.0f));

	float d = max(dot(normal, -lightDir), 0.0f);

	float3 sphereColor = closestSphere->material.color;
	sphereColor = sphereColor * d;

	outBufferPtr[j * cameraData.viewportWidth + i] = toColorUInt(sphereColor);

}

void RenderViewport(std::shared_ptr<PixelBuffer> pixelBuffer)
{
	checkCudaErrors(cudaGraphicsMapResources(1, &pixelBuffer->GetCudaResource()));
	size_t size = 0;
	uint32_t* devicePtr = 0;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&devicePtr, &size, pixelBuffer->GetCudaResource()));

	uint32_t tx = 16, ty = 16;
	dim3 blocks(pixelBuffer->GetWidth() / tx + 1, pixelBuffer->GetHeight() / ty + 1);
	dim3 threads(tx, ty);

	traceRay<<<blocks, threads>>>(devicePtr);

	checkCudaErrors(cudaGraphicsUnmapResources(1, &pixelBuffer->GetCudaResource(), 0));
}

void SendCameraDataToDevice(Camera* camera)
{
	float3 position = camera->GetPosition();
	float3 forwardDirection = camera->GetForwardDirection();
	float3 rightDirection = camera->GetRightDirection();
	float3 upDirection = cross(rightDirection, forwardDirection);

	float aspectRatio = camera->GetViewportWidth() / (float)camera->GetViewportHeight();
	float halfHeight = tanf(camera->GetVerticalFOV() / 2.0f * M_PI / 180.0f);
	float halfWidth = aspectRatio * halfHeight;

	float3 lowerLeftCorner = position - halfWidth * rightDirection - halfHeight * upDirection + forwardDirection;
	float3 horizontal = 2 * halfWidth * rightDirection;
	float3 vertical = 2 * halfHeight * upDirection;

	CameraData data = {
		position,
		forwardDirection,
		lowerLeftCorner,
		horizontal,
		vertical,
		camera->GetViewportWidth(),
		camera->GetViewportHeight()
	};
	checkCudaErrors(cudaMemcpyToSymbol(cameraData, &data, sizeof(CameraData)));
}

void SendSceneDataToDevice(Scene* scene)
{
	SceneData data;
	std::vector<Sphere> spheres = scene->GetSpheres();
	data.nSpheres = spheres.size();
	for (int i = 0; i < spheres.size(); i++)
	{
		data.spheres[i] = spheres[i];
	}
	// TODO: change the size of copy
	checkCudaErrors(cudaMemcpyToSymbol(sceneData, &data, sizeof(unsigned int) + sizeof(Sphere) * data.nSpheres));
}
