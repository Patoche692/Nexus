#include "Renderer.cuh"
#include "cuda/cuda_math.h"
#include "../Utils.h"

__device__ __constant__ CameraData cameraData;
__device__ __constant__ SceneData sceneData;


__global__ void traceRay(void *bufferDevicePtr)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	float x = (float)i / (float)cameraData.viewportWidth;
	float y = (float)j / (float)cameraData.viewportHeight;

	if (i >= cameraData.viewportWidth || j >= cameraData.viewportHeight)
		return;

	uint32_t* imagePtr = (uint32_t*)bufferDevicePtr;

	float3 rayOrigin = cameraData.position;
	float3 rayDirection = cameraData.lowerLeftCorner + x * cameraData.horizontal + y * cameraData.vertical - rayOrigin;

	SphereData* closestSphere = nullptr;
	float hitDistance = FLT_MAX;

	for (int i = 0; i < sceneData.nSpheres; i++)
	{
		float3 origin = rayOrigin - sceneData.spheres[i].position;

		float radius = sceneData.spheres[i].radius;

		float a = dot(rayDirection, rayDirection);
		float b = dot(origin, rayDirection);
		float c = dot(origin, origin) - radius * radius;

		float discriminant = b * b - a * c;

		if (discriminant < 0.0f)
		{
			imagePtr[j * cameraData.viewportWidth + i] = 0xff000000;
			continue;
		}

		float closestT = (-b - sqrt(discriminant)) / a;

		if (closestT < hitDistance && closestT > 0.0f)
		{
			hitDistance = closestT;
			closestSphere = &sceneData.spheres[i];
		}
	}

	if (closestSphere == nullptr)
	{
		imagePtr[j * cameraData.viewportWidth + i] = 0xff000000;
		return;
	}

	float3 hitPoint = rayOrigin + rayDirection * hitDistance;
	float3 normal = (hitPoint - closestSphere->position) / closestSphere->radius;

	float3 lightDir = normalize(make_float3(-1.0f));

	float d = max(dot(normal, -lightDir), 0.0f);

	float3 sphereColor = closestSphere->material.color;
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

	float aspectRatio = camera->GetViewportWidth() / (float)camera->GetViewportHeight();
	float halfHeight = tanf(camera->GetVerticalFOV() / 2.0f * M_PI / 180.0f);
	float halfWidth = aspectRatio * halfHeight;

	glm::vec3 lowerLeftCorner = position - halfWidth * rightDirection - halfHeight * upDirection + forwardDirection;
	glm::vec3 horizontal = 2 * halfWidth * rightDirection;
	glm::vec3 vertical = 2 * halfHeight * upDirection;

	CameraData data = {
		make_float3(position),
		make_float3(forwardDirection),
		make_float3(lowerLeftCorner),
		make_float3(horizontal),
		make_float3(vertical),
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
		data.spheres[i] = {
			spheres[i].radius,
			make_float3(spheres[i].position),
			{ make_float3(spheres[i].material.color) }
		};
	}
	// TODO: change the size of copy
	checkCudaErrors(cudaMemcpyToSymbol(sceneData, &data, (sizeof(unsigned int) + sizeof(Sphere)) * data.nSpheres));
}
