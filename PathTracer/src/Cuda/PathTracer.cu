#include "PathTracer.cuh"
#include "Random.cuh"
#include "BRDF.cuh"
#include "../Utils/cuda_math.h"
#include "../Utils/Utils.h"
#include "../Camera.h"

__device__ __constant__ CameraData cameraData;
__device__ __constant__ SceneData sceneData;
extern __constant__ __device__ Material* materials;

inline __device__ uint32_t toColorUInt(float3 color)
{
	float4 clamped = clamp(make_float4(color, 1.0f), make_float4(0.0f), make_float4(1.0f));
	uint8_t red = (uint8_t)(clamped.x * 255.0f);
	uint8_t green = (uint8_t)(clamped.y * 255.0f);
	uint8_t blue = (uint8_t)(clamped.z * 255.0f);
	uint8_t alpha = (uint8_t)(clamped.w * 255.0f);
	 
	return alpha << 24 | blue << 16 | green << 8 | red;
}

inline __device__ float3 color(Ray& r, unsigned int& rngState)
{
	Ray currentRay = r;
	float3 currentAttenuation = make_float3(1.0f);

	for (int j = 0; j < 10; j++)
	{
		Sphere* closestSphere = nullptr;
		float hitDistance = FLT_MAX;
		float t;

		for (int i = 0; i < sceneData.nSpheres; i++)
		{
			if (sceneData.spheres[i].Hit(currentRay, t) && t < hitDistance)
			{
				hitDistance = t;
				closestSphere = &sceneData.spheres[i];
			}
		}

		if (closestSphere)
		{
			HitResult hitResult;
			hitResult.p = currentRay.origin + currentRay.direction * hitDistance;
			hitResult.rIn = currentRay;
			hitResult.normal = (hitResult.p - closestSphere->position) / closestSphere->radius;
			hitResult.material = materials[closestSphere->materialId];
			
			switch (hitResult.material.type)
			{
			case Material::Type::DIFFUSE:
				diffuseScatter(hitResult, currentAttenuation, currentRay, rngState);
				break;
			case Material::Type::PLASTIC:
				plasticScattter(hitResult, currentAttenuation, currentRay, rngState);
				break;
			default:
				break;
			}
		}
		else
		{
			float3 unitDirection = normalize(currentRay.direction);
			float t = 0.5 * (unitDirection.y + 1.0f);
			return currentAttenuation * ((1.0 - t) * make_float3(1.0f) + t * make_float3(0.5f, 0.7f, 1.0f));
		}
	}

	return make_float3(0.0f);
}

__global__ void traceRay(uint32_t* outBufferPtr, uint32_t frameNumber, float3* accumulationBuffer)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	uint2 pixel = make_uint2(i, j);

	uint2 resolution = cameraData.resolution;
	unsigned int rngState = Random::InitRNG(pixel, resolution, frameNumber);

	// Avoid using modulo, it significantly impacts performance
	float x = (pixel.x + Random::Rand(rngState)) / (float)resolution.x;
	float y = (pixel.y + Random::Rand(rngState)) / (float)resolution.y;

	if (pixel.x >= resolution.x || pixel.y >= resolution.y)
		return;

	Ray ray(
		cameraData.position,
		cameraData.lowerLeftCorner + x * cameraData.horizontal + y * cameraData.vertical - cameraData.position
	);

	float3 c = color(ray, rngState);
	if (frameNumber == 1)
		accumulationBuffer[pixel.y * resolution.x + pixel.x] = c;
	else
		accumulationBuffer[pixel.y * resolution.x + pixel.x] += c;

	c = accumulationBuffer[pixel.y * resolution.x + pixel.x] / frameNumber;

	// Gamma correction
	c = make_float3(sqrt(c.x), sqrt(c.y), sqrt(c.z));
	outBufferPtr[pixel.y * resolution.x + pixel.x] = toColorUInt(c);
}

void RenderViewport(std::shared_ptr<PixelBuffer> pixelBuffer, uint32_t frameNumber, float3* accumulationBuffer)
{
	checkCudaErrors(cudaGraphicsMapResources(1, &pixelBuffer->GetCudaResource()));
	size_t size = 0;
	uint32_t* devicePtr = 0;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&devicePtr, &size, pixelBuffer->GetCudaResource()));

	uint32_t tx = 16, ty = 16;
	dim3 blocks(pixelBuffer->GetWidth() / tx + 1, pixelBuffer->GetHeight() / ty + 1);
	dim3 threads(tx, ty);

	traceRay<<<blocks, threads>>>(devicePtr, frameNumber, accumulationBuffer);

	checkCudaErrors(cudaGetLastError());
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
		make_uint2(camera->GetViewportWidth(), camera->GetViewportHeight())
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
	size_t size = sizeof(unsigned int) + sizeof(Sphere) * data.nSpheres;
	checkCudaErrors(cudaMemcpyToSymbol(sceneData, &data, sizeof(SceneData)));
}

