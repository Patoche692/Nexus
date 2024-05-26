#include "PathTracer.cuh"
#include "Random.cuh"
#include "BSDF/DielectricBSDF.cuh"
#include "BSDF/LambertianBSDF.cuh"
#include "BSDF/BSDF.cuh"
#include "Utils/cuda_math.h"
#include "Utils/Utils.h"
#include "Camera.h"
#include "Geometry/BVH/TLAS.h"
#include "texture_indirect_functions.h"
#include "BSDF/ConductorBSDF.cuh"
#include "BVH/TLAS.cuh"


__constant__ __device__ CameraData cameraData;
__constant__ __device__ SceneData sceneData;
extern __constant__ __device__ Material* materials;
extern __constant__ __device__ cudaTextureObject_t* diffuseMaps;
extern __constant__ __device__ cudaTextureObject_t* emissiveMaps;
extern __constant__ __device__ TLAS tlas;

inline __device__ uint32_t toColorUInt(float3 color)
{
	float4 clamped = clamp(make_float4(color, 1.0f), make_float4(0.0f), make_float4(1.0f));
	uint8_t red = (uint8_t)(clamped.x * 255.0f);
	uint8_t green = (uint8_t)(clamped.y * 255.0f);
	uint8_t blue = (uint8_t)(clamped.z * 255.0f);
	uint8_t alpha = (uint8_t)(clamped.w * 255.0f);
	 
	return alpha << 24 | blue << 16 | green << 8 | red;
}

// Approximated ACES tonemapping by Krzysztof Narkowicz. See https://graphics-programming.org/resources/tonemapping/index.html
inline __device__ float3 tonemap(float3 color)
{
	// Tungsten renderer filmic tonemapping to compare my results
	//float3 x = fmaxf(make_float3(0.0f), color - 0.004f);
	//return (x * (6.2f * x + 0.5f)) / (x * (6.2f * x + 1.7f) + 0.06f);

	color *= 0.6f; // Exposure
	const float a = 2.51f;
	const float b = 0.03f;
	const float c = 2.43f;
	const float d = 0.59f;
	const float e = 0.14f;
	return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0f, 1.0f);
}

// If necessary, sample the HDR map (from spherical to equirectangular projection)
inline __device__ float3 sampleBackground(float3 direction)
{
	float3 backgroundColor;
	if (sceneData.hasHdrMap)
	{
		// Theta goes from -PI to PI, phi from -PI/2 to PI/2
		const float theta = atan2(direction.z, direction.x);
		const float phi = asin(direction.y);

		// Equirectangular projection
		const float u = (theta + M_PI) * INV_PI * 0.5;
		const float v = 1.0f - (phi + M_PI * 0.5f) * INV_PI;

		backgroundColor = make_float3(tex2D<float4>(sceneData.hdrMap, u, v));
	}
	else
		backgroundColor = make_float3(0.02f);
	return backgroundColor;
}

inline __device__ float3 color(Ray& r, unsigned int& rngState)
{
	Ray currentRay = r;
	float3 currentThroughput = make_float3(1.0f);
	float3 emission = make_float3(0.0f);

	for (int j = 0; j < 15; j++)
	{
		// Reset the hit position and calculate the inverse of the new direction
		currentRay.hit.t = 1e30f;
		currentRay.invDirection = 1 / currentRay.direction;

		IntersectTLAS(tlas, currentRay);

		// If no intersection, sample background
		if (currentRay.hit.t == 1e30f)
		{ 
			float3 backgroundColor = sampleBackground(currentRay.direction);
			return currentThroughput * backgroundColor + emission;
		}

		HitResult hitResult;
		hitResult.p = currentRay.origin + currentRay.direction * currentRay.hit.t;
		hitResult.rIn = currentRay;

		BVHInstance& instance = tlas.blas[currentRay.hit.instanceIdx];
		Triangle& triangle = instance.bvh->triangles[currentRay.hit.triIdx];
		float u = currentRay.hit.u, v = currentRay.hit.v;

		// Interpolating and rotating the normal
		hitResult.normal = u * triangle.normal1 + v * triangle.normal2 + (1 - (u + v)) * triangle.normal0;
		hitResult.normal = normalize(instance.transform.TransformVector(hitResult.normal));

		float3 gNormal = normalize(instance.transform.TransformVector(triangle.Normal()));

		hitResult.material = materials[instance.materialId];

		if (hitResult.material.diffuseMapId == -1)
			hitResult.albedo = hitResult.material.diffuse.albedo;
		else
		{
			float2 uv = u * triangle.texCoord1 + v * triangle.texCoord2 + (1 - (u + v)) * triangle.texCoord0;
			hitResult.material.diffuse.albedo = make_float3(tex2D<float4>(diffuseMaps[hitResult.material.diffuseMapId], uv.x, uv.y));

		}
		if (hitResult.material.emissiveMapId != -1) {
			float2 uv = u * triangle.texCoord1 + v * triangle.texCoord2 + (1 - (u + v)) * triangle.texCoord0;
			hitResult.material.emissive = make_float3(tex2D<float4>(emissiveMaps[hitResult.material.emissiveMapId], uv.x, uv.y));
			//if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
			//	printf("emissive map: %f, %f, %f\n", hitResult.material.emissive.x, hitResult.material.emissive.y, hitResult.material.emissive.z);
		}

		// Normal flipping
		//if (dot(hitResult.normal, currentRay.direction) > 0.0f)
		//	hitResult.normal = -hitResult.normal;

		// Invert normals for non transmissive material if the primitive is backfacing the ray
		if (dot(gNormal, currentRay.direction) > 0.0f && (hitResult.material.type != Material::Type::DIELECTRIC || hitResult.material.dielectric.transmittance == 0.0f))
		{
			hitResult.normal = -hitResult.normal;
			gNormal = -gNormal;
		}

		if (fmaxf(hitResult.material.emissive) > 0.0f)
			emission += hitResult.material.emissive * hitResult.material.intensity * currentThroughput;


		// Transform the incoming ray to local space (positive Z axis aligned with shading normal)
		float4 qRotationToZ = getRotationToZAxis(hitResult.normal);
		float3 wi = rotatePoint(qRotationToZ, -hitResult.rIn.direction);

		//bool wiGeometryBackSide = dot(wi, gNormal) < 0.0f;
		//bool wiShadingBackSide = dot(wi, hitResult.normal) < 0.0f;

		//if (wiGeometryBackSide != wiShadingBackSide)
		//	continue;

		float3 throughput;
		float3 wo;

		bool scattered = false;
		switch (hitResult.material.type)
		{
		case Material::Type::DIFFUSE:
			scattered = BSDF::Sample<LambertianBSDF>(hitResult, wi, wo, throughput, rngState);
			break;
		case Material::Type::DIELECTRIC:
			scattered = BSDF::Sample<DielectricBSDF>(hitResult, wi, wo, throughput, rngState);
			break;
		case Material::Type::CONDUCTOR:
			scattered = BSDF::Sample<ConductorBSDF>(hitResult, wi, wo, throughput, rngState);
			break;
		default:
			break;
		}

		if (scattered)
		{
			// Inverse ray transformation to world space
			wo = normalize(rotatePoint(invertRotation(qRotationToZ), wo));
			bool woGeometryBackSide = dot(wo, gNormal) < 0.0f;
			bool woShadingBackSide = dot(wo, hitResult.normal) < 0.0f;

			if (woGeometryBackSide == woShadingBackSide)
			{
				currentThroughput *= throughput;
				float offsetDirection = woGeometryBackSide ? -1.0f : 1.0f;
				currentRay.origin = hitResult.p + offsetDirection * 1.0e-4 * hitResult.normal;
				currentRay.direction = wo;
			}
		}

		// Russian roulette
		if (j > 2)
		{
			float p = clamp(fmax(currentThroughput.x, fmax(currentThroughput.y, currentThroughput.z)), 0.01f, 1.0f);
			if (Random::Rand(rngState) < p)
			{
				// To get unbiased results, we need to increase the contribution of
				// the non-terminated rays with their probability of being terminated
				currentThroughput *= 1.0f / p;
			}
			else
				return emission;
		}
	}

	return emission;
}

__global__ void traceRay(uint32_t* outBufferPtr, uint32_t frameNumber, float3* accumulationBuffer)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	uint2 pixel = make_uint2(i, j);

	uint2 resolution = cameraData.resolution;

	if (pixel.x >= resolution.x || pixel.y >= resolution.y)
		return;

	unsigned int rngState = Random::InitRNG(pixel, resolution, frameNumber);

	// Avoid using modulo, it significantly impacts performance
	float x = (pixel.x + Random::Rand(rngState)) / (float)resolution.x;
	float y = (pixel.y + Random::Rand(rngState)) / (float)resolution.y;

	float2 rd = cameraData.lensRadius * Random::RandomInUnitDisk(rngState);
	float3 offset = cameraData.right * rd.x + cameraData.up * rd.y;

	Ray ray(
		//make_float3(2, 0, 0), make_float3(-1, 0.001, 0.001)
		cameraData.position + offset,
		normalize(cameraData.lowerLeftCorner + x * cameraData.viewportX + y * cameraData.viewportY - cameraData.position - offset)
	);

	float3 c = color(ray, rngState);
	if (frameNumber == 1)
		accumulationBuffer[pixel.y * resolution.x + pixel.x] = c;
	else
		accumulationBuffer[pixel.y * resolution.x + pixel.x] += c;

	c = accumulationBuffer[pixel.y * resolution.x + pixel.x] / frameNumber;

	outBufferPtr[pixel.y * resolution.x + pixel.x] = toColorUInt(Utils::LinearToGamma(tonemap(c)));
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

void InitDeviceSceneData()
{
	SceneData scene;
	scene.hasHdrMap = false;
	checkCudaErrors(cudaMemcpyToSymbol(sceneData, &scene, sizeof(SceneData)));
}

void SendHDRMapToDevice(const Texture& map)
{
	// Channel descriptor for 4 Channels (RGBA)
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	cudaArray_t cuArray;
	checkCudaErrors(cudaMallocArray(&cuArray, &channelDesc, map.width, map.height));

	const size_t spitch = map.width * 4 * sizeof(float);
	checkCudaErrors(cudaMemcpy2DToArray(cuArray, 0, 0, map.pixels, spitch, map.width * 4 * sizeof(float), map.height, cudaMemcpyHostToDevice));

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

	SceneData scene;
	scene.hasHdrMap = true;
	scene.hdrMap = texObject;
	checkCudaErrors(cudaMemcpyToSymbol(sceneData, &scene, sizeof(SceneData)));
}

void SendCameraDataToDevice(Camera* camera)
{
	float3 position = camera->GetPosition();
	float3 forwardDirection = camera->GetForwardDirection();
	float3 rightDirection = camera->GetRightDirection();
	float3 upDirection = cross(rightDirection, forwardDirection);

	float aspectRatio = camera->GetViewportWidth() / (float)camera->GetViewportHeight();
	float halfHeight = camera->GetFocusDist() * tanf(camera->GetVerticalFOV() / 2.0f * M_PI / 180.0f);
	float halfWidth = aspectRatio * halfHeight;

	float3 viewportX = 2 * halfWidth * rightDirection;
	float3 viewportY = 2 * halfHeight * upDirection;
	float3 lowerLeftCorner = position - viewportX / 2.0f - viewportY / 2.0f + forwardDirection * camera->GetFocusDist();

	float lensRadius = camera->GetFocusDist() * tanf(camera->GetDefocusAngle() / 2.0f * M_PI / 180.0f);

	CameraData data = {
		position,
		rightDirection,
		upDirection,
		lensRadius,
		lowerLeftCorner,
		viewportX,
		viewportY,
		make_uint2(camera->GetViewportWidth(), camera->GetViewportHeight())
	};
	checkCudaErrors(cudaMemcpyToSymbol(cameraData, &data, sizeof(CameraData)));
}