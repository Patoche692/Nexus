#include "PathTracer.cuh"
#include "Cuda/Random.cuh"
#include "Cuda/BSDF/LambertianBSDF.cuh"
#include "Cuda/BSDF/DielectricBSDF.cuh"
#include "Cuda/BSDF/PlasticBSDF.cuh"
#include "Cuda/BSDF/ConductorBSDF.cuh"
#include "Cuda/BSDF/BSDF.cuh"
#include "Utils/cuda_math.h"
#include "Utils/Utils.h"
#include "texture_indirect_functions.h"
#include "Cuda/BVH/BVH8Traversal.cuh"
#include "Cuda/Scene/Scene.cuh"
#include "Cuda/Scene/Camera.cuh"
#include "Cuda/Sampler.cuh"


__device__ __constant__ uint32_t frameNumber;
__device__ __constant__ uint32_t bounce;

__device__ __constant__ float3* accumulationBuffer;
__device__ __constant__ uint32_t* renderBuffer;

__device__ __constant__ D_Scene scene;
__device__ __constant__ D_PathStateSAO pathState;
__device__ __constant__ D_TraceRequestSAO traceRequest;
__device__ __constant__ D_ShadowTraceRequestSAO shadowTraceRequest;

__device__ __constant__ D_MaterialRequestSAO diffuseMaterialBuffer;
__device__ __constant__ D_MaterialRequestSAO plasticMaterialBuffer;
__device__ __constant__ D_MaterialRequestSAO dielectricMaterialBuffer;
__device__ __constant__ D_MaterialRequestSAO conductorMaterialBuffer;

__device__ D_PixelQuery pixelQuery;
__device__ D_QueueSize queueSize;


inline __device__ uint32_t ToColorUInt(float3 color)
{
	float4 clamped = clamp(make_float4(color, 1.0f), make_float4(0.0f), make_float4(1.0f));
	uint8_t red = (uint8_t)(clamped.x * 255.0f);
	uint8_t green = (uint8_t)(clamped.y * 255.0f);
	uint8_t blue = (uint8_t)(clamped.z * 255.0f);
	uint8_t alpha = (uint8_t)(clamped.w * 255.0f);
	 
	return alpha << 24 | blue << 16 | green << 8 | red;
}

// Approximated ACES tonemapping by Krzysztof Narkowicz. See https://graphics-programming.org/resources/tonemapping/index.html
inline __device__ float3 Tonemap(float3 color)
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
inline __device__ float3 SampleBackground(const D_Scene& scene, float3 direction)
{
	float3 backgroundColor;
	if (scene.hasHdrMap)
	{
		// Theta goes from -PI to PI, phi from -PI/2 to PI/2
		const float theta = atan2(direction.z, direction.x);
		const float phi = asin(direction.y);

		// Equirectangular projection
		const float u = (theta + PI) * INV_PI * 0.5;
		const float v = 1.0f - (phi + PI * 0.5f) * INV_PI;

		backgroundColor = make_float3(tex2D<float4>(scene.hdrMap, u, v));
	}
	else
		backgroundColor = scene.renderSettings.backgroundColor * scene.renderSettings.backgroundIntensity;
	return backgroundColor;
}

__global__ void GenerateKernel()
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

	const D_Camera camera = scene.camera;
	uint2 resolution = camera.resolution;

	if (index >= resolution.x * resolution.y)
		return;

	const uint32_t j = index / resolution.x;
	const uint32_t i = index - j * resolution.x;

	const uint2 pixel = make_uint2(i, j);

	unsigned int rngState = Random::InitRNG(pixel, resolution, frameNumber);

	// Normalized jittered coordinates
	const float x = (pixel.x + Random::Rand(rngState)) / (float)resolution.x;
	const float y = (pixel.y + Random::Rand(rngState)) / (float)resolution.y;

	float2 rd = camera.lensRadius * Random::RandomInUnitDisk(rngState);
	float3 offset = camera.right * rd.x + camera.up * rd.y;

	D_Ray ray(
		camera.position + offset,
		normalize(camera.lowerLeftCorner + x * camera.viewportX + y * camera.viewportY - camera.position - offset)
	);

	if (index == 0)
		queueSize.traceSize[0] = resolution.x * resolution.y;

	traceRequest.ray.origin[index] = ray.origin;
	traceRequest.ray.direction[index] = ray.direction;
	traceRequest.pixelIdx[index] = index;
}


__global__ void TraceKernel()
{
	BVH8Trace(traceRequest, queueSize.traceSize[bounce], &queueSize.traceCount[bounce]);
}

__global__ void TraceShadowKernel()
{
	BVH8TraceShadow(shadowTraceRequest, queueSize.traceShadowSize[bounce], &queueSize.traceShadowCount[bounce], pathState.radiance);
}


__global__ void LogicKernel()
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= queueSize.traceSize[bounce - 1])
		return;

	uint32_t rngState = Random::InitRNG(index, scene.camera.resolution, frameNumber);

	const D_Intersection intersection = traceRequest.intersection.Get(index);
	const D_Ray ray(traceRequest.ray.origin[index], traceRequest.ray.direction[index]);
	const uint32_t pixelIdx = traceRequest.pixelIdx[index];

	const float3 throughput = bounce == 1 ? make_float3(1.0f) : pathState.throughput[pixelIdx];

	// If no intersection, sample background
	if (intersection.hitDistance == 1e30f)
	{
		float3 backgroundColor = SampleBackground(scene, ray.direction);
		if (bounce == 1)
			pathState.radiance[pixelIdx] = throughput * backgroundColor;
		else
			pathState.radiance[pixelIdx] += throughput * backgroundColor;

		if (bounce == 1 && pixelQuery.pixelIdx == pixelIdx)
			pixelQuery.instanceIdx = -1;

		return;
	}

	// Russian roulette
	float probability = fmaxf(throughput);// clamp(fmaxf(currentThroughput), 0.01f, 1.0f);
	if (Random::Rand(rngState) < probability)
	{
		// To get unbiased results, we need to increase the contribution of
		// the non-terminated rays with their probability of being terminated
		pathState.throughput[pixelIdx] = throughput / probability;
	}
	else
		return;

	const D_BVHInstance instance = blas[intersection.instanceIdx];
	const D_Material material = scene.materials[instance.materialId];

	int32_t requestIdx;
	switch (material.type)
	{
	case D_Material::D_Type::DIFFUSE:
		requestIdx = atomicAdd(&queueSize.diffuseSize[bounce], 1);
		diffuseMaterialBuffer.intersection.Set(requestIdx, intersection);
		diffuseMaterialBuffer.rayDirection[requestIdx] = ray.direction;
		diffuseMaterialBuffer.pixelIdx[requestIdx] = pixelIdx;
		break;
	case D_Material::D_Type::PLASTIC:
		requestIdx = atomicAdd(&queueSize.plasticSize[bounce], 1);
		plasticMaterialBuffer.intersection.Set(requestIdx, intersection);
		plasticMaterialBuffer.rayDirection[requestIdx] = ray.direction;
		plasticMaterialBuffer.pixelIdx[requestIdx] = pixelIdx;
		break;
	case D_Material::D_Type::DIELECTRIC:
		requestIdx = atomicAdd(&queueSize.dielectricSize[bounce], 1);
		dielectricMaterialBuffer.intersection.Set(requestIdx, intersection);
		dielectricMaterialBuffer.rayDirection[requestIdx] = ray.direction;
		dielectricMaterialBuffer.pixelIdx[requestIdx] = pixelIdx;
		break;
	case D_Material::D_Type::CONDUCTOR:
		requestIdx = atomicAdd(&queueSize.conductorSize[bounce], 1);
		conductorMaterialBuffer.intersection.Set(requestIdx, intersection);
		conductorMaterialBuffer.rayDirection[requestIdx] = ray.direction;
		conductorMaterialBuffer.pixelIdx[requestIdx] = pixelIdx;
		break;
	default:
		break;
	}
}


template<typename BSDF>
inline __device__ void NextEventEstimation(
	const float3 wi,
	const D_Material& material,
	const D_Intersection& intersection,
	const float3 hitPoint,
	const float3 normal,
	const float3 hitGNormal,
	const float3 throughput,
	const uint32_t pixelIdx,
	unsigned int& rngState
) {
	D_Light light = Sampler::UniformSampleLights(scene.lights, scene.lightCount, rngState);

	if (light.type == D_Light::Type::MESH_LIGHT)
	{
		D_BVHInstance instance = blas[light.mesh.meshId];

		uint32_t triangleIdx;
		float2 uv;
		Sampler::UniformSampleMesh(bvhs[instance.bvhIdx], rngState, triangleIdx, uv);

		D_Triangle triangle = bvhs[instance.bvhIdx].triangles[triangleIdx];

		float3 p = Barycentric(triangle.pos0, triangle.pos1, triangle.pos2, uv);
		p = instance.transform.TransformPoint(p);

		const float3 lightGNormal = normalize(instance.invTransform.Transposed().TransformVector(triangle.Normal()));

		float3 lightNormal = Barycentric(triangle.normal0, triangle.normal1, triangle.normal2, uv);
		lightNormal = normalize(instance.invTransform.Transposed().TransformVector(lightNormal));

		D_Ray shadowRay;

		float3 toLight = p - hitPoint;
		float offsetDirection = Utils::SgnE(dot(toLight, normal));
		shadowRay.origin = OffsetRay(hitPoint, hitGNormal * offsetDirection);

		offsetDirection = Utils::SgnE(dot(-toLight, lightNormal));
		p = OffsetRay(p, lightGNormal * offsetDirection);

		toLight = p - shadowRay.origin;
		const float distance = length(toLight);
		shadowRay.direction = toLight / distance;
		shadowRay.invDirection = 1.0f / shadowRay.direction;

		float4 qRotationToZ = getRotationToZAxis(normal);
		const float3 wo = rotatePoint(qRotationToZ, shadowRay.direction);

		const float cosThetaO = fabs(dot(lightNormal, shadowRay.direction));

		const float dSquared = dot(toLight, toLight);

		const D_Triangle triangleTransformed(
			instance.transform.TransformPoint(triangle.pos0),
			instance.transform.TransformPoint(triangle.pos1),
			instance.transform.TransformPoint(triangle.pos2)
		);

		float lightPdf = 1.0f / (scene.lightCount * bvhs[instance.bvhIdx].triCount * triangleTransformed.Area());
		// Transform pdf over an area to pdf over directions
		lightPdf *= dSquared / cosThetaO;

		if (!Sampler::IsPdfValid(lightPdf))
			return;

		const D_Material lightMaterial = scene.materials[instance.materialId];

		float3 sampleThroughput;
		float bsdfPdf;

		bool sampleIsValid = D_BSDF::Eval<BSDF>(material, wi, wo, sampleThroughput, bsdfPdf);

		if (!sampleIsValid)
			return;

		const float weight = Sampler::PowerHeuristic(lightPdf, bsdfPdf);

		float3 emissive;
		if (lightMaterial.emissiveMapId != -1)
		{
			float2 texUv = Barycentric(triangle.texCoord0, triangle.texCoord1, triangle.texCoord2, uv);
			emissive = make_float3(tex2D<float4>(scene.emissiveMaps[lightMaterial.emissiveMapId], texUv.x, texUv.y));
		}
		else
			emissive = lightMaterial.emissive;

		const float3 radiance = weight * throughput * sampleThroughput * emissive * lightMaterial.intensity / lightPdf;

		const int32_t index = atomicAdd(&queueSize.traceShadowSize[bounce], 1);
		shadowTraceRequest.hitDistance[index] = distance;
		shadowTraceRequest.radiance[index] = radiance;
		shadowTraceRequest.ray.Set(index, shadowRay);
		shadowTraceRequest.pixelIdx[index] = pixelIdx;
	}
}


template<typename BSDF>
inline __device__ void Shade(D_MaterialRequestSAO materialRequest, int32_t size)
{
	const int32_t requestIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (requestIdx >= size)
		return;

	const D_Intersection intersection = materialRequest.intersection.Get(requestIdx);
	const float3 rayDirection = materialRequest.rayDirection[requestIdx];
	const uint32_t pixelIdx = materialRequest.pixelIdx[requestIdx];

	float3 throughput = bounce == 1 ? make_float3(1.0f) : pathState.throughput[pixelIdx];

	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t rngState = Random::InitRNG(index, scene.camera.resolution, frameNumber);

	const D_BVHInstance instance = blas[intersection.instanceIdx];
	const D_Triangle triangle = bvhs[instance.bvhIdx].triangles[intersection.triIdx];

	D_Material material = scene.materials[instance.materialId];

	const float2 uv = make_float2(intersection.u, intersection.v);

	float3 p = Barycentric(triangle.pos0, triangle.pos1, triangle.pos2, uv);
	p = instance.transform.TransformPoint(p);

	float3 normal = Barycentric(triangle.normal0, triangle.normal1, triangle.normal2, uv);
	float2 texUv = Barycentric(triangle.texCoord0, triangle.texCoord1, triangle.texCoord2, uv);

	// We use the transposed of the inverse matrix to transform normals.
	// See https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/transforming-normals.html
	normal = normalize(instance.invTransform.Transposed().TransformVector(normal));

	float3 gNormal = normalize(instance.invTransform.Transposed().TransformVector(triangle.Normal()));

	if (material.emissiveMapId != -1)
	{
		material.emissive = make_float3(tex2D<float4>(scene.emissiveMaps[material.emissiveMapId], texUv.x, texUv.y));
	}

	float3 radiance = make_float3(0.0f);

	if (fmaxf(material.emissive * material.intensity) > 0.0f)
	{
		float weight = 1.0f;

		// Not using MIS for primary rays
		if (scene.renderSettings.useMIS && bounce > 1)
		{
			const float lastPdf = pathState.lastPdf[pixelIdx];

			const float cosThetaO = fabs(dot(normal, rayDirection));

			const float dSquared = Square(intersection.hitDistance);

			const D_Triangle triangleTransformed(
				instance.transform.TransformPoint(triangle.pos0),
				instance.transform.TransformPoint(triangle.pos1),
				instance.transform.TransformPoint(triangle.pos2)
			);

			float lightPdf = 1.0f / (scene.lightCount * bvhs[instance.bvhIdx].triCount * triangleTransformed.Area());
			// Transform pdf over an area to pdf over directions
			lightPdf *= dSquared / cosThetaO;

			weight = Sampler::PowerHeuristic(lastPdf, lightPdf);
		}
		radiance = weight * material.emissive * material.intensity * throughput;
	}

	if (bounce == 1)
		pathState.radiance[pixelIdx] = radiance;
	else
		pathState.radiance[pixelIdx] += radiance;

	if (bounce == scene.renderSettings.pathLength)
		return;

	if (bounce == 1 && pixelQuery.pixelIdx == pixelIdx)
		pixelQuery.instanceIdx = intersection.instanceIdx;

	float4 color = make_float4(1.0f);
	if (material.diffuseMapId != -1)
	{
		color = tex2D<float4>(scene.diffuseMaps[material.diffuseMapId], texUv.x, texUv.y);
		material.diffuse.albedo = make_float3(color);
	}

	// Invert normals for non transmissive material if the primitive is backfacing the ray
	if (dot(gNormal, rayDirection) > 0.0f && material.type != D_Material::D_Type::DIELECTRIC)
	{
		normal = -normal;
		gNormal = -gNormal;
	}

	float4 qRotationToZ = getRotationToZAxis(normal);
	float3 wi = rotatePoint(qRotationToZ, -rayDirection);

	float3 wo, sampleThroughput;
	float pdf;

	// TODO: handle use of MIS for semi-transparent diffuse textures
	//bool useMIS = material.opacity == 1.0f;

	bool scattered = true;
	// Handle texture transparency
	if (Random::Rand(rngState) > material.opacity || (material.diffuseMapId != -1 && Random::Rand(rngState) > color.w))
	{
		wo = -wi;
		sampleThroughput = make_float3(1.0f);
		pdf = 1.0f;
	}
	else
	{
		if (scene.renderSettings.useMIS)
			NextEventEstimation<BSDF>(wi, material, intersection, p, normal, gNormal, throughput, pixelIdx, rngState);

		scattered = D_BSDF::Sample<BSDF>(material, wi, wo, sampleThroughput, pdf, rngState);
	}

	if (!scattered)
		return;

	throughput *= sampleThroughput;

	wo = normalize(rotatePoint(invertRotation(qRotationToZ), wo));

	// If sample is valid, write trace request in the path state
	const float offsetDirection = Utils::SgnE(dot(wo, normal));
	const float3 offsetOrigin = OffsetRay(p, gNormal * offsetDirection);
	const D_Ray scatteredRay(offsetOrigin, wo);

	const int32_t traceRequestIdx = atomicAdd(&queueSize.traceSize[bounce], 1);

	traceRequest.ray.Set(traceRequestIdx, scatteredRay);
	traceRequest.pixelIdx[traceRequestIdx] = pixelIdx;

	pathState.throughput[pixelIdx] = throughput;
	pathState.lastPdf[pixelIdx] = pdf;
}

__global__ void DiffuseMaterialKernel()
{
	Shade<D_LambertianBSDF>(diffuseMaterialBuffer, queueSize.diffuseSize[bounce]);
}

__global__ void PlasticMaterialKernel()
{
	Shade<D_PlasticBSDF>(plasticMaterialBuffer, queueSize.plasticSize[bounce]);
}

__global__ void DielectricMaterialKernel()
{
	Shade<D_DielectricBSDF>(dielectricMaterialBuffer, queueSize.dielectricSize[bounce]);
}

__global__ void ConductorMaterialKernel()
{
	//Shade<D_ConductorBSDF>(conductorMaterialBuffer, queueSize.conductorSize[bounce]);
}

__global__ void AccumulateKernel()
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

	const uint2 resolution = scene.camera.resolution;

	if (index >= resolution.x * resolution.y)
		return;

	if (frameNumber == 1)
		accumulationBuffer[index] = pathState.radiance[index];
	else
		accumulationBuffer[index] += (pathState.radiance[index] - accumulationBuffer[index]) / frameNumber;


	renderBuffer[index] = ToColorUInt(Utils::LinearToGamma(Tonemap(accumulationBuffer[index])));
}

D_Scene* GetDeviceSceneAddress()
{
	D_Scene* deviceScene;
	CheckCudaErrors(cudaGetSymbolAddress((void**)&deviceScene, scene));
	return deviceScene;
}

float3** GetDeviceAccumulationBufferAddress()
{
	float3** buffer;
	CheckCudaErrors(cudaGetSymbolAddress((void**)&buffer, accumulationBuffer));
	return buffer;
}

uint32_t** GetDeviceRenderBufferAddress()
{
	uint32_t** buffer;
	CheckCudaErrors(cudaGetSymbolAddress((void**)&buffer, renderBuffer));
	return buffer;
}

uint32_t* GetDeviceFrameNumberAddress()
{
	uint32_t* target;
	CheckCudaErrors(cudaGetSymbolAddress((void**)&target, frameNumber));
	return target;
}

uint32_t* GetDeviceBounceAddress()
{
	uint32_t* target;
	CheckCudaErrors(cudaGetSymbolAddress((void**)&target, bounce));
	return target;
}

D_BVH8* GetDeviceTLASAddress()
{
	D_BVH8* target;
	CheckCudaErrors(cudaGetSymbolAddress((void**)&target, tlas));
	return target;
}

D_BVH8** GetDeviceBVHAddress()
{
	D_BVH8** target;
	CheckCudaErrors(cudaGetSymbolAddress((void**)&target, bvhs));
	return target;
}

D_BVHInstance** GetDeviceBLASAddress()
{
	D_BVHInstance** target;
	CheckCudaErrors(cudaGetSymbolAddress((void**)&target, blas));
	return target;
}

D_PathStateSAO* GetDevicePathStateAddress()
{
	D_PathStateSAO* target;
	CheckCudaErrors(cudaGetSymbolAddress((void**)&target, pathState));
	return target;
}

D_ShadowTraceRequestSAO* GetDeviceShadowTraceRequestAddress()
{
	D_ShadowTraceRequestSAO* target;
	CheckCudaErrors(cudaGetSymbolAddress((void**)&target, shadowTraceRequest));
	return target;
}

D_TraceRequestSAO* GetDeviceTraceRequestAddress()
{
	D_TraceRequestSAO* target;
	CheckCudaErrors(cudaGetSymbolAddress((void**)&target, traceRequest));
	return target;
}

D_MaterialRequestSAO* GetDeviceDiffuseRequestAddress()
{
	D_MaterialRequestSAO* target;
	CheckCudaErrors(cudaGetSymbolAddress((void**)&target, diffuseMaterialBuffer));
	return target;
}

D_MaterialRequestSAO* GetDevicePlasticRequestAddress()
{
	D_MaterialRequestSAO* target;
	CheckCudaErrors(cudaGetSymbolAddress((void**)&target, plasticMaterialBuffer));
	return target;
}

D_MaterialRequestSAO* GetDeviceDielectricRequestAddress()
{
	D_MaterialRequestSAO* target;
	CheckCudaErrors(cudaGetSymbolAddress((void**)&target, dielectricMaterialBuffer));
	return target;
}

D_MaterialRequestSAO* GetDeviceConductorRequestAddress()
{
	D_MaterialRequestSAO* target;
	CheckCudaErrors(cudaGetSymbolAddress((void**)&target, conductorMaterialBuffer));
	return target;
}

D_QueueSize* GetDeviceQueueSizeAddress()
{
	D_QueueSize* target;
	CheckCudaErrors(cudaGetSymbolAddress((void**)&target, queueSize));
	return target;
}

D_PixelQuery* GetDevicePixelQueryAddress()
{
	D_PixelQuery* target;
	CheckCudaErrors(cudaGetSymbolAddress((void**)&target, pixelQuery));
	return target;
}
