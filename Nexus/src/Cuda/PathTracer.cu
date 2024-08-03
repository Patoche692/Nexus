#include "PathTracer.cuh"
#include "Random.cuh"
#include "BSDF/LambertianBSDF.cuh"
#include "BSDF/DielectricBSDF.cuh"
#include "BSDF/PlasticBSDF.cuh"
#include "BSDF/ConductorBSDF.cuh"
#include "BSDF/BSDF.cuh"
#include "Utils/cuda_math.h"
#include "Utils/Utils.h"
#include "texture_indirect_functions.h"
#include "BVH/TLASTraversal.cuh"
#include "Scene/Scene.cuh"
#include "Scene/Camera.cuh"
#include "Sampler.cuh"

__constant__ __device__ D_Settings settings;


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
		const float u = (theta + M_PI) * INV_PI * 0.5;
		const float v = 1.0f - (phi + M_PI * 0.5f) * INV_PI;

		backgroundColor = make_float3(tex2D<float4>(scene.hdrMap, u, v));
	}
	else
		backgroundColor = make_float3(0.02f);
	return backgroundColor;
}

inline __device__ float3 NextEventEstimation(const D_Scene& scene, const D_Ray& r, const D_HitResult& hitResult, const float3 hitGNormal, unsigned int& rngState)
{
	D_Light light = Sampler::UniformSampleLights(scene.lights, scene.lightCount, rngState);

	if (light.type == D_Light::Type::MESH_LIGHT)
	{
		D_BVHInstance instance = scene.tlas.blas[light.mesh.meshId];

		uint32_t triangleIdx;
		float2 uv;
		Sampler::UniformSampleMesh(scene.tlas.bvhs[instance.bvhIdx], rngState, triangleIdx, uv);

		D_Triangle triangle = scene.tlas.bvhs[instance.bvhIdx].triangles[triangleIdx];

		const float3 edge1 = triangle.pos1 - triangle.pos0;
		const float3 edge2 = triangle.pos2 - triangle.pos0;
		float3 p = triangle.pos0 + uv.x * edge1 + uv.y * edge2;
		p = instance.transform.TransformPoint(p);

		float4 qRotationToZ = getRotationToZAxis(hitResult.normal);
		const float3 wi = rotatePoint(qRotationToZ, -hitResult.rIn.direction);

		const float3 gNormal = normalize(instance.transform.TransformVector(triangle.Normal()));

		bool wiShadingBackSide = dot(-hitResult.rIn.direction, hitResult.normal) < 0.0f;
		bool wiGeometryBackSide = dot(-hitResult.rIn.direction, hitGNormal) < 0.0f;

		//if (wiGeometryBackSide != wiShadingBackSide)
		//	return make_float3(0.0f);

		D_Ray shadowRay;
		float offsetDirection = wiGeometryBackSide ? -1.0f : 1.0f;
		shadowRay.origin = hitResult.p + offsetDirection * 1.0e-4f * hitGNormal;

		const float3 toLight = p - shadowRay.origin;
		const float distance = length(toLight);
		shadowRay.direction = toLight / distance;
		shadowRay.invDirection = 1.0f / shadowRay.direction;
		shadowRay.hit.t = distance;

		const float3 wo = rotatePoint(qRotationToZ, shadowRay.direction);

		bool anyHit = TLASTraceShadow(scene.tlas, shadowRay);

		if (anyHit)
			return make_float3(0.0f);

		const float cosThetaO = fabs(dot(gNormal, shadowRay.direction));

		const float dSquared = dot(toLight, toLight);

		float lightPdf = 1.0f / (scene.lightCount * scene.tlas.bvhs[instance.bvhIdx].triCount * triangle.Area());
		// Transform pdf over an area to pdf over directions
		lightPdf *= dSquared / cosThetaO;

		if (!Sampler::IsPdfValid(lightPdf))
			return make_float3(0.0f);

		const D_Material& material = scene.materials[instance.materialId];

		float3 throughput;
		float bsdfPdf;
		bool sampleIsValid;

		switch (hitResult.material.type)
		{
		case D_Material::D_Type::PLASTIC:
			sampleIsValid = D_BSDF::Eval<D_PlasticBSDF>(hitResult, wi, wo, throughput, bsdfPdf);
			break;
		case D_Material::D_Type::DIELECTRIC:
			sampleIsValid = D_BSDF::Eval<D_DielectricBSDF>(hitResult, wi, wo, throughput, bsdfPdf);
			break;
		}

		if (!sampleIsValid)
			return make_float3(0.0f);

		//const float weight = 1.0f;
		const float weight = Sampler::PowerHeuristic(lightPdf, bsdfPdf);

		float3 emissive;
		if (material.emissiveMapId != -1)
			emissive = make_float3(tex2D<float4>(scene.emissiveMaps[material.emissiveMapId], uv.x, uv.y));
		else
			emissive = material.emissive;

		return weight * throughput * emissive * material.intensity / lightPdf;
	}
}

// Incoming radiance estimate on ray origin and in ray direction
inline __device__ float3 Radiance(const D_Scene& scene, const D_Ray& r, unsigned int& rngState)
{
	D_Ray currentRay = r;
	float3 currentThroughput = make_float3(1.0f);
	float3 emission = make_float3(0.0f);
	float bsdfPdf = 1e10f;

	for (int j = 0; j < MAX_BOUNCES; j++)
	{
		// Reset the hit position and calculate the inverse of the new direction
		currentRay.hit.t = 1e30f;
		currentRay.invDirection = 1.0f / currentRay.direction;

		TLASTrace(scene.tlas, currentRay);

		// If no intersection, sample background
		if (currentRay.hit.t == 1e30f)
		{ 
			float3 backgroundColor = SampleBackground(scene, currentRay.direction);
			return currentThroughput * backgroundColor + emission;
		}

		D_HitResult hitResult;
		hitResult.p = currentRay.origin + currentRay.direction * currentRay.hit.t;
		hitResult.rIn = currentRay;

		const D_BVHInstance& instance = scene.tlas.blas[currentRay.hit.instanceIdx];
		const D_Triangle& triangle = scene.tlas.bvhs[instance.bvhIdx].triangles[currentRay.hit.triIdx];
		float u = currentRay.hit.u, v = currentRay.hit.v;

		// Interpolating and rotating the normal
		hitResult.normal = u * triangle.normal1 + v * triangle.normal2 + (1 - (u + v)) * triangle.normal0;
		hitResult.normal = normalize(instance.transform.TransformVector(hitResult.normal));

		float3 gNormal = normalize(instance.transform.TransformVector(triangle.Normal()));

		hitResult.material = scene.materials[instance.materialId];

		if (hitResult.material.diffuseMapId == -1)
			hitResult.albedo = hitResult.material.diffuse.albedo;
		else
		{
			float2 uv = u * triangle.texCoord1 + v * triangle.texCoord2 + (1 - (u + v)) * triangle.texCoord0;
			hitResult.material.diffuse.albedo = make_float3(tex2D<float4>(scene.diffuseMaps[hitResult.material.diffuseMapId], uv.x, uv.y));
		}
		if (hitResult.material.emissiveMapId != -1) {
			float2 uv = u * triangle.texCoord1 + v * triangle.texCoord2 + (1 - (u + v)) * triangle.texCoord0;
			hitResult.material.emissive = make_float3(tex2D<float4>(scene.emissiveMaps[hitResult.material.emissiveMapId], uv.x, uv.y));
		}

		// Normal flipping
		//if (dot(hitResult.normal, currentRay.direction) > 0.0f)
		//	hitResult.normal = -hitResult.normal;

		// Invert normals for non transmissive material if the primitive is backfacing the ray
		if (dot(gNormal, currentRay.direction) > 0.0f && (hitResult.material.type != D_Material::D_Type::DIELECTRIC))
		{
			hitResult.normal = -hitResult.normal;
			gNormal = -gNormal;
		}

		bool hitLight = false;
		if (fmaxf(hitResult.material.emissive * hitResult.material.intensity) > 0.0f)
		{
			float weight = 1.0f;

			if (settings.useMIS)
			{
				hitLight = true;
				const float3 toLight = hitResult.p - currentRay.origin;
				const float cosThetaO = fabs(dot(gNormal, currentRay.direction));

				const float dSquared = dot(toLight, toLight);

				float lightPdf = 1.0f / (scene.lightCount * scene.tlas.bvhs[instance.bvhIdx].triCount * triangle.Area());
				// Transform pdf over an area to pdf over directions
				lightPdf *= dSquared / cosThetaO;

				weight = settings.useMIS ? Sampler::PowerHeuristic(bsdfPdf, lightPdf) : 1.0f;
			}
			//weight = 0.0f;

			emission += weight * hitResult.material.emissive * hitResult.material.intensity * currentThroughput;
		}


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
		case D_Material::D_Type::DIFFUSE:
			scattered = D_BSDF::Sample<D_LambertianBSDF>(hitResult, wi, wo, throughput, bsdfPdf, rngState);
			break;
		case D_Material::D_Type::DIELECTRIC:
			scattered = D_BSDF::Sample<D_DielectricBSDF>(hitResult, wi, wo, throughput, bsdfPdf, rngState);
			break;
		case D_Material::D_Type::PLASTIC:
			scattered = D_BSDF::Sample<D_PlasticBSDF>(hitResult, wi, wo, throughput, bsdfPdf, rngState);
			break;
		case D_Material::D_Type::CONDUCTOR:
			scattered = D_BSDF::Sample<D_ConductorBSDF>(hitResult, wi, wo, throughput, bsdfPdf, rngState);
			break;
		default:
			break;
		}

		if (scattered)
		{
			if (!hitLight && settings.useMIS)
				emission += currentThroughput * NextEventEstimation(scene, currentRay, hitResult, gNormal, rngState);

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

__global__ void TraceRay(const D_Scene scene, uint32_t* outBuffer, uint32_t frameNumber, float3* accumulationBuffer)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	uint2 pixel = make_uint2(i, j);

	D_Camera camera = scene.camera;
	uint2 resolution = camera.resolution;

	if (pixel.x >= resolution.x || pixel.y >= resolution.y)
		return;

	unsigned int rngState = Random::InitRNG(pixel, resolution, frameNumber);

	// Normalized jittered coordinates
	float x = (pixel.x + Random::Rand(rngState)) / (float)resolution.x;
	float y = (pixel.y + Random::Rand(rngState)) / (float)resolution.y;

	float2 rd = camera.lensRadius * Random::RandomInUnitDisk(rngState);
	float3 offset = camera.right * rd.x + camera.up * rd.y;

	D_Ray ray(
		camera.position + offset,
		normalize(camera.lowerLeftCorner + x * camera.viewportX + y * camera.viewportY - camera.position - offset)
	);

	float3 c = Radiance(scene, ray, rngState);

	if (frameNumber == 1)
		accumulationBuffer[pixel.y * resolution.x + pixel.x] = c;
	else
		accumulationBuffer[pixel.y * resolution.x + pixel.x] += c;

	c = accumulationBuffer[pixel.y * resolution.x + pixel.x] / frameNumber;

	outBuffer[pixel.y * resolution.x + pixel.x] = ToColorUInt(Utils::LinearToGamma(Tonemap(c)));
}

void RenderViewport(PixelBuffer& pixelBuffer, const D_Scene& scene,
	uint32_t frameNumber, float3* accumulationBuffer)
{
	checkCudaErrors(cudaGraphicsMapResources(1, &pixelBuffer.GetCudaResource()));
	size_t size = 0;
	uint32_t* devicePtr = 0;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&devicePtr, &size, pixelBuffer.GetCudaResource()));

	dim3 blocks(pixelBuffer.GetWidth() / BLOCK_SIZE + 1, pixelBuffer.GetHeight() / BLOCK_SIZE + 1);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	TraceRay<<<blocks, threads>>>(scene, devicePtr, frameNumber, accumulationBuffer);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaGraphicsUnmapResources(1, &pixelBuffer.GetCudaResource(), 0));
}

void SetSettings(const D_Settings& s) 
{
	D_Settings* deviceSettings;
	checkCudaErrors(cudaGetSymbolAddress((void**)&deviceSettings, settings));
	checkCudaErrors(cudaMemcpy(deviceSettings, &s, sizeof(D_Settings), cudaMemcpyHostToDevice));
}
