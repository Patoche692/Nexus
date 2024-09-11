#include "PathTracer.h"
#include "Device/CudaMemory.h"


PathTracer::PathTracer(uint32_t width, uint32_t height)
	: m_ViewportWidth(width), m_ViewportHeight(height),
	m_PixelBuffer(width, height), m_AccumulationBuffer(GetDeviceAccumulationBufferAddress()),
	m_RenderBuffer(GetDeviceRenderBufferAddress()), m_Scene(GetDeviceSceneAddress()),
	m_DeviceFrameNumber(GetDeviceFrameNumberAddress()), m_PathState(GetDevicePathStateAddress()),
	m_ShadowRayState(GetDeviceShadowRayStateAddress()), m_DiffuseMaterialBuffer(GetDeviceDiffuseRequestAddress()),
	m_PlasticMaterialBuffer(GetDevicePlasticRequestAddress()), m_DielectricMaterialBuffer(GetDeviceDielectricRequestAddress()),
	m_ConductorMaterialBuffer(GetDeviceConductorRequestAddress())
{
	m_AccumulationBuffer = CudaMemory::Allocate<float3>(width * height);

	Reset();
}

PathTracer::~PathTracer()
{
	FreeDeviceBuffers();
	CudaMemory::Free(m_AccumulationBuffer.Instance());
}

void PathTracer::FreeDeviceBuffers()
{
	CudaMemory::FreeAsync(m_PathState->intersection.hitDistance);
	CudaMemory::FreeAsync(m_PathState->intersection.instanceIdx);
	CudaMemory::FreeAsync(m_PathState->intersection.triIdx);
	CudaMemory::FreeAsync(m_PathState->intersection.u);
	CudaMemory::FreeAsync(m_PathState->intersection.v);
	CudaMemory::FreeAsync(m_PathState->lastPdf);
	CudaMemory::FreeAsync(m_PathState->pixelIdx);
	CudaMemory::FreeAsync(m_PathState->radiance);
	CudaMemory::FreeAsync(m_PathState->ray.origin);
	CudaMemory::FreeAsync(m_PathState->ray.direction);
	CudaMemory::FreeAsync(m_PathState->throughput);

	CudaMemory::FreeAsync(m_ShadowRayState->hitDistance);
	CudaMemory::FreeAsync(m_ShadowRayState->pixelIdx);
	CudaMemory::FreeAsync(m_ShadowRayState->radiance);
	CudaMemory::FreeAsync(m_ShadowRayState->ray.origin);
	CudaMemory::FreeAsync(m_ShadowRayState->ray.direction);

	CudaMemory::FreeAsync(m_DiffuseMaterialBuffer->intersection.hitDistance);
	CudaMemory::FreeAsync(m_DiffuseMaterialBuffer->intersection.instanceIdx);
	CudaMemory::FreeAsync(m_DiffuseMaterialBuffer->intersection.triIdx);
	CudaMemory::FreeAsync(m_DiffuseMaterialBuffer->intersection.u);
	CudaMemory::FreeAsync(m_DiffuseMaterialBuffer->intersection.v);
	CudaMemory::FreeAsync(m_DiffuseMaterialBuffer->rayDirection);

	CudaMemory::FreeAsync(m_PlasticMaterialBuffer->intersection.hitDistance);
	CudaMemory::FreeAsync(m_PlasticMaterialBuffer->intersection.instanceIdx);
	CudaMemory::FreeAsync(m_PlasticMaterialBuffer->intersection.triIdx);
	CudaMemory::FreeAsync(m_PlasticMaterialBuffer->intersection.u);
	CudaMemory::FreeAsync(m_PlasticMaterialBuffer->intersection.v);
	CudaMemory::FreeAsync(m_PlasticMaterialBuffer->rayDirection);

	CudaMemory::FreeAsync(m_DielectricMaterialBuffer->intersection.hitDistance);
	CudaMemory::FreeAsync(m_DielectricMaterialBuffer->intersection.instanceIdx);
	CudaMemory::FreeAsync(m_DielectricMaterialBuffer->intersection.triIdx);
	CudaMemory::FreeAsync(m_DielectricMaterialBuffer->intersection.u);
	CudaMemory::FreeAsync(m_DielectricMaterialBuffer->intersection.v);
	CudaMemory::FreeAsync(m_DielectricMaterialBuffer->rayDirection);

	CudaMemory::FreeAsync(m_ConductorMaterialBuffer->intersection.hitDistance);
	CudaMemory::FreeAsync(m_ConductorMaterialBuffer->intersection.instanceIdx);
	CudaMemory::FreeAsync(m_ConductorMaterialBuffer->intersection.triIdx);
	CudaMemory::FreeAsync(m_ConductorMaterialBuffer->intersection.u);
	CudaMemory::FreeAsync(m_ConductorMaterialBuffer->intersection.v);
	CudaMemory::FreeAsync(m_ConductorMaterialBuffer->rayDirection);
}

void PathTracer::Reset()
{
	dim3 gridSize(m_ViewportWidth / BLOCK_SIZE + 1, m_ViewportHeight / BLOCK_SIZE + 1);
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

	m_GenerateKernel = CUDAKernel((void*)GenerateKernel, gridSize, blockSize);
	m_LogicKernel = CUDAKernel((void*)LogicKernel, gridSize, blockSize);
	m_DiffuseMaterialKernel = CUDAKernel((void*)DiffuseMaterialKernel, gridSize, blockSize);
	m_PlasticMaterialKernel = CUDAKernel((void*)PlasticMaterialKernel, gridSize, blockSize);
	m_DielectricMaterialKernel = CUDAKernel((void*)DielectricMaterialKernel, gridSize, blockSize);
	m_ConductorMaterialKernel = CUDAKernel((void*)ConductorMaterialKernel, gridSize, blockSize);
	m_TraceKernel = CUDAKernel((void*)TraceKernel, gridSize, blockSize);
	m_TraceShadowKernel = CUDAKernel((void*)TraceShadowKernel, gridSize, blockSize);

	cudaGraphNode_t logicNode = m_RenderGraph.AddKernelNode(m_LogicKernel);
	cudaGraphNode_t diffuseNode = m_RenderGraph.AddKernelNode(m_DiffuseMaterialKernel, &logicNode, 1);
	cudaGraphNode_t plasticNode = m_RenderGraph.AddKernelNode(m_PlasticMaterialKernel, &logicNode, 1);
	cudaGraphNode_t dielectricNode = m_RenderGraph.AddKernelNode(m_DielectricMaterialKernel, &logicNode, 1);
	cudaGraphNode_t conductorNode = m_RenderGraph.AddKernelNode(m_ConductorMaterialKernel, &logicNode, 1);
	cudaGraphNode_t materialNodes[4] = { diffuseNode, plasticNode, dielectricNode, conductorNode };

	m_RenderGraph.AddKernelNode(m_TraceKernel, materialNodes, 4);
	m_RenderGraph.AddKernelNode(m_TraceShadowKernel, materialNodes, 4);
	m_RenderGraph.BuildGraph();

	const uint32_t count = m_ViewportWidth * m_ViewportHeight;

	D_IntersectionSAO intersectionSAO;
	intersectionSAO.hitDistance = CudaMemory::AllocateAsync<float>(count);
	intersectionSAO.instanceIdx = CudaMemory::AllocateAsync<uint32_t>(count);
	intersectionSAO.triIdx = CudaMemory::AllocateAsync<uint32_t>(count);
	intersectionSAO.u = CudaMemory::AllocateAsync<float>(count);
	intersectionSAO.v = CudaMemory::AllocateAsync<float>(count);

	D_RaySAO raySAO;
	raySAO.origin = CudaMemory::AllocateAsync<float3>(count);
	raySAO.direction = CudaMemory::AllocateAsync<float3>(count);

	float* lastPdf = CudaMemory::AllocateAsync<float>(count);
	uint32_t* pixelIdx = CudaMemory::AllocateAsync<uint32_t>(count);
	float3* radiance = CudaMemory::AllocateAsync<float3>(count);
	float3* throughput = CudaMemory::AllocateAsync<float3>(count);

	D_PathStateSAO pathState;
	pathState.intersection = intersectionSAO;
	pathState.ray = raySAO;
	pathState.lastPdf = lastPdf;
	pathState.pixelIdx = pixelIdx;
	pathState.radiance = radiance;
	pathState.throughput = throughput;
	pathState.size = 0;

	m_PathState = pathState;

	float* hitDistance = CudaMemory::AllocateAsync<float>(count);
	pixelIdx = CudaMemory::AllocateAsync<uint32_t>(count);
	radiance = CudaMemory::AllocateAsync<float3>(count);
	raySAO.origin = CudaMemory::AllocateAsync<float3>(count);
	raySAO.direction = CudaMemory::AllocateAsync<float3>(count);

	D_ShadowRayStateSAO shadowRayState;
	shadowRayState.hitDistance = hitDistance;
	shadowRayState.pixelIdx = pixelIdx;
	shadowRayState.radiance = radiance;
	shadowRayState.ray = raySAO;
	shadowRayState.size = 0;

	m_ShadowRayState = shadowRayState;

	intersectionSAO.hitDistance = CudaMemory::AllocateAsync<float>(count);
	intersectionSAO.instanceIdx = CudaMemory::AllocateAsync<uint32_t>(count);
	intersectionSAO.triIdx = CudaMemory::AllocateAsync<uint32_t>(count);
	intersectionSAO.u = CudaMemory::AllocateAsync<float>(count);
	intersectionSAO.v = CudaMemory::AllocateAsync<float>(count);
	float3* rayDirection = CudaMemory::AllocateAsync<float3>(count);

	D_MaterialRequestSAO materialRequest;
	materialRequest.intersection = intersectionSAO;
	materialRequest.rayDirection = rayDirection;
	materialRequest.size = 0;

	m_DiffuseMaterialBuffer = materialRequest;

	intersectionSAO.hitDistance = CudaMemory::AllocateAsync<float>(count);
	intersectionSAO.instanceIdx = CudaMemory::AllocateAsync<uint32_t>(count);
	intersectionSAO.triIdx = CudaMemory::AllocateAsync<uint32_t>(count);
	intersectionSAO.u = CudaMemory::AllocateAsync<float>(count);
	intersectionSAO.v = CudaMemory::AllocateAsync<float>(count);
	rayDirection = CudaMemory::AllocateAsync<float3>(count);

	materialRequest.intersection = intersectionSAO;
	materialRequest.rayDirection = rayDirection;
	materialRequest.size = 0;

	m_PlasticMaterialBuffer = materialRequest;

	intersectionSAO.hitDistance = CudaMemory::AllocateAsync<float>(count);
	intersectionSAO.instanceIdx = CudaMemory::AllocateAsync<uint32_t>(count);
	intersectionSAO.triIdx = CudaMemory::AllocateAsync<uint32_t>(count);
	intersectionSAO.u = CudaMemory::AllocateAsync<float>(count);
	intersectionSAO.v = CudaMemory::AllocateAsync<float>(count);
	rayDirection = CudaMemory::AllocateAsync<float3>(count);

	materialRequest.intersection = intersectionSAO;
	materialRequest.rayDirection = rayDirection;
	materialRequest.size = 0;

	m_DielectricMaterialBuffer = materialRequest;

	intersectionSAO.hitDistance = CudaMemory::AllocateAsync<float>(count);
	intersectionSAO.instanceIdx = CudaMemory::AllocateAsync<uint32_t>(count);
	intersectionSAO.triIdx = CudaMemory::AllocateAsync<uint32_t>(count);
	intersectionSAO.u = CudaMemory::AllocateAsync<float>(count);
	intersectionSAO.v = CudaMemory::AllocateAsync<float>(count);
	rayDirection = CudaMemory::AllocateAsync<float3>(count);

	materialRequest.intersection = intersectionSAO;
	materialRequest.rayDirection = rayDirection;
	materialRequest.size = 0;

	m_ConductorMaterialBuffer = materialRequest;
}

void PathTracer::ResetFrameNumber()
{
	m_FrameNumber = 0;
}

void PathTracer::Render(const Scene& scene)
{
	m_FrameNumber++;

	m_DeviceFrameNumber = m_FrameNumber;

	CheckCudaErrors(cudaGraphicsMapResources(1, &m_PixelBuffer.GetCudaResource()));
	size_t size = 0;
	uint32_t* devicePtr = 0;
	CheckCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&devicePtr, &size, m_PixelBuffer.GetCudaResource()));
	m_RenderBuffer = devicePtr;

	CheckCudaErrors(cudaDeviceSynchronize());

	m_GenerateKernel.Launch();

	CheckCudaErrors(cudaDeviceSynchronize());

	m_TraceKernel.Launch();

	CheckCudaErrors(cudaDeviceSynchronize());

	for (uint32_t depth = 0; depth < scene.GetRenderSettings().pathLength; depth++)
	{
	  m_RenderGraph.Execute();
	}

	CheckCudaErrors(cudaGetLastError());
	CheckCudaErrors(cudaGraphicsUnmapResources(1, &m_PixelBuffer.GetCudaResource(), 0));
}

void PathTracer::OnResize(uint32_t width, uint32_t height)
{
	if ((m_ViewportWidth != width || m_ViewportHeight != height) && width != 0 && height != 0)
	{
		m_FrameNumber = 0;
		m_PixelBuffer.OnResize(width, height);

		CudaMemory::Free(m_AccumulationBuffer.Instance());
		m_AccumulationBuffer = CudaMemory::Allocate<float3>(width * height);

		m_ViewportWidth = width;
		m_ViewportHeight = height;

		FreeDeviceBuffers();
		Reset();
		m_RenderGraph.Reset();
	}
}

void PathTracer::UpdateDeviceScene(const Scene& scene)
{
	m_Scene = scene;
}

