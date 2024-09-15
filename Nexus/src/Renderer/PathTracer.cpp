#include "PathTracer.h"
#include "Device/CudaMemory.h"


PathTracer::PathTracer(uint32_t width, uint32_t height)
	: m_ViewportWidth(width),
	m_ViewportHeight(height),
	m_PixelBuffer(width, height),
	m_AccumulationBuffer(GetDeviceAccumulationBufferAddress()),
	m_RenderBuffer(GetDeviceRenderBufferAddress()),
	m_Scene(GetDeviceSceneAddress()),
	m_DeviceFrameNumber(GetDeviceFrameNumberAddress()),
	m_DeviceBounce(GetDeviceBounceAddress()),
	m_PathState(GetDevicePathStateAddress()),
	m_ShadowTraceRequest(GetDeviceShadowTraceRequestAddress()),
	m_TraceRequest(GetDeviceTraceRequestAddress()),
	m_DiffuseMaterialRequest(GetDeviceDiffuseRequestAddress()),
	m_PlasticMaterialRequest(GetDevicePlasticRequestAddress()),
	m_DielectricMaterialRequest(GetDeviceDielectricRequestAddress()),
	m_ConductorMaterialRequest(GetDeviceConductorRequestAddress()),
	m_QueueSize(GetDeviceQueueSizeAddress())
{
	m_AccumulationBuffer = CudaMemory::Allocate<float3>(width * height);

	Reset();
}

PathTracer::~PathTracer()
{
	FreeDeviceBuffers();
	CheckCudaErrors(cudaDeviceSynchronize());
}

void PathTracer::FreeDeviceBuffers()
{
	CudaMemory::FreeAsync(m_AccumulationBuffer.Instance());
	CudaMemory::FreeAsync(m_PathState->lastPdf);
	CudaMemory::FreeAsync(m_PathState->throughput);
	CudaMemory::FreeAsync(m_PathState->radiance);

	CudaMemory::FreeAsync(m_TraceRequest->intersection.hitDistance);
	CudaMemory::FreeAsync(m_TraceRequest->intersection.instanceIdx);
	CudaMemory::FreeAsync(m_TraceRequest->intersection.triIdx);
	CudaMemory::FreeAsync(m_TraceRequest->intersection.u);
	CudaMemory::FreeAsync(m_TraceRequest->intersection.v);
	CudaMemory::FreeAsync(m_TraceRequest->pixelIdx);
	CudaMemory::FreeAsync(m_TraceRequest->ray.origin);
	CudaMemory::FreeAsync(m_TraceRequest->ray.direction);

	CudaMemory::FreeAsync(m_ShadowTraceRequest->hitDistance);
	CudaMemory::FreeAsync(m_ShadowTraceRequest->pixelIdx);
	CudaMemory::FreeAsync(m_ShadowTraceRequest->radiance);
	CudaMemory::FreeAsync(m_ShadowTraceRequest->ray.origin);
	CudaMemory::FreeAsync(m_ShadowTraceRequest->ray.direction);

	CudaMemory::FreeAsync(m_DiffuseMaterialRequest->intersection.hitDistance);
	CudaMemory::FreeAsync(m_DiffuseMaterialRequest->intersection.instanceIdx);
	CudaMemory::FreeAsync(m_DiffuseMaterialRequest->intersection.triIdx);
	CudaMemory::FreeAsync(m_DiffuseMaterialRequest->intersection.u);
	CudaMemory::FreeAsync(m_DiffuseMaterialRequest->intersection.v);
	CudaMemory::FreeAsync(m_DiffuseMaterialRequest->rayDirection);
	CudaMemory::FreeAsync(m_DiffuseMaterialRequest->pixelIdx);

	CudaMemory::FreeAsync(m_PlasticMaterialRequest->intersection.hitDistance);
	CudaMemory::FreeAsync(m_PlasticMaterialRequest->intersection.instanceIdx);
	CudaMemory::FreeAsync(m_PlasticMaterialRequest->intersection.triIdx);
	CudaMemory::FreeAsync(m_PlasticMaterialRequest->intersection.u);
	CudaMemory::FreeAsync(m_PlasticMaterialRequest->intersection.v);
	CudaMemory::FreeAsync(m_PlasticMaterialRequest->rayDirection);
	CudaMemory::FreeAsync(m_PlasticMaterialRequest->pixelIdx);

	CudaMemory::FreeAsync(m_DielectricMaterialRequest->intersection.hitDistance);
	CudaMemory::FreeAsync(m_DielectricMaterialRequest->intersection.instanceIdx);
	CudaMemory::FreeAsync(m_DielectricMaterialRequest->intersection.triIdx);
	CudaMemory::FreeAsync(m_DielectricMaterialRequest->intersection.u);
	CudaMemory::FreeAsync(m_DielectricMaterialRequest->intersection.v);
	CudaMemory::FreeAsync(m_DielectricMaterialRequest->rayDirection);
	CudaMemory::FreeAsync(m_DielectricMaterialRequest->pixelIdx);

	CudaMemory::FreeAsync(m_ConductorMaterialRequest->intersection.hitDistance);
	CudaMemory::FreeAsync(m_ConductorMaterialRequest->intersection.instanceIdx);
	CudaMemory::FreeAsync(m_ConductorMaterialRequest->intersection.triIdx);
	CudaMemory::FreeAsync(m_ConductorMaterialRequest->intersection.u);
	CudaMemory::FreeAsync(m_ConductorMaterialRequest->intersection.v);
	CudaMemory::FreeAsync(m_ConductorMaterialRequest->rayDirection);
	CudaMemory::FreeAsync(m_ConductorMaterialRequest->pixelIdx);
}

void PathTracer::Reset()
{
	m_AccumulationBuffer = CudaMemory::Allocate<float3>(m_ViewportWidth * m_ViewportHeight);

	dim3 gridSize(m_ViewportWidth * m_ViewportHeight / BLOCK_SIZE + 1, 1, 1);
	dim3 blockSize(BLOCK_SIZE, 1, 1);

	m_GenerateKernel = CUDAKernel((void*)GenerateKernel, gridSize, blockSize);
	m_LogicKernel = CUDAKernel((void*)LogicKernel, gridSize, blockSize);
	m_DiffuseMaterialKernel = CUDAKernel((void*)DiffuseMaterialKernel, gridSize, blockSize);
	m_PlasticMaterialKernel = CUDAKernel((void*)PlasticMaterialKernel, gridSize, blockSize);
	m_DielectricMaterialKernel = CUDAKernel((void*)DielectricMaterialKernel, gridSize, blockSize);
	m_ConductorMaterialKernel = CUDAKernel((void*)ConductorMaterialKernel, gridSize, blockSize);
	m_TraceKernel = CUDAKernel((void*)TraceKernel);
	m_TraceShadowKernel = CUDAKernel((void*)TraceShadowKernel);
	m_AccumulateKernel = CUDAKernel((void*)AccumulateKernel, gridSize, blockSize);

	// Set minimal launch configuration for trace kernels.
	// Inactive threads will fetch new rays in the trace queue.
	m_TraceKernel.SetMinimalLaunchConfigurationWithBlockSize(BLOCK_SIZE);
	m_TraceShadowKernel.SetMinimalLaunchConfigurationWithBlockSize(BLOCK_SIZE);

	m_RenderGraph.Reset();
	cudaGraphNode_t logicNode = m_RenderGraph.AddKernelNode(m_LogicKernel);
	cudaGraphNode_t diffuseNode = m_RenderGraph.AddKernelNode(m_DiffuseMaterialKernel, &logicNode, 1);
	cudaGraphNode_t plasticNode = m_RenderGraph.AddKernelNode(m_PlasticMaterialKernel, &logicNode, 1);
	cudaGraphNode_t dielectricNode = m_RenderGraph.AddKernelNode(m_DielectricMaterialKernel, &logicNode, 1);
	cudaGraphNode_t conductorNode = m_RenderGraph.AddKernelNode(m_ConductorMaterialKernel, &logicNode, 1);
	cudaGraphNode_t materialNodes[4] = { diffuseNode, plasticNode, dielectricNode, conductorNode };
	cudaGraphNode_t traceNode = m_RenderGraph.AddKernelNode(m_TraceKernel, materialNodes, 4);
	cudaGraphNode_t traceShadowNode = m_RenderGraph.AddKernelNode(m_TraceShadowKernel, materialNodes, 4);

	m_RenderGraph.BuildGraph();

	const uint32_t count = m_ViewportWidth * m_ViewportHeight;

	float* lastPdf = CudaMemory::AllocateAsync<float>(count);
	float3* throughput = CudaMemory::AllocateAsync<float3>(count);
	float3* radiance = CudaMemory::AllocateAsync<float3>(count);

	D_PathStateSAO pathState;
	pathState.lastPdf = lastPdf;
	pathState.throughput = throughput;
	pathState.radiance = radiance;

	m_PathState = pathState;

	D_IntersectionSAO intersectionSAO;
	intersectionSAO.hitDistance = CudaMemory::AllocateAsync<float>(count);
	intersectionSAO.instanceIdx = CudaMemory::AllocateAsync<uint32_t>(count);
	intersectionSAO.triIdx = CudaMemory::AllocateAsync<uint32_t>(count);
	intersectionSAO.u = CudaMemory::AllocateAsync<float>(count);
	intersectionSAO.v = CudaMemory::AllocateAsync<float>(count);

	D_RaySAO raySAO;
	raySAO.origin = CudaMemory::AllocateAsync<float3>(count);
	raySAO.direction = CudaMemory::AllocateAsync<float3>(count);

	uint32_t* pixelIdx = CudaMemory::AllocateAsync<uint32_t>(count);

	D_TraceRequestSAO traceRequest;

	traceRequest.intersection = intersectionSAO;
	traceRequest.ray = raySAO;
	traceRequest.pixelIdx = pixelIdx;

	m_TraceRequest = traceRequest;

	float* hitDistance = CudaMemory::AllocateAsync<float>(count);
	pixelIdx = CudaMemory::AllocateAsync<uint32_t>(count);
	radiance = CudaMemory::AllocateAsync<float3>(count);
	raySAO.origin = CudaMemory::AllocateAsync<float3>(count);
	raySAO.direction = CudaMemory::AllocateAsync<float3>(count);

	D_ShadowTraceRequestSAO shadowTraceRequest;
	shadowTraceRequest.hitDistance = hitDistance;
	shadowTraceRequest.pixelIdx = pixelIdx;
	shadowTraceRequest.radiance = radiance;
	shadowTraceRequest.ray = raySAO;

	m_ShadowTraceRequest = shadowTraceRequest;

	intersectionSAO.hitDistance = CudaMemory::AllocateAsync<float>(count);
	intersectionSAO.instanceIdx = CudaMemory::AllocateAsync<uint32_t>(count);
	intersectionSAO.triIdx = CudaMemory::AllocateAsync<uint32_t>(count);
	intersectionSAO.u = CudaMemory::AllocateAsync<float>(count);
	intersectionSAO.v = CudaMemory::AllocateAsync<float>(count);
	float3* rayDirection = CudaMemory::AllocateAsync<float3>(count);
	pixelIdx = CudaMemory::AllocateAsync<uint32_t>(count);

	D_MaterialRequestSAO materialRequest;
	materialRequest.intersection = intersectionSAO;
	materialRequest.rayDirection = rayDirection;
	materialRequest.pixelIdx = pixelIdx;

	m_DiffuseMaterialRequest = materialRequest;

	intersectionSAO.hitDistance = CudaMemory::AllocateAsync<float>(count);
	intersectionSAO.instanceIdx = CudaMemory::AllocateAsync<uint32_t>(count);
	intersectionSAO.triIdx = CudaMemory::AllocateAsync<uint32_t>(count);
	intersectionSAO.u = CudaMemory::AllocateAsync<float>(count);
	intersectionSAO.v = CudaMemory::AllocateAsync<float>(count);
	rayDirection = CudaMemory::AllocateAsync<float3>(count);
	pixelIdx = CudaMemory::AllocateAsync<uint32_t>(count);

	materialRequest.intersection = intersectionSAO;
	materialRequest.rayDirection = rayDirection;
	materialRequest.pixelIdx = pixelIdx;

	m_PlasticMaterialRequest = materialRequest;

	intersectionSAO.hitDistance = CudaMemory::AllocateAsync<float>(count);
	intersectionSAO.instanceIdx = CudaMemory::AllocateAsync<uint32_t>(count);
	intersectionSAO.triIdx = CudaMemory::AllocateAsync<uint32_t>(count);
	intersectionSAO.u = CudaMemory::AllocateAsync<float>(count);
	intersectionSAO.v = CudaMemory::AllocateAsync<float>(count);
	rayDirection = CudaMemory::AllocateAsync<float3>(count);
	pixelIdx = CudaMemory::AllocateAsync<uint32_t>(count);

	materialRequest.intersection = intersectionSAO;
	materialRequest.rayDirection = rayDirection;
	materialRequest.pixelIdx = pixelIdx;

	m_DielectricMaterialRequest = materialRequest;

	intersectionSAO.hitDistance = CudaMemory::AllocateAsync<float>(count);
	intersectionSAO.instanceIdx = CudaMemory::AllocateAsync<uint32_t>(count);
	intersectionSAO.triIdx = CudaMemory::AllocateAsync<uint32_t>(count);
	intersectionSAO.u = CudaMemory::AllocateAsync<float>(count);
	intersectionSAO.v = CudaMemory::AllocateAsync<float>(count);
	rayDirection = CudaMemory::AllocateAsync<float3>(count);
	pixelIdx = CudaMemory::AllocateAsync<uint32_t>(count);

	materialRequest.intersection = intersectionSAO;
	materialRequest.rayDirection = rayDirection;
	materialRequest.pixelIdx = pixelIdx;

	m_ConductorMaterialRequest = materialRequest;
}

void PathTracer::ResetFrameNumber()
{
	m_FrameNumber = 0;
}

void PathTracer::Render(const Scene& scene)
{
	m_FrameNumber++;

	CheckCudaErrors(cudaGraphicsMapResources(1, &m_PixelBuffer.GetCudaResource()));
	size_t size = 0;
	uint32_t* devicePtr = 0;
	CheckCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&devicePtr, &size, m_PixelBuffer.GetCudaResource()));

	m_RenderBuffer = devicePtr;

	m_DeviceFrameNumber = m_FrameNumber;
	m_DeviceBounce = 0;

	// Reset queue sizes
	CudaMemory::MemsetAsync(m_QueueSize.Data(), 0, sizeof(D_QueueSize));

	// Primary rays
	m_GenerateKernel.Launch();
	m_TraceKernel.Launch();

	m_DeviceBounce = 1;

	// Secondary rays
	for (uint32_t i = 0; i < scene.GetRenderSettings().pathLength; i++)
	{
		m_RenderGraph.Execute();
		m_DeviceBounce = i + 2;
	}

	m_AccumulateKernel.Launch();

	CheckCudaErrors(cudaGetLastError());
	CheckCudaErrors(cudaGraphicsUnmapResources(1, &m_PixelBuffer.GetCudaResource(), 0));
}

void PathTracer::OnResize(uint32_t width, uint32_t height)
{
	if ((m_ViewportWidth != width || m_ViewportHeight != height) && width != 0 && height != 0)
	{
		m_FrameNumber = 0;
		m_PixelBuffer.OnResize(width, height);

		m_ViewportWidth = width;
		m_ViewportHeight = height;

		FreeDeviceBuffers();
		Reset();
	}
}

void PathTracer::UpdateDeviceScene(const Scene& scene)
{
	m_Scene = scene;
}

