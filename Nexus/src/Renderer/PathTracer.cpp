#include "PathTracer.h"
#include "Device/CudaMemory.h"


PathTracer::PathTracer(uint32_t width, uint32_t height)
	: m_ViewportWidth(width), m_ViewportHeight(height),
	m_PixelBuffer(width, height), m_AccumulationBuffer(GetDeviceAccumulationBufferAddress()),
	m_RenderBuffer(GetDeviceRenderBufferAddress()), m_Scene(GetDeviceSceneAddress()),
	m_DeviceFrameNumber(GetDeviceFrameNumberAddress())
{
	m_AccumulationBuffer = CudaMemory::Allocate<float3>(width * height);

	dim3 gridSize(width / BLOCK_SIZE + 1, height / BLOCK_SIZE + 1);
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

	m_Kernel = CUDAKernel((void*)TraceRay, gridSize, blockSize);
	m_RenderGraph.AddKernelNode(m_Kernel);
	m_RenderGraph.BuildGraph();

	Reset();
}

PathTracer::~PathTracer()
{
	CudaMemory::Free(m_AccumulationBuffer.Instance());
}

void PathTracer::Reset()
{
	m_FrameNumber = 0;
}

void PathTracer::Render()
{
	m_FrameNumber++;

	m_DeviceFrameNumber = m_FrameNumber;

	checkCudaErrors(cudaGraphicsMapResources(1, &m_PixelBuffer.GetCudaResource()));
	size_t size = 0;
	uint32_t* devicePtr = 0;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&devicePtr, &size, m_PixelBuffer.GetCudaResource()));
	m_RenderBuffer = devicePtr;

	m_RenderGraph.Execute();

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaGraphicsUnmapResources(1, &m_PixelBuffer.GetCudaResource(), 0));
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

		dim3 gridSize(width / BLOCK_SIZE + 1, height / BLOCK_SIZE + 1);
		dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
		m_Kernel = CUDAKernel((void*)TraceRay, gridSize, blockSize);
		m_RenderGraph.Reset();
		m_RenderGraph.AddKernelNode(m_Kernel);
		m_RenderGraph.BuildGraph();
	}
}

void PathTracer::UpdateDeviceScene(const Scene& scene)
{
	m_Scene = scene;
}

