#pragma once
#include "Device/Kernels/CUDAKernel.h"
#include "OpenGL/PixelBuffer.h"
#include "Cuda/PathTracer/PathTracer.cuh"
#include "Device/DeviceVector.h"
#include "Device/Kernels/CUDAGraph.h"


class PathTracer
{
public:
	PathTracer(uint32_t width, uint32_t height);
	~PathTracer();

	void Reset();
	void Render();
	void OnResize(uint32_t width, uint32_t height);

	void UpdateDeviceScene(const Scene& scene);

	uint32_t GetFrameNumber() { return m_FrameNumber; }
	const PixelBuffer& GetPixelBuffer() { return m_PixelBuffer; }
	const CUDAKernel& GetKernel() { return m_Kernel; }

private:
	CUDAKernel m_Kernel;
	CUDAGraph m_RenderGraph;


	uint32_t m_FrameNumber = 0;

	uint32_t m_ViewportWidth = 0, m_ViewportHeight = 0;

	// Device members
	DeviceInstance<float3*> m_AccumulationBuffer;
	DeviceInstance<uint32_t> m_DeviceFrameNumber;
	DeviceInstance<uint32_t*> m_RenderBuffer;

	PixelBuffer m_PixelBuffer;

	DeviceInstance<Scene, D_Scene> m_Scene;
};