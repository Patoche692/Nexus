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

	void FreeDeviceBuffers();
	void Reset();
	void ResetFrameNumber();
	void Render(const Scene& scene);
	void OnResize(uint32_t width, uint32_t height);

	void UpdateDeviceScene(const Scene& scene);

	uint32_t GetFrameNumber() { return m_FrameNumber; }
	const PixelBuffer& GetPixelBuffer() { return m_PixelBuffer; }

private:
	CUDAKernel m_GenerateKernel;
	CUDAKernel m_LogicKernel;
	CUDAKernel m_TraceKernel;
	CUDAKernel m_TraceShadowKernel;
	CUDAKernel m_DiffuseMaterialKernel;
	CUDAKernel m_PlasticMaterialKernel;
	CUDAKernel m_DielectricMaterialKernel;
	CUDAKernel m_ConductorMaterialKernel;

	CUDAGraph m_RenderGraph;


	uint32_t m_FrameNumber = 0;

	uint32_t m_ViewportWidth = 0, m_ViewportHeight = 0;

	// Device members
	DeviceInstance<float3*> m_AccumulationBuffer;
	DeviceInstance<uint32_t> m_DeviceFrameNumber;
	DeviceInstance<uint32_t*> m_RenderBuffer;

	PixelBuffer m_PixelBuffer;

	DeviceInstance<Scene, D_Scene> m_Scene;

	DeviceInstance<D_PathStateSAO> m_PathState;

	DeviceInstance<D_ShadowRayStateSAO> m_ShadowRayState;

	DeviceInstance<D_MaterialRequestSAO> m_DiffuseMaterialBuffer;
	DeviceInstance<D_MaterialRequestSAO> m_PlasticMaterialBuffer;
	DeviceInstance<D_MaterialRequestSAO> m_DielectricMaterialBuffer;
	DeviceInstance<D_MaterialRequestSAO> m_ConductorMaterialBuffer;
};