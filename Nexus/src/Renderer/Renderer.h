#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime_api.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "Scene/Camera.h"
#include "OpenGL/PixelBuffer.h"
#include "OpenGL/OGLTexture.h"
#include "Scene/Scene.h"
#include "Panels/SceneHierarchyPanel.h"
#include "Panels/MetricsPanel.h"
#include "Cuda/PathTracer/PathTracer.cuh"
#include "Device/Kernels/CUDAGraph.h"

class Renderer
{
public:
	Renderer(uint32_t width, uint32_t height, GLFWwindow* window, Scene* scene);
	~Renderer();

	void Reset();
	void OnResize(uint32_t width, uint32_t height);
	void SaveScreenshot();
	void Render(Scene& scene, float deltaTime);
	void RenderUI(Scene& scene);
	void UnpackToTexture();

	PixelBuffer GetPixelBuffer() { return m_PixelBuffer; }
	OGLTexture GetTexture() { return m_Texture; }

private:
	uint32_t m_ViewportWidth, m_ViewportHeight;
	PixelBuffer m_PixelBuffer;
	OGLTexture m_Texture;
	uint32_t m_FrameNumber = 0;
	float3* m_AccumulationBuffer;
	Scene* m_Scene;
	SceneHierarchyPanel m_HierarchyPannel;
	MetricsPanel m_MetricsPanel;

	//CUDAGraph m_RenderGraph;
};

