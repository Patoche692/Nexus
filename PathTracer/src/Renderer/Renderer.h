#pragma once

#include <glm.hpp>
#include <cuda_runtime_api.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "../Camera.h"
#include "../OpenGL/PixelBuffer.h"
#include "../OpenGL/Texture.h"
#include "../Scene.h"

class Renderer
{
public:
	Renderer(uint32_t width, uint32_t height, GLFWwindow* window);
	~Renderer();

	void OnResize(std::shared_ptr<Camera> camera, uint32_t width, uint32_t height);
	void Render(Scene& scene, float deltaTime);
	void RenderUI(Scene& scene, float deltaTime);
	void UnpackToTexture();
	void UpdateTimer(float deltaTime);

	std::shared_ptr<PixelBuffer> GetPixelBuffer() { return m_PixelBuffer; };
	std::shared_ptr<Texture> GetTexture() { return m_Texture; };

private:
	uint32_t m_ViewportWidth, m_ViewportHeight;
	std::shared_ptr<PixelBuffer> m_PixelBuffer;
	std::shared_ptr<Texture> m_Texture;
	float m_DisplayFPSTimer;
	float m_DeltaTime = 0.0f;
	float m_AccumulatedTime = 0.0f;
	uint32_t m_NAccumulatedFrame = 0;
	uint32_t m_FrameNumber = 0;
	float3* m_AccumulationBuffer;
};

