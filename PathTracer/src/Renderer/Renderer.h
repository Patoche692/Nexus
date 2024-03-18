#pragma once

#include <glm.hpp>
#include <cuda_runtime_api.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "../Camera.h"
#include "../OpenGL_API/PixelBuffer.h"
#include "../OpenGL_API/Texture.h"

class Renderer
{
public:
	Renderer(uint32_t width, uint32_t height, GLFWwindow* window);
	~Renderer();

	void OnResize(Camera& camera, uint32_t width, uint32_t height);
	void Render(Camera& camera, float deltaTime);
	void RenderUI(Camera& camera, float deltaTime);
	void UnpackToTexture();

	std::shared_ptr<PixelBuffer> GetPixelBuffer() { return m_PixelBuffer; };
	std::shared_ptr<Texture> GetTexture() { return m_Texture; };

private:
	uint32_t m_ViewportWidth, m_ViewportHeight;
	std::shared_ptr<PixelBuffer> m_PixelBuffer;
	std::shared_ptr<Texture> m_Texture;
};

