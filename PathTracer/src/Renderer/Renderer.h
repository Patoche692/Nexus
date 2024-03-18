#pragma once

#include <glm.hpp>
#include <cuda_runtime_api.h>
#include "../OpenGL_API/TextureRenderer.h"
#include <GLFW/glfw3.h>

class Renderer
{
public:
	Renderer(uint32_t width, uint32_t height, GLFWwindow* window);
	~Renderer();

	void OnResize(uint32_t width, uint32_t height);
	void Render(float deltaTime);
	void RenderUI(float deltaTime);

private:
	uint32_t m_ViewportWidth, m_ViewportHeight;
	std::shared_ptr<TextureRenderer> m_TextureRenderer;
};

