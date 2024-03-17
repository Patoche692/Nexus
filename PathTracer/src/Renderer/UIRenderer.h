#pragma once
#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "../OpenGL_API/Texture.h"

class UIRenderer
{
public:
	UIRenderer(GLFWwindow* window, unsigned int viewportTextureHandle, uint32_t viewportTextureWidth, uint32_t viewportTextureHeight);
	~UIRenderer();

	void Render(std::shared_ptr<Texture> texture);

private:
	unsigned int m_ViewportTextureHandle = 0;
	uint32_t m_ViewportTextureWidth, m_ViewportTextureHeight;
};
