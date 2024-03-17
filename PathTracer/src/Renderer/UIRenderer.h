#pragma once
#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "../OpenGL_API/Texture.h"
#include "../OpenGL_API/PixelBuffer.h"

class UIRenderer
{
public:
	UIRenderer(GLFWwindow* window, unsigned int viewportTextureHandle, uint32_t viewportWidth, uint32_t viewportHeight);
	~UIRenderer();

	void Render(std::shared_ptr<Texture> texture, std::shared_ptr<PixelBuffer> pixelBuffer);

private:
	unsigned int m_ViewportTextureHandle = 0;
	uint32_t m_ViewportWidth, m_ViewportHeight;
};
