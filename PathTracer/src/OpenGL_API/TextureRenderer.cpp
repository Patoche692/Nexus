#include "TextureRenderer.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

TextureRenderer::TextureRenderer(uint32_t width, uint32_t height)
	:m_Width(width), m_Height(height)
{
	m_PixelBuffer = std::make_shared<PixelBuffer>(width, height);
	m_Texture = std::make_shared<Texture>(width, height);
	m_Framebuffer = std::make_shared<Framebuffer>(width, height, m_Texture->GetHandle());
}

void TextureRenderer::Render()
{
	m_Texture->Bind();
	m_PixelBuffer->Bind();
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_Width, m_Height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
}
