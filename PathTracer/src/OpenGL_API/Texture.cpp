#include "Texture.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>

Texture::Texture(uint32_t width, uint32_t height)
    :m_Width(width), m_Height(height)
{
	glGenTextures(1, &m_Handle);
    Bind();
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_Width, m_Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
}

void Texture::Bind()
{
	glBindTexture(GL_TEXTURE_2D, m_Handle);
}

void Texture::OnResize(uint32_t width, uint32_t height)
{
	Bind();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_Width, m_Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	m_Width = width;
	m_Height = height;
	Bind();
	//glDeleteTextures(1, &m_Handle);
	//glGenTextures(1, &m_Handle);
 //   Bind();
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_Width, m_Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
}

Texture::~Texture()
{
    glDeleteTextures(1, &m_Handle);
}
