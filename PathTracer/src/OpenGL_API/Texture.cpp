#include "Texture.h"
#include <GLFW/glfw3.h>

Texture::Texture(uint32_t width, uint32_t height)
{
    glGenTextures(1, &m_Handle);
    Bind();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
}

void Texture::Bind()
{
	glBindTexture(GL_TEXTURE_2D, m_Handle);
}

Texture::~Texture()
{
    glDeleteTextures(1, &m_Handle);
}
