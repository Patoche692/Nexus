#pragma once
#include <iostream>

class Texture
{
public:
	Texture(uint32_t width, uint32_t height);
	~Texture();

	void Bind();
	unsigned int GetHandle() { return m_Handle; };
	uint32_t GetWidth() { return m_Width; };
	uint32_t GetHeight() { return m_Height; };

private:
	unsigned int m_Handle = 0;
	uint32_t m_Width, m_Height;
};