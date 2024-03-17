#pragma once
#include <iostream>

class Texture
{
public:
	Texture(uint32_t width, uint32_t height);
	~Texture();

	void Bind();
	unsigned int GetHandle() { return m_Handle; };

private:
	unsigned int m_Handle = 0;
};