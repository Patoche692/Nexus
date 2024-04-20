#pragma once
#include <iostream>

struct Texture
{
	~Texture();
	Texture(uint32_t w, uint32_t h, uint32_t c, unsigned char* d);

	uint32_t width;
	uint32_t height;
	uint32_t channels;

	unsigned char* data;
};