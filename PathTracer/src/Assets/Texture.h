#pragma once
#include <iostream>

class Texture
{
public:
	~Texture();
	Texture(uint32_t w, uint32_t h, unsigned char* d);

private:
	uint32_t width;
	uint32_t height;

	unsigned char* data;


};