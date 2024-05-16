#define STB_IMAGE_IMPLEMENTATION
#include "IMGLoader.h"
#include "stb_image.h"

IMGLoader::IMGLoader()
{
}

IMGLoader::~IMGLoader()
{
}

Texture IMGLoader::LoadIMG(std::string filepath)
{
	int width, height, channels;

	float3* pixels = (float3*)stbi_loadf(filepath.c_str(), &width, &height, &channels, 4);

	return Texture(width, height, channels, pixels);
}


