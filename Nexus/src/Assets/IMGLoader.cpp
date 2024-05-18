#define STB_IMAGE_IMPLEMENTATION
#include "IMGLoader.h"
#include "stb_image.h"

IMGLoader::IMGLoader()
{
}

IMGLoader::~IMGLoader()
{
}

Texture IMGLoader::LoadIMG(const std::string& filepath)
{
	int width, height, channels;

	float3* pixels = (float3*)stbi_loadf(filepath.c_str(), &width, &height, &channels, 4);

	if (pixels == NULL)
		std::cout << "IMGLoader: Failed to load texture " << filepath << std::endl;

	return Texture(width, height, channels, pixels);
}

Texture IMGLoader::LoadIMG(const aiTexture* texture)
{
	int width, height, channels;
	float3* pixels = (float3*)stbi_loadf_from_memory((const stbi_uc*)texture->pcData, texture->mWidth, &width, &height, &channels, 4);

	if (pixels == NULL)
		std::cout << "IMGLoader: Failed to load an embedded texture" << std::endl;

	return Texture(width, height, channels, pixels);
}


