#include "IMGLoader.h"
#include "stb/stb_image.h"

IMGLoader::IMGLoader()
{
}

IMGLoader::~IMGLoader()
{
}

inline Texture IMGLoader::LoadIMG(std::string filepath)
{
	int width, height, channels;

	unsigned char* image = stbi_load(filepath.c_str(), &width, &height, &channels, 0);

	return Texture(width, height, channels, image);
}


