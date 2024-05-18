#pragma once
#include <iostream>
#include "Texture.h"
#include "assimp/scene.h"

class IMGLoader
{
public:
	IMGLoader();
	~IMGLoader();

	static Texture LoadIMG(const std::string& pathfile);
	static Texture LoadIMG(const aiTexture* texture);

private:

};

