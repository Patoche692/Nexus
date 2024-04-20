#pragma once
#include <iostream>
#include "Texture.h"

class IMGLoader
{
public:
	IMGLoader();
	~IMGLoader();

	static Texture LoadIMG(std::string pathfile);

private:

};

