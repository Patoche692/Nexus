#pragma once
#include <iostream>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "Geometry/Triangle.h"

class OBJLoader
{
public:
	static std::vector<Triangle> LoadOBJ(const std::string& filename);

private:
	static Assimp::Importer m_Importer;
};
