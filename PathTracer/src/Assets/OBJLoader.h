#pragma once
#include <iostream>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "Geometry/Triangle.h"
#include "Geometry/BVH/BVH.h"
#include "Geometry/Mesh.h"

class OBJLoader
{
public:
	static Mesh LoadOBJ(const std::string& filename);

private:
	static Assimp::Importer m_Importer;
};
