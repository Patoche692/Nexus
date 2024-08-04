#pragma once
#include <iostream>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "Geometry/Triangle.h"
#include "Geometry/BVH/BVH.h"
#include "Assets/Mesh.h"
#include "Assets/AssetManager.h"

class OBJLoader
{
public:
	static std::vector<Mesh> LoadOBJ(const std::string& path, const std::string& filename, AssetManager* assetManager);

private:
	static Assimp::Importer m_Importer;
};
