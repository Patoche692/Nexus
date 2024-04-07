#pragma once
#include <iostream>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

class OBJLoader
{
public:
	static void LoadOBJ(const std::string& filename);

private:
	static Assimp::Importer m_Importer;
};
