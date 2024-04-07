#include "OBJLoader.h"

Assimp::Importer OBJLoader::m_Importer;

void OBJLoader::LoadOBJ(const std::string& filename)
{
	const aiScene* scene = m_Importer.ReadFile(filename, aiProcess_CalcTangentSpace | aiProcess_Triangulate);
	
	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
	{
		std::cout << "Error loading model " << filename << std::endl;
	}


	
}
