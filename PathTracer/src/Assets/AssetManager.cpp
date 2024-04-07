#include "AssetManager.h"
#include "OBJLoader.h"

void AssetManager::AddMesh(const std::string& filename)
{
	OBJLoader::LoadOBJ(filename);
}
