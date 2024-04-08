#pragma once

#include <iostream>
#include "../Geometry/Mesh.h"

class AssetManager
{
public:
	AssetManager() = default;

	void AddMesh(const std::string& filename);
	std::vector<Mesh>& GetMeshes() { return m_Meshes; }

private:
	std::vector<Mesh> m_Meshes;
};
