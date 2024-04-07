#pragma once

#include <iostream>

class AssetManager
{
public:
	AssetManager() = default;

	void AddMesh(const std::string& filename);
};
