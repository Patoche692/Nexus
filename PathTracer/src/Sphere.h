#pragma once
#include <glm.hpp>
#include "Material.h"

struct Sphere
{
	float radius;
	glm::vec3 position;
	Material material;
};

