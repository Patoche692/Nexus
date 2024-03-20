#pragma once
#include <glm.hpp>
#include "Material.h"

#define MAX_SPHERES 50

struct Sphere
{
	float radius;
	glm::vec3 position;
	Material material;
};

