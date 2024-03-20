#include "Scene.h"

Scene::Scene(uint32_t width, uint32_t height)
	:m_Camera(std::make_shared<Camera>(glm::vec3(0.0f, 0.0f, 2.0f), glm::vec3(0.0f, 0.0f, -1.0f), 45.0f, width, height))
{
	Material material = { glm::vec3(1.0f) };
	Sphere sphere = {
		0.5f,
		glm::vec3(0.0f),
		material
	};
	m_Spheres.emplace_back(sphere);
}
