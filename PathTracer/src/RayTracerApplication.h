#pragma once
#include "Renderer/Renderer.h"
#include "Scene.h"

class RayTracerApplication {

public:
	RayTracerApplication(int width, int height, GLFWwindow* window);
	~RayTracerApplication();

	void Update(float deltaTime);

	void Display(float deltaTime);

	void OnResize(int width, int height);

private:
	Renderer m_Renderer;
	Scene m_Scene;
	std::vector<Material*> m_Materials;
};
