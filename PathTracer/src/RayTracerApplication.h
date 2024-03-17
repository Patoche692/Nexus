#pragma once
#include "Renderer/Renderer.h"

class RayTracerApplication {

public:
	RayTracerApplication(int width, int height, GLFWwindow* window);

	void Update(double deltaTime);

	void Display();

	void OnResize(int width, int height);

private:
	Renderer m_Renderer;
};
