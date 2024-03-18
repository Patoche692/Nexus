#pragma once
#include "Renderer/Renderer.h"
#include "Camera.h"

class RayTracerApplication {

public:
	RayTracerApplication(int width, int height, GLFWwindow* window);

	void Update(float deltaTime);

	void Display(double deltaTime);

	void OnResize(int width, int height);

private:
	Renderer m_Renderer;
	Camera m_Camera;
};
