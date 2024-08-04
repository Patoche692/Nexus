#pragma once
#include "Renderer/Renderer.h"
#include "Scene/Scene.h"

class RayTracerApplication {

public:
	RayTracerApplication(int width, int height, GLFWwindow* window);

	void Update(float deltaTime);

	void Display(float deltaTime);

	void OnResize(int width, int height);

	static GLFWwindow* GetNativeWindow() { return m_Window; }

private:
	Renderer m_Renderer;
	Scene m_Scene;
	static GLFWwindow* m_Window;
};
