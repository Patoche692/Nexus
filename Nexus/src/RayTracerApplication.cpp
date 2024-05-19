#include "RayTracerApplication.h"

GLFWwindow* RayTracerApplication::m_Window;

RayTracerApplication::RayTracerApplication(int width, int height, GLFWwindow *window)
	: m_Scene(width, height), m_Renderer(width, height, window, &m_Scene) {
	m_Window = window;
}

void RayTracerApplication::Update(float deltaTime)
{
	m_Scene.GetCamera()->OnUpdate(deltaTime);
	Display(deltaTime);
}

void RayTracerApplication::Display(float deltaTime)
{
	m_Renderer.Render(m_Scene, deltaTime);
}

void RayTracerApplication::OnResize(int width, int height)
{
}

