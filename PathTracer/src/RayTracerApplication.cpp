#include "RayTracerApplication.h"

RayTracerApplication::RayTracerApplication(int width, int height, GLFWwindow *window)
	:m_Renderer(width, height, window), m_Scene(width, height)
{

}

void RayTracerApplication::Update(float deltaTime)
{
	m_Scene.GetCamera()->OnUpdate(deltaTime);
	Display(deltaTime);
}

void RayTracerApplication::Display(double deltaTime)
{
	m_Renderer.Render(m_Scene, deltaTime);
}

void RayTracerApplication::OnResize(int width, int height)
{
}

