#include "RayTracerApplication.h"

RayTracerApplication::RayTracerApplication(int width, int height, GLFWwindow *window)
	:m_Renderer(width, height, window), m_Camera(45.0f)
{

}

void RayTracerApplication::Update(float deltaTime)
{
	m_Camera.OnUpdate(deltaTime);
	Display(deltaTime);
}

void RayTracerApplication::Display(double deltaTime)
{
	m_Renderer.Render(&m_Camera, deltaTime);
}

void RayTracerApplication::OnResize(int width, int height)
{
}

