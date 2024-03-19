#include "RayTracerApplication.h"

RayTracerApplication::RayTracerApplication(int width, int height, GLFWwindow *window)
	:m_Renderer(width, height, window), m_Camera(glm::vec3(0.0f, 0.0f, 2.0f), glm::vec3(0.0f, 0.0f, -1.0f), 45.0f, width, height)
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

