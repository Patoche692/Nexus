#include "RayTracerApplication.h"

RayTracerApplication::RayTracerApplication(int width, int height, GLFWwindow *window)
	:m_Renderer(width, height, window), m_Camera(45.0f, 0.1f, 100.0f)
{

}

void RayTracerApplication::Update(float deltaTime)
{
	m_Camera.OnUpdate(deltaTime);
	//std::cout << deltaTime * 1000 << std::endl;
	Display(deltaTime);
}

void RayTracerApplication::Display(double deltaTime)
{
	m_Renderer.Render(m_Camera, deltaTime);
}

void RayTracerApplication::OnResize(int width, int height)
{
	//if (width == )
}

