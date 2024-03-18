#include "RayTracerApplication.h"

RayTracerApplication::RayTracerApplication(int width, int height, GLFWwindow *window)
	:m_Renderer(width, height, window)
{
}

void RayTracerApplication::Update(float deltaTime)
{
	
	//std::cout << deltaTime * 1000 << std::endl;
	Display(deltaTime);
}

void RayTracerApplication::Display(double deltaTime)
{
	m_Renderer.Render(deltaTime);
}

void RayTracerApplication::OnResize(int width, int height)
{
	//if (width == )
}

