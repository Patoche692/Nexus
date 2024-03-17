#include "RayTracerApplication.h"

RayTracerApplication::RayTracerApplication(int width, int height, GLFWwindow *window)
	:m_Renderer(width, height, window)
{
}

void RayTracerApplication::Update(double deltaTime)
{
	
	std::cout << deltaTime * 1000 << std::endl;
	Display();
}

void RayTracerApplication::Display()
{
	m_Renderer.Render();
}

void RayTracerApplication::OnResize(int width, int height)
{
	//if (width == )
}

