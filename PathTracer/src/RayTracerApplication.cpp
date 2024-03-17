#include "RayTracerApplication.h"

RayTracerApplication::RayTracerApplication(int width, int height)
	:m_Renderer(width, height)
{
}

void RayTracerApplication::Update(double deltaTime)
{

	Display();
}

void RayTracerApplication::Display()
{
	m_Renderer.Render();
}

void RayTracerApplication::OnResize(int width, int height)
{
}

