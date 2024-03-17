#pragma once
#include "Renderer.h"

class RayTracerApplication {

public:
	RayTracerApplication(int width, int height);

	void Update(double deltaTime);

	void Display();

	void OnResize(int width, int height);

private:
	Renderer m_Renderer;
};
