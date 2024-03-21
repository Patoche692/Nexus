#pragma once
#include "GLFW/glfw3.h"
#include "cuda/cuda_math.h"

class Input
{
public:
	static void Init(GLFWwindow* window);
	static float2 GetMousePosition();
	static bool IsKeyDown(int key);
	static bool IsMouseButtonDown(int key);
	static void SetCursorMode(int mode);
		

private:
	static GLFWwindow* m_Window;
};

