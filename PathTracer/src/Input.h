#pragma once
#include "GLFW/glfw3.h"
#include "glm.hpp"

class Input
{
public:
	static void Init(GLFWwindow* window);
	static glm::vec2 GetMousePosition();
	static bool IsKeyDown(int key);
	static bool IsMouseButtonDown(int key);
	static void SetCursorMode(int mode);
		

private:
	static GLFWwindow* m_Window;
};

