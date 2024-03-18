#include "Input.h"

GLFWwindow* Input::m_Window;

void Input::Init(GLFWwindow* window)
{
	m_Window = window;
}

glm::vec2 Input::GetMousePosition()
{
	double xpos, ypos;
	glfwGetCursorPos(m_Window, &xpos, &ypos);
	return glm::vec2(xpos, ypos);
}

bool Input::IsKeyDown(int key)
{
	int state = glfwGetKey(m_Window, key);
	return state == GLFW_PRESS;
}

void Input::SetCursorMode(int mode)
{
	glfwSetInputMode(m_Window, GLFW_CURSOR, mode);
}


