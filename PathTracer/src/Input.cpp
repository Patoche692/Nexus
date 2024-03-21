#include "Input.h"
#include "cuda/cuda_math.h"

GLFWwindow* Input::m_Window;

void Input::Init(GLFWwindow* window)
{
	m_Window = window;
}

float2 Input::GetMousePosition()
{
	double xpos, ypos;
	glfwGetCursorPos(m_Window, &xpos, &ypos);
	return make_float2(xpos, ypos);
}

bool Input::IsKeyDown(int key)
{
	int state = glfwGetKey(m_Window, key);
	return state == GLFW_PRESS;
}

bool Input::IsMouseButtonDown(int key)
{
	int state = glfwGetMouseButton(m_Window, key);
	return state == GLFW_PRESS;
}

void Input::SetCursorMode(int mode)
{
	glfwSetInputMode(m_Window, GLFW_CURSOR, mode);
}


