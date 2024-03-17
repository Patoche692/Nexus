#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>

class UIRenderer
{
public:
	UIRenderer(GLFWwindow *window);
	~UIRenderer();

	void Render();

private:

};
