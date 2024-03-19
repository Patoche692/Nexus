#include "Camera.h"

#include <GL/glew.h>
#include "GLFW/glfw3.h"
#include <gtc/matrix_transform.hpp>
#include <gtc/quaternion.hpp>
#include <gtx/quaternion.hpp>

#include "../Utils.h"
#include "Input.h"
#include "Renderer/Renderer.cuh"


Camera::Camera(float verticalFOV)
{
	m_CameraData = {
		verticalFOV,
		glm::vec3(0.0f, 0.0f, 2.0f),
		glm::vec3(0.0f, 0.0f, -1.0f),
		glm::vec3(1.0f, 0.0f, 0.0f),
		glm::vec3(0.0f, 1.0f, 0.0f)
	};
	SendDataToDevice();
}


void Camera::OnUpdate(float ts)
{
	glm::vec2 mousePos = Input::GetMousePosition();
	glm::vec2 delta = (mousePos - m_LastMousePosition) * 2.0f;
	m_LastMousePosition = mousePos;

	if (!Input::IsMouseButtonDown(GLFW_MOUSE_BUTTON_RIGHT))
	{
		Input::SetCursorMode(GLFW_CURSOR_NORMAL);
		return;
	}

	Input::SetCursorMode(GLFW_CURSOR_DISABLED);

	m_Moved = false;

	constexpr glm::vec3 upDirection(0.0f, 1.0f, 0.0f);

	float speed = 0.005f;

	if (Input::IsKeyDown(GLFW_KEY_W))
	{
		m_CameraData.position += ts * speed * m_CameraData.forwardDirection;
		m_Moved = true;
	}
	else if (Input::IsKeyDown(GLFW_KEY_S))
	{
		m_CameraData.position -= ts * speed * m_CameraData.forwardDirection;
		m_Moved = true;
	}
	if (Input::IsKeyDown(GLFW_KEY_A))
	{
		m_CameraData.position -= ts * speed * m_CameraData.rightDirection;
		m_Moved = true;
	}
	else if (Input::IsKeyDown(GLFW_KEY_D))
	{
		m_CameraData.position += ts * speed * m_CameraData.rightDirection;
		m_Moved = true;
	}
	if (Input::IsKeyDown(GLFW_KEY_Q))
	{
		m_CameraData.position -= ts * speed * upDirection;
		m_Moved = true;
	}
	else if (Input::IsKeyDown(GLFW_KEY_E))
	{
		m_CameraData.position += ts * speed * upDirection;
		m_Moved = true;
	}

	if (delta.x != 0.0f || delta.y != 0.0f)
	{
		float pitchDelta = delta.y * GetRotationSpeed();
		float yawDelta = delta.x * GetRotationSpeed();

		glm::quat q = glm::normalize(glm::cross(glm::angleAxis(-pitchDelta, m_CameraData.rightDirection),
			glm::angleAxis(-yawDelta, glm::vec3(0.0f, 1.0f, 0.0f))));
		m_CameraData.forwardDirection = glm::rotate(q, m_CameraData.forwardDirection);
		m_CameraData.rightDirection = glm::cross(m_CameraData.forwardDirection, upDirection);
		m_CameraData.upDirection = glm::cross(m_CameraData.rightDirection, m_CameraData.forwardDirection);

		m_Moved = true;
	}

	if (m_Moved)
	{
		//SendDataToDevice();
	}
}

void Camera::OnResize(uint32_t width, uint32_t height)
{
	if (width == m_ViewportWidth && height == m_ViewportHeight)
		return;

	m_ViewportWidth = width;
	m_ViewportHeight = height;
}

float Camera::GetRotationSpeed()
{
	return 0.0008f;
}

