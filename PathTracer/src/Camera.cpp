#include "Camera.h"

#include <GL/glew.h>
#include "GLFW/glfw3.h"
#include <gtc/matrix_transform.hpp>
#include <gtc/quaternion.hpp>
#include <gtx/quaternion.hpp>
#include <math.h>

#include "../Utils.h"
#include "Input.h"
#include "Renderer/Renderer.cuh"


Camera::Camera(float verticalFOV, uint32_t width, uint32_t height)
{
	m_VerticalFOV = verticalFOV;
	m_ViewportWidth = width;
	m_ViewportHeight = height;
	m_Position = glm::vec3(0.0f, 0.0f, 2.0f);
	m_ForwardDirection = glm::vec3(0.0f, 0.0f, -1.0f);
	m_RightDirection = glm::vec3(1.0f, 0.0f, 0.0f);
}

Camera::Camera(glm::vec3 position, glm::vec3 forward, float verticalFOV, uint32_t width, uint32_t height)
	:m_Position(position), m_ForwardDirection(forward), m_VerticalFOV(verticalFOV), m_ViewportWidth(width),
	m_ViewportHeight(height), m_RightDirection(glm::cross(m_ForwardDirection, glm::vec3(0.0f, 1.0f, 0.0f)))
{
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

	constexpr glm::vec3 upDirection(0.0f, 1.0f, 0.0f);

	float speed = 0.003f;

	if (Input::IsKeyDown(GLFW_KEY_W))
	{
		m_Position += ts * speed * m_ForwardDirection;
		m_Invalid = true;
	}
	else if (Input::IsKeyDown(GLFW_KEY_S))
	{
		m_Position -= ts * speed * m_ForwardDirection;
		m_Invalid = true;
	}
	if (Input::IsKeyDown(GLFW_KEY_A))
	{
		m_Position -= ts * speed * m_RightDirection;
		m_Invalid = true;
	}
	else if (Input::IsKeyDown(GLFW_KEY_D))
	{
		m_Position += ts * speed * m_RightDirection;
		m_Invalid = true;
	}
	if (Input::IsKeyDown(GLFW_KEY_Q))
	{
		m_Position -= ts * speed * upDirection;
		m_Invalid = true;
	}
	else if (Input::IsKeyDown(GLFW_KEY_E))
	{
		m_Position += ts * speed * upDirection;
		m_Invalid = true;
	}

	if (delta.x != 0.0f || delta.y != 0.0f)
	{
		float pitchDelta = delta.y * GetRotationSpeed();
		float yawDelta = delta.x * GetRotationSpeed();

		glm::quat q = glm::normalize(glm::cross(glm::angleAxis(-pitchDelta, m_RightDirection),
			glm::angleAxis(-yawDelta, glm::vec3(0.0f, 1.0f, 0.0f))));
		m_ForwardDirection = glm::normalize(glm::rotate(q, m_ForwardDirection));
		m_RightDirection = glm::normalize(glm::cross(m_ForwardDirection, upDirection));

		m_Invalid = true;
	}
}

void Camera::OnResize(uint32_t width, uint32_t height)
{
	if (width == m_ViewportWidth && height == m_ViewportHeight)
		return;

	m_ViewportWidth = width;
	m_ViewportHeight = height;
	Invalidate();
}

void Camera::SetVerticalFOV(float verticalFOV)
{
	m_VerticalFOV = verticalFOV;
}

float Camera::GetRotationSpeed()
{
	return 0.0008f;
}

void Camera::SendDataToDevice()
{
	m_Invalid = false;
	SendCameraDataToDevice(this);
}

