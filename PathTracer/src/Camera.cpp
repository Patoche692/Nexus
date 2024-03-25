#include "Camera.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/quaternion.hpp>
#include <gtx/quaternion.hpp>
#include <math.h>

#include "../Utils/Utils.h"
#include "Utils/cuda_math.h"
#include "Input.h"
#include "Cuda/PathTracer.cuh"


Camera::Camera(float verticalFOV, uint32_t width, uint32_t height)
{
	m_VerticalFOV = verticalFOV;
	m_ViewportWidth = width;
	m_ViewportHeight = height;
	m_Position = make_float3(0.0f, 0.0f, 2.0f);
	m_ForwardDirection = make_float3(0.0f, 0.0f, -1.0f);
	m_RightDirection = make_float3(1.0f, 0.0f, 0.0f);
}

Camera::Camera(float3 position, float3 forward, float verticalFOV, uint32_t width, uint32_t height)
	:m_Position(position), m_ForwardDirection(forward), m_VerticalFOV(verticalFOV), m_ViewportWidth(width),
	m_ViewportHeight(height), m_RightDirection(cross(m_ForwardDirection, make_float3(0.0f, 1.0f, 0.0f)))
{
}


void Camera::OnUpdate(float ts)
{
	float2 mousePos = Input::GetMousePosition();
	float2 delta = (mousePos - m_LastMousePosition) * 2.0f;
	m_LastMousePosition = mousePos;

	if (!Input::IsMouseButtonDown(GLFW_MOUSE_BUTTON_RIGHT))
	{
		Input::SetCursorMode(GLFW_CURSOR_NORMAL);
		return;
	}

	Input::SetCursorMode(GLFW_CURSOR_DISABLED);

	float3 upDirection = make_float3(0.0f, 1.0f, 0.0f);

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

		glm::vec3 rightDirection(m_RightDirection.x, m_RightDirection.y, m_RightDirection.z);
		glm::vec3 forwardDirection(m_ForwardDirection.x, m_ForwardDirection.y, m_ForwardDirection.z);
		glm::quat q = glm::normalize(glm::cross(glm::angleAxis(-pitchDelta, rightDirection),
			glm::angleAxis(-yawDelta, glm::vec3(0.0f, 1.0f, 0.0f))));
		m_ForwardDirection = make_float3(glm::normalize(glm::rotate(q, forwardDirection)));
		m_RightDirection = normalize(cross(m_ForwardDirection, upDirection));

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

