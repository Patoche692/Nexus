#include "Camera.h"

#include "GLFW/glfw3.h"
#include <gtc/matrix_transform.hpp>
#include <gtc/quaternion.hpp>
#include <gtx/quaternion.hpp>

#include "../Utils.h"
#include "Input.h"


Camera::Camera(float verticalFOV, float nearClip, float farClip)
	:m_VerticalFOV(verticalFOV), m_NearClip(nearClip), m_FarClip(farClip)
{
	m_ForwardDirection = glm::vec3(0.0f, 0.0f, -1.0f);
	m_Position = glm::vec3(0.0f, 0.0f, 3.0f);
	//RecalculateView();
	//RecalculateProjection();

	checkCudaErrors(cudaMalloc((void**)&m_DevicePtr, sizeof(Camera)));
	SendDataToDevice();
}

Camera::~Camera()
{
	checkCudaErrors(cudaFree(m_DevicePtr));
}



void Camera::OnUpdate(float ts)
{
	glm::vec2 mousePos = Input::GetMousePosition();
	glm::vec2 delta = (mousePos - m_LastMousePosition) * 2.0f;
	m_LastMousePosition = mousePos;

	if (!Input::IsKeyDown(GLFW_MOUSE_BUTTON_RIGHT))
	{
		Input::SetCursorMode(GLFW_CURSOR_NORMAL);
		return;
	}

	Input::SetCursorMode(GLFW_CURSOR_DISABLED);

	bool moved = false;

	constexpr glm::vec3 upDirection(0.0f, 1.0f, 0.0f);
	glm::vec3 rightDirection = glm::cross(m_ForwardDirection, upDirection);

	float speed = 0.05f;

	if (Input::IsKeyDown(GLFW_KEY_W))
	{
		m_Position += ts * speed * m_ForwardDirection;
		moved = true;
	}
	else if (Input::IsKeyDown(GLFW_KEY_S))
	{
		m_Position -= ts * speed * m_ForwardDirection;
		moved = true;
	}
	if (Input::IsKeyDown(GLFW_KEY_A))
	{
		m_Position -= ts * speed * rightDirection;
		moved = true;
	}
	else if (Input::IsKeyDown(GLFW_KEY_D))
	{
		m_Position += ts * speed * rightDirection;
		moved = true;
	}
	if (Input::IsKeyDown(GLFW_KEY_Q))
	{
		m_Position -= ts * speed * upDirection;
		moved = true;
	}
	else if (Input::IsKeyDown(GLFW_KEY_E))
	{
		m_Position += ts * speed * upDirection;
		moved = true;
	}

	if (delta.x != 0.0f || delta.y != 0.0f)
	{
		float pitchDelta = delta.y * GetRotationSpeed();
		float yawDelta = delta.x * GetRotationSpeed();

		glm::quat q = glm::normalize(glm::cross(glm::angleAxis(-pitchDelta, rightDirection),
			glm::angleAxis(-yawDelta, glm::vec3(0.0f, 1.0f, 0.0f))));
		m_ForwardDirection = glm::rotate(q, m_ForwardDirection);

		moved = true;
	}

	if (moved)
	{
		RecalculateView();
		
		SendDataToDevice();
		//RecalculateRayDirections();
	}
}

void Camera::OnResize(uint32_t width, uint32_t height)
{
	if (width == m_ViewportWidth && height == m_ViewportHeight)
		return;

	m_ViewportWidth = width;
	m_ViewportHeight = height;

	RecalculateProjection();
	SendDataToDevice();
	//RecalculateRayDirections();
}

float Camera::GetRotationSpeed()
{
	return 0.8f;
}

void Camera::RecalculateProjection()
{
	m_Projection = glm::perspectiveFov(glm::radians(m_VerticalFOV), (float)m_ViewportWidth, (float)m_ViewportHeight, m_NearClip, m_FarClip);
	m_InverseProjection = glm::inverse(m_Projection);
}

void Camera::RecalculateView()
{
	m_View = glm::lookAt(m_Position, m_Position + m_ForwardDirection, glm::vec3(0.0f, 1.0f, 0.0f));
	m_InverseView = glm::inverse(m_View);
}

void Camera::RecalculateRayDirections()
{
	//m_RayDirections.resize(m_ViewportWidth * m_ViewportHeight);

	for (uint32_t y = 0; y < m_ViewportHeight; y++)
	{
		for (uint32_t x = 0; x < m_ViewportWidth; x++)
		{
			glm::vec2 coord = { (float)x / (float)m_ViewportWidth, (float)y / (float)m_ViewportHeight };
			coord = coord * 2.0f - 1.0f;

			glm::vec4 target = m_InverseProjection * glm::vec4(coord.x, coord.y, 1.0f, 1.0f);
			glm::vec3 rayDirection = glm::vec3(m_InverseView * glm::vec4(glm::normalize(glm::vec3(target) / target.w), 0));
			//m_RayDirections[y * m_ViewportWidth + x] = rayDirection;
		}
	}
}

void Camera::SendDataToDevice()
{
	checkCudaErrors(cudaMemcpy(m_DevicePtr, this, sizeof(Camera), cudaMemcpyHostToDevice));
}
