#pragma once

#include <glm.hpp>
#include <vector>
#include <cuda_runtime_api.h>

class Camera 
{
public:
	Camera(float verticalFOV, uint32_t width, uint32_t height);

	void OnUpdate(float ts);
	void OnResize(uint32_t width, uint32_t height);

	void SetVerticalFOV(float verticalFOV);

	float GetRotationSpeed();
	float& GetVerticalFOV() { return m_VerticalFOV; };
	uint32_t GetViewportWidth() { return m_ViewportWidth; };
	uint32_t GetViewportHeight() { return m_ViewportHeight; };
	glm::vec3& GetPosition() { return m_Position; };
	glm::vec3& GetForwardDirection() { return m_ForwardDirection; };
	glm::vec3& GetRightDirection() { return m_RightDirection; };

	bool IsInvalid() { return m_Invalid; };
	void Invalidate() { m_Invalid = true; };

	void sendDataToDevice();

private:
	glm::vec2 m_LastMousePosition{ 0.0f, 0.0 };

	float m_VerticalFOV;
	uint32_t m_ViewportWidth;
	uint32_t m_ViewportHeight;
	glm::vec3 m_Position;
	glm::vec3 m_ForwardDirection;
	glm::vec3 m_RightDirection;

	bool m_Invalid = true;
};
