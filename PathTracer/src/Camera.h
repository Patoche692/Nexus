#pragma once

#include <glm.hpp>
#include <vector>
#include <cuda_runtime_api.h>

class Camera 
{
public:
	Camera(float verticalFOV, uint32_t width, uint32_t height);
	Camera(float3 position, float3 forward, float verticalFOV, uint32_t width, uint32_t height, float focusDistance, float defocusAngle);

	void OnUpdate(float ts);
	void OnResize(uint32_t width, uint32_t height);

	void SetVerticalFOV(float verticalFOV);

	float GetRotationSpeed();
	float& GetVerticalFOV() { return m_VerticalFOV; }
	float& GetDefocusAngle() { return m_DefocusAngle; }
	float& GetFocusDist() { return m_FocusDist; }
	uint32_t GetViewportWidth() { return m_ViewportWidth; }
	uint32_t GetViewportHeight() { return m_ViewportHeight; }
	float3& GetPosition() { return m_Position; }
	float3& GetForwardDirection() { return m_ForwardDirection; }
	float3& GetRightDirection() { return m_RightDirection; }

	bool IsInvalid() { return m_Invalid; }
	void Invalidate() { m_Invalid = true; }

	bool SendDataToDevice();

private:
	float2 m_LastMousePosition{ 0.0f, 0.0 };

	float m_VerticalFOV;
	float m_DefocusAngle;
	float m_FocusDist;
	uint32_t m_ViewportWidth;
	uint32_t m_ViewportHeight;
	float3 m_Position;
	float3 m_ForwardDirection;
	float3 m_RightDirection;

	bool m_Invalid = true;
};
