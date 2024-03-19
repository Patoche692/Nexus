#pragma once

#include <glm.hpp>
#include <vector>
#include <cuda_runtime_api.h>

struct CameraData
{
	float verticalFOV;
	glm::vec3 position;
	glm::vec3 forwardDirection;
	glm::vec3 rightDirection;
	glm::vec3 upDirection;
};

class Camera 
{
public:
	Camera(float verticalFOV);

	void OnUpdate(float ts);
	void OnResize(uint32_t width, uint32_t height);

	float GetRotationSpeed();

	bool HasMoved() { return m_Moved; };

	CameraData& GetCameraData() { return m_CameraData; };

private:
	CameraData m_CameraData;

	glm::vec2 m_LastMousePosition{ 0.0f, 0.0 };

	uint32_t m_ViewportWidth = 0, m_ViewportHeight = 0;

	bool m_Moved = true;
};
