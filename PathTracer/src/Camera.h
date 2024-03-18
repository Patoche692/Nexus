#pragma once

#include <glm.hpp>
#include <vector>
#include <cuda_runtime_api.h>

class Camera {
public:
	Camera(float verticalFOV, float nearClip, float farClip);
	~Camera();

	void OnUpdate(float ts);
	void OnResize(uint32_t width, uint32_t height);

	__host__ __device__ glm::mat4 GetProjection() const { return m_Projection; };
	__host__ __device__ glm::mat4 GetInverseProjection() const { return m_InverseProjection; };
	__host__ __device__ glm::mat4 GetView() const  { return m_View; };
	__host__ __device__ glm::mat4 GetInverseView() const { return m_InverseView; };
 
	__host__ __device__ glm::vec3 GetPosition() const { return m_Position; };

	Camera* GetDevicePtr() const { return m_DevicePtr; };

	float GetRotationSpeed();

private:
	void RecalculateProjection();
	void RecalculateView();
	void RecalculateRayDirections();
	void SendDataToDevice();

private:
	glm::mat4 m_Projection{ 1.0f };
	glm::mat4 m_View{ 1.0f };
	glm::mat4 m_InverseProjection{ 1.0f };
	glm::mat4 m_InverseView{ 1.0f };

	float m_VerticalFOV = 45.0f;
	float m_NearClip = 0.1f;
	float m_FarClip = 100.0f;

	glm::vec3 m_Position{ 0.0f, 0.0f, 0.0f };
	glm::vec3 m_ForwardDirection{ 0.0f, 0.0f, 0.0f };

	glm::vec2 m_LastMousePosition{ 0.0f, 0.0 };

	uint32_t m_ViewportWidth = 0, m_ViewportHeight = 0;
	Camera* m_DevicePtr = nullptr;
};
