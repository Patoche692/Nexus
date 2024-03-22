#pragma once

#include <iostream>

#include "Camera.h"
#include "Geometry/Sphere.h"

class Scene
{
public:
	Scene(uint32_t width, uint32_t height);

	bool IsInvalid() const { return m_Invalid; };
	void Invalidate() { m_Invalid = true; };

	std::vector<Sphere>& GetSpheres() { return m_Spheres; };
	std::shared_ptr<Camera> GetCamera() { return m_Camera; };

	void SendDataToDevice();

private:
	bool m_Invalid = true;
	std::shared_ptr<Camera> m_Camera;
	std::vector<Sphere> m_Spheres;

};
