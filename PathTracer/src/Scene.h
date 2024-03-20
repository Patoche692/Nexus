#pragma once

#include <iostream>

#include "Camera.h"
#include "Sphere.h"

class Scene
{
public:
	Scene(uint32_t width, uint32_t height);

	bool IsInvalid() const { return m_Invalid; };
	bool Invalidate() { m_Invalid = true; };
	std::shared_ptr<Camera> GetCamera() { return m_Camera; };

private:
	bool m_Invalid = true;
	std::shared_ptr<Camera> m_Camera;
	std::vector<Sphere> m_Spheres;

};
