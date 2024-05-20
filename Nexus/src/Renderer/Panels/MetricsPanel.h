#pragma once
#include "Scene.h"

class MetricsPanel 
{
public:
	MetricsPanel(Scene* context);

	void Reset();
	void UpdateMetrics(float deltaTime);
	void OnImGuiRender(uint32_t frameNumber);

private:

	Scene* m_Context;

	uint32_t m_NAccumulatedFrame = 0;
	float m_AccumulatedTime = 0.0f;
	float m_DisplayFPSTimer = 0.0f;
	float m_DeltaTime = 0.0f;

	float m_MRPS = 0.0f;
	uint32_t m_NumRaysProcessed = 0;

};