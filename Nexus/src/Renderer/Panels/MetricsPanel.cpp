#include "MetricsPanel.h"

#include <GLFW/glfw3.h>
#include <imgui.h>

MetricsPanel::MetricsPanel(Scene* context) : m_Context(context)
{
	Reset();
}

void MetricsPanel::Reset()
{
	m_NAccumulatedFrame = 0;
	m_AccumulatedTime = 0.0f;
	m_DeltaTime = 0.0f;

	m_MRPS = 0.0f;
	m_NumRaysProcessed = 0;

	m_DisplayFPSTimer = glfwGetTime();
}

void MetricsPanel::UpdateMetrics(float deltaTime)
{
	std::shared_ptr<Camera> camera = m_Context->GetCamera();

	m_NAccumulatedFrame++;
	m_NumRaysProcessed += camera->GetViewportWidth() * camera->GetViewportHeight();

	m_AccumulatedTime += deltaTime;
	if (glfwGetTime() - m_DisplayFPSTimer >= 0.2f || m_DeltaTime == 0)
	{
		m_DisplayFPSTimer = glfwGetTime();
		m_DeltaTime = m_AccumulatedTime / m_NAccumulatedFrame;
		m_MRPS = static_cast<float>(m_NumRaysProcessed) / m_AccumulatedTime / 1000.0f;		// millisecond * 1.000.000
		
		m_NAccumulatedFrame = 0;
		m_AccumulatedTime = 0.0f;
		m_NumRaysProcessed = 0;
	}

}

void MetricsPanel::OnImGuiRender(uint32_t frameNumber, D_Settings& settings)
{
	std::shared_ptr<Camera> camera = m_Context->GetCamera();

	ImGui::Begin("Metrics");

	ImGui::Spacing();
	ImGui::Separator();
	ImGui::Text("Time info");
	ImGui::Text("Render time millisec: %.3f", m_DeltaTime);
	ImGui::Text("FPS: %d", (int)(1000.0f / m_DeltaTime));
	ImGui::Text("Frame: %d", frameNumber);
	ImGui::Text("Megarays/sec: %.2f", m_MRPS);

	// TODO: move camera settings to another panel
	ImGui::Spacing();
	ImGui::Separator();
	ImGui::Text("Camera");
	if (ImGui::SliderFloat("Vertical FOV", &camera->GetVerticalFOV(), 1.0f, 180.0f))
		camera->Invalidate();
	if (ImGui::DragFloat("Focus distance", &camera->GetFocusDist(), 0.02f, 0.01f, 1000.0f))
		camera->Invalidate();
	if (ImGui::DragFloat("Defocus angle", &camera->GetDefocusAngle(), 0.2f, 0.0f, 180.0f))
		camera->Invalidate();

	ImGui::Text("Settings");
	ImGui::Checkbox("Use MIS", &settings.useMIS);

	ImGui::End();
}
