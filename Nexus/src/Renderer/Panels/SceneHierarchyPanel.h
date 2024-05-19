#pragma once
#include "Scene.h"

class SceneHierarchyPanel
{
public:
	SceneHierarchyPanel(Scene* context);

	void SetContext(Scene* context);

	void SetSelectionContext(int selectionContext);

	void OnImGuiRender();

private:
	void DrawProperties(int selectionContext);

private:
	Scene* m_Context;
	int m_SelectionContext = -1;
};
