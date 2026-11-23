#pragma once

namespace Evans::Techniques { struct Settings; }

namespace Evans::Editor
{
class EditorCore;
struct Rect;
// Shared by the native Windows and macOS editor panels.
void DrawTechniqueControls(EditorCore& editor);
bool DrawDragonMaterialControls(EditorCore& editor,Techniques::Settings& settings);
bool DrawDragonReflectionControls(EditorCore& editor,Techniques::Settings& settings);
bool DrawTransparencyControls(EditorCore& editor,Techniques::Settings& settings);
bool DrawHairControls(EditorCore& editor,Techniques::Settings& settings);
bool DrawVolumeControls(EditorCore& editor,Techniques::Settings& settings);
bool DrawWaterControls(EditorCore& editor,Techniques::Settings& settings);
bool DrawWorldEnvironmentControls(EditorCore& editor,Techniques::Settings& settings);
void DrawViewportStatus(const EditorCore& editor,const Rect& viewport,int clientHeight);
}
