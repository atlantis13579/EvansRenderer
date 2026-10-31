#pragma once

#include "Editor/Editor.hpp"

namespace Evans::Editor
{

void RegisterModuleAdapters(EditorCore& editor);
[[nodiscard]] PickHandler CreatePickHandler();

} // namespace Evans::Editor
