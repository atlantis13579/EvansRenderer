#pragma once

#include "Editor/Editor.hpp"

namespace Evans::Editor
{

void RegisterModuleAdapters(EditorCore& editor);
[[nodiscard]] PickHandler CreatePickHandler();
[[nodiscard]] bool CookNaniteAssets(const SceneSnapshot&,std::size_t& families,std::string& error);
[[nodiscard]] bool CookBakedLightingAssets(const SceneSnapshot&,const Techniques::Settings&,std::string& error);

} // namespace Evans::Editor
