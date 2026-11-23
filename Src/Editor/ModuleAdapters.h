#pragma once

#include "Editor/Editor.hpp"

namespace Evans::Editor
{

void RegisterModuleAdapters(EditorCore& editor);
[[nodiscard]] PickHandler CreatePickHandler();
[[nodiscard]] bool CookNaniteAssets(const SceneSnapshot&,std::size_t& families,std::string& error);
[[nodiscard]] bool BakeSceneWithArvo(const SceneSnapshot&,const Techniques::Settings&,
    const Renderer::ArvoBakeOptions&,std::string& error,const std::function<bool()>& cancelled={},
    const std::function<void(const std::string&)>& status={});

} // namespace Evans::Editor
