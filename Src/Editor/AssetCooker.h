#pragma once

#include <optional>
#include <string>
#include <vector>

namespace Evans::Editor
{
    [[nodiscard]] std::optional<int> RunAssetCookCommand(const std::vector<std::string>& arguments);
}
