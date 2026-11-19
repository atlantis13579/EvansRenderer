#pragma once
#include <functional>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>
namespace Evans::Renderer
{
    // The caller owns the current ES context. All vertex attributes and fragment
    // outputs use explicit locations in these independently authored shaders.
    unsigned CompileCachedGLESProgram(const std::function<void*(const char*)>& resolve,
        std::initializer_list<std::pair<unsigned,std::string_view>> stages,std::string& error);
}
