#pragma once

#include "Renderer/Types.h"

namespace Evans::Renderer
{
    bool IntersectsFrustum(const Float3& minimum, const Float3& maximum,
        const Matrix4& world, const Matrix4& viewProjection);
}
