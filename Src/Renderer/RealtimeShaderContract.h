#pragma once

#include "Renderer/Types.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace Evans::Renderer
{
	// Keep native shadow-map rasterization consistent. A zero clamp is portable
	// to OpenGL ES and Vulkan devices without optional depth-bias-clamp support.
	inline constexpr int ShadowDepthBiasUnits = 1024;
	inline constexpr float ShadowDepthBiasClamp = 0.0f;
	inline constexpr float ShadowSlopeScaledDepthBias = 2.0f;

	struct ShaderUInt4
	{
		std::uint32_t X = 0;
		std::uint32_t Y = 0;
		std::uint32_t Z = 0;
		std::uint32_t W = 0;
	};

	struct ShaderLight
	{
		Float4 PositionType;
		Float4 DirectionIntensity;
		Float4 ColorCastShadows;
		Float4 AreaSizePadding;
	};

	struct RealtimeDrawConstants
	{
		Matrix4 World;
		Matrix4 NormalWorld;
		Matrix4 ViewProjection;
		Matrix4 LightViewProjection;
		Float4 PipelineColor;
		Float4 MaterialDiffuseColor;
		Float4 MaterialAmbientColor;
		Float4 MaterialSpecularColor;
		Float4 MaterialEmissiveColor;
		Float4 MaterialParameters;
		Float4 AmbientColorStrength;
		Float4 CameraLightCount;
		Float4 RenderParameters;
		ShaderUInt4 TextureFlags;
		std::array<ShaderLight, MaximumSceneLights> Lights;
	};

	static_assert(sizeof(Vertex) == 48,
		"The realtime shader input layout must match Renderer::Vertex");
	static_assert(sizeof(ShaderLight) == 64,
		"The realtime light constants must match the shader contract");
	static_assert(offsetof(RealtimeDrawConstants, NormalWorld) == 64
		&& offsetof(RealtimeDrawConstants, ViewProjection) == 128
		&& offsetof(RealtimeDrawConstants, LightViewProjection) == 192
		&& offsetof(RealtimeDrawConstants, PipelineColor) == 256
		&& offsetof(RealtimeDrawConstants, MaterialDiffuseColor) == 272
		&& offsetof(RealtimeDrawConstants, MaterialParameters) == 336
		&& offsetof(RealtimeDrawConstants, TextureFlags) == 400
		&& offsetof(RealtimeDrawConstants, Lights) == 416,
		"The realtime draw constants must preserve the shader packing contract");
	static_assert(sizeof(RealtimeDrawConstants) == 672,
		"The realtime draw constants must match the shader contract");

	inline Matrix4 BuildRealtimeNormalMatrix(const Matrix4& world)
	{
		const float a00 = world.Values[0];
		const float a01 = world.Values[1];
		const float a02 = world.Values[2];
		const float a10 = world.Values[4];
		const float a11 = world.Values[5];
		const float a12 = world.Values[6];
		const float a20 = world.Values[8];
		const float a21 = world.Values[9];
		const float a22 = world.Values[10];
		const float determinant =
			a00 * (a11 * a22 - a12 * a21) -
			a01 * (a10 * a22 - a12 * a20) +
			a02 * (a10 * a21 - a11 * a20);
		if (!std::isfinite(determinant) || std::fabs(determinant) <= 1.0e-12f)
		{
			return Matrix4::Identity();
		}

		const float inverseDeterminant = 1.0f / determinant;
		Matrix4 result;
		result.Values = {
			(a11 * a22 - a12 * a21) * inverseDeterminant,
			(a12 * a20 - a10 * a22) * inverseDeterminant,
			(a10 * a21 - a11 * a20) * inverseDeterminant,
			0.0f,
			(a02 * a21 - a01 * a22) * inverseDeterminant,
			(a00 * a22 - a02 * a20) * inverseDeterminant,
			(a01 * a20 - a00 * a21) * inverseDeterminant,
			0.0f,
			(a01 * a12 - a02 * a11) * inverseDeterminant,
			(a02 * a10 - a00 * a12) * inverseDeterminant,
			(a00 * a11 - a01 * a10) * inverseDeterminant,
			0.0f,
			0.0f, 0.0f, 0.0f, 1.0f };
		return result;
	}

	inline RealtimeDrawConstants BuildRealtimeDrawConstants(
		const PipelineState& pipeline,
		const Matrix4& world,
		const Matrix4& viewProjection,
		const Matrix4& lightViewProjection,
		PassType pass,
		bool shadowAvailable,
		bool diffuseAvailable,
		bool normalAvailable,
		bool opacityAvailable)
	{
		RealtimeDrawConstants constants {};
		constants.World = world;
		constants.NormalWorld = BuildRealtimeNormalMatrix(world);
		constants.ViewProjection = viewProjection;
		constants.LightViewProjection = lightViewProjection;
		constants.PipelineColor = pipeline.Color;
		constants.MaterialDiffuseColor = pipeline.MaterialDiffuseColor;
		constants.MaterialAmbientColor = pipeline.MaterialAmbientColor;
		constants.MaterialSpecularColor = pipeline.MaterialSpecularColor;
		constants.MaterialEmissiveColor = pipeline.MaterialEmissiveColor;
		const float outlineThickness = std::isfinite(pipeline.OutlineThickness)
			? std::clamp(pipeline.OutlineThickness, 0.0f, 255.0f)
			: 0.0f;
		constants.MaterialParameters = {
			pipeline.MaterialShininess,
			pipeline.MaterialOpacity,
			pipeline.AlphaCutoff,
			outlineThickness };

		const LightingState& lighting = pipeline.Lighting;
		const std::uint32_t lightCount = std::min(
			lighting.LightCount, MaximumSceneLights);
		constants.AmbientColorStrength = {
			lighting.AmbientColor.X,
			lighting.AmbientColor.Y,
			lighting.AmbientColor.Z,
			lighting.AmbientStrength };
		constants.CameraLightCount = {
			lighting.CameraPosition.X,
			lighting.CameraPosition.Y,
			lighting.CameraPosition.Z,
			static_cast<float>(lightCount) };
		const float renderMode = pass == PassType::Outline || pipeline.OutlineExtrusion
			? 2.0f
			: (pipeline.Fill == FillMode::Wireframe ? 1.0f : 0.0f);
		constants.RenderParameters = {
			shadowAvailable ? 1.0f : 0.0f,
			renderMode,
			static_cast<float>(lighting.ShadowLightIndex),
			pipeline.BlendEnabled ? 1.0f : 0.0f };
		constants.TextureFlags = {
			diffuseAvailable ? 1u : 0u,
			normalAvailable ? 1u : 0u,
			opacityAvailable ? 1u : 0u,
			0u };

		for (std::uint32_t lightIndex = 0; lightIndex < lightCount; ++lightIndex)
		{
			const SceneLight& light = lighting.Lights[lightIndex];
			ShaderLight& gpuLight = constants.Lights[lightIndex];
			gpuLight.PositionType = {
				light.Position.X,
				light.Position.Y,
				light.Position.Z,
				static_cast<float>(light.Type) };
			gpuLight.DirectionIntensity = {
				light.Direction.X,
				light.Direction.Y,
				light.Direction.Z,
				light.Intensity };
			gpuLight.ColorCastShadows = {
				light.Color.X,
				light.Color.Y,
				light.Color.Z,
				light.CastShadows ? 1.0f : 0.0f };
			gpuLight.AreaSizePadding = {
				light.AreaSize.X,
				light.AreaSize.Y,
				0.0f,
				0.0f };
		}
		return constants;
	}
}
