#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace Evans::Renderer
{
	enum class RealtimeShaderLanguage : std::uint8_t
	{
		HLSL,
		GLSLES,
		Metal,
		SPIRV
	};

	enum class RealtimeShaderStage : std::uint8_t
	{
		Vertex,
		Fragment,
		Shadow,
		Outline
	};

	enum class RealtimeShaderResourceKind : std::uint8_t
	{
		ConstantBuffer,
		DepthTexture,
		Texture,
		Sampler
	};

	struct RealtimeShaderDiagnostic
	{
		std::size_t Line = 1;
		std::size_t Column = 1;
		std::string Message;
	};

	struct RealtimeShaderResource
	{
		RealtimeShaderResourceKind Kind =
			RealtimeShaderResourceKind::Texture;
		std::string Name;
		std::uint32_t Binding = 0;
		std::uint32_t VulkanBinding = 0;
		std::size_t Size = 0;
		std::string ColorSpace;
		std::string Filter;
		std::string AddressMode;
	};

	struct RealtimeShaderStageEntry
	{
		RealtimeShaderStage Stage = RealtimeShaderStage::Vertex;
		std::string HLSL;
		std::string GLSLES;
		std::string Metal;
		std::string SPIRV;

		const std::string& Entry(RealtimeShaderLanguage language) const;
	};

	struct RealtimeShaderProgram
	{
		std::string Name;
		std::uint32_t Version = 0;
		std::uint32_t MaximumLights = 0;
		std::uint32_t ShadowKernelRadius = 0;
		float ShadowStrength = 0.0f;
		std::string OutlineSpace;
		std::vector<RealtimeShaderResource> Resources;
		std::vector<RealtimeShaderStageEntry> Stages;

		const RealtimeShaderResource* FindResourceByName(
			std::string_view name) const;
		const RealtimeShaderStageEntry* FindStage(
			RealtimeShaderStage stage) const;
	};

	struct RealtimeShaderParseResult
	{
		RealtimeShaderProgram Program;
		std::vector<RealtimeShaderDiagnostic> Diagnostics;

		explicit operator bool() const
		{
			return Diagnostics.empty();
		}

		std::string ErrorText() const;
	};

	RealtimeShaderParseResult ParseRealtimeShader(
		std::string_view source);

	const char* BuiltinRealtimeShaderManifest();
	const RealtimeShaderParseResult& BuiltinRealtimeShader();

	bool ValidateRealtimeShaderContract(
		const RealtimeShaderProgram& program,
		std::size_t constantBufferSize,
		std::uint32_t maximumLights,
		std::string& error);

	bool MaterializeRealtimeShader(
		const RealtimeShaderProgram& program,
		RealtimeShaderLanguage language,
		std::string_view sourceTemplate,
		const std::vector<RealtimeShaderStage>& requiredStages,
		std::string& output,
		std::string& error);

	bool PrepareBuiltinRealtimeShader(
		RealtimeShaderLanguage language,
		std::string_view sourceTemplate,
		const std::vector<RealtimeShaderStage>& requiredStages,
		std::size_t constantBufferSize,
		std::uint32_t maximumLights,
		std::string& output,
		std::string& error);
}
