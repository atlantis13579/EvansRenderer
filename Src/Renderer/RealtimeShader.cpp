#include "Renderer/RealtimeShader.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <locale>
#include <set>
#include <sstream>
#include <system_error>
#include <unordered_map>
#include <utility>

namespace Evans::Renderer
{
	namespace
	{
		enum class TokenKind : std::uint8_t
		{
			Identifier,
			Number,
			LeftBrace,
			RightBrace,
			Semicolon,
			Equal,
			End,
			Invalid
		};

		struct Token
		{
			TokenKind Kind = TokenKind::Invalid;
			std::string Text;
			std::size_t Line = 1;
			std::size_t Column = 1;
		};

		class Lexer
		{
		public:
			explicit Lexer(std::string_view source)
				: Source_(source)
			{
			}

			Token Next()
			{
				SkipTrivia();
				const std::size_t line = Line_;
				const std::size_t column = Column_;
				if (Position_ >= Source_.size())
				{
					return {TokenKind::End, {}, line, column};
				}
				const char value = Source_[Position_];
				if (IsIdentifierStart(value))
				{
					const std::size_t start = Position_;
					Advance();
					while (Position_ < Source_.size() &&
						IsIdentifierContinue(Source_[Position_]))
					{
						Advance();
					}
					return {
						TokenKind::Identifier,
						std::string(Source_.substr(
							start,
							Position_ - start)),
						line,
						column };
				}
				if ((value >= '0' && value <= '9') ||
					(value == '.' &&
						Position_ + 1 < Source_.size() &&
						Source_[Position_ + 1] >= '0' &&
						Source_[Position_ + 1] <= '9'))
				{
					const std::size_t start = Position_;
					bool hasDot = value == '.';
					Advance();
					while (Position_ < Source_.size())
					{
						const char digit = Source_[Position_];
						if (digit >= '0' && digit <= '9')
						{
							Advance();
							continue;
						}
						if (digit == '.' && !hasDot)
						{
							hasDot = true;
							Advance();
							continue;
						}
						break;
					}
					return {
						TokenKind::Number,
						std::string(Source_.substr(
							start,
							Position_ - start)),
						line,
						column };
				}
				Advance();
				switch (value)
				{
				case '{':
					return {TokenKind::LeftBrace, "{", line, column};
				case '}':
					return {TokenKind::RightBrace, "}", line, column};
				case ';':
					return {TokenKind::Semicolon, ";", line, column};
				case '=':
					return {TokenKind::Equal, "=", line, column};
				default:
					return {
						TokenKind::Invalid,
						std::string(1, value),
						line,
						column };
				}
			}

		private:
			static bool IsIdentifierStart(char value)
			{
				return
					(value >= 'a' && value <= 'z') ||
					(value >= 'A' && value <= 'Z') ||
					value == '_';
			}

			static bool IsIdentifierContinue(char value)
			{
				return IsIdentifierStart(value) ||
					(value >= '0' && value <= '9');
			}

			void Advance()
			{
				if (Position_ >= Source_.size())
				{
					return;
				}
				if (Source_[Position_] == '\n')
				{
					++Line_;
					Column_ = 1;
				}
				else
				{
					++Column_;
				}
				++Position_;
			}

			void SkipTrivia()
			{
				for (;;)
				{
					while (Position_ < Source_.size())
					{
						const char value = Source_[Position_];
						if (value != ' ' && value != '\t' &&
							value != '\r' && value != '\n')
						{
							break;
						}
						Advance();
					}
					if (Position_ + 1 >= Source_.size() ||
						Source_[Position_] != '/')
					{
						return;
					}
					if (Source_[Position_ + 1] == '/')
					{
						Advance();
						Advance();
						while (Position_ < Source_.size() &&
							Source_[Position_] != '\n')
						{
							Advance();
						}
						continue;
					}
					if (Source_[Position_ + 1] == '*')
					{
						Advance();
						Advance();
						while (Position_ + 1 < Source_.size() &&
							!(Source_[Position_] == '*' &&
								Source_[Position_ + 1] == '/'))
						{
							Advance();
						}
						if (Position_ + 1 < Source_.size())
						{
							Advance();
							Advance();
						}
						continue;
					}
					return;
				}
			}

			std::string_view Source_;
			std::size_t Position_ = 0;
			std::size_t Line_ = 1;
			std::size_t Column_ = 1;
		};

		class Parser
		{
		public:
			explicit Parser(std::string_view source)
				: Lexer_(source)
			{
				Advance();
			}

			RealtimeShaderParseResult Parse()
			{
				if (!ConsumeIdentifier("shader"))
				{
					AddError(Current_, "Expected 'shader'");
					return std::move(Result_);
				}
				if (Current_.Kind != TokenKind::Identifier)
				{
					AddError(Current_, "Expected a shader name");
					return std::move(Result_);
				}
				Result_.Program.Name = Current_.Text;
				Advance();
				if (!Consume(TokenKind::LeftBrace))
				{
					AddError(Current_, "Expected '{' after the shader name");
					return std::move(Result_);
				}

				while (Current_.Kind != TokenKind::RightBrace &&
					Current_.Kind != TokenKind::End)
				{
					if (Current_.Kind == TokenKind::Invalid)
					{
						AddError(
							Current_,
							"Unexpected character '" +
								Current_.Text + "'");
						Advance();
						continue;
					}
					if (Current_.Kind != TokenKind::Identifier)
					{
						AddError(Current_, "Expected a shader declaration");
						Synchronize();
						continue;
					}
					const Token declaration = Current_;
					Advance();
					if (declaration.Text == "version")
					{
						ParseVersion();
					}
					else if (declaration.Text == "constant_buffer")
					{
						ParseConstantBuffer();
					}
					else if (declaration.Text == "resource")
					{
						ParseResource();
					}
					else if (declaration.Text == "sampler")
					{
						ParseSampler();
					}
					else if (declaration.Text == "stage")
					{
						ParseStage();
					}
					else if (declaration.Text == "option")
					{
						ParseOption();
					}
					else
					{
						AddError(
							declaration,
							"Unknown declaration '" +
								declaration.Text + "'");
						Synchronize();
					}
				}
				if (!Consume(TokenKind::RightBrace))
				{
					AddError(Current_, "Expected '}' at the end of the shader");
				}
				if (Current_.Kind != TokenKind::End)
				{
					AddError(Current_, "Unexpected tokens after the shader");
				}
				Validate();
				return std::move(Result_);
			}

		private:
			void Advance()
			{
				Current_ = Lexer_.Next();
			}

			bool Consume(TokenKind kind)
			{
				if (Current_.Kind != kind)
				{
					return false;
				}
				Advance();
				return true;
			}

			bool ConsumeIdentifier(std::string_view identifier)
			{
				if (Current_.Kind != TokenKind::Identifier ||
					Current_.Text != identifier)
				{
					return false;
				}
				Advance();
				return true;
			}

			bool ExpectIdentifier(
				const char* description,
				std::string& output)
			{
				if (Current_.Kind != TokenKind::Identifier)
				{
					AddError(
						Current_,
						std::string("Expected ") + description);
					return false;
				}
				output = Current_.Text;
				Advance();
				return true;
			}

			bool ExpectUnsigned(
				const char* description,
				std::uint32_t& output)
			{
				if (Current_.Kind != TokenKind::Number ||
					Current_.Text.find('.') != std::string::npos)
				{
					AddError(
						Current_,
						std::string("Expected ") + description);
					return false;
				}
				const char* first = Current_.Text.data();
				const char* last = first + Current_.Text.size();
				const std::from_chars_result parsed =
					std::from_chars(first, last, output);
				if (parsed.ec != std::errc {} ||
					parsed.ptr != last)
				{
					AddError(
						Current_,
						std::string("Invalid ") + description);
					return false;
				}
				Advance();
				return true;
			}

			bool ExpectFloat(
				const char* description,
				float& output)
			{
				if (Current_.Kind != TokenKind::Number)
				{
					AddError(
						Current_,
						std::string("Expected ") + description);
					return false;
				}
				char* end = nullptr;
				output = std::strtof(Current_.Text.c_str(), &end);
				if (end == nullptr ||
					end != Current_.Text.c_str() +
						Current_.Text.size() ||
					!std::isfinite(output))
				{
					AddError(
						Current_,
						std::string("Invalid ") + description);
					return false;
				}
				Advance();
				return true;
			}

			bool ExpectKeyword(std::string_view keyword)
			{
				if (!ConsumeIdentifier(keyword))
				{
					AddError(
						Current_,
						"Expected '" + std::string(keyword) + "'");
					return false;
				}
				return true;
			}

			bool ExpectSemicolon()
			{
				if (!Consume(TokenKind::Semicolon))
				{
					AddError(Current_, "Expected ';'");
					Synchronize();
					return false;
				}
				return true;
			}

			void ParseVersion()
			{
				if (VersionDeclared_)
				{
					AddSemanticError("Duplicate shader version declaration");
				}
				VersionDeclared_ = true;
				if (!ExpectUnsigned(
						"a positive shader version",
						Result_.Program.Version))
				{
					Synchronize();
					return;
				}
				ExpectSemicolon();
			}

			bool ParseBindingFields(
				std::uint32_t& binding,
				std::uint32_t& vulkanBinding)
			{
				if (!ExpectKeyword("binding") ||
					!ExpectUnsigned("a binding index", binding) ||
					!ExpectKeyword("vulkan_binding") ||
					!ExpectUnsigned(
						"a Vulkan binding index",
						vulkanBinding))
				{
					Synchronize();
					return false;
				}
				return true;
			}

			void ParseConstantBuffer()
			{
				RealtimeShaderResource resource;
				resource.Kind =
					RealtimeShaderResourceKind::ConstantBuffer;
				if (!ExpectIdentifier(
						"a constant-buffer name",
						resource.Name) ||
					!ParseBindingFields(
						resource.Binding,
						resource.VulkanBinding) ||
					!ExpectKeyword("size"))
				{
					return;
				}
				std::uint32_t size = 0;
				if (!ExpectUnsigned("a constant-buffer size", size))
				{
					Synchronize();
					return;
				}
				resource.Size = size;
				if (ExpectSemicolon())
				{
					Result_.Program.Resources.push_back(
						std::move(resource));
				}
			}

			void ParseResource()
			{
				std::string kind;
				RealtimeShaderResource resource;
				if (!ExpectIdentifier("a resource kind", kind) ||
					!ExpectIdentifier(
						"a resource name",
						resource.Name))
				{
					Synchronize();
					return;
				}
				if (kind == "depth_texture")
				{
					resource.Kind =
						RealtimeShaderResourceKind::DepthTexture;
				}
				else if (kind == "texture")
				{
					resource.Kind =
						RealtimeShaderResourceKind::Texture;
				}
				else
				{
					AddError(
						Current_,
						"Unknown resource kind '" + kind + "'");
					Synchronize();
					return;
				}
				if (!ParseBindingFields(
						resource.Binding,
						resource.VulkanBinding))
				{
					return;
				}
				if (ConsumeIdentifier("colorspace"))
				{
					if (!ExpectIdentifier(
							"a resource color space",
							resource.ColorSpace))
					{
						Synchronize();
						return;
					}
				}
				if (ExpectSemicolon())
				{
					Result_.Program.Resources.push_back(
						std::move(resource));
				}
			}

			void ParseSampler()
			{
				RealtimeShaderResource resource;
				resource.Kind =
					RealtimeShaderResourceKind::Sampler;
				if (!ExpectIdentifier(
						"a sampler name",
						resource.Name) ||
					!ParseBindingFields(
						resource.Binding,
						resource.VulkanBinding) ||
					!ExpectKeyword("filter") ||
					!ExpectIdentifier(
						"a sampler filter",
						resource.Filter) ||
					!ExpectKeyword("address") ||
					!ExpectIdentifier(
						"a sampler address mode",
						resource.AddressMode))
				{
					Synchronize();
					return;
				}
				if (ExpectSemicolon())
				{
					Result_.Program.Resources.push_back(
						std::move(resource));
				}
			}

			void ParseStage()
			{
				std::string stageName;
				RealtimeShaderStageEntry stage;
				if (!ExpectIdentifier("a shader stage", stageName))
				{
					Synchronize();
					return;
				}
				if (stageName == "vertex")
				{
					stage.Stage = RealtimeShaderStage::Vertex;
				}
				else if (stageName == "fragment")
				{
					stage.Stage = RealtimeShaderStage::Fragment;
				}
				else if (stageName == "shadow")
				{
					stage.Stage = RealtimeShaderStage::Shadow;
				}
				else if (stageName == "outline")
				{
					stage.Stage = RealtimeShaderStage::Outline;
				}
				else
				{
					AddError(
						Current_,
						"Unknown shader stage '" + stageName + "'");
					Synchronize();
					return;
				}
				if (!ExpectKeyword("hlsl") ||
					!ExpectIdentifier(
						"an HLSL entry point",
						stage.HLSL) ||
					!ExpectKeyword("spirv") ||
					!ExpectIdentifier(
						"a SPIR-V entry point",
						stage.SPIRV) ||
					!ExpectKeyword("glsl_es") ||
					!ExpectIdentifier(
						"a GLSL ES entry point",
						stage.GLSLES) ||
					!ExpectKeyword("metal") ||
					!ExpectIdentifier(
						"a Metal entry point",
						stage.Metal))
				{
					Synchronize();
					return;
				}
				if (ExpectSemicolon())
				{
					Result_.Program.Stages.push_back(std::move(stage));
				}
			}

			void ParseOption()
			{
				std::string name;
				if (!ExpectIdentifier("an option name", name) ||
					!Consume(TokenKind::Equal))
				{
					AddError(Current_, "Expected '=' after the option name");
					Synchronize();
					return;
				}
				if (!Options_.insert(name).second)
				{
					AddSemanticError(
						"Duplicate shader option '" + name + "'");
				}
				if (name == "maximum_lights")
				{
					if (!ExpectUnsigned(
							"an integer maximum_lights value",
							Result_.Program.MaximumLights))
					{
						Synchronize();
						return;
					}
				}
				else if (name == "shadow_kernel_radius")
				{
					if (!ExpectUnsigned(
							"an integer shadow_kernel_radius value",
							Result_.Program.ShadowKernelRadius))
					{
						Synchronize();
						return;
					}
				}
				else if (name == "shadow_strength")
				{
					if (!ExpectFloat(
							"a numeric shadow_strength value",
							Result_.Program.ShadowStrength))
					{
						Synchronize();
						return;
					}
				}
				else if (name == "outline_space")
				{
					if (!ExpectIdentifier(
							"an outline_space value",
							Result_.Program.OutlineSpace))
					{
						Synchronize();
						return;
					}
				}
				else
				{
					AddError(
						Current_,
						"Unknown shader option '" + name + "'");
					Synchronize();
					return;
				}
				ExpectSemicolon();
			}

			void Validate()
			{
				if (Result_.Program.Name.empty())
				{
					AddSemanticError("The shader requires a name");
				}
				if (Result_.Program.Version == 0)
				{
					AddSemanticError(
						"The shader version must be greater than zero");
				}
				for (const char* requiredOption : {
					"maximum_lights",
					"shadow_kernel_radius",
					"shadow_strength",
					"outline_space"})
				{
					if (Options_.count(requiredOption) == 0)
					{
						AddSemanticError(
							std::string("Missing required shader option '") +
								requiredOption + "'");
					}
				}
				if (Result_.Program.MaximumLights == 0 ||
					Result_.Program.MaximumLights > 16)
				{
					AddSemanticError(
						"maximum_lights must be between 1 and 16");
				}
				if (Result_.Program.ShadowKernelRadius > 4)
				{
					AddSemanticError(
						"shadow_kernel_radius must be between 0 and 4");
				}
				if (!std::isfinite(Result_.Program.ShadowStrength) ||
					Result_.Program.ShadowStrength < 0.0f ||
					Result_.Program.ShadowStrength > 1.0f)
				{
					AddSemanticError(
						"shadow_strength must be between 0 and 1");
				}
				if (Result_.Program.OutlineSpace != "screen")
				{
					AddSemanticError(
						"outline_space must be 'screen'");
				}

				std::set<std::string> names;
				std::set<std::pair<int, std::uint32_t>> bindings;
				std::set<std::uint32_t> vulkanBindings;
				std::size_t constantBufferCount = 0;
				for (const RealtimeShaderResource& resource :
					Result_.Program.Resources)
				{
					if (!names.insert(resource.Name).second)
					{
						AddSemanticError(
							"Duplicate resource name '" +
								resource.Name + "'");
					}
					int bindingSpace = 1;
					if (resource.Kind ==
						RealtimeShaderResourceKind::ConstantBuffer)
					{
						bindingSpace = 0;
						++constantBufferCount;
					}
					else if (resource.Kind ==
						RealtimeShaderResourceKind::Sampler)
					{
						bindingSpace = 2;
					}
					if (!bindings.insert({
							bindingSpace,
							resource.Binding}).second)
					{
						AddSemanticError(
							"Duplicate API binding " +
								std::to_string(resource.Binding));
					}
					if (!vulkanBindings.insert(
							resource.VulkanBinding).second)
					{
						AddSemanticError(
							"Duplicate Vulkan binding " +
								std::to_string(
									resource.VulkanBinding));
					}
					if (resource.Kind ==
							RealtimeShaderResourceKind::ConstantBuffer &&
						resource.Size == 0)
					{
						AddSemanticError(
							"Constant buffers require a nonzero size");
					}
					if ((resource.Kind ==
							RealtimeShaderResourceKind::Texture ||
							resource.Kind ==
								RealtimeShaderResourceKind::DepthTexture) &&
						resource.ColorSpace != "linear" &&
						resource.ColorSpace != "srgb")
					{
						AddSemanticError(
							"Texture resource '" + resource.Name +
								"' requires colorspace linear or srgb");
					}
				}
				if (constantBufferCount != 1)
				{
					AddSemanticError(
						"The realtime shader requires exactly one constant buffer");
				}

				std::set<RealtimeShaderStage> stages;
				for (const RealtimeShaderStageEntry& stage :
					Result_.Program.Stages)
				{
					if (!stages.insert(stage.Stage).second)
					{
						AddSemanticError("Duplicate shader stage");
					}
				}
				for (RealtimeShaderStage required : {
					RealtimeShaderStage::Vertex,
					RealtimeShaderStage::Fragment,
					RealtimeShaderStage::Shadow,
					RealtimeShaderStage::Outline})
				{
					if (stages.count(required) == 0)
					{
						AddSemanticError(
							"Missing required shader stage");
					}
				}
			}

			void Synchronize()
			{
				while (Current_.Kind != TokenKind::End &&
					Current_.Kind != TokenKind::RightBrace &&
					Current_.Kind != TokenKind::Semicolon)
				{
					Advance();
				}
				if (Current_.Kind == TokenKind::Semicolon)
				{
					Advance();
				}
			}

			void AddError(const Token& token, std::string message)
			{
				Result_.Diagnostics.push_back({
					token.Line,
					token.Column,
					std::move(message) });
			}

			void AddSemanticError(std::string message)
			{
				Result_.Diagnostics.push_back({
					1,
					1,
					std::move(message) });
			}

			Lexer Lexer_;
			Token Current_;
			RealtimeShaderParseResult Result_;
			bool VersionDeclared_ = false;
			std::set<std::string> Options_;
		};

		const char* StageName(RealtimeShaderStage stage)
		{
			switch (stage)
			{
			case RealtimeShaderStage::Vertex:
				return "vertex";
			case RealtimeShaderStage::Fragment:
				return "fragment";
			case RealtimeShaderStage::Shadow:
				return "shadow";
			case RealtimeShaderStage::Outline:
				return "outline";
			}
			return "unknown";
		}

		std::string FloatLiteral(float value)
		{
			std::ostringstream stream;
			stream.imbue(std::locale::classic());
			stream << std::fixed << std::setprecision(6) << value;
			std::string literal = stream.str();
			while (literal.size() > 2 &&
				literal.back() == '0' &&
				literal[literal.size() - 2] != '.')
			{
				literal.pop_back();
			}
			return literal;
		}

		bool IsIdentifierCharacter(char value)
		{
			return
				(value >= 'a' && value <= 'z') ||
				(value >= 'A' && value <= 'Z') ||
				(value >= '0' && value <= '9') ||
				value == '_';
		}

		bool ContainsIdentifier(
			std::string_view source,
			std::string_view identifier)
		{
			std::size_t position = 0;
			while ((position = source.find(identifier, position)) !=
				std::string_view::npos)
			{
				const bool leftBoundary =
					position == 0 ||
					!IsIdentifierCharacter(source[position - 1]);
				const std::size_t end = position + identifier.size();
				const bool rightBoundary =
					end == source.size() ||
					!IsIdentifierCharacter(source[end]);
				if (leftBoundary && rightBoundary)
				{
					return true;
				}
				position = end;
			}
			return false;
		}
	}

	const std::string& RealtimeShaderStageEntry::Entry(
		RealtimeShaderLanguage language) const
	{
		switch (language)
		{
		case RealtimeShaderLanguage::HLSL:
			return HLSL;
		case RealtimeShaderLanguage::GLSLES:
			return GLSLES;
		case RealtimeShaderLanguage::Metal:
			return Metal;
		case RealtimeShaderLanguage::SPIRV:
			return SPIRV;
		}
		return HLSL;
	}

	const RealtimeShaderResource* RealtimeShaderProgram::FindResourceByName(
		std::string_view name) const
	{
		const auto found = std::find_if(
			Resources.begin(),
			Resources.end(),
			[name](const RealtimeShaderResource& resource)
			{
				return resource.Name == name;
			});
		return found == Resources.end() ? nullptr : &*found;
	}

	const RealtimeShaderStageEntry* RealtimeShaderProgram::FindStage(
		RealtimeShaderStage stage) const
	{
		const auto found = std::find_if(
			Stages.begin(),
			Stages.end(),
			[stage](const RealtimeShaderStageEntry& entry)
			{
				return entry.Stage == stage;
			});
		return found == Stages.end() ? nullptr : &*found;
	}

	std::string RealtimeShaderParseResult::ErrorText() const
	{
		std::ostringstream stream;
		for (std::size_t index = 0;
			index < Diagnostics.size();
			++index)
		{
			const RealtimeShaderDiagnostic& diagnostic =
				Diagnostics[index];
			if (index != 0)
			{
				stream << '\n';
			}
			stream << diagnostic.Line << ':' << diagnostic.Column
				<< ": " << diagnostic.Message;
		}
		return stream.str();
	}

	RealtimeShaderParseResult ParseRealtimeShader(
		std::string_view source)
	{
		return Parser(source).Parse();
	}

	const char* BuiltinRealtimeShaderManifest()
	{
		return R"CSL(
// Evans Shader Language: one contract shared by every realtime backend.
shader EvansRealtime {
    version 1;

    constant_buffer DrawConstants binding 0 vulkan_binding 0 size 672;
    resource depth_texture ShadowMap binding 0 vulkan_binding 1 colorspace linear;
    resource texture DiffuseMap binding 1 vulkan_binding 2 colorspace srgb;
    resource texture NormalMap binding 2 vulkan_binding 3 colorspace linear;
    resource texture OpacityMap binding 3 vulkan_binding 4 colorspace linear;
    sampler SurfaceSampler binding 0 vulkan_binding 5 filter linear address repeat;

    stage vertex hlsl VSMain spirv VSMain glsl_es main metal EvansVertexMain;
    stage fragment hlsl PSMain spirv PSMain glsl_es main metal EvansFragmentMain;
    stage shadow hlsl PSShadow spirv PSShadow glsl_es main metal EvansShadowFragment;
    stage outline hlsl PSOutline spirv PSOutline glsl_es main metal EvansFragmentMain;

    option maximum_lights = 4;
    option shadow_kernel_radius = 1;
    option shadow_strength = 0.55;
    option outline_space = screen;
}
)CSL";
	}

	const RealtimeShaderParseResult& BuiltinRealtimeShader()
	{
		static const RealtimeShaderParseResult result =
			ParseRealtimeShader(BuiltinRealtimeShaderManifest());
		return result;
	}

	bool ValidateRealtimeShaderContract(
		const RealtimeShaderProgram& program,
		std::size_t constantBufferSize,
		std::uint32_t maximumLights,
		std::string& error)
	{
		const auto requireResource = [&program, &error](
			const char* name,
			RealtimeShaderResourceKind kind,
			std::uint32_t binding,
			std::uint32_t vulkanBinding,
			const char* colorSpace)
		{
			const RealtimeShaderResource* resource =
				program.FindResourceByName(name);
			if (resource == nullptr ||
				resource->Kind != kind ||
				resource->Binding != binding ||
				resource->VulkanBinding != vulkanBinding ||
				(colorSpace != nullptr &&
					resource->ColorSpace != colorSpace))
			{
				error =
					std::string("Realtime shader resource contract mismatch: ") +
					name;
				return false;
			}
			return true;
		};
		if (program.MaximumLights != maximumLights)
		{
			error =
				"Realtime shader maximum_lights does not match the renderer ABI";
			return false;
		}
		const RealtimeShaderResource* constants =
			program.FindResourceByName("DrawConstants");
		if (constants == nullptr ||
			constants->Kind !=
				RealtimeShaderResourceKind::ConstantBuffer ||
			constants->Binding != 0 ||
			constants->VulkanBinding != 0 ||
			constants->Size != constantBufferSize)
		{
			error =
				"Realtime shader DrawConstants does not match the renderer ABI";
			return false;
		}
		if (!requireResource(
				"ShadowMap",
				RealtimeShaderResourceKind::DepthTexture,
				0,
				1,
				"linear") ||
			!requireResource(
				"DiffuseMap",
				RealtimeShaderResourceKind::Texture,
				1,
				2,
				"srgb") ||
			!requireResource(
				"NormalMap",
				RealtimeShaderResourceKind::Texture,
				2,
				3,
				"linear") ||
			!requireResource(
				"OpacityMap",
				RealtimeShaderResourceKind::Texture,
				3,
				4,
				"linear") ||
			!requireResource(
				"SurfaceSampler",
				RealtimeShaderResourceKind::Sampler,
				0,
				5,
				nullptr))
		{
			return false;
		}
		for (RealtimeShaderStage stage : {
			RealtimeShaderStage::Vertex,
			RealtimeShaderStage::Fragment,
			RealtimeShaderStage::Shadow,
			RealtimeShaderStage::Outline})
		{
			if (program.FindStage(stage) == nullptr)
			{
				error =
					std::string("Realtime shader is missing the ") +
					StageName(stage) + " stage";
				return false;
			}
		}
		if (program.OutlineSpace != "screen")
		{
			error =
				"Realtime shader outline_space does not match the renderer";
			return false;
		}
		error.clear();
		return true;
	}

	bool MaterializeRealtimeShader(
		const RealtimeShaderProgram& program,
		RealtimeShaderLanguage language,
		std::string_view sourceTemplate,
		const std::vector<RealtimeShaderStage>& requiredStages,
		std::string& output,
		std::string& error)
	{
		const std::uint32_t kernelWidth =
			program.ShadowKernelRadius * 2U + 1U;
		const std::uint32_t sampleCount =
			kernelWidth * kernelWidth;
		std::unordered_map<std::string, std::string> values = {
			{"MAX_LIGHTS", std::to_string(program.MaximumLights)},
			{"SHADOW_KERNEL_RADIUS",
				std::to_string(program.ShadowKernelRadius)},
			{"SHADOW_MIN_OFFSET",
				"-" + std::to_string(program.ShadowKernelRadius)},
			{"SHADOW_MAX_OFFSET",
				std::to_string(program.ShadowKernelRadius)},
			{"SHADOW_SAMPLE_COUNT", std::to_string(sampleCount)},
			{"SHADOW_STRENGTH",
				FloatLiteral(program.ShadowStrength)} };
		for (const RealtimeShaderResource& resource :
			program.Resources)
		{
			std::string key;
			key.reserve(resource.Name.size() + 16);
			for (char value : resource.Name)
			{
				if (value >= 'a' && value <= 'z')
				{
					key.push_back(
						static_cast<char>(value - 'a' + 'A'));
				}
				else
				{
					key.push_back(value);
				}
			}
			values.emplace(
				key + "_BINDING",
				std::to_string(resource.Binding));
			values.emplace(
				key + "_VULKAN_BINDING",
				std::to_string(resource.VulkanBinding));
			if (resource.Kind ==
				RealtimeShaderResourceKind::ConstantBuffer)
			{
				values.emplace(
					key + "_SIZE",
					std::to_string(resource.Size));
			}
		}
		for (const RealtimeShaderStageEntry& stage : program.Stages)
		{
			std::string key = StageName(stage.Stage);
			std::transform(
				key.begin(),
				key.end(),
				key.begin(),
				[](char value)
				{
					return static_cast<char>(
						value >= 'a' && value <= 'z'
							? value - 'a' + 'A'
							: value);
				});
			values.emplace(
				key + "_ENTRY",
				stage.Entry(language));
		}

		output.assign(sourceTemplate);
		std::size_t position = 0;
		while ((position = output.find("{{", position)) !=
			std::string::npos)
		{
			const std::size_t end = output.find("}}", position + 2);
			if (end == std::string::npos)
			{
				error = "Realtime shader template has an unterminated placeholder";
				return false;
			}
			const std::string key =
				output.substr(position + 2, end - position - 2);
			const auto value = values.find(key);
			if (value == values.end())
			{
				error =
					"Realtime shader template has unknown placeholder {{" +
					key + "}}";
				return false;
			}
			output.replace(
				position,
				end + 2 - position,
				value->second);
			position += value->second.size();
		}

		for (RealtimeShaderStage stage : requiredStages)
		{
			const RealtimeShaderStageEntry* entry =
				program.FindStage(stage);
			if (entry == nullptr ||
				!ContainsIdentifier(
					output,
					entry->Entry(language)))
			{
				error =
					std::string("Realtime shader template is missing the ") +
					StageName(stage) + " entry point";
				return false;
			}
		}
		error.clear();
		return true;
	}

	bool PrepareBuiltinRealtimeShader(
		RealtimeShaderLanguage language,
		std::string_view sourceTemplate,
		const std::vector<RealtimeShaderStage>& requiredStages,
		std::size_t constantBufferSize,
		std::uint32_t maximumLights,
		std::string& output,
		std::string& error)
	{
		const RealtimeShaderParseResult& shader =
			BuiltinRealtimeShader();
		if (!shader)
		{
			error =
				"Built-in realtime shader parse failed:\n" +
				shader.ErrorText();
			return false;
		}
		if (!ValidateRealtimeShaderContract(
				shader.Program,
				constantBufferSize,
				maximumLights,
				error))
		{
			return false;
		}
		return MaterializeRealtimeShader(
			shader.Program,
			language,
			sourceTemplate,
			requiredStages,
			output,
			error);
	}
}
