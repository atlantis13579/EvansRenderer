param(
    [string]$DxcPath
)

$ErrorActionPreference = 'Stop'

$buildDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = (Resolve-Path (Join-Path $buildDirectory '..')).Path
$utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
$newline = "`n"

function Resolve-DxcPath {
    param([string]$RequestedPath)

    if ($RequestedPath) {
        return (Resolve-Path $RequestedPath).Path
    }
    $kitsBin = 'C:\Program Files (x86)\Windows Kits\10\bin'
    $candidates = @(Get-ChildItem -LiteralPath $kitsBin -Directory -ErrorAction SilentlyContinue |
        ForEach-Object { Join-Path $_.FullName 'x64\dxc.exe' } |
        Where-Object { Test-Path -LiteralPath $_ } |
        Sort-Object -Descending)
    if ($candidates.Count -eq 0) {
        throw 'Could not find an x64 DirectX Shader Compiler. Pass -DxcPath.'
    }
    return $candidates[0]
}

function Invoke-Compiler {
    param(
        [string]$Compiler,
        [string[]]$Arguments
    )

    & $Compiler @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Shader compiler failed with exit code $LASTEXITCODE`: $Compiler"
    }
}

function Format-ByteLines {
    param([byte[]]$Bytes)

    $result = New-Object System.Collections.Generic.List[string]
    for ($offset = 0; $offset -lt $Bytes.Length; $offset += 16) {
        $count = [Math]::Min(16, $Bytes.Length - $offset)
        $values = for ($index = 0; $index -lt $count; ++$index) {
            '0x{0:x2}' -f $Bytes[$offset + $index]
        }
        $result.Add('        ' + ($values -join ', ') + ',')
    }
    return $result
}

function Write-TextFile {
    param(
        [string]$Path,
        [string[]]$Lines
    )

    [System.IO.File]::WriteAllText(
        $Path,
        ($Lines -join $newline) + $newline,
        $utf8WithoutBom)
}

$resolvedDxc = Resolve-DxcPath $DxcPath
$temporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
    'EvansRayShaders-' + [Guid]::NewGuid().ToString('N'))
[void](New-Item -ItemType Directory -Path $temporaryRoot)

try {
    $dxrCppPath = Join-Path $repositoryRoot 'Src\Renderer\DX12\DxrRayTracingRHI.cpp'
    $dxrCpp = [System.IO.File]::ReadAllText($dxrCppPath)
    $dxrPattern = 'constexpr const char\* DxrShaderSource = R"DXR\(\r?\n(?<shader>.*?)\r?\n\)DXR";'
    $dxrMatch = [regex]::Match(
        $dxrCpp,
        $dxrPattern,
        [System.Text.RegularExpressions.RegexOptions]::Singleline)
    if (-not $dxrMatch.Success) {
        throw 'Could not extract DxrShaderSource from Renderer/DX12/DxrRayTracingRHI.cpp.'
    }
    $dxrShader = [regex]::Replace(
        $dxrMatch.Groups['shader'].Value,
        '\r?\n\)DXR"\s*R"DXR\(\r?\n',
        $newline)
    $dxrSourcePath = Join-Path $temporaryRoot 'DxrShader.hlsl'
    [System.IO.File]::WriteAllText(
        $dxrSourcePath,
        $dxrShader + $newline,
        $utf8WithoutBom)
    $dxrBinaryPath = Join-Path $temporaryRoot 'DxrShader.dxil'
    Invoke-Compiler $resolvedDxc @(
        '-T', 'lib_6_3',
        '-HV', '2021',
        '-O3',
        '-Qstrip_debug',
        '-Fo', $dxrBinaryPath,
        $dxrSourcePath)
    $dxrBytes = [System.IO.File]::ReadAllBytes($dxrBinaryPath)
    $dxrOutput = @(
        '#pragma once',
        '',
        '#include <array>',
        '#include <cstdint>',
        '',
        'namespace Evans',
        '{',
        'namespace RayTracing',
        '{',
        'namespace Detail',
        '{',
        '    // Generated from the renderer-owned lib_6_3 DXR source with DirectX Shader Compiler.',
        ('    inline constexpr std::array<std::uint8_t, {0}> DxrShaderBytecode = {{' -f $dxrBytes.Length)
    )
    $dxrOutput += Format-ByteLines $dxrBytes
    $dxrOutput += '    };'
    $dxrOutput += '}'
    $dxrOutput += '}'
    $dxrOutput += '}'
    Write-TextFile (Join-Path $repositoryRoot 'Src\Renderer\DX12\DxrShaderBytecode.h') $dxrOutput
    Write-Host "Regenerated DXR shaders with $resolvedDxc"
}
finally {
    $systemTemporaryRoot = [System.IO.Path]::GetFullPath(
        [System.IO.Path]::GetTempPath())
    $resolvedTemporaryRoot = [System.IO.Path]::GetFullPath($temporaryRoot)
    if ($resolvedTemporaryRoot.StartsWith(
            $systemTemporaryRoot,
            [System.StringComparison]::OrdinalIgnoreCase)) {
        [System.IO.Directory]::Delete($resolvedTemporaryRoot, $true)
    }
}
