param(
    [string]$NvccPath,
    [string]$ClPath
)

$ErrorActionPreference = 'Stop'

$buildDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = (Resolve-Path (Join-Path $buildDirectory '..')).Path
$utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
$newline = "`n"

function Resolve-NvccPath {
    param([string]$RequestedPath)

    if ($RequestedPath) {
        return (Resolve-Path $RequestedPath).Path
    }
    $command = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        throw 'Could not find nvcc on PATH. Pass -NvccPath.'
    }
    return $command.Source
}

function Resolve-ClPath {
    param([string]$RequestedPath)

    if ($RequestedPath) {
        return (Resolve-Path $RequestedPath).Path
    }
    $visualStudioRoot = 'C:\Program Files\Microsoft Visual Studio\2022'
    $candidates = @(Get-ChildItem -LiteralPath $visualStudioRoot -Directory -ErrorAction SilentlyContinue |
        ForEach-Object {
            Get-ChildItem -LiteralPath (Join-Path $_.FullName 'VC\Tools\MSVC') `
                -Directory -ErrorAction SilentlyContinue
        } |
        Sort-Object Name -Descending |
        ForEach-Object { Join-Path $_.FullName 'bin\Hostx64\x64\cl.exe' } |
        Where-Object { Test-Path -LiteralPath $_ })
    if ($candidates.Count -eq 0) {
        throw 'Could not find the x64 MSVC compiler required by nvcc. Pass -ClPath.'
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

$resolvedNvcc = Resolve-NvccPath $NvccPath
$resolvedCl = Resolve-ClPath $ClPath
$temporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
    'EvansRasterShaders-' + [Guid]::NewGuid().ToString('N'))
[void](New-Item -ItemType Directory -Path $temporaryRoot)

try {
    $cudaSource = Join-Path $repositoryRoot 'Src\Rasterizer\CudaRasterKernel.cu'
    $ptxPath = Join-Path $temporaryRoot 'CudaRasterKernel.ptx'
    Invoke-Compiler $resolvedNvcc @(
        '--ptx',
        '--std=c++17',
        '-O3',
        '--gpu-architecture=compute_75',
        '-ccbin', (Split-Path -Parent $resolvedCl),
        '-o', $ptxPath,
        $cudaSource)
    $ptx = [System.IO.File]::ReadAllText($ptxPath).Replace("`r`n", "`n").Replace("`r", "`n")
    $ptx = $ptx.TrimEnd("`n")
    foreach ($entryPoint in @('TransformKernel', 'SetupKernel', 'ShadowDepthKernel', 'RasterKernel', 'OutlineKernel')) {
        if ($ptx -notmatch ('(?m)^\.visible \.entry ' + [regex]::Escape($entryPoint) + '\(')) {
            throw "Generated PTX does not contain the required $entryPoint entry point."
        }
    }
    # CUDA 13.3 labels this output PTX 9.3 although this kernel uses an older
    # instruction subset. Preserve compatibility with currently supported drivers.
    $ptx = [regex]::Replace($ptx, '(?m)^\.version\s+\S+', '.version 8.8', 1)
    $ptx += $newline

    $output = @(
        '#pragma once',
        '',
        'namespace Evans::Rasterizer::Detail',
        '{',
        '    // Generated from CudaRasterKernel.cu with the CUDA Toolkit PTX compiler.',
        '    // The embedded copy declares PTX 8.8 for current driver compatibility.',
        '    inline constexpr const char* CudaRasterPtxChunks[] =',
        '    {'
    )
    # MSVC limits an individual string literal even when the complete PTX is
    # assembled dynamically. Stay well below that implementation limit.
    $maximumPtxChunkLength = 8000
    for ($offset = 0; $offset -lt $ptx.Length; $offset += $maximumPtxChunkLength) {
        $length = [Math]::Min($maximumPtxChunkLength, $ptx.Length - $offset)
        $output += '        R"PTX(' + $ptx.Substring($offset, $length) + ')PTX",'
    }
    $output += '    };'
    $output += '}'

    Write-TextFile (
        Join-Path $repositoryRoot 'Src\Rasterizer\CudaRasterPtx.h') $output
    Write-Host "Regenerated raster shaders with:`n  $resolvedNvcc`n  $resolvedCl"
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
