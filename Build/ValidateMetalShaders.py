#!/usr/bin/env python3
"""Compile every Metal shader with every preprocessor variant the renderer injects.

MetalRHI compiles each technique shader as a standalone translation unit and
prepends `#define` lines to select a variant (shading model, shadow filter,
composite/moments, ...). A variant that fails to compile stalls the frame, so
this harness mirrors the injection sites in Src/Renderer/Metal and reports every
failing (file, macro set) pair.
"""
import itertools
import os
import tempfile
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHADERS = ROOT / "Src/Shaders/Metal"
OUT = Path(tempfile.gettempdir()) / "evans-metal-variants"
FLAGS = ["-c", "-O3", "-fno-fast-math", "-fpreserve-invariance"]

MATERIAL_FILES = ["BasicMaterials", "TextureMapping", "NormalMapping", "AlphaTesting", "AlphaBlending"]
MODELS = range(5)          # ShadingModel: Phong, Flat, Gouraud, PBR, Toon
PROGRAMS = range(8)        # deferred/screen program ids used by PCF/PCSS/VSM/RayShadow
FILTERS = range(4)         # ShadowFilterMode: Hard, PCF, PCSS, VSM


def variants(name):
    """Yield (label, [macro=value, ...]) pairs for one shader file."""
    stem = Path(name).stem
    # Material technique shaders are always compiled with an injected program id.
    if stem not in MATERIAL_FILES:
        yield ("default", [])
    if stem in MATERIAL_FILES:
        for model, gbuffer in itertools.product(MODELS, (0, 1)):
            macros = [f"EVANS_MATERIAL_PROGRAM={model}"]
            if gbuffer:
                macros.append("EVANS_MATERIAL_GBUFFER=1")
            yield (f"material model={model} gbuffer={gbuffer}", macros)
    if stem in ("ForwardPlus", "ClusteredForward"):
        prefix = "EVANS_FORWARD_PLUS" if stem == "ForwardPlus" else "EVANS_CLUSTERED_FORWARD"
        for f, contact, multi in itertools.product(FILTERS, (0, 1), (0, 1)):
            if multi and not contact:
                continue
            yield (f"surface filter={f} contact={contact} multisample={multi}",
                   [f"{prefix}_SURFACE=1", f"{prefix}_FILTER={f}",
                    f"EVANS_CONTACT_SHADOWS={contact}", f"EVANS_CONTACT_MULTISAMPLE={multi}"])
    if stem == "TiledDeferred":
        for f, contact, multi in itertools.product(FILTERS, (0, 1), (0, 1)):
            if multi and not contact:
                continue
            yield (f"tiled filter={f} contact={contact} multisample={multi}",
                   ["EVANS_TILED_DEFERRED_SURFACE=1", "EVANS_TILED_DEFERRED_LIGHTING=1",
                    f"EVANS_TILED_DEFERRED_FILTER={f}",
                    f"EVANS_CONTACT_SHADOWS={contact}", f"EVANS_CONTACT_MULTISAMPLE={multi}"])
    program_macro = {"PCF": "EVANS_PCF_PROGRAM", "PCSS": "EVANS_PCSS_PROGRAM",
                     "VSM": "EVANS_VSM_PROGRAM", "RayTracedShadows": "EVANS_RAY_SHADOW_PROGRAM"}.get(stem)
    if program_macro:
        for p in PROGRAMS:
            yield (f"program={p}", [f"{program_macro}={p}"])
    composite = {"ParticipatingMedia": ("EVANS_MEDIA_COMPOSITE", "EVANS_MEDIA_MOMENTS"),
                 "VolumetricClouds": ("EVANS_CLOUD_COMPOSITE", "EVANS_CLOUD_MOMENTS")}.get(stem)
    if composite:
        yield ("composite", [f"{composite[0]}=1"])
        yield ("moments", [f"{composite[1]}=1"])
    if stem == "Lumen":
        # MetalLumenHardwareRayTracing.inl compiles this file at MTLLanguageVersion2_4.
        yield ("hardware rays", ["EVANS_LUMEN_HARDWARE_RAYS=1"], "2.4")
    if stem == "OIT":
        yield ("capture", ["EVANS_OIT_CAPTURE=1"])
        yield ("accumulate", ["EVANS_OIT_ACCUMULATE=1"])
    if stem == "Refraction":
        yield ("capture", ["EVANS_REFRACTION_CAPTURE=1"])
    if stem == "HairFur":
        for macro in ("EVANS_HAIR_VELOCITY", "EVANS_HAIR_CAPTURE", "EVANS_HAIR_VSM_SHADOW",
                      "EVANS_HAIR_DEPTH_ONLY", "EVANS_HAIR_MOMENT_RECEIVER",
                      "EVANS_HAIR_CONTACT_RECEIVER", "EVANS_HAIR_CONTACT_SAMPLES"):
            yield (macro, [f"{macro}=1"])
        # Contact receivers are emitted with or without per-sample coverage, and a
        # moment receiver can coincide with both (MetalHairFur.inl).
        yield ("capture+contact", ["EVANS_HAIR_CAPTURE=1", "EVANS_HAIR_CONTACT_RECEIVER=1"])
        yield ("capture+contact samples",
               ["EVANS_HAIR_CAPTURE=1", "EVANS_HAIR_CONTACT_RECEIVER=1", "EVANS_HAIR_CONTACT_SAMPLES=1"])
        yield ("moment+contact samples",
               ["EVANS_HAIR_MOMENT_RECEIVER=1", "EVANS_HAIR_CONTACT_RECEIVER=1",
                "EVANS_HAIR_CONTACT_SAMPLES=1"])


def compile_one(job):
    path, label, macros, index, std = job
    source = path
    if macros:
        OUT.mkdir(parents=True, exist_ok=True)
        prelude = "".join(f"#define {m.split('=')[0]} {m.split('=')[1]}\n" for m in macros)
        source = OUT / f"{path.stem}_{index}.metal"
        source.write_text(prelude + path.read_text())
    air = OUT / f"{path.stem}_{index}.air"
    # MetalMeshShaders.inl compiles the mesh stage at MTLLanguageVersion3_0.
    version = std or ("3.0" if path.stem == "MeshShaders" else "2.1")
    # 3.0 has no macos- prefix; 1.x-2.4 are spelled macos-metalX.Y.
    spelling = f"metal{version}" if version.startswith("3.") else f"macos-metal{version}"
    flags = [*FLAGS, f"-std={spelling}"]
    cmd = ["xcrun", "--sdk", "macosx", "metal", *flags, str(source), "-o", str(air)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Catch diagnostics that carry no file:line prefix (e.g. an unsupported -std).
    errors = [line.replace(str(source), f"{path.name}") for line in result.stderr.splitlines()
              if "error:" in line]
    return (path.name, label, errors)


def main():
    only = set(sys.argv[1:])
    jobs, index = [], 0
    for path in sorted(SHADERS.glob("*.metal")):
        if only and path.stem not in only:
            continue
        for variant in variants(path.name):
            label, macros = variant[0], variant[1]
            std = variant[2] if len(variant) > 2 else None
            jobs.append((path, label, macros, index, std))
            index += 1
    print(f"{len(jobs)} variant(s) across {len({j[0] for j in jobs})} shader(s)")
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        results = list(pool.map(compile_one, jobs))
    failures = [r for r in results if r[2]]
    for name, label, errors in failures:
        print(f"\nFAIL {name} [{label}]")
        for line in errors[:5]:
            print("   " + line)
    print(f"\n{len(results) - len(failures)}/{len(results)} variants compiled")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
