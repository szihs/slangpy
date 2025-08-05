[![docs][1]][2] [![ci][3]][4] [![pypi][5]][6]

# SlangPy

[1]: https://readthedocs.org/projects/slangpy/badge/?version=latest
[2]: https://slangpy.readthedocs.io/en/latest/
[3]: https://github.com/shader-slang/slangpy/actions/workflows/ci.yml/badge.svg
[4]: https://github.com/shader-slang/slangpy/actions/workflows/ci.yml
[5]: https://img.shields.io/pypi/v/slangpy.svg?color=green
[6]: https://pypi.org/pypi/slangpy

## Introduction

SlangPy is a cross-platform library designed to make calling GPU code written in Slang extremely simple and easy. It's core objectives are to:
- Make it quick and simple to call Slang functions on the GPU from Python
- Eradicate the boilerplate and bugs associated with writing compute kernels
- Grant easy access to Slang's auto-diff features
- Provide optional PyTorch support out of the box

## Documentation

See the [documentation][2] for more detailed information and examples.

More information about Slang in general can be found [here](https://shader-slang.com).

## Installation

SlangPy is available as pre-compiled wheels via PyPi. Installing SlangPy is as simple as running:

```bash
pip install slangpy
```

To enable PyTorch integration, simply ``pip install pytorch`` as usual and it will be detected automatically by SlangPy.

You can also compile SlangPy from source:

```bash
git clone https://github.com/shader-slang/slangpy.git --recursive
cd slangpy
pip install .
```

## GPU Backend Dependencies

SlangPy supports multiple GPU backends and will automatically use the best available option on your system. The following external dependencies are optional but recommended for optimal performance:

### CUDA Backend (Windows/Linux)

For CUDA GPU acceleration on Windows and Linux systems:

- **CUDA Toolkit**: Version 11.8 or later (12.8 recommended)
- **Platforms**: Windows and Linux only
- **Installation**: CUDA toolkit is **not required** for SlangPy installation. SlangPy uses dynamic loading and will automatically detect and use CUDA if available at runtime.
- **Fallback**: If CUDA is not available, SlangPy automatically falls back to Vulkan or D3D12 backends.

### Metal Backend (macOS)

For Metal GPU acceleration on macOS systems:

- **Metal SDK**: No specific version requirements beyond macOS system requirements
- **macOS Version**: 13.0 (Ventura) or later required
- **Xcode**: Required for development (Xcode 16+ recommended), but not needed for using pre-built SlangPy wheels
- **Fallback**: Vulkan backend is available as an alternative on macOS.

### Other Backends

- **Vulkan**: Available on all platforms (Windows, Linux, macOS) - no additional dependencies required
- **D3D12**: Available on Windows - no additional dependencies required

**Note**: SlangPy will work without any of these external dependencies installed, using the most appropriate available GPU backend for your system.

## License

SlangPy source code is licensed under the Apache-2.0 License - see the [LICENSE.txt](LICENSE.txt) file for details.

SlangPy depends on the following third-party libraries, which have their own license:

- [argparse](https://github.com/p-ranav/argparse) (MIT)
- [AsmJit](https://github.com/asmjit/asmjit) (Zlib)
- [BS::thread-pool](https://github.com/bshoshany/thread-pool) (MIT)
- [Dear ImGui](https://github.com/ocornut/imgui) (MIT)
- [doctest](https://github.com/doctest/doctest) (MIT)
- [fmt](https://fmt.dev/latest/index.html) (MIT)
- [glfw3](https://www.glfw.org/) (Zlib)
- [libjpeg-turbo](https://libjpeg-turbo.org/) (BSD)
- [libpng](http://www.libpng.org/pub/png/libpng.html) (libpng)
- [nanobind](https://github.com/wjakob/nanobind) (BSD)
- [NVAPI](https://github.com/NVIDIA/nvapi) (MIT)
- [OpenEXR](https://openexr.com/en/latest/) (BSD)
- [pugixml](https://pugixml.org/) (MIT)
- [RenderDoc API](https://github.com/baldurk/renderdoc) (MIT)
- [Slang](https://github.com/shader-slang/slang) (MIT)
- [stb](https://github.com/nothings/stb) (MIT)
- [tevclient](https://github.com/skallweitNV/tevclient) (BSD)
- [tinyexr](https://github.com/syoyo/tinyexr) (BSD)
- [vcpkg](https://vcpkg.io/en/) (MIT)
- [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) (MIT)

SlangPy releases additionally include pre-built binaries of the following third-party components, which have their own license:

- [DirectXShaderCompiler](https://github.com/microsoft/DirectXShaderCompiler) (LLVM Release License)
- [Agility SDK](https://devblogs.microsoft.com/directx/directx12agility) (MICROSOFT DIRECTX License)

## Citation

If you use SlangPy in a research project leading to a publication, please cite the project. The BibTex entry is:

```bibtex
@software{slangpy,
    title = {SlangPy},
    author = {Simon Kallweit and Chris Cummings and Benedikt Bitterli and Sai Bangaru and Yong He},
    note = {https://github.com/shader-slang/slangpy},
    version = {0.31.0},
    year = 2025
}
```
