Serpens SpMV Accelerator — Optimized for Xilinx Alveo U280

Project: High-performance Sparse Matrix–Vector multiply (SpMV) accelerator optimized for the Xilinx Alveo U280.

This repository contains the host code, FPGA bitstreams and helper scripts needed to build and run the Serpens SpMV accelerator. Presentation slides and the project report were removed from this fork to keep the repository focused and production-ready.

Repository layout
- `src/` : FPGA kernel sources and host helper headers.
- `scripts/` : Host and utility programs (e.g. `host_xrt.cpp`, `spmvgen.py`, run helpers).
- `bitstreams/` : Prebuilt FPGA binaries (`.xclbin`, `.xo`) for deployment.
- `docs/` : Short docs and run notes (this file includes the main run instructions).

Prerequisites
- Xilinx Vitis / Vivado toolchain (for rebuilding kernels) and XRT (runtime) for target U280.
- A Linux build host (recommended) or WSL on Windows with required Xilinx toolchain installed.
- `g++` (or `clang++`) for host compilation, Python 3 for scripts.
- Access to an Alveo U280 or appropriate emulation environment.

Quick start  Build host and run (recommended: Linux / WSL)

1. Prepare environment
 - Ensure XRT is installed and the `xbutil` tool is available. For U280, set up the target and user permissions.

2. Build the host application

```bash
cd scripts
# Example: compile on Linux (adjust include/lib paths for your XRT install)
g++ -std=c++14 -O2 -I/opt/xilinx/xrt/include -o host_xrt host_xrt.cpp -lOpenCL
```

3. Deploy bitstream to the board

 - Copy `bitstreams/Serpens.xclbin` to the host that manages the U280 and program the card using your usual tooling (for example, `xbutil program` on systems with XRT):

```bash
xbutil program --device 0000:00:00.0 --file bitstreams/Serpens.xclbin
```

4. Run the host

 - Basic run (example):

```bash
./host_xrt bitstreams/Serpens.xclbin [other args]
# See `scripts/instruction_to_run.txt` for recommended command-line arguments and ordering
```

Rebuilding the FPGA kernel
 - Use the Vitis build flow (kernel sources are in `src/`) to recompile if you need to rebuild for a different device or optimization flags. Typical flow:

```bash
# From project root, run your Vitis build script / v++ commands that target the Alveo U280
# Example (illustrative):
# v++ -t xilinx_u280_xdma_201920_3 --link -o bitstreams/Serpens.xclbin <kernel XO files>
```

Notes & troubleshooting
- Line endings: source files were normalized for cross-platform use  configure your git core.autocrlf appropriately.
- If host fails to find the xclbin, ensure the environment variable `XCLBIN` or the command line path points to `bitstreams/Serpens.xclbin`.
- For performance tuning, review the kernel compile flags and the `src/` sources. Profiling tools in XRT and `xbutil` can help diagnose bottlenecks.

Contact / Attribution
- This fork is maintained by the repository owner. Please check upstream for original license and contributors.

---
This README replaces presentation materials and focuses on delivering a clean, maintainable repo layout and clear run/build instructions. If you want more detailed step-by-step Vitis build scripts or CI integration, tell me and I will add them.
