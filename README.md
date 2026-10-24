# 3D WCSPH fluid simulation

Implemented through **M2** of `CUDA_Fluid_Simulation_Project_Spec.md`: a C++17 CPU reference
and a naive CUDA all-pairs solver for a seeded 3D dam break. Both support density, Tait
pressure, viscosity, gravity, semi-implicit Euler integration, box collisions and binary
PLY export. CMake/CTest builds require no third-party test framework.

M3 and later milestones are not implemented. There is no grid search, CUB sorting, GUI,
Vulkan interop or ML dependency. CUDA is optional; the CPU reference remains usable alone.
The original specification was read during M0/M1 but is not present in the current checkout.

## Windows build

Verified on Windows x64 with CMake/CTest 4.2.1, Visual Studio Build Tools 2022 / MSVC 19.42,
CUDA Toolkit 13.0.3 (`nvcc` 13.0.88), driver 582.05 and an RTX 4050 Laptop GPU (SM 8.9).
Use PowerShell in this repository. The Visual Studio generator locates MSVC without
putting `cl` on PATH. Explicitly choose VS 2022 on machines with multiple VS versions.

```powershell
cmake -S . -B build-cuda -G "Visual Studio 17 2022" -A x64 -T cuda=13.0 -DFLUID_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda --config Release --parallel 1
.\build-cuda\Release\fluid_sim.exe --device-info
ctest --test-dir build-cuda -C Release --output-on-failure
```

If `nvcc` is not recognized after installation, restart the terminal/editor. To refresh
only the current PowerShell session for the installed 13.0 Toolkit:

```powershell
$env:CUDA_PATH = [Environment]::GetEnvironmentVariable("CUDA_PATH", "Machine")
$env:CUDA_PATH_V13_0 = [Environment]::GetEnvironmentVariable("CUDA_PATH_V13_0", "Machine")
$env:Path = "$env:CUDA_PATH\bin;$env:CUDA_PATH\bin\x64;$env:Path"
nvcc --version
```

The versioned variable is used by the VS CUDA integration; `bin\x64` contains CUDA 13
runtime DLLs. These commands change only this process. A driver's CUDA compatibility
banner does not mean the Toolkit is installed. If a failed CUDA probe was cached, use a
fresh build directory or reconfigure with `-U CMAKE_CUDA_COMPILER`. Use a Toolkit-supported
host compiler and an architecture appropriate to your GPU on other machines.

CPU-only build and tests:

```powershell
cmake -S . -B build-cpu -G "Visual Studio 17 2022" -A x64 -DFLUID_ENABLE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build-cpu --config Release --parallel 1
ctest --test-dir build-cpu -C Release --output-on-failure
```

With `FLUID_ENABLE_CUDA=ON` (the default), a missing CUDA compiler produces a warning and
a CPU-only build. That build rejects `--backend cuda`; its device query exits 2.
An installed compiler with a broken Toolkit/host-toolchain configuration must be fixed
or explicitly disabled. The single-worker build was used in this sandbox.

Debug builds synchronize after every CUDA kernel to localize errors:

```powershell
cmake --build build-cuda --config Debug --parallel 1
ctest --test-dir build-cuda -C Debug --output-on-failure
```

## Linux / single-configuration generators

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFLUID_ENABLE_CUDA=OFF
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/fluid_sim --help
```

Requires CMake >= 3.24 and a C++17 compiler. For CUDA, configure a separate directory with
`-DFLUID_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<your SM number>` and a compatible
Toolkit/host compiler/driver. Linux portability has not been exercised; validation here
used Windows/MSVC.

## Run and export

```powershell
.\build-cuda\Release\fluid_sim.exe --help
.\build-cuda\Release\fluid_sim.exe --backend cuda --particles 512 --steps 1000 --dt 0.0005 --seed 42 --dump-every 250 --output results/m2/dam_break
.\build-cpu\Release\fluid_sim.exe --backend cpu --particles 256 --steps 1000 --dump-every 50 --output results/cpu_dam_break
```

The CUDA example was run successfully and writes five frames from `frame_000000.ply`
through `frame_001000.ply`. Export includes the initial state, requested intervals and
the final state even when the interval does not divide the step count. PLY files contain
vertex-only xyz float32 values in binary little-endian PLY 1.0. Output directories are
created; matching filenames are overwritten, so use a fresh directory for each run.

No arguments selects CPU, 256 particles, 1000 steps and no exports. `--dump-every 0`
creates no files; `--steps 0 --dump-every 1` exports the initial state. The CLI prints
resolved parameters and final density/speed diagnostics. Its wall time includes exports,
readback and final field evaluation; use the separate stage timing executable below
for M2 measurements. `fluid_sim --benchmark` remains unsupported.

Exit codes: 0 success, 1 invalid input/simulation/I/O failure, 2 unavailable device query.

## Numerical preset

| Parameter | Default |
|---|---:|
| Box / gravity | `[0,1]^3` / `(0,-9.81,0)` |
| Particles / seed / jitter | 256 / 42 / 0.02 of lattice spacing |
| Spacing / support radius h | 0.05 / 0.1 |
| Rest density / mass | 1000 / 0.125 |
| Tait stiffness B / exponent | 100000 / 7 |
| Kinematic viscosity | 0.01 |
| Fixed dt / wall padding / restitution | 0.0005 / 0.025 / 0.3 |

`--spacing` sets mass to `rho0 * spacing^3`, and defaults h to twice spacing and padding
to half spacing; explicit `--h` and `--padding` override these defaults. Parameters are
validated before allocation. The lattice must fit the padded box.

Both solvers are O(N^2), capped at **4096 particles**. CPU correctness workloads use
128-512 particles. Larger M2 runs use smaller spacing and dt, as in the timing preset.
This is a compressible reference model, without boundary particles, density correction,
adaptive timesteps or an incompressibility solve. Custom parameters can be unstable.
The [M2 report](M2_VALIDATION.md) describes equations, error handling and numerical limits.

## M2 stage timings

The CUDA build adds `fluid_cuda_benchmark`, a small baseline timing harness. Run Release
without competing GPU tests or sanitizer jobs:

```powershell
foreach ($n in @(1024, 2048, 4096)) {
    .\build-cuda\Release\fluid_cuda_benchmark.exe --particles $n --warmup 100 --steps 500 --output "results/m2/n$n"
}
```

It uses a fixed preset (spacing 0.025, h 0.05, dt 0.0001, padding 0.0125, seed 42),
100 warmup steps by default and preallocated CUDA events for density/pressure, forces,
integration and total step latency. `samples.csv` records every sample; `summary.csv`
includes median/p95, parameters, GPU/runtime/build metadata, command and host loop time.
The output filenames are overwritten on repeated runs in the same directory.

On the verified GPU, medians of three run medians were **0.231 ms / 0.422 ms / 0.770 ms**
for 1024 / 2048 / 4096 particles. These are instrumented CUDA-event stage spans, including
possible submission gaps; they are not rendering FPS or isolated profiler kernel times.
The [report](M2_VALIDATION.md#timing-results) includes all runs and measurement limitations.

## Validation and architecture

CUDA Release passed **26/26** CTest cases, CUDA Debug **25/25**, and CPU-only Release
**16/16**. Debug excludes the timing harness contract. CPU/GPU comparisons cover
128/256/512 particles, three seeds, initial fields and each of five steps, with documented
FP32 tolerances. Edge fixtures, independent streams, asynchronous errors, PLY output
and 1000-step stability are also checked. Compute Sanitizer memcheck, racecheck,
initcheck and synccheck all passed on the dedicated small-case suite.

| Module | Responsibility |
|---|---|
| `fluid_core` | Parameters, seeded host initialization, CPU reference, shared math |
| `fluid_cuda` | Private device buffers/stream/events and three all-pairs CUDA stages |
| `fluid_io` | Renderer-independent binary PLY export |
| `fluid_device` | CUDA device diagnostics or CPU-only fallback |
| `fluid_sim` | Backend selection, fixed-step loop, export and diagnostics |
| `fluid_cuda_benchmark` | M2 stage timing and CSV output |
| `tests/` | Analytic CPU checks, CUDA comparisons, stability and CLI contracts |

The CUDA public API hides CUDA types. Positions and velocities use separate preallocated
A/B `float4` buffers; density, pressure and acceleration have their own arrays. No device
allocation, export or readback occurs in ordinary `step()`. Each solver owns one stream.
Shared SPH and collision math keeps both backends consistent. Explicit downloads provide
the future renderer boundary without coupling physics to graphics resources.

See [M2 validation and M3 handoff](M2_VALIDATION.md) for sanitizer commands, ownership
contracts and next steps. **Implementation stops after M2.**
