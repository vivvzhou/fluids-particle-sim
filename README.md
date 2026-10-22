# 3D WCSPH fluid reference

Implementation follows `CUDA_Fluid_Simulation_Project_Spec.md`, scoped to **M0 and M1 only**.
This is a reusable C++17 CPU reference for a deterministic 3D dam break: all-pairs density,
Tait pressure, symmetric pressure acceleration, viscosity, gravity, semi-implicit Euler,
and padded box collisions. Binary PLY export and CTest checks are included.

CUDA is optional for M0 device diagnostics. **No CUDA simulation kernels or later milestones
are implemented.** There are no third-party libraries, GUI, Vulkan, or ML dependencies.

## Windows build (observed environment)

Visual Studio Build Tools 2022 with MSVC v143 and the Windows SDK are installed. CMake 4.2.1 is available.
Use PowerShell from this repository; the Visual Studio generator locates MSVC without putting `cl` on PATH:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --parallel 1
ctest --test-dir build -C Release --output-on-failure
.\build\Release\fluid_sim.exe --help
.\build\Release\fluid_sim.exe --device-info
```

CUDA detection falls back to a CPU build if `nvcc` is missing; explicitly select this with
`-DFLUID_ENABLE_CUDA=OFF`. The CPU build's `--device-info` reports unavailability and exits 2.
Initial inspection found no CUDA Toolkit. After installation, CUDA Toolkit 13.0.3 was verified
with MSVC 19.42, the RTX 4050 Laptop GPU, and driver 582.05: **17/17 Release tests passed**,
including the CUDA runtime device query. The original CPU-only builds remain usable.
The driver's advertised CUDA compatibility is **not** an installed toolkit/runtime version.

When a supported Toolkit is installed, configure a fresh directory with `-DFLUID_ENABLE_CUDA=ON`.
The CUDA target queries runtime version, device name, compute capability and free/total device memory.
Use a Toolkit-supported MSVC version; do not assume the newer VS 2026 compiler is supported.
For this GPU you can explicitly pass `-DCMAKE_CUDA_ARCHITECTURES=89`. If detection was cached
as NOTFOUND, use a new build directory. CUDA diagnostics now compile and run in `build-cuda`.

Verified CUDA build:

```powershell
cmake -S . -B build-cuda -G "Visual Studio 17 2022" -A x64 -T cuda=13.0 -DFLUID_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda --config Release --parallel 1
.\build-cuda\Release\fluid_sim.exe --device-info
ctest --test-dir build-cuda -C Release --output-on-failure
```

If `nvcc` is not recognized immediately after installation, fully restart the terminal/editor
so it inherits the new environment. To refresh just the current PowerShell session:

```powershell
$env:CUDA_PATH = [Environment]::GetEnvironmentVariable("CUDA_PATH", "Machine")
$env:CUDA_PATH_V13_0 = [Environment]::GetEnvironmentVariable("CUDA_PATH_V13_0", "Machine")
$env:Path = "$env:CUDA_PATH\bin;$env:CUDA_PATH\bin\x64;$env:Path"
nvcc --version
```

The versioned variable is needed by VS 2022's CUDA integration, while `bin\x64` contains
CUDA 13 runtime DLLs. These commands change only the current process; the installer already
saved the correct system variables. If CMake cached a failed CUDA probe, reconfigure with
`-U CMAKE_CUDA_COMPILER` or use a fresh build directory.

The single-worker build above is the verified command in this sandbox. A normal unrestricted
developer shell may use more workers. Debug validation:

```powershell
cmake --build build --config Debug --parallel 1
ctest --test-dir build -C Debug --output-on-failure
```

## Linux / single-configuration generators

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFLUID_ENABLE_CUDA=OFF
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/fluid_sim --help
```

Requires CMake >= 3.24 and a C++17 compiler. CUDA diagnostics additionally require a compatible
CUDA compiler, runtime, host compiler and driver. No third-party test framework is needed.
The Linux commands are provided for portability; only Windows/MSVC was exercised here.

## Run and export

```powershell
.\build\Release\fluid_sim.exe --scene dam-break --backend cpu --particles 256 --steps 1000 --dt 0.0005 --seed 42 --dump-every 50 --output results/dam_break
```

For Linux, use `./build/fluid_sim` with the same arguments. Running with no arguments uses
256 particles and 1000 steps with exports disabled. Use `--help` for all parameters.

With `--dump-every 50`, the example writes `frame_000000.ply`, `frame_000050.ply`, ...,
`frame_001000.ply`. The initial state, requested intervals, and final state are exported;
the final state is also exported when the step count is not divisible by the interval.
Files contain vertex-only `x y z` IEEE-754 float32 values in binary little-endian PLY 1.0.
Parent directories are created, and existing files with matching frame names are overwritten.
Use a fresh output directory for each run to avoid mixing old and new snapshots.
`--steps 0 --dump-every 1` exports just the initial state. `--dump-every 0` creates no files.

The CLI logs resolved parameters, final density range, maximum final speed and CPU wall time.
That time includes exports and final field evaluation; it is **not** a GPU benchmark.
Exit codes: 0 success, 1 invalid input/simulation/I/O failure, 2 unavailable device query.
Unsupported options such as `--benchmark` and `--backend cuda` are rejected.

## Default numerical preset

| Parameter | Value |
|---|---:|
| Box | `[0,1]^3`, gravity `(0,-9.81,0)` |
| Particles / seed / jitter | 256 / 42 / 0.02 of lattice spacing |
| Spacing / support radius | 0.05 / 0.1 |
| Rest density / mass | 1000 / 0.125 |
| Tait stiffness B / exponent | 100000 / 7 |
| Kinematic viscosity | 0.01 |
| Fixed dt / wall padding / restitution | 0.0005 / 0.025 / 0.3 |

`--spacing` derives mass as `rho0 * spacing^3`, and defaults `h` to `2 * spacing` and
padding to `spacing / 2`; explicit `--h` and `--padding` override those defaults.
Other numerical parameters are exposed in `SimParams` and via CLI flags where useful.
Recommended CPU workloads are **128-512 particles**. The CLI/API reject more than 4096
to guard against accidental O(N²) workloads; large lattices may also require reduced spacing
to fit the box. The 10K-100K examples in the full spec belong to later CUDA milestones.

## Validation and limitations

The 16 CPU tests cover 3D kernel normalization/support/gradients; self and neighbor density;
pressure sign and momentum balance; viscosity damping and density weights; gravity and
integration; all six walls; seeded reproducibility; invalid input/overflow; binary PLY bytes;
CLI/export behavior; and 1000-step stability at 128, 256 and 512 particles.
See [validation results](docs/validation.md) for executed commands, measured values and blockers.

The stable preset is a small reference simulation, not calibrated water or an incompressible
solver. Free surfaces and walls have missing neighbors; walls use clamping/reflection without
boundary particles. Density errors and compressibility remain. Larger stiffness, smaller
spacing, or higher viscosity may require a smaller fixed timestep. No automatic stability
guarantee or timestep adaptation is provided for custom parameters.

See [solver math](docs/numerical-method.md) for equations, units, singularity policy and references,
and [architecture](docs/architecture.md) for module boundaries and the **M2 handoff**.
Implementation stops after M1; CUDA runtime verification is complete, with GPU simulation deferred to M2.
