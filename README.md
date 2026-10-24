# 3D WCSPH fluid simulation

Implemented through **M3** of `docs/CUDA_Fluid_Simulation_Project_Spec.md`: a C++17 CPU reference,
a naive CUDA all-pairs reference and a CUDA uniform-grid solver for a seeded 3D dam break.
All support density, Tait
pressure, viscosity, gravity, semi-implicit Euler integration, box collisions and binary
PLY export. CMake/CTest builds require no third-party test framework.

M3 adds cell keys, CUB radix sorting and 27-cell neighbor traversal. CUB comes with the
CUDA Toolkit; no separate installation is needed. M4 and later milestones are not
implemented. There is no GUI, Vulkan interop or ML dependency. CUDA is optional; the
CPU reference remains usable alone. Local specification and detailed reports live in
`docs/`, which this checkout currently ignores in Git.

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
.\build-cuda\Release\fluid_sim.exe --backend cuda --neighbor-search grid --particles 25000 --spacing 0.0125 --dt 0.0001 --steps 1000 --seed 42 --dump-every 250 --output results/m3/dam_break
.\build-cuda\Release\fluid_sim.exe --backend cuda --neighbor-search all-pairs --particles 512 --steps 1000
.\build-cpu\Release\fluid_sim.exe --backend cpu --particles 256 --steps 1000 --dump-every 50 --output results/cpu_dam_break
```

The grid example writes five frames from `frame_000000.ply` through `frame_001000.ply`.
Export includes the initial state, requested intervals and
the final state even when the interval does not divide the step count. PLY files contain
vertex-only xyz float32 values in binary little-endian PLY 1.0. Output directories are
created; matching filenames are overwritten, so use a fresh directory for each run.

No arguments selects CPU, 256 particles, 1000 steps and no exports. With `--backend cuda`,
neighbor search defaults to `grid`; select `all-pairs` explicitly for the M2 baseline.
`--neighbor-search` is rejected for the CPU backend. `--dump-every 0`
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

CPU and naive CUDA are O(N^2), capped at **4096 particles**. The M3 grid path permits
**50000 particles**; use the tested spacing/dt below so the larger lattice fits the box.
The initializer permits 50K, while each solver independently enforces its own cap.
Dense cell tables are limited to 4,194,304 cells (32 MiB for both range arrays); a smaller
h or larger box that exceeds this limit is rejected before allocating grid buffers.
CPU correctness workloads use 128-512 particles.
This is a compressible reference model, without boundary particles, density correction,
adaptive timesteps or an incompressibility solve. Custom parameters can be unstable.
The [M2 report](docs/M2_VALIDATION.md) describes equations, error handling and numerical limits.

## M3 grid pipeline and scale checks

Each step computes collision-free keys `cx + nx * (cy + ny * cz)` for cells of width h,
sorts `(cell key, original particle ID)` with `cub::DeviceRadixSort::SortPairs`, clears
start/end tables to -1, and builds half-open ranges. Density and force kernels inspect
the 27 neighboring cells and filter by distance `< h`. They access the original arrays
through sorted IDs; physical state is not reordered. Boundary coordinates are clamped
to valid cells, including closed upper box faces. Empty/outside cells are skipped.

All operations use the solver's stream. CUB scratch and grid buffers are allocated once
during construction. Indexing is rebuilt before every field evaluation, so a new state
cannot use stale cell ranges. Integration retains separate input/output buffers and
swaps only after launch. Grid construction costs O(N) for fixed-width radix sorting plus
O(number of cells) for range clearing; traversal costs the number of nearby candidates.
Dense clustering can still produce O(N^2) work.

The API keeps its backward-compatible all-pairs default; opt into the grid explicitly:

```cpp
fluid::CudaSphSolver solver(params, initial, fluid::CudaNeighborSearch::uniform_grid);
solver.step();
solver.synchronize(); // Reports asynchronous runtime and numerical errors.
auto state = solver.download_state(); // Host copy outside the hot path.
```

The fields after `step()` describe its input state. Call `evaluate_fields()` to evaluate
the updated positions/velocities. Keep the construction CUDA device current throughout
the solver lifetime; one instance is not thread-safe. An observed numerical failure
invalidates the solver, and destructors cannot report errors: synchronize explicitly.

Reproduce the M3 acceptance runs (seed 42, jitter 0.02, no exports):

```powershell
foreach ($n in @(10000, 25000, 50000)) {
    .\build-cuda\Release\fluid_sim.exe --backend cuda --neighbor-search grid --particles $n --spacing 0.0125 --dt 0.0001 --steps 1000
}
ctest --test-dir build-cuda -C Release -L grid --output-on-failure
```

The scale tests inspect every step, including finite acceleration/velocity/positions,
positive density, nonnegative pressure and padded bounds. The tested preset has
h=0.025, mass=0.001953125, padding=0.00625, rho0=1000, B=100000, nu=0.01,
gravity `(0,-9.81,0)`, restitution 0.3 and 1000 fixed steps of dt=0.0001 (0.1 seconds).

| Particles | Peak sampled density | Peak speed | Result |
|---:|---:|---:|---|
| 10000 | 1031.67 | 1.16785 | 1000 steps passed |
| 25000 | 1033.99 | 1.30554 | 1000 steps passed |
| 50000 | 1038.30 | 1.37093 | 1000 steps passed |

These are stability checks for this preset, not controlled timing results or a guarantee
for custom parameters. The 100K preset, extended stress/tuning and a full indexed
benchmark command belong to M4. The M2 timing harness below continues to use all-pairs.

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
The [report](docs/M2_VALIDATION.md#timing-results) includes all runs and measurement limitations.

## Validation and architecture

CUDA Release passed **40/40** CTest cases, CUDA Debug **39/39**, and CPU-only Release
**16/16**. Debug excludes the timing harness contract. All four Compute Sanitizer tools
passed the neighbor suite, grid physics suite and a 10K smoke run: no errors or leaks,
and no shared-memory race hazards. Tests compare
grid, naive CUDA and CPU for 128/256/512 particles, three seeds, initial fields and each
of five steps. Component tolerances are `abs(gpu-reference) <= atol + 1e-4*abs(reference)`,
with atol=1e-5 position, 1e-4 velocity, 1e-2 density, 1 pressure and 1e-2 acceleration.
No M2 tolerances were loosened. Edge fixtures, independent streams, asynchronous errors,
PLY output and 1000-step stability are also checked.

Grid tests compare exact neighbor masks with brute force, detecting missing and duplicate
visits. They verify sorted keys/IDs and every occupied/empty range on randomized scenes,
overlaps, cell-edge/box-face points, shifted domains, partial end cells, one-cell grids
and particles moving out of occupied cells. FP32 summation order changes with traversal;
short-step agreement does not imply matching long trajectories.

| Module | Responsibility |
|---|---|
| `fluid_core` | Parameters, seeded host initialization, CPU reference, shared math |
| `fluid_cuda` | Buffers/stream/events, grid indexing/CUB sorting and shared CUDA physics |
| `fluid_io` | Renderer-independent binary PLY export |
| `fluid_device` | CUDA device diagnostics or CPU-only fallback |
| `fluid_sim` | Backend selection, fixed-step loop, export and diagnostics |
| `fluid_cuda_benchmark` | M2 stage timing and CSV output |
| `tests/` | CPU physics, exact grid neighbors, backend comparisons, stability and CLI contracts |

The CUDA public API hides CUDA types. Positions and velocities use separate preallocated
A/B `float4` buffers; density, pressure and acceleration have their own arrays. No device
allocation, export or readback occurs in ordinary `step()`. Each solver owns one stream.
Shared SPH and collision math keeps both backends consistent. Explicit downloads provide
the future renderer boundary without coupling physics to graphics resources.

See [M3 validation and M4 handoff](docs/M3_VALIDATION.md) for detailed results and sanitizer
commands. **Implementation stops after M3.** M4 should add a validated 100K preset,
longer stress cases and reproducible stage benchmarks with identical small-N comparisons.
