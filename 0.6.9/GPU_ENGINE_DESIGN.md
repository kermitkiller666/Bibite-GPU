# GPU-resident Bibites engine

## Measurable target

The engine passes its offload target when CUDA event time accounts for at least
99% of the complete blocking `bgf_world_step` interval:

```
GPU share = GPU kernel milliseconds /
            (GPU kernel milliseconds + host launch/synchronisation overhead)
```

Menus, drawing, save/load and optional snapshot downloads are outside the
simulation-step interval. The benchmark reports both this share and the
end-to-end simulated-time multiplier so a high offload percentage cannot be
mistaken for high simulation speed.

## Architecture

```
Windows application / benchmark
             |
             | batched step request; no per-tick state transfer
             v
CUDA world context (persistent device allocations)
             |
             v
cooperative persistent world kernel
  1. update density-adaptive contact/sensing hashes on their cadence
  2. diffuse and decay pheromone field
  3. refresh cached food, neighbour and pheromone perception
  4. evaluate active inherited FP16 brain nodes with FP32 accumulation
  5. resolve steering, contact response, feeding and metabolism
  6. reproduce, mutate, die and incrementally rehash recycled pellets
  7. repeat entirely on-device
```

World state uses structure-of-arrays storage. Positions, velocities, energy,
age and accumulation stay FP32. Neural weights use FP16. IDs, occupancy and
counters use integers. Grid-wide barriers make each phase observe a consistent
previous phase while the entire requested batch remains one GPU launch.

## Interfaces

- `bgf_world_create`: allocate and initialise a world directly on a selected
  CUDA device.
- `bgf_world_step`: execute many fixed simulation steps in a single persistent
  launch and return measured GPU/host timing.
- `bgf_world_get_stats`: retrieve infrequent aggregate statistics.
- `bgf_world_spawn_template_bibite`: copy a stock-template projection, lineage
  metadata and recurrent NEAT graph into a free GPU slot.
- `bgf_world_download_snapshot`: pack aggregate statistics, live Bibites and
  pellets on GPU, then download the compact presentation snapshot.
- `bgf_world_download_bibites`: retrieve an optional display/save snapshot.
- `bgf_world_destroy`: release device state.

All failures return a CUDA-style integer error code. A failed context can be
destroyed safely.

## Constraints and trade-offs

- This is a new data-oriented simulation core, not Unity Physics2D running on
  CUDA. The compiled Unity game remains useful as a behavioural reference.
- Evolutionary outcomes will not be bit-identical to Unity because parallel
  collision and feeding order differs.
- The selectable Extreme profile additionally amortises collision, full-vision
  and brain work across multiple fixed steps. Exact retains the compact
  engine's normal per-tick collision cadence.
- The initial backend is NVIDIA CUDA. A later portable backend would require
  separate Vulkan/DirectCompute kernels.
- On Direct3D 11, CUDA writes shared GPU vertex buffers directly. CPU snapshots
  remain capacity-aware at 10, 4 or 1 Hz for charts and selection; rendering
  time stays outside the native simulation-step offload percentage.
- The native safety ceiling is 500,000 Bibites, but device and presentation
  arrays are sized to the player's selected per-world cap rather than that
  ceiling. This keeps ordinary worlds small while allowing deliberate large
  experiments on GPUs with enough memory.
- A single world may not fill a large GPU at very small populations. Batched
  steps amortise launch overhead; parallel multi-world evolution is the next
  scaling path if one world becomes too small.

## Growth points

Revisit exact 0.6.4 save conversion, arbitrary-topology grouping and parallel
multi-world evolution after the single-world core. The current engine has
passed the measured 99% target and state-invariant tests.
