#include "bgf_world.h"

#include <cooperative_groups.h>
#include <cuda_d3d11_interop.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <d3d11_4.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include <fcntl.h>
#include <io.h>
#include <share.h>
#include <sys/stat.h>

namespace cg = cooperative_groups;

namespace {

constexpr int32_t kSensorCount = 16;
constexpr int32_t kOutputCount = 6;
constexpr int32_t kTemplateOutputCount = 15;
constexpr int32_t kWeightCount = kSensorCount * kOutputCount;
constexpr int32_t kHiddenWidth = 12;
constexpr int32_t kHiddenNodeCount = kHiddenWidth * 2;
constexpr int32_t kHiddenFirstEnd = kSensorCount * kHiddenWidth;
constexpr int32_t kHiddenSecondEnd = kHiddenFirstEnd + kHiddenWidth * kHiddenWidth;
constexpr int32_t kHiddenConnectionCount = kHiddenSecondEnd + kHiddenWidth * kOutputCount;
constexpr int32_t kHiddenMaskWords = (kHiddenConnectionCount + 31) / 32;
constexpr int32_t kLegacyNativeBrainNodes = kSensorCount + kHiddenNodeCount + kOutputCount;
constexpr int32_t kStockSensorCount = 34;
constexpr int32_t kExtraSensorCount = kStockSensorCount - kSensorCount;
constexpr int32_t kExtraOutputCount = kTemplateOutputCount - kOutputCount;
constexpr int32_t kExtensionInputEnd = kExtraSensorCount * kHiddenWidth;
constexpr int32_t kExtensionConnectionCount =
    kExtensionInputEnd + kExtraOutputCount * kHiddenWidth;
constexpr int32_t kExtensionMaskWords = (kExtensionConnectionCount + 31) / 32;
constexpr int32_t kFullNativeBrainNodes =
    kStockSensorCount + kHiddenNodeCount + kTemplateOutputCount;
constexpr int32_t kMaximumNativeSynapses =
    kWeightCount + kHiddenConnectionCount + kExtensionConnectionCount;
constexpr int32_t kMaximumSelectedSynapses =
    kMaximumNativeSynapses > 512 ? kMaximumNativeSynapses : 512;
static_assert(kMaximumNativeSynapses <= 1024,
    "native brain inspector exceeds its synapse capacity");
constexpr int32_t kMaxTemplateNodes = 256;
constexpr int32_t kMaxTemplateSynapses = 512;
constexpr int32_t kMaximumBibites = 500000;
constexpr int32_t kMaximumPellets = 32768;
constexpr int32_t kMaximumFoodZones = 64;
constexpr int32_t kMaximumSnapshotBibites = 8192;
constexpr int32_t kMaximumVisibleBibites = 2048;
constexpr int32_t kInitialTemplateTopologyCapacity = 8;
constexpr int32_t kBrainUpdateFactor = 2;
constexpr int32_t kSenseUpdateFactor = 10;
constexpr int32_t kVisionLookupFactor = 20;
constexpr int32_t kPheromoneUpdateFactor = 4;
constexpr int32_t kContactBucketCapacity = 16;
constexpr int32_t kSenseBucketCapacity = 16;
constexpr int32_t kPelletBucketCapacity = 32;
constexpr int32_t kPelletRespawnWheelSize = 512;
constexpr float kPelletRespawnSeconds = 5.0f;
// A pellet is taken in small bites. Units are integral so competing Bibites
// can claim portions atomically without duplicating food.
constexpr int32_t kPelletFoodUnits = 1024;
constexpr int32_t kMaximumPelletFoodUnits = 50 * kPelletFoodUnits;
constexpr float kPelletEatingSeconds = 2.0f;
constexpr float kPelletDigestionSeconds = 5.0f;
constexpr float kRegularMaximumBodySize = 2.5f;
constexpr float kEcologyReferenceHalfExtent = 500.0f;
constexpr float kEcologyReferencePellets = 8192.0f;
constexpr float kMaximumEcologyScale = 32.0f;
constexpr int32_t kThreads = 256;

struct WorldCounters {
    unsigned long long completed_steps;
    unsigned long long births;
    unsigned long long deaths;
    unsigned long long pellets_eaten;
    unsigned long long starvation_deaths;
    unsigned long long age_deaths;
    unsigned long long invalid_state_deaths;
    unsigned long long phase_prepare_cycles;
    unsigned long long phase_spatial_index_cycles;
    unsigned long long phase_contact_cycles;
    unsigned long long phase_decision_cycles;
    unsigned long long phase_motion_cycles;
    unsigned long long phase_lifecycle_cycles;
    unsigned long long phase_started_cycle;
    int32_t living_bibites;
};

struct SnapshotCounters {
    int32_t bibites;
    int32_t pellets;
    int32_t plants;
    int32_t meats;
    float total_energy;
    float plant_energy;
    float meat_energy;
    int32_t visible_bibites;
};

struct RenderVertex {
    float position_x;
    float position_y;
    float position_z;
    float color_r;
    float color_g;
    float color_b;
    float color_a;
    float uv_x;
    float uv_y;
};

static_assert(sizeof(RenderVertex) == 9u * sizeof(float), "Unity render vertex layout mismatch");

struct RuntimeTiming {
    unsigned long long food_cycles, brain_cycles, post_cycles;
};

struct TemplateFoodSenses {
    int32_t plant = -1;
    int32_t meat = -1;
    int32_t plants_seen = 0;
    int32_t meats_seen = 0;
    float plant_distance_squared = 0.0f;
    float meat_distance_squared = 0.0f;
    float2 plant_direction{0.0f, 0.0f};
    float2 meat_direction{0.0f, 0.0f};
};


struct ContactTileTask { int32_t cell,begin,cursor; };

struct DeviceWorld {
    // Derived runtime state is not part of the checkpoint counter layout.
    int32_t* live_items;
    int32_t* live_scratch;
    int32_t* live_count;
    int32_t* live_scratch_count;
    float2* digestion_efficiency;
    float* drag_retention;
    int32_t* pellet_overflow_heads; // Segment offsets, not linked-list heads.
    int32_t* pellet_overflow_next;  // Contiguous overflow pellet IDs.
    int32_t* pellet_overflow_counts;
    int32_t* pellet_overflow_tile_prefix;
    TemplateFoodSenses* food_senses_cache;
    int32_t* food_senses_cache_active; // Uniform tick flag, never a mutable food count.
    ContactTileTask* contact_tile_tasks;
    int32_t* contact_tile_count;
    float* brain_inputs;
    float* brain_outputs;
    RuntimeTiming* timing;
    int32_t max_bibites;
    int32_t initial_bibites;
    int32_t pellet_count;
    int32_t pellet_limit;
    int32_t spatial_grid_width;
    int32_t contact_grid_width;
    int32_t sense_grid_width;
    int32_t pheromone_grid_width;
    int32_t pheromone_grid_height;
    int32_t sense_cell_radius;
    uint32_t seed;
    float world_half_extent;
    float fixed_delta_time;
    float initial_energy;
    float pellet_energy;
    float food_growth_factor;
    float linear_drag;
    float collision_damage_constant;
    float collision_damage_threshold;
    float biting_damage_factor;
    float biting_pressure;
    BgfWorldDigestionSettings digestion;
    float reproduction_energy;
    float sense_radius;
    float ecology_scale;
    float mobility_scale;
    float pheromone_diffusion;
    float pheromone_decay;
    float mutation_strength;
    uint32_t diagnostic_mask;
    int32_t contact_grid_update_factor;
    int32_t contact_solve_factor;
    int32_t vision_lookup_factor;
    int32_t brain_update_factor;
    int32_t lock_food_target;

    int32_t* alive;
    // Runtime-only slot incarnation. Inspector commands must not act on a new
    // Bibite that reused the selected slot between the click and worker drain.
    uint32_t* instance_id;
    int32_t* generation;
    float2* positions;
    float2* velocities;
    float* headings;
    float* energy;
    float* age;
    float* size;
    float* reproduction_cooldown;
    float4* traits;
    float3* colors;
    float* gene_mutation_strength;
    float* brain_mutation_strength;
    float* diet;
    float* health;
    float* bite_cooldown;
    float* pending_damage;
    float* last_damage;
    // adult size, accumulated egg progress, resettable clock, clock period.
    float4* life_state;
    // Egg laying period and womb capacity inherited by descendants.
    float2* reproductive_traits;
    int32_t* held_target; // 0=none, positive=Bibite slot+1, negative=pellet slot+1.
    uint32_t* held_instance;
    uint64_t* lineage_id;
    uint64_t* tag_id;
    uint32_t* rng;
    __half* weights;
    // Starter brains retain their direct 16x6 path and grow two sparse 12-node
    // hidden layers. Masks and weights are synapse-major for coalesced lanes.
    uint32_t* hidden_masks;
    __half* hidden_weights;
    __half* hidden_biases;
    __half* hidden_last_input;
    __half* hidden_last_output;
    uint16_t* hidden_synapse_count;
    // New native-born brains keep the fast legacy 16x6 path and add sparse
    // links for the remaining stock sensors/actions. Version zero preserves
    // the exact behavior of pre-v10 GPU checkpoints.
    uint8_t* native_brain_version;
    uint32_t* extension_masks;
    __half* extension_weights;
    uint16_t* extension_synapse_count;

    // Zero selects a native evolving brain. Positive values are one-based
    // indices into the shared immutable topology table below. The large
    // mutable template arrays are allocated lazily only if a placed stock
    // brain is actually introduced into the world.
    int32_t* template_brain;
    int32_t template_instance_capacity;
    int32_t* template_instance;
    int32_t* template_free_items;
    int32_t* template_free_count;
    int32_t* template_living_count;
    int32_t template_topology_capacity;
    uint16_t* template_topology_node_counts;
    uint16_t* template_topology_synapse_counts;
    uint16_t* template_topology_active_node_counts;
    uint8_t* template_active_nodes;
    uint32_t* template_node_descriptors;
    uint16_t* template_synapse_edges;
    __half* template_node_biases;
    float* template_node_accum;
    float* template_node_last_input;
    float* template_node_last_output;
    __half* template_synapse_weights;

    float4* actions_a;
    float2* actions_b;
    float4* actions_c; // eat, digest, grab, attack
    float4* actions_d; // herding, egg production, grow, heal
    float* clock_reset_output;
    float2* repulsion;
    float* food_gain; // Undigested food energy in this Bibite's stomach.
    float* meat_gain;
    int32_t* wants_reproduction;
    int32_t* reproduction_items;
    int32_t* reproduction_count;
    int32_t* free_bibite_items;
    int32_t* free_bibite_count;
    int32_t* dead_bibite_items;
    int32_t* dead_bibite_count;
    int32_t* meat_cursor;

    int32_t* contact_cell_counts;
    int32_t* contact_cell_items;
    int32_t* contact_overflow_heads;
    int32_t* contact_overflow_next;
    uint32_t* contact_occupied;
    int32_t* contact_occupied_cells;
    int32_t* contact_occupied_count;
    int32_t* contact_overflow_count;
    int32_t* large_contact_items;
    int32_t* large_contact_count;
    int32_t* contact_grid_dirty;
    int32_t* sense_grid_dirty;
    int32_t* contact_home_cell;
    uint8_t* contact_is_overflow;

    int32_t* sense_bibite_cell_counts;
    int32_t* sense_bibite_cell_items;
    int32_t* sense_bibite_overflow_heads;
    int32_t* sense_bibite_overflow_next;
    uint32_t* sense_bibite_occupied;
    int32_t* sense_bibite_occupied_cells;
    int32_t* sense_bibite_occupied_count;
    int32_t* sense_bibite_overflow_count;

    uint32_t* pellet_cell_masks;
    int32_t* pellet_cell_items;
    int32_t* pellet_hash_cell;
    int32_t* pellet_hash_slot;
    uint32_t* pellet_occupied;
    int32_t* pellet_overflow_count;
    int32_t* eaten_pellet_items;
    int32_t* eaten_pellet_count;
    int32_t* pellet_respawn_items;
    int32_t* pellet_respawn_counts;
    int32_t* pellet_pending;

    int32_t* cached_food;
    int32_t* cached_neighbour;
    int32_t* cached_neighbour_count;
    float2* cached_food_direction;
    float2* cached_neighbour_direction;
    float* cached_food_distance_squared;
    float* cached_neighbour_distance_squared;
    float3* cached_neighbour_color;

    int32_t* pellet_active;
    int32_t* pellet_food_units;
    int32_t* pellet_nominal_units;
    int32_t* pellet_held_by;
    float2* pellet_positions;
    uint32_t* pellet_rng;
    BgfWorldFoodZone* food_zones;
    int32_t food_zone_count;

    float3* pheromone_a;
    float3* pheromone_b;
    float3* pheromone_heading_x_a;
    float3* pheromone_heading_x_b;
    float3* pheromone_heading_y_a;
    float3* pheromone_heading_y_b;
    WorldCounters* counters;
};

struct ContactGridView {
    int32_t* home_cell;
    uint8_t* is_overflow;
    int32_t* cell_counts;
    int32_t* cell_items;
    int32_t* overflow_heads;
    int32_t* overflow_next;
    uint32_t* occupied;
    int32_t* occupied_cells;
    int32_t* occupied_count;
    int32_t* overflow_count;
    int32_t* large_items;
    int32_t* large_count;
};

struct TemplateTopologyHost {
    std::vector<uint32_t> node_descriptors;
    std::vector<uint8_t> active_nodes;
    std::vector<uint16_t> synapse_edges;
};

struct CheckpointHeader {
    char magic[8];
    uint32_t format_version;
    uint32_t header_bytes;
    BgfWorldConfig config;
    uint32_t topology_count;
    uint32_t has_template_instance_storage;
    uint32_t manual_spawn_sequence;
};

struct FoodCheckpointState {
    int32_t target;
    float growth_factor;
    float pellet_energy;
    float ecology_scale;
    float mobility_scale;
    float sense_radius;
    int32_t sense_cell_radius;
};

struct RuntimeCheckpointState {
    float linear_drag;
};

struct CombatCheckpointState {
    float collision_damage_constant;
    float collision_damage_threshold;
    float biting_damage_factor;
    float biting_pressure;
};

constexpr char kCheckpointMagic[8] = {'B', 'G', 'F', 'C', 'P', '0', '1', '\0'};
constexpr uint32_t kCheckpointFormatVersion = 12u;
constexpr uint64_t kCheckpointFooter = 0x31444e4546504742ull;

struct RenderBufferSet {
    cudaGraphicsResource_t render_bibite_vertices = nullptr;
    cudaGraphicsResource_t render_pellet_vertices = nullptr;
    BgfWorldD3D11RenderConfig render_config{};
    int32_t rendered_bibites = 0;
    int32_t rendered_pellets = 0;
    RenderVertex* captured_bibites = nullptr;
    RenderVertex* captured_pellets = nullptr;
    SnapshotCounters* captured_counters = nullptr;
    cudaStream_t render_stream = nullptr;
    int32_t captured_bibite_count = 0, captured_pellet_count = 0;
    bool capture_ready = false;
    bool render_buffers_initialized = false;
    bool faulted = false;
};

struct StepGraphCacheEntry {
    cudaGraphExec_t graph = nullptr;
    int32_t steps = 0, cadence_phase = 0;
    uint64_t last_used = 0;
};

struct GpuWorldContext {
    int32_t device_index = 0;
    int32_t grid_blocks = 0;
    int32_t maximum_grid_blocks = 0;
    int32_t allocated_contact_width = 0, allocated_sense_width = 0;
    cudaDeviceProp properties{};
    BgfWorldConfig config{};
    DeviceWorld world{};
    bool has_stepped = false;
    bool live_list_dirty = false;
    int32_t latest_living = 0, latest_active_pellets = 0;
    cudaGraphExec_t step_graph = nullptr; // Non-owning alias of one cache entry.
    StepGraphCacheEntry graph_cache[8]{};
    uint64_t graph_cache_clock = 0;
    uint64_t host_steps = 0;
    bool graph_world_valid = false;
    float graph_build_milliseconds = 0.0f;
    int32_t graph_cache_hits = 0;
    DeviceWorld graph_world{};
    uint8_t* checkpoint_staging = nullptr;
    void* checkpoint_device_staging = nullptr;
    int32_t* checkpoint_slots = nullptr;
    int32_t* checkpoint_template_slots = nullptr;
    std::vector<int32_t> checkpoint_live;
    std::vector<int32_t> checkpoint_imported;
    cudaEvent_t step_started = nullptr;
    cudaEvent_t step_finished = nullptr;
    BgfWorldBibite* snapshot_bibites = nullptr;
    int32_t snapshot_bibite_capacity = 0;
    BgfWorldBibite* snapshot_visible_bibites = nullptr;
    int32_t snapshot_visible_bibite_capacity = 0;
    BgfWorldPellet* snapshot_pellets = nullptr;
    SnapshotCounters* snapshot_counters = nullptr;
    BgfWorldBibiteDetail* selected_detail = nullptr;
    BgfWorldBrainNodeState* selected_nodes = nullptr;
    BgfWorldBrainSynapseState* selected_synapses = nullptr;
    int32_t* command_result = nullptr;
    RenderBufferSet render_buffers[2];
    uint32_t manual_spawn_sequence = 0;
    std::vector<TemplateTopologyHost> template_topologies;
};

__host__ __device__ float clampf(float value, float minimum, float maximum)
{
    return fminf(maximum, fmaxf(minimum, value));
}

__host__ __device__ uint32_t mix_bits(uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value == 0 ? 0x9e3779b9u : value;
}

__device__ uint32_t next_random(uint32_t& state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ float uniform01(uint32_t& state)
{
    return static_cast<float>(next_random(state) & 0x00ffffffu) * (1.0f / 16777216.0f);
}

__device__ float signed_uniform(uint32_t& state)
{
    return uniform01(state) * 2.0f - 1.0f;
}

__device__ float2 sample_food_zone_position(
    const DeviceWorld& world, uint32_t& random_state, bool regrowth,
    float* pellet_size = nullptr)
{
    if (pellet_size) *pellet_size = 1.0f;
    float total = 0.0f;
    for (int32_t i = 0; i < world.food_zone_count; ++i) {
        const BgfWorldFoodZone& zone = world.food_zones[i];
        total += regrowth ? zone.growth_weight : zone.seed_weight;
    }
    // A scenario can stop growth while retaining biomass (or the reverse).
    // Keep positions in its zones even if the selected weight set is zero.
    const bool alternate = total <= 0.0f;
    if (alternate) {
        for (int32_t i = 0; i < world.food_zone_count; ++i) {
            const BgfWorldFoodZone& zone = world.food_zones[i];
            total += regrowth ? zone.seed_weight : zone.growth_weight;
        }
    }
    if (total <= 0.0f) {
        return {signed_uniform(random_state) * world.world_half_extent,
            signed_uniform(random_state) * world.world_half_extent};
    }
    float choice = uniform01(random_state) * total;
    int32_t selected = world.food_zone_count - 1;
    for (int32_t i = 0; i < world.food_zone_count; ++i) {
        const BgfWorldFoodZone& zone = world.food_zones[i];
        choice -= alternate
            ? (regrowth ? zone.seed_weight : zone.growth_weight)
            : (regrowth ? zone.growth_weight : zone.seed_weight);
        if (choice < 0.0f) {
            selected = i;
            break;
        }
    }
    const BgfWorldFoodZone& zone = world.food_zones[selected];
    if (pellet_size) *pellet_size = zone.pellet_size > 0.0f
        ? zone.pellet_size : 1.0f;
    float2 position{};
    if (zone.distribution == 5) {
        position = {zone.center_x + signed_uniform(random_state) * zone.half_width,
            zone.center_y + signed_uniform(random_state) * zone.half_height};
    } else {
        const float angle = uniform01(random_state) * 6.283185307179586f;
        float fraction = sqrtf(uniform01(random_state));
        if (zone.distribution == 1) {
            fraction = 1.0f - sqrtf(1.0f - fraction);
        } else if (zone.distribution == 2) {
            const float remainder = 1.0f - fraction;
            fraction = 1.0f - remainder * remainder;
        } else if (zone.distribution == 3 || zone.distribution == 4) {
            const float inner = zone.inner_radius;
            const float flat_radius = sqrtf(inner * inner +
                uniform01(random_state) * (1.0f - inner * inner));
            if (zone.distribution == 3) {
                // Match the stock 0.6.3.1 Ring warp (RandomPointInRing,
                // warp=3), which biases pellets toward both ring edges.
                const float radial = (flat_radius - inner) / (1.0f - inner);
                fraction = inner + (1.0f - inner) * 0.5f *
                    (1.0f + powf(radial, 1.0f / 3.0f) -
                        powf(1.0f - radial, 1.0f / 3.0f));
            } else {
                fraction = flat_radius;
            }
        }
        const float distance = zone.radius * fraction;
        position = {zone.center_x + cosf(angle) * distance,
            zone.center_y + sinf(angle) * distance};
    }
    const float limit = world.world_half_extent -
        fmaxf(0.0001f, world.world_half_extent * 0.000001f);
    return {clampf(position.x, -world.world_half_extent, limit),
        clampf(position.y, -world.world_half_extent, limit)};
}

__device__ __forceinline__ void finish_profile_phase(
    WorldCounters* counters,
    unsigned long long* phase_cycles)
{
    unsigned long long now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    *phase_cycles += now - counters->phase_started_cycle;
    counters->phase_started_cycle = now;
}

__device__ float mutation_noise(uint32_t& state)
{
    return (signed_uniform(state) + signed_uniform(state) + signed_uniform(state)) * (1.0f / 3.0f);
}

__device__ float wrap_coordinate(float value, float half_extent)
{
    // An invalid or very large coordinate must not trap a cooperative grid in
    // an endless subtraction loop. Invalid values reach the existing death
    // check later in the tick; ordinary in-range coordinates keep the fast path.
    if (!isfinite(value)) return value;
    const float extent = half_extent * 2.0f;
    if (value >= half_extent || value < -half_extent) {
        value = fmodf(value, extent);
        if (value >= half_extent) value -= extent;
        if (value < -half_extent) value += extent;
    }
    return value;
}

__device__ float2 wrapped_delta(float2 from, float2 to, float half_extent)
{
    float2 result{to.x - from.x, to.y - from.y};
    const float extent = half_extent * 2.0f;
    if (result.x > half_extent) result.x -= extent;
    if (result.x < -half_extent) result.x += extent;
    if (result.y > half_extent) result.y -= extent;
    if (result.y < -half_extent) result.y += extent;
    return result;
}

__device__ int32_t wrap_index(int32_t value, int32_t size)
{
    value %= size;
    return value < 0 ? value + size : value;
}

__device__ __forceinline__ int32_t grid_cell(
    const DeviceWorld& world,
    float2 position,
    int32_t grid_width)
{
    const float inverse_extent = 1.0f / (world.world_half_extent * 2.0f);
    int32_t x = static_cast<int32_t>((position.x + world.world_half_extent) *
        inverse_extent * grid_width);
    int32_t y = static_cast<int32_t>((position.y + world.world_half_extent) *
        inverse_extent * grid_width);
    x = max(0, min(grid_width - 1, x));
    y = max(0, min(grid_width - 1, y));
    return y * grid_width + x;
}

__host__ __device__ __forceinline__ size_t dense_weight_slot(
    const DeviceWorld& world,
    int32_t bibite,
    int32_t weight)
{
    // Weight pairs are major and Bibites are minor, so consecutive lanes fetch
    // the same two synapses for different brains in coalesced 32-bit loads.
    const int32_t pair = weight >> 1;
    const int32_t lane = weight & 1;
    return (static_cast<size_t>(pair) * world.max_bibites + bibite) * 2u + lane;
}

__host__ __device__ __forceinline__ size_t hidden_column_slot(
    const DeviceWorld& world, int32_t bibite, int32_t column)
{
    return static_cast<size_t>(column) * world.max_bibites + bibite;
}

__device__ __forceinline__ bool enable_hidden_link(
    const DeviceWorld& world, int32_t bibite, int32_t edge, uint32_t& random_state)
{
    const size_t mask_slot = hidden_column_slot(world, bibite, edge >> 5);
    const uint32_t bit = 1u << (edge & 31);
    if ((world.hidden_masks[mask_slot] & bit) != 0u) return false;
    const size_t weight_slot = dense_weight_slot(world, bibite, edge);
    // An inactive link has zero *effective* weight. Its FP16 value is kept as
    // dormant genetic state, so reconnecting restores exactly that strength.
    // NaN is only used for an edge which has never had a weight.
    if (isnan(__half2float(world.hidden_weights[weight_slot]))) {
        world.hidden_weights[weight_slot] =
            __float2half_rn(signed_uniform(random_state) * 0.5f);
    }
    world.hidden_masks[mask_slot] |= bit;
    ++world.hidden_synapse_count[bibite];
    return true;
}

__device__ void initialise_hidden_brain(
    const DeviceWorld& world, int32_t bibite, uint32_t& random_state,
    bool enable_output_links)
{
    world.hidden_synapse_count[bibite] = 0;
    for (int32_t word = 0; word < kHiddenMaskWords; ++word)
        world.hidden_masks[hidden_column_slot(world, bibite, word)] = 0u;
    for (int32_t edge = 0; edge < kHiddenConnectionCount; ++edge)
        world.hidden_weights[dense_weight_slot(world, bibite, edge)] =
            __float2half_rn(nanf(""));
    for (int32_t node = 0; node < kHiddenNodeCount; ++node) {
        const size_t slot = hidden_column_slot(world, bibite, node);
        world.hidden_biases[slot] = __float2half_rn(signed_uniform(random_state) * 0.15f);
        world.hidden_last_input[slot] = __float2half_rn(0.0f);
        world.hidden_last_output[slot] = __float2half_rn(0.0f);
    }
    // Guarantee each hidden neuron receives an input. Retaining the original
    // 16x6 path keeps a newly enabled hidden connection from destroying a
    // previously viable parent behaviour.
    for (int32_t destination = 0; destination < kHiddenWidth; ++destination) {
        const int32_t first_source = static_cast<int32_t>(
            uniform01(random_state) * kSensorCount);
        const int32_t second_source = static_cast<int32_t>(
            uniform01(random_state) * kHiddenWidth);
        enable_hidden_link(world, bibite,
            destination * kSensorCount + first_source, random_state);
        enable_hidden_link(world, bibite,
            kHiddenFirstEnd + destination * kHiddenWidth + second_source,
            random_state);
    }
    if (enable_output_links) {
        for (int32_t destination = 0; destination < kOutputCount; ++destination) {
            const int32_t source = static_cast<int32_t>(
                uniform01(random_state) * kHiddenWidth);
            enable_hidden_link(world, bibite,
                kHiddenSecondEnd + destination * kHiddenWidth + source,
                random_state);
        }
    }
    const int32_t starts[3] = {0, kHiddenFirstEnd, kHiddenSecondEnd};
    const int32_t ends[3] = {
        kHiddenFirstEnd, kHiddenSecondEnd, kHiddenConnectionCount};
    const int32_t extras[3] = {12, 12, enable_output_links ? 6 : 0};
    for (int32_t layer = 0; layer < 3; ++layer) {
        for (int32_t extra = 0; extra < extras[layer]; ++extra) {
            for (int32_t attempt = 0; attempt < 8; ++attempt) {
                const int32_t edge = starts[layer] + static_cast<int32_t>(
                    uniform01(random_state) * (ends[layer] - starts[layer]));
                if (enable_hidden_link(world, bibite, edge, random_state)) break;
            }
        }
    }
}

__device__ void mutate_hidden_brain(
    const DeviceWorld& world, int32_t child, int32_t parent,
    float mutation_strength, uint32_t& random_state)
{
    world.hidden_synapse_count[child] = world.hidden_synapse_count[parent];
    // Inherit dormant links too: a later structural mutation must revive the
    // ancestral weight, not create a fresh random one.
    for (int32_t edge = 0; edge < kHiddenConnectionCount; ++edge) {
        world.hidden_weights[dense_weight_slot(world, child, edge)] =
            world.hidden_weights[dense_weight_slot(world, parent, edge)];
    }
    for (int32_t word = 0; word < kHiddenMaskWords; ++word) {
        uint32_t bits = world.hidden_masks[hidden_column_slot(world, parent, word)];
        world.hidden_masks[hidden_column_slot(world, child, word)] = bits;
        while (bits != 0u) {
            const int32_t edge = word * 32 + __ffs(bits) - 1;
            const float inherited = __half2float(world.hidden_weights[
                dense_weight_slot(world, parent, edge)]);
            world.hidden_weights[dense_weight_slot(world, child, edge)] =
                __float2half_rn(clampf(
                    inherited + mutation_noise(random_state) * mutation_strength,
                    -4.0f, 4.0f));
            bits &= bits - 1u;
        }
    }
    for (int32_t node = 0; node < kHiddenNodeCount; ++node) {
        const size_t source = hidden_column_slot(world, parent, node);
        const size_t destination = hidden_column_slot(world, child, node);
        world.hidden_biases[destination] = __float2half_rn(clampf(
            __half2float(world.hidden_biases[source]) +
                mutation_noise(random_state) * mutation_strength * 0.25f,
            -4.0f, 4.0f));
        world.hidden_last_input[destination] = __float2half_rn(0.0f);
        world.hidden_last_output[destination] = __float2half_rn(0.0f);
    }
    // A small structural bias toward new links permits complexity to grow,
    // while occasional deletion stops every lineage converging on a dense net.
    if (uniform01(random_state) < clampf(mutation_strength * 2.5f, 0.0f, 0.5f)) {
        const int32_t layer = static_cast<int32_t>(uniform01(random_state) * 3.0f);
        const int32_t starts[3] = {0, kHiddenFirstEnd, kHiddenSecondEnd};
        const int32_t ends[3] = {
            kHiddenFirstEnd, kHiddenSecondEnd, kHiddenConnectionCount};
        for (int32_t attempt = 0; attempt < 16; ++attempt) {
            const int32_t edge = starts[layer] + static_cast<int32_t>(
                uniform01(random_state) * (ends[layer] - starts[layer]));
            if (enable_hidden_link(world, child, edge, random_state)) break;
        }
    }
    if (world.hidden_synapse_count[child] > 0 &&
        uniform01(random_state) < clampf(mutation_strength * 0.5f, 0.0f, 0.2f)) {
        int32_t ordinal = static_cast<int32_t>(uniform01(random_state) *
            world.hidden_synapse_count[child]);
        for (int32_t word = 0; word < kHiddenMaskWords; ++word) {
            const size_t slot = hidden_column_slot(world, child, word);
            uint32_t bits = world.hidden_masks[slot];
            while (bits != 0u) {
                const uint32_t bit = bits & (~bits + 1u);
                if (ordinal-- == 0) {
                    world.hidden_masks[slot] &= ~bit;
                    --world.hidden_synapse_count[child];
                    return;
                }
                bits &= bits - 1u;
            }
        }
    }
}

__device__ __forceinline__ bool enable_extension_link(
    const DeviceWorld& world, int32_t bibite, int32_t edge,
    uint32_t& random_state)
{
    const size_t mask_slot = hidden_column_slot(world, bibite, edge >> 5);
    const uint32_t bit = 1u << (edge & 31);
    if ((world.extension_masks[mask_slot] & bit) != 0u) return false;
    const size_t weight_slot = dense_weight_slot(world, bibite, edge);
    if (isnan(__half2float(world.extension_weights[weight_slot])))
        world.extension_weights[weight_slot] =
            __float2half_rn(signed_uniform(random_state) * 0.5f);
    world.extension_masks[mask_slot] |= bit;
    ++world.extension_synapse_count[bibite];
    return true;
}

__device__ void initialise_extension_brain(
    const DeviceWorld& world, int32_t bibite, uint32_t& random_state)
{
    world.extension_synapse_count[bibite] = 0;
    for (int32_t word = 0; word < kExtensionMaskWords; ++word)
        world.extension_masks[hidden_column_slot(world, bibite, word)] = 0u;
    for (int32_t edge = 0; edge < kExtensionConnectionCount; ++edge)
        world.extension_weights[dense_weight_slot(world, bibite, edge)] =
            __float2half_rn(nanf(""));
    // Every stock input and action is present and has an initial live route.
    // Evolution may remove a link while preserving its dormant FP16 strength.
    for (int32_t source = 0; source < kExtraSensorCount; ++source) {
        const int32_t destination = static_cast<int32_t>(
            uniform01(random_state) * kHiddenWidth);
        enable_extension_link(world, bibite,
            destination * kExtraSensorCount + source, random_state);
    }
    for (int32_t output = 0; output < kExtraOutputCount; ++output) {
        const int32_t source = static_cast<int32_t>(
            uniform01(random_state) * kHiddenWidth);
        enable_extension_link(world, bibite,
            kExtensionInputEnd + output * kHiddenWidth + source,
            random_state);
    }
    for (int32_t extra = 0; extra < 12; ++extra) {
        const int32_t edge = static_cast<int32_t>(
            uniform01(random_state) * kExtensionConnectionCount);
        enable_extension_link(world, bibite, edge, random_state);
    }
}

__device__ void mutate_extension_brain(
    const DeviceWorld& world, int32_t child, int32_t parent,
    float mutation_strength, uint32_t& random_state)
{
    world.extension_synapse_count[child] = world.extension_synapse_count[parent];
    for (int32_t edge = 0; edge < kExtensionConnectionCount; ++edge)
        world.extension_weights[dense_weight_slot(world, child, edge)] =
            world.extension_weights[dense_weight_slot(world, parent, edge)];
    for (int32_t word = 0; word < kExtensionMaskWords; ++word) {
        uint32_t bits = world.extension_masks[hidden_column_slot(world, parent, word)];
        world.extension_masks[hidden_column_slot(world, child, word)] = bits;
        while (bits != 0u) {
            const int32_t edge = word * 32 + __ffs(bits) - 1;
            const float inherited = __half2float(world.extension_weights[
                dense_weight_slot(world, parent, edge)]);
            world.extension_weights[dense_weight_slot(world, child, edge)] =
                __float2half_rn(clampf(
                    inherited + mutation_noise(random_state) * mutation_strength,
                    -4.0f, 4.0f));
            bits &= bits - 1u;
        }
    }
    if (uniform01(random_state) < clampf(mutation_strength * 2.5f, 0.0f, 0.5f)) {
        for (int32_t attempt = 0; attempt < 16; ++attempt) {
            const int32_t edge = static_cast<int32_t>(
                uniform01(random_state) * kExtensionConnectionCount);
            if (enable_extension_link(world, child, edge, random_state)) break;
        }
    }
    if (world.extension_synapse_count[child] > 0 &&
        uniform01(random_state) < clampf(mutation_strength * 0.5f, 0.0f, 0.2f)) {
        int32_t ordinal = static_cast<int32_t>(uniform01(random_state) *
            world.extension_synapse_count[child]);
        for (int32_t word = 0; word < kExtensionMaskWords; ++word) {
            const size_t slot = hidden_column_slot(world, child, word);
            uint32_t bits = world.extension_masks[slot];
            while (bits != 0u) {
                const uint32_t bit = bits & (~bits + 1u);
                if (ordinal-- == 0) {
                    world.extension_masks[slot] &= ~bit;
                    --world.extension_synapse_count[child];
                    return;
                }
                bits &= bits - 1u;
            }
        }
    }
}

__host__ __device__ __forceinline__ size_t template_node_slot(
    const DeviceWorld& world,
    int32_t bibite,
    int32_t node)
{
    return static_cast<size_t>(node) * world.template_instance_capacity +
        world.template_instance[bibite];
}

__host__ __device__ __forceinline__ size_t template_synapse_slot(
    const DeviceWorld& world,
    int32_t bibite,
    int32_t synapse)
{
    return static_cast<size_t>(synapse) * world.template_instance_capacity +
        world.template_instance[bibite];
}

__host__ __device__ __forceinline__ size_t topology_node_slot(
    int32_t topology,
    int32_t node)
{
    return static_cast<size_t>(topology) * kMaxTemplateNodes + node;
}

__host__ __device__ __forceinline__ size_t topology_synapse_slot(
    int32_t topology,
    int32_t synapse)
{
    return static_cast<size_t>(topology) * kMaxTemplateSynapses + synapse;
}

__host__ __device__ __forceinline__ uint32_t pack_node_descriptor(
    int32_t type,
    int32_t sensor,
    int32_t action)
{
    return static_cast<uint32_t>(type & 0xff) |
        (static_cast<uint32_t>((sensor + 1) & 0xff) << 8) |
        (static_cast<uint32_t>((action + 1) & 0xff) << 16);
}

__host__ __device__ __forceinline__ int32_t descriptor_node_type(uint32_t descriptor)
{
    return static_cast<int32_t>(descriptor & 0xffu);
}

__host__ __device__ __forceinline__ int32_t descriptor_sensor(uint32_t descriptor)
{
    return static_cast<int32_t>((descriptor >> 8) & 0xffu) - 1;
}

__host__ __device__ __forceinline__ int32_t descriptor_action(uint32_t descriptor)
{
    return static_cast<int32_t>((descriptor >> 16) & 0xffu) - 1;
}

__host__ __device__ __forceinline__ uint16_t pack_synapse_edge(
    int32_t source,
    int32_t destination)
{
    return static_cast<uint16_t>(
        static_cast<uint16_t>(source & 0xff) |
        static_cast<uint16_t>((destination & 0xff) << 8));
}

__host__ __device__ __forceinline__ int32_t edge_source(uint16_t edge)
{
    return static_cast<int32_t>(edge & 0xffu);
}

__host__ __device__ __forceinline__ int32_t edge_destination(uint16_t edge)
{
    return static_cast<int32_t>((edge >> 8) & 0xffu);
}

__device__ __forceinline__ ContactGridView contact_grid_view(
    const DeviceWorld& world,
    int32_t buffer)
{
    const size_t cells = static_cast<size_t>(world.contact_grid_width) *
        world.contact_grid_width;
    const size_t bitmap_words = (cells + 31u) / 32u;
    const size_t bibites = static_cast<size_t>(world.max_bibites);
    return {
        world.contact_home_cell + bibites * buffer,
        world.contact_is_overflow + bibites * buffer,
        world.contact_cell_counts + cells * buffer,
        world.contact_cell_items + cells * kContactBucketCapacity * buffer,
        world.contact_overflow_heads + cells * buffer,
        world.contact_overflow_next + bibites * buffer,
        world.contact_occupied + bitmap_words * buffer,
        world.contact_occupied_cells + cells * buffer,
        world.contact_occupied_count + buffer,
        world.contact_overflow_count + buffer,
        world.large_contact_items + bibites * buffer,
        world.large_contact_count + buffer};
}

__device__ __forceinline__ int32_t wrap_spatial_index(int32_t value, int32_t size)
{
    return value & (size - 1);
}

__device__ __forceinline__ bool occupied_cell(const uint32_t* bitmap, int32_t cell)
{
    return (bitmap[cell >> 5] & (1u << (cell & 31))) != 0u;
}

__device__ __forceinline__ void mark_occupied(uint32_t* bitmap, int32_t cell)
{
    atomicOr(&bitmap[cell >> 5], 1u << (cell & 31));
}

__device__ __forceinline__ void mark_contact_occupied(
    const ContactGridView& grid,
    int32_t cell)
{
    const uint32_t bit = 1u << (cell & 31);
    const uint32_t previous = atomicOr(&grid.occupied[cell >> 5], bit);
    if ((previous & bit) == 0u) {
        const int32_t slot = atomicAdd(grid.occupied_count, 1);
        grid.occupied_cells[slot] = cell;
    }
}

__device__ __forceinline__ void insert_sense_item(
    const DeviceWorld& world,
    int32_t cell,
    int32_t item)
{
    const uint32_t bit = 1u << (cell & 31);
    const uint32_t previous = atomicOr(
        &world.sense_bibite_occupied[cell >> 5],
        bit);
    if ((previous & bit) == 0u) {
        const int32_t occupied_slot = atomicAdd(
            world.sense_bibite_occupied_count,
            1);
        world.sense_bibite_occupied_cells[occupied_slot] = cell;
    }
    const int32_t slot = atomicAdd(&world.sense_bibite_cell_counts[cell], 1);
    if (slot < kSenseBucketCapacity) {
        world.sense_bibite_cell_items[cell * kSenseBucketCapacity + slot] = item;
    } else {
        atomicAdd(world.sense_bibite_overflow_count, 1);
        world.sense_bibite_overflow_next[item] = atomicExch(
            &world.sense_bibite_overflow_heads[cell],
            item);
    }
}

__device__ __forceinline__ void insert_contact_item(
    const ContactGridView& grid,
    int32_t cell,
    int32_t item)
{
    mark_contact_occupied(grid, cell);
    const int32_t slot = atomicAdd(&grid.cell_counts[cell], 1);
    grid.home_cell[item] = cell;
    grid.is_overflow[item] = slot >= kContactBucketCapacity;
    if (slot < kContactBucketCapacity) {
        grid.cell_items[cell * kContactBucketCapacity + slot] = item;
    } else {
        atomicAdd(grid.overflow_count, 1);
        grid.overflow_next[item] = atomicExch(
            &grid.overflow_heads[cell],
            item);
    }
}

__device__ __forceinline__ void insert_pellet(
    const DeviceWorld& world,
    int32_t pellet)
{
    const int32_t cell = grid_cell(
        world,
        world.pellet_positions[pellet],
        world.sense_grid_width);
    uint32_t observed = world.pellet_cell_masks[cell];
    while (observed != 0xffffffffu) {
        const uint32_t available = ~observed;
        const int32_t slot = __ffs(static_cast<int32_t>(available)) - 1;
        const uint32_t bit = 1u << slot;
        const uint32_t previous = atomicCAS(
            &world.pellet_cell_masks[cell],
            observed,
            observed | bit);
        if (previous == observed) {
            world.pellet_cell_items[cell * kPelletBucketCapacity + slot] = pellet;
            world.pellet_hash_cell[pellet] = cell;
            world.pellet_hash_slot[pellet] = slot;
            mark_occupied(world.pellet_occupied, cell);
            return;
        }
        observed = previous;
    }

    world.pellet_hash_cell[pellet] = -2;
    world.pellet_hash_slot[pellet] = -1;
    atomicAdd(world.pellet_overflow_count, 1);
}

__device__ __forceinline__ void remove_pellet(
    const DeviceWorld& world,
    int32_t pellet)
{
    const int32_t cell = world.pellet_hash_cell[pellet];
    const int32_t slot = world.pellet_hash_slot[pellet];
    if (cell >= 0 && slot >= 0) {
        const uint32_t bit = 1u << slot;
        const uint32_t previous = atomicAnd(&world.pellet_cell_masks[cell], ~bit);
        if ((previous & ~bit) == 0u) {
            atomicAnd(
                &world.pellet_occupied[cell >> 5],
                ~(1u << (cell & 31)));
        }
    } else if (cell == -2) {
        atomicSub(world.pellet_overflow_count, 1);
    }
    world.pellet_hash_cell[pellet] = -1;
    world.pellet_hash_slot[pellet] = -1;
}

__device__ int32_t pheromone_cell(const DeviceWorld& world, float2 position)
{
    const float inverse_extent = 1.0f / (world.world_half_extent * 2.0f);
    int32_t x = static_cast<int32_t>((position.x + world.world_half_extent) *
        inverse_extent * world.pheromone_grid_width);
    int32_t y = static_cast<int32_t>((position.y + world.world_half_extent) *
        inverse_extent * world.pheromone_grid_height);
    x = max(0, min(world.pheromone_grid_width - 1, x));
    y = max(0, min(world.pheromone_grid_height - 1, y));
    return y * world.pheromone_grid_width + x;
}

__device__ void release_template_instance(const DeviceWorld& world, int index)
{
    const int instance = atomicExch(&world.template_instance[index], -1);
    if (instance >= 0) {
        const int free_slot = atomicAdd(world.template_free_count, 1);
        world.template_free_items[free_slot] = instance;
        atomicSub(world.template_living_count, 1);
    }
}

__global__ void assign_template_instance_kernel(DeviceWorld world, int index)
{
    if (threadIdx.x || blockIdx.x) return;
    const int free_slot = atomicSub(world.template_free_count,1)-1;
    world.template_instance[index] = world.template_free_items[free_slot];
    atomicAdd(world.template_living_count,1);
}

__device__ void cache_digestion(const DeviceWorld& world, int32_t index)
{
    const float diet = clampf(world.diet[index], 0.0f, 1.0f);
    const BgfWorldDigestionSettings d = world.digestion;
    world.digestion_efficiency[index] = {
        d.plant_min_efficiency + powf(1.0f - diet, d.plant_affinity_power) *
            (d.plant_max_efficiency - d.plant_min_efficiency),
        d.meat_min_efficiency + powf(diet, d.meat_affinity_power) *
            (d.meat_max_efficiency - d.meat_min_efficiency)};
}

__global__ void rebuild_runtime_kernel(DeviceWorld world)
{
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index == 0) *world.drag_retention =
        expf(-world.linear_drag * world.fixed_delta_time);
    if (index < world.max_bibites && world.alive[index] == 1) {
        const int32_t slot = atomicAdd(world.live_count, 1);
        world.live_items[slot] = index;
        cache_digestion(world, index);
    }
}

#include "bgf_food_segments.cuh"

__device__ void initialise_bibite(
    const DeviceWorld& world,
    int32_t index,
    uint32_t random_state,
    float2 position,
    float energy,
    int32_t generation,
    int32_t parent)
{
    uint32_t incarnation = world.instance_id[index] + 1u;
    world.instance_id[index] = incarnation == 0u ? 1u : incarnation;
    world.positions[index] = position;
    world.velocities[index] = {0.0f, 0.0f};
    world.headings[index] = uniform01(random_state) * 6.283185307179586f;
    world.energy[index] = energy;
    world.age[index] = 0.0f;
    world.reproduction_cooldown[index] = 5.0f + uniform01(random_state) * 10.0f;
    world.generation[index] = generation;
    world.rng[index] = random_state;
    world.actions_a[index] = {0.0f, 0.0f, 0.0f, 0.0f};
    world.actions_b[index] = {0.0f, 0.0f};
    world.actions_c[index] = {1.0f, 1.0f, 0.0f, 0.0f};
    world.actions_d[index] = {0.0f, 0.0f, 0.0f, 0.0f};
    world.repulsion[index] = {0.0f, 0.0f};
    world.food_gain[index] = 0.0f;
    world.meat_gain[index] = 0.0f;
    world.bite_cooldown[index] = 0.0f;
    world.pending_damage[index] = 0.0f;
    world.last_damage[index] = 0.0f;
    world.held_target[index] = 0;
    world.held_instance[index] = 0u;
    world.clock_reset_output[index] = 0.0f;
    world.wants_reproduction[index] = 0;
    world.cached_food[index] = -1;
    world.cached_neighbour[index] = -1;
    world.cached_neighbour_count[index] = 0;
    world.cached_food_direction[index] = {0.0f, 0.0f};
    world.cached_neighbour_direction[index] = {0.0f, 0.0f};
    world.cached_food_distance_squared[index] = world.sense_radius * world.sense_radius;
    world.cached_neighbour_distance_squared[index] = world.sense_radius * world.sense_radius;
    world.cached_neighbour_color[index] = {0.0f, 0.0f, 0.0f};

    if (parent < 0) {
        world.gene_mutation_strength[index] = world.mutation_strength;
        world.brain_mutation_strength[index] = world.mutation_strength;
        world.lineage_id[index] = 0ull;
        world.tag_id[index] = 0ull;
        world.template_brain[index] = 0;
        world.template_instance[index] = -1;
        world.native_brain_version[index] = 1;
        // An initially plant-fed world needs viable founders. Carnivory can
        // increase by inherited mutation after meat becomes available.
        world.diet[index] = uniform01(random_state) * 0.15f;
        const float body_size = 0.65f + uniform01(random_state) * 0.7f;
        world.size[index] = body_size;
        world.life_state[index] = {body_size, 0.0f, 0.0f, 1.0f};
        world.reproductive_traits[index] = {15.0f, 2.0f};
        world.traits[index] = {
            (3.0f + uniform01(random_state) * 3.0f) * world.mobility_scale,
            1.5f + uniform01(random_state) * 2.5f,
            0.15f + uniform01(random_state) * 0.2f,
            1800.0f + uniform01(random_state) * 3600.0f};
        world.colors[index] = {
            0.2f + uniform01(random_state) * 0.8f,
            0.2f + uniform01(random_state) * 0.8f,
            0.2f + uniform01(random_state) * 0.8f};
        for (int32_t weight = 0; weight < kWeightCount; ++weight) {
            world.weights[dense_weight_slot(world, index, weight)] =
                __float2half_rn(signed_uniform(random_state));
        }
        initialise_hidden_brain(world, index, random_state, true);
        initialise_extension_brain(world, index, random_state);
    } else {
        const float gene_mutation = world.gene_mutation_strength[parent];
        const float brain_mutation = world.brain_mutation_strength[parent];
        world.gene_mutation_strength[index] = gene_mutation;
        world.brain_mutation_strength[index] = brain_mutation;
        world.lineage_id[index] = world.lineage_id[parent];
        world.tag_id[index] = world.tag_id[parent];
        world.template_brain[index] = world.template_brain[parent];
        world.native_brain_version[index] = world.native_brain_version[parent];
        world.diet[index] = clampf(
            world.diet[parent] + mutation_noise(random_state) * gene_mutation * 0.1f,
            0.0f, 1.0f);
        const float4 parent_life = world.life_state[parent];
        const float adult_size = clampf(parent_life.x +
            mutation_noise(random_state) * gene_mutation * 0.2f,
            0.35f, 2.5f);
        world.size[index] = clampf(adult_size * 0.45f, 0.2f, adult_size);
        world.life_state[index] = {adult_size, 0.0f, 0.0f,
            fmaxf(0.05f, parent_life.w)};
        world.reproductive_traits[index] = world.reproductive_traits[parent];
        const float4 source_traits = world.traits[parent];
        world.traits[index] = {
            clampf(
                source_traits.x + mutation_noise(random_state) * gene_mutation *
                    world.mobility_scale,
                1.0f * world.mobility_scale,
                10.0f * world.mobility_scale),
            clampf(source_traits.y + mutation_noise(random_state) * gene_mutation, 0.5f, 8.0f),
            clampf(source_traits.z + mutation_noise(random_state) * gene_mutation * 0.05f, 0.05f, 0.8f),
            clampf(
                source_traits.w + mutation_noise(random_state) * gene_mutation * 600.0f,
                600.0f,
                21600.0f)};
        const float3 source_color = world.colors[parent];
        world.colors[index] = {
            clampf(source_color.x + mutation_noise(random_state) * gene_mutation * 0.1f, 0.0f, 1.0f),
            clampf(source_color.y + mutation_noise(random_state) * gene_mutation * 0.1f, 0.0f, 1.0f),
            clampf(source_color.z + mutation_noise(random_state) * gene_mutation * 0.1f, 0.0f, 1.0f)};
        if (world.template_brain[parent] != 0) {
            const int free_slot = atomicSub(world.template_free_count,1)-1;
            world.template_instance[index] = world.template_free_items[free_slot];
            atomicAdd(world.template_living_count,1);
            world.hidden_synapse_count[index] = 0;
            world.extension_synapse_count[index] = 0;
            for (int32_t word = 0; word < kHiddenMaskWords; ++word)
                world.hidden_masks[hidden_column_slot(world, index, word)] = 0u;
            const int32_t topology = world.template_brain[parent] - 1;
            const int32_t node_count = static_cast<int32_t>(
                world.template_topology_node_counts[topology]);
            const int32_t synapse_count = static_cast<int32_t>(
                world.template_topology_synapse_counts[topology]);
            for (int32_t node = 0; node < node_count; ++node) {
                const size_t source = template_node_slot(world, parent, node);
                const size_t destination = template_node_slot(world, index, node);
                const float bias = __half2float(
                    world.template_node_biases[source]);
                world.template_node_biases[destination] =
                    __float2half_rn(clampf(
                        bias + mutation_noise(random_state) * brain_mutation * 0.25f,
                        -16.0f,
                        16.0f));
                world.template_node_last_input[destination] = 0.0f;
                world.template_node_last_output[destination] = 0.0f;
            }
            for (int32_t synapse = 0; synapse < synapse_count; ++synapse) {
                const size_t source = template_synapse_slot(world, parent, synapse);
                const size_t destination = template_synapse_slot(world, index, synapse);
                const float inherited = __half2float(
                    world.template_synapse_weights[source]);
                world.template_synapse_weights[destination] =
                    __float2half_rn(clampf(
                        inherited + mutation_noise(random_state) * brain_mutation,
                        -16.0f,
                        16.0f));
            }
        } else {
            world.template_instance[index] = -1;
            for (int32_t weight = 0; weight < kWeightCount; ++weight) {
                const float inherited = __half2float(
                    world.weights[dense_weight_slot(world, parent, weight)]);
                const float mutated = inherited + mutation_noise(random_state) * brain_mutation;
                world.weights[dense_weight_slot(world, index, weight)] =
                    __float2half_rn(clampf(mutated, -4.0f, 4.0f));
            }
            mutate_hidden_brain(world, index, parent, brain_mutation, random_state);
            if (world.native_brain_version[parent] != 0)
                mutate_extension_brain(world, index, parent, brain_mutation, random_state);
            else world.extension_synapse_count[index] = 0;
        }
    }
    world.rng[index] = random_state;
    world.health[index] = 100.0f * world.size[index];
    cache_digestion(world, index);
    const int32_t live_slot = atomicAdd(world.live_count, 1);
    if (live_slot < world.max_bibites) world.live_items[live_slot] = index;
}

__global__ void initialise_world_kernel(DeviceWorld world)
{
    const int32_t thread = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t stride = static_cast<int32_t>(blockDim.x * gridDim.x);
    const ContactGridView initial_contact_grid = contact_grid_view(world, 0);
    if (thread == 0) {
        world.counters->living_bibites = world.initial_bibites;
        *world.free_bibite_count = world.max_bibites - world.initial_bibites;
        *world.reproduction_count = 0;
    }
    for (int32_t index = thread; index < world.max_bibites; index += stride) {
        if (index < world.initial_bibites) {
            uint32_t random_state = mix_bits(world.seed ^ (0x9e3779b9u * static_cast<uint32_t>(index + 1)));
            const float2 position{
                signed_uniform(random_state) * world.world_half_extent,
                signed_uniform(random_state) * world.world_half_extent};
            initialise_bibite(
                world,
                index,
                random_state,
                position,
                world.initial_energy * (0.8f + uniform01(random_state) * 0.4f),
                0,
                -1);
            world.alive[index] = 1;
            const int32_t contact_cell = grid_cell(
                world,
                position,
                world.contact_grid_width);
            insert_contact_item(initial_contact_grid, contact_cell, index);
        } else {
            // Host-side statistics and the legacy snapshot API transfer these
            // arrays in capacity-sized blocks before filtering on `alive`.
            // Give never-used slots deterministic values so those transfers
            // cannot expose stale device memory (and so CUDA initcheck remains
            // clean even though the host ignores dead records).
            world.alive[index] = 0;
            world.generation[index] = 0;
            world.positions[index] = {0.0f, 0.0f};
            world.velocities[index] = {0.0f, 0.0f};
            world.headings[index] = 0.0f;
            world.energy[index] = 0.0f;
            world.age[index] = 0.0f;
            world.size[index] = 0.0f;
            world.life_state[index] = {0.0f, 0.0f, 0.0f, 1.0f};
            world.reproductive_traits[index] = {15.0f, 2.0f};
            world.held_target[index] = 0;
            world.held_instance[index] = 0u;
            world.colors[index] = {0.0f, 0.0f, 0.0f};
            world.lineage_id[index] = 0ull;
            world.tag_id[index] = 0ull;
            world.template_brain[index] = 0;
            world.wants_reproduction[index] = 0;
            world.free_bibite_items[index - world.initial_bibites] = index;
        }
    }
    for (int32_t index = thread; index < world.pellet_count; index += stride) {
        uint32_t random_state = mix_bits(
            world.seed ^ 0xa511e9b3u ^ (0x85ebca6bu * static_cast<uint32_t>(index + 1)));
        float pellet_size = 1.0f;
        world.pellet_positions[index] = sample_food_zone_position(
            world, random_state, false, &pellet_size);
        world.pellet_rng[index] = random_state;
        world.pellet_active[index] = 1;
        world.pellet_held_by[index] = -1;
        world.pellet_nominal_units[index] = max(1, __float2int_rn(
            pellet_size * kPelletFoodUnits));
        world.pellet_food_units[index] = world.pellet_nominal_units[index];
        insert_pellet(world, index);
    }
}

__global__ void reseed_food_zone_pellets_kernel(
    DeviceWorld world, int32_t reseed_existing)
{
    for (int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < world.pellet_count; index += blockDim.x * gridDim.x) {
        if (world.pellet_active[index] == 2) continue;
        if (world.pellet_held_by[index] >= 0) continue;
        const bool active = world.pellet_active[index] == 1;
        if (active && reseed_existing == 0) continue;
        uint32_t random_state = world.pellet_rng[index];
        float pellet_size = 1.0f;
        world.pellet_positions[index] = sample_food_zone_position(
            world, random_state, !active, &pellet_size);
        world.pellet_nominal_units[index] = max(1, __float2int_rn(
            pellet_size * kPelletFoodUnits));
        if (active && reseed_existing != 0)
            world.pellet_food_units[index] = world.pellet_nominal_units[index];
        world.pellet_rng[index] = random_state;
    }
}

__global__ void move_initial_bibites_into_food_zones_kernel(DeviceWorld world)
{
    for (int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < world.initial_bibites; index += blockDim.x * gridDim.x) {
        if (world.alive[index] == 0) continue;
        uint32_t random_state = world.rng[index];
        world.positions[index] = sample_food_zone_position(
            world, random_state, false);
        world.rng[index] = random_state;
    }
}

__global__ void update_food_target_kernel(
    DeviceWorld world,
    int32_t old_target,
    int32_t new_target)
{
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (new_target < old_target) {
        const int32_t pellet = new_target + index;
        if (pellet >= old_target) return;
        if (world.pellet_active[pellet] == 1) {
            const int32_t holder = world.pellet_held_by[pellet];
            if (holder >= 0 && holder < world.max_bibites)
                atomicCAS(&world.held_target[holder], -pellet - 1, 0);
            world.pellet_held_by[pellet] = -1;
            world.pellet_active[pellet] = 0;
            world.pellet_food_units[pellet] = 0;
            remove_pellet(world, pellet);
        }
    } else {
        const int32_t pellet = old_target + index;
        if (pellet >= new_target) return;
        // An eaten pellet retains its place in the timing wheel. Do not
        // reactivate it early or allow two pending respawns for one slot.
        if (world.pellet_active[pellet] == 0 && world.pellet_pending[pellet] == 0) {
            world.pellet_food_units[pellet] = world.pellet_nominal_units[pellet];
            insert_pellet(world, pellet);
            world.pellet_active[pellet] = 1;
        }
    }
}

__global__ void rebuild_food_pending_kernel(DeviceWorld world)
{
    const int32_t bucket = blockIdx.x;
    const int32_t count = world.pellet_respawn_counts[bucket];
    for (int32_t slot = threadIdx.x; slot < count; slot += blockDim.x) {
        const int32_t pellet = world.pellet_respawn_items[
            static_cast<size_t>(bucket) * world.pellet_count + slot];
        if (pellet >= 0 && pellet < world.pellet_count) {
            world.pellet_pending[pellet] = 1;
        }
    }
}

__global__ void spawn_bibite_kernel(
    DeviceWorld world,
    float2 position,
    float heading,
    uint32_t random_seed,
    int32_t* spawned_slot)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    *spawned_slot = -1;
    const int32_t free_slot = atomicSub(world.free_bibite_count, 1) - 1;
    if (free_slot < 0) {
        atomicAdd(world.free_bibite_count, 1);
        return;
    }
    const int32_t candidate = world.free_bibite_items[free_slot];
    if (atomicCAS(&world.alive[candidate], 0, 2) != 0) {
        atomicAdd(world.free_bibite_count, 1);
        return;
    }
    uint32_t random_state = mix_bits(
        random_seed ^ static_cast<uint32_t>(candidate * 0x9e3779b9u));
    position.x = wrap_coordinate(position.x, world.world_half_extent);
    position.y = wrap_coordinate(position.y, world.world_half_extent);
    initialise_bibite(
        world,
        candidate,
        random_state,
        position,
        world.initial_energy,
        0,
        -1);
    world.headings[candidate] = heading;
    __threadfence();
    world.alive[candidate] = 1;
    const ContactGridView current_contact_grid = contact_grid_view(
        world,
        static_cast<int32_t>(world.counters->completed_steps & 1ull));
    if (world.size[candidate] <= kRegularMaximumBodySize) {
        const int32_t contact_cell = grid_cell(
            world,
            world.positions[candidate],
            world.contact_grid_width);
        insert_contact_item(current_contact_grid, contact_cell, candidate);
    } else {
        const int32_t slot = atomicAdd(current_contact_grid.large_count, 1);
        current_contact_grid.large_items[slot] = candidate;
    }
    atomicAdd(&world.counters->living_bibites, 1);
    atomicAdd(&world.counters->births, 1ull);
    *spawned_slot = candidate;
}

__global__ void set_hidden_link_enabled_kernel(
    DeviceWorld world, int32_t slot, int32_t edge,
    int32_t enabled, int32_t* result)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    *result = 0;
    if (world.alive[slot] != 1 || world.template_brain[slot] != 0) return;
    const bool extension = edge >= kHiddenConnectionCount;
    if (extension && world.native_brain_version[slot] == 0) return;
    const int32_t local_edge = extension ? edge - kHiddenConnectionCount : edge;
    const size_t mask_slot = hidden_column_slot(world, slot, local_edge >> 5);
    const uint32_t bit = 1u << (local_edge & 31);
    if (enabled != 0) {
        uint32_t random_state = world.rng[slot];
        if (extension)
            enable_extension_link(world, slot, local_edge, random_state);
        else enable_hidden_link(world, slot, local_edge, random_state);
        world.rng[slot] = random_state;
    } else if (extension) {
        if ((world.extension_masks[mask_slot] & bit) != 0u) {
            world.extension_masks[mask_slot] &= ~bit;
            --world.extension_synapse_count[slot];
        }
    } else if ((world.hidden_masks[mask_slot] & bit) != 0u) {
        world.hidden_masks[mask_slot] &= ~bit;
        --world.hidden_synapse_count[slot];
    }
    *result = 1;
}

__global__ void kill_bibite_kernel(
    DeviceWorld world,
    int32_t slot,
    int32_t* removed)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    *removed = 0;
    if (slot < 0 || slot >= world.max_bibites ||
        atomicCAS(&world.alive[slot], 1, 0) != 1) {
        return;
    }
    release_template_instance(world,slot);
    world.wants_reproduction[slot] = 0;
    const int32_t held = world.held_target[slot];
    if (held < 0) {
        const int32_t pellet = -held - 1;
        if (pellet >= 0 && pellet < world.pellet_count &&
            world.pellet_held_by[pellet] == slot)
            world.pellet_held_by[pellet] = -1;
    }
    world.held_target[slot] = 0;
    world.repulsion[slot] = {0.0f, 0.0f};
    const int32_t free_slot = atomicAdd(world.free_bibite_count, 1);
    world.free_bibite_items[free_slot] = slot;
    atomicAdd(&world.counters->deaths, 1ull);
    // Keep the existing public death-cause accounting invariant intact. A
    // user removal is closest to an externally forced starvation in the
    // compact native lifecycle model.
    atomicAdd(&world.counters->starvation_deaths, 1ull);
    atomicSub(&world.counters->living_bibites, 1);
    *world.contact_grid_dirty = 1;
    *removed = 1;
}

__global__ void release_dead_bibite_grabs_kernel(DeviceWorld world, int32_t slot)
{
    const int32_t target = slot + 1;
    for (int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < world.max_bibites; index += blockDim.x * gridDim.x) {
        if (world.alive[index] == 1)
            atomicCAS(&world.held_target[index], target, 0);
    }
}

__global__ void force_reproduction_kernel(
    DeviceWorld world,
    int32_t parent,
    uint32_t command_seed,
    int32_t* spawned_slot)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    *spawned_slot = -1;
    if (parent < 0 || parent >= world.max_bibites || world.alive[parent] != 1) {
        *spawned_slot = -2;
        return;
    }
    if (world.energy[parent] < world.reproduction_energy) {
        *spawned_slot = -3;
        return;
    }
    const int32_t free_slot = atomicSub(world.free_bibite_count, 1) - 1;
    if (free_slot < 0) {
        atomicAdd(world.free_bibite_count, 1);
        *spawned_slot = -4;
        return;
    }
    const int32_t child = world.free_bibite_items[free_slot];
    if (atomicCAS(&world.alive[child], 0, 2) != 0) {
        atomicAdd(world.free_bibite_count, 1);
        *spawned_slot = -5;
        return;
    }

    uint32_t random_state = mix_bits(
        world.rng[parent] ^ command_seed ^
        static_cast<uint32_t>(child * 0x9e3779b9u));
    const float child_energy = world.reproduction_energy * 0.32f;
    const float angle = uniform01(random_state) * 6.283185307179586f;
    float2 child_position = world.positions[parent];
    child_position.x = wrap_coordinate(
        child_position.x + sinf(angle) * world.size[parent] * 2.2f,
        world.world_half_extent);
    child_position.y = wrap_coordinate(
        child_position.y + cosf(angle) * world.size[parent] * 2.2f,
        world.world_half_extent);
    initialise_bibite(
        world,
        child,
        random_state,
        child_position,
        child_energy,
        world.generation[parent] + 1,
        parent);
    __threadfence();
    world.alive[child] = 1;
    world.energy[parent] -= child_energy;
    world.reproduction_cooldown[parent] = 12.0f + uniform01(random_state) * 12.0f;
    world.wants_reproduction[parent] = 0;
    world.rng[parent] = random_state;
    atomicAdd(&world.counters->living_bibites, 1);
    atomicAdd(&world.counters->births, 1ull);
    *world.contact_grid_dirty = 1;
    *spawned_slot = child;
}

__global__ void pack_bibite_detail_kernel(
    DeviceWorld world,
    int32_t slot,
    BgfWorldBibiteDetail* detail,
    BgfWorldBrainNodeState* nodes,
    BgfWorldBrainSynapseState* synapses)
{
    const int32_t thread = static_cast<int32_t>(threadIdx.x);
    if (thread == 0) {
        BgfWorldBibiteDetail result{};
        result.slot = slot;
        if (slot >= 0 && slot < world.max_bibites && world.alive[slot] == 1) {
            const float2 position = world.positions[slot];
            const float2 velocity = world.velocities[slot];
            const float4 traits = world.traits[slot];
            const float3 color = world.colors[slot];
            const float4 action_a = world.actions_a[slot];
            const float2 action_b = world.actions_b[slot];
            const float2 food_direction = world.cached_food_direction[slot];
            const float2 neighbour_direction = world.cached_neighbour_direction[slot];
            const float3 neighbour_color = world.cached_neighbour_color[slot];
            const int32_t topology = world.template_brain[slot] - 1;
            result.alive = 1;
            result.reserved = static_cast<int32_t>(world.instance_id[slot]);
            result.generation = world.generation[slot];
            const bool full_native = topology < 0 &&
                world.native_brain_version[slot] != 0;
            result.template_brain = topology >= 0 ? 1 : (full_native ? 2 : 0);
            result.brain_nodes = topology >= 0
                ? static_cast<int32_t>(world.template_topology_node_counts[topology])
                : (full_native ? kFullNativeBrainNodes : kLegacyNativeBrainNodes);
            result.brain_synapses = topology >= 0
                ? static_cast<int32_t>(world.template_topology_synapse_counts[topology])
                : kWeightCount + world.hidden_synapse_count[slot] +
                    (full_native ? world.extension_synapse_count[slot] : 0);
            result.cached_food = world.cached_food[slot];
            result.cached_neighbour = world.cached_neighbour[slot];
            result.neighbours_seen = world.cached_neighbour_count[slot];
            result.lineage_id = world.lineage_id[slot];
            result.tag_id = world.tag_id[slot];
            result.position_x = position.x;
            result.position_y = position.y;
            result.velocity_x = velocity.x;
            result.velocity_y = velocity.y;
            result.heading = world.headings[slot];
            result.energy = world.energy[slot];
            result.age = world.age[slot];
            result.size = world.size[slot];
            result.reproduction_cooldown = world.reproduction_cooldown[slot];
            result.maximum_speed = traits.x;
            result.turn_speed = traits.y;
            result.metabolism = traits.z;
            result.lifespan = traits.w;
            result.color_r = color.x;
            result.color_g = color.y;
            result.color_b = color.z;
            result.gene_mutation_strength = world.gene_mutation_strength[slot];
            result.brain_mutation_strength = world.brain_mutation_strength[slot];
            result.diet = world.diet[slot];
            result.health = world.health[slot];
            result.plant_stomach = world.food_gain[slot];
            result.meat_stomach = world.meat_gain[slot];
            result.eat_output = world.actions_c[slot].x;
            result.digestion_output = world.actions_c[slot].y;
            result.attack_output = world.actions_c[slot].w;
            result.acceleration_output = action_a.x;
            result.rotation_output = action_a.y;
            result.pheromone_1_output = action_a.z;
            result.pheromone_2_output = action_a.w;
            result.pheromone_3_output = action_b.x;
            result.reproduction_output = action_b.y;
            result.food_direction_x = food_direction.x;
            result.food_direction_y = food_direction.y;
            result.food_distance_squared = world.cached_food_distance_squared[slot];
            result.neighbour_direction_x = neighbour_direction.x;
            result.neighbour_direction_y = neighbour_direction.y;
            result.neighbour_distance_squared =
                world.cached_neighbour_distance_squared[slot];
            result.neighbour_color_r = neighbour_color.x;
            result.neighbour_color_g = neighbour_color.y;
            result.neighbour_color_b = neighbour_color.z;
            const float4 life = world.life_state[slot];
            result.adult_size = life.x;
            result.egg_progress = life.y;
            result.clock_time = life.z;
            result.clock_tic = static_cast<int32_t>(floorf(result.age /
                fmaxf(life.w, 0.05f))) & 1;
            result.growth_output = world.actions_d[slot].z;
            result.egg_production_output = world.actions_d[slot].y;
            result.grab_output = world.actions_c[slot].z;
            result.herding_output = world.actions_d[slot].x;
            result.clock_reset_output = world.clock_reset_output[slot];
            result.held_count = world.held_target[slot] != 0 ? 1 : 0;
        }
        *detail = result;
    }
    __syncthreads();
    if (slot < 0 || slot >= world.max_bibites || world.alive[slot] != 1) return;

    const int32_t topology = world.template_brain[slot] - 1;
    const bool full_native = topology < 0 && world.native_brain_version[slot] != 0;
    const int32_t native_sensor_count = full_native ? kStockSensorCount : kSensorCount;
    const int32_t node_count = topology >= 0
        ? static_cast<int32_t>(world.template_topology_node_counts[topology])
        : (full_native ? kFullNativeBrainNodes : kLegacyNativeBrainNodes);
    const int32_t synapse_count = topology >= 0
        ? static_cast<int32_t>(world.template_topology_synapse_counts[topology])
        : kWeightCount + world.hidden_synapse_count[slot] +
            (full_native ? world.extension_synapse_count[slot] : 0);

    for (int32_t node = thread; node < node_count; node += blockDim.x) {
        BgfWorldBrainNodeState state{};
        if (topology >= 0) {
            const uint32_t descriptor = world.template_node_descriptors[
                topology_node_slot(topology, node)];
            const size_t value_slot = template_node_slot(world, slot, node);
            state.type = descriptor_node_type(descriptor);
            state.sensor = descriptor_sensor(descriptor);
            state.action = descriptor_action(descriptor);
            state.base_activation = __half2float(world.template_node_biases[value_slot]);
            state.last_input = world.template_node_last_input[value_slot];
            state.last_output = world.template_node_last_output[value_slot];
        } else if (node < native_sensor_count) {
            state.type = 0;
            state.sensor = node;
            state.action = -1;
        } else if (node < native_sensor_count + kHiddenNodeCount) {
            const int32_t hidden = node - native_sensor_count;
            const size_t value_slot = hidden_column_slot(world, slot, hidden);
            state.type = 3;
            state.sensor = -1;
            state.action = -1;
            state.base_activation = __half2float(world.hidden_biases[value_slot]);
            state.last_input = __half2float(world.hidden_last_input[value_slot]);
            state.last_output = __half2float(world.hidden_last_output[value_slot]);
        } else {
            const int32_t action = node - native_sensor_count - kHiddenNodeCount;
            const float4 action_a = world.actions_a[slot];
            const float2 action_b = world.actions_b[slot];
            const float4 action_c = world.actions_c[slot];
            const float4 action_d = world.actions_d[slot];
            state.type = 3;
            state.sensor = -1;
            state.action = action;
            if (action == 0) state.last_output = action_a.x;
            else if (action == 1) state.last_output = action_a.y;
            else if (action == 2) state.last_output = action_a.z;
            else if (action == 3) state.last_output = action_a.w;
            else if (action == 4) state.last_output = action_b.x;
            else if (action == 5) state.last_output = action_b.y;
            else if (action == 6) state.last_output = action_d.x;
            else if (action == 7) state.last_output = action_d.y;
            else if (action == 8) state.last_output = action_c.x;
            else if (action == 9) state.last_output = action_c.y;
            else if (action == 10) state.last_output = action_c.z;
            else if (action == 11) state.last_output = world.clock_reset_output[slot];
            else if (action == 12) state.last_output = action_d.z;
            else if (action == 13) state.last_output = action_d.w;
            else state.last_output = action_c.w;
        }
        nodes[node] = state;
    }

    const int32_t direct_count = topology >= 0 ? synapse_count : kWeightCount;
    for (int32_t synapse = thread; synapse < direct_count; synapse += blockDim.x) {
        BgfWorldBrainSynapseState state{};
        if (topology >= 0) {
            const uint16_t edge = world.template_synapse_edges[
                topology_synapse_slot(topology, synapse)];
            state.node_in = edge_source(edge);
            state.node_out = edge_destination(edge);
            state.weight = __half2float(world.template_synapse_weights[
                template_synapse_slot(world, slot, synapse)]);
        } else {
            const int32_t output = synapse / kSensorCount;
            const int32_t sensor = synapse % kSensorCount;
            state.node_in = sensor;
            state.node_out = native_sensor_count + kHiddenNodeCount + output;
            state.weight = __half2float(world.weights[
                dense_weight_slot(world, slot, synapse)]);
        }
        synapses[synapse] = state;
    }
    if (topology < 0 && thread == 0) {
        int32_t destination = kWeightCount;
        for (int32_t word = 0; word < kHiddenMaskWords; ++word) {
            uint32_t bits = world.hidden_masks[hidden_column_slot(world, slot, word)];
            while (bits != 0u && destination < synapse_count) {
                const int32_t edge = word * 32 + __ffs(bits) - 1;
                BgfWorldBrainSynapseState state{};
                if (edge < kHiddenFirstEnd) {
                    state.node_in = edge % kSensorCount;
                    state.node_out = native_sensor_count + edge / kSensorCount;
                } else if (edge < kHiddenSecondEnd) {
                    const int32_t local = edge - kHiddenFirstEnd;
                    state.node_in = native_sensor_count + local % kHiddenWidth;
                    state.node_out = native_sensor_count + kHiddenWidth + local / kHiddenWidth;
                } else if (edge < kHiddenConnectionCount) {
                    const int32_t local = edge - kHiddenSecondEnd;
                    state.node_in = native_sensor_count + kHiddenWidth + local % kHiddenWidth;
                    state.node_out = native_sensor_count + kHiddenNodeCount + local / kHiddenWidth;
                } else {
                    break;
                }
                state.weight = __half2float(world.hidden_weights[
                    dense_weight_slot(world, slot, edge)]);
                synapses[destination++] = state;
                bits &= bits - 1u;
            }
        }
        if (full_native) {
            for (int32_t word = 0; word < kExtensionMaskWords; ++word) {
                uint32_t bits = world.extension_masks[hidden_column_slot(world, slot, word)];
                while (bits != 0u && destination < synapse_count) {
                    const int32_t edge = word * 32 + __ffs(bits) - 1;
                    if (edge >= kExtensionConnectionCount) break;
                    BgfWorldBrainSynapseState state{};
                    if (edge < kExtensionInputEnd) {
                        state.node_in = kSensorCount + edge % kExtraSensorCount;
                        state.node_out = native_sensor_count + edge / kExtraSensorCount;
                    } else {
                        const int32_t local = edge - kExtensionInputEnd;
                        state.node_in = native_sensor_count + kHiddenWidth +
                            local % kHiddenWidth;
                        state.node_out = native_sensor_count + kHiddenNodeCount +
                            kOutputCount + local / kHiddenWidth;
                    }
                    state.weight = __half2float(world.extension_weights[
                        dense_weight_slot(world, slot, edge)]);
                    synapses[destination++] = state;
                    bits &= bits - 1u;
                }
            }
        }
    }
}

__global__ void finish_template_spawn_kernel(DeviceWorld world)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    *world.contact_grid_dirty = 1;
}

__device__ float2 relative_direction(float2 direction, float heading)
{
    const float sine = sinf(heading);
    const float cosine = cosf(heading);
    const float2 forward{sine, cosine};
    const float2 right{cosine, -sine};
    return {
        direction.x * right.x + direction.y * right.y,
        direction.x * forward.x + direction.y * forward.y};
}

__device__ float2 safe_normalize(float2 value)
{
    const float squared = value.x * value.x + value.y * value.y;
    if (squared <= 1.0e-12f) return {0.0f, 0.0f};
    const float inverse = rsqrtf(squared);
    return {value.x * inverse, value.y * inverse};
}

__device__ float soft_latch(
    float bias,
    float last_output,
    float last_input,
    float input)
{
    if (input >= 0.999f) return 1.0f;
    if (input <= 0.001f) return 0.0f;
    if (last_input >= 0.999f) last_input = 1.0f;
    if (last_input <= 0.001f) last_input = 0.0f;
    if (input < last_input) {
        const float denominator = 1.0f - expf(-bias * last_input);
        return fabsf(denominator) > 1.0e-8f
            ? last_output / denominator * (1.0f - expf(-bias * input))
            : last_output;
    }
    const float exponent = expf(bias * (last_input - 1.0f));
    const float denominator = 1.0f - exponent;
    const float transition = fabsf(denominator) > 1.0e-8f
        ? (last_output - exponent) / denominator
        : last_output;
    return transition + (1.0f - transition) * expf(bias * (input - 1.0f));
}

__device__ float execute_template_neuron(
    int32_t type,
    float input,
    float bias,
    float last_input,
    float last_output,
    float period)
{
    if (type != 7 && type != 10 && type != 11 && type != 12 && type != 13) {
        input += bias;
    } else if (type == 10) {
        input *= bias;
    }
    float result = 0.0f;
    switch (type) {
        case 0: result = input; break;
        case 1: result = 1.0f / (1.0f + expf(-input)); break;
        case 2: result = input; break;
        case 3: result = tanhf(input); break;
        case 4: result = sinf(input); break;
        case 5: result = fmaxf(input, 0.0f); break;
        case 6: result = 1.0f / (1.0f + input * input); break;
        case 7:
            result = input >= 1.0f ? 1.0f : (input <= 0.0f ? 0.0f : last_output);
            break;
        case 8: result = (input - last_input) / fmaxf(period, 1.0e-6f); break;
        case 9: result = fabsf(input); break;
        case 10: result = input; break;
        case 11: result = last_output + input * period; break;
        case 12:
            result = (input - last_input) + last_output * expf(-bias * period);
            break;
        case 13: result = soft_latch(bias, last_output, last_input, input); break;
        default: result = 0.0f; break;
    }
    return clampf(result, -1024.0f, 1024.0f);
}

__device__ __forceinline__ void accumulate_contact_repulsion(
    const DeviceWorld& world,
    int32_t index,
    int32_t other,
    float2 position,
    float2& repulsion)
{
    if (other == index || world.alive[other] != 1) return;
    const float2 direction = wrapped_delta(
        position,
        world.positions[other],
        world.world_half_extent);
    const float distance_squared = direction.x * direction.x + direction.y * direction.y;
    if (distance_squared <= 1.0e-10f) return;
    const float contact_distance = world.size[index] + world.size[other];
    if (distance_squared >= contact_distance * contact_distance) return;
    const float distance = sqrtf(distance_squared);
    const float overlap = contact_distance - distance;
    repulsion.x -= direction.x / distance * overlap;
    repulsion.y -= direction.y / distance * overlap;
    const float2 relative{
        world.velocities[index].x - world.velocities[other].x,
        world.velocities[index].y - world.velocities[other].y};
    const float impact_speed = hypotf(relative.x, relative.y);
    if (impact_speed > world.collision_damage_threshold)
        atomicAdd(&world.pending_damage[index],
            (impact_speed - world.collision_damage_threshold) *
                world.collision_damage_constant * fminf(world.size[index],
                    world.size[other]));
}


__device__ void calculate_contact_repulsion(
    const DeviceWorld& world,
    const ContactGridView& grid,
    int32_t index,
    float2& repulsion)
{
    const float2 position = world.positions[index];
    const int32_t width = world.contact_grid_width;
    const int32_t origin_cell = grid_cell(world, position, width);
    const int32_t origin_x = origin_cell % width;
    const int32_t origin_y = origin_cell / width;
    const float cell_width = world.world_half_extent * 2.0f / static_cast<float>(width);
    const float search_distance = world.size[index] + kRegularMaximumBodySize;
    const int32_t radius = min(
        width / 2,
        max(1, static_cast<int32_t>(ceilf(search_distance / cell_width))));
    const bool scan_all = radius * 2 + 1 >= width;
    const int32_t span = scan_all ? width : radius * 2 + 1;

    repulsion = {0.0f, 0.0f};
    for (int32_t scan_y = 0; scan_y < span; ++scan_y) {
        const int32_t cell_y = scan_all
            ? scan_y
            : wrap_spatial_index(origin_y + scan_y - radius, width);
        for (int32_t scan_x = 0; scan_x < span; ++scan_x) {
            const int32_t cell_x = scan_all
                ? scan_x
                : wrap_spatial_index(origin_x + scan_x - radius, width);
            const int32_t cell = cell_y * width + cell_x;
            if (!occupied_cell(grid.occupied, cell)) continue;

            const int32_t fixed_count = min(
                grid.cell_counts[cell],
                kContactBucketCapacity);
            const int32_t item_offset = cell * kContactBucketCapacity;
            for (int32_t slot = 0; slot < fixed_count; ++slot) {
                accumulate_contact_repulsion(
                    world,
                    index,
                    grid.cell_items[item_offset + slot],
                    position,
                    repulsion);
            }
            const int32_t overflow_limit = max(0, min(world.max_bibites,
                grid.cell_counts[cell] - kContactBucketCapacity));
            int32_t visited = 0;
            for (int32_t other = grid.overflow_heads[cell];
                 other >= 0 && other < world.max_bibites && visited < overflow_limit;
                 ++visited) {
                accumulate_contact_repulsion(world, index, other, position, repulsion);
                 const int32_t next = grid.overflow_next[other];
                 if (next == other) break;
                 other = next;
            }
        }
    }
    const int32_t large_count = max(0, min(*grid.large_count, world.max_bibites));
    for (int32_t slot = 0; slot < large_count; ++slot) {
        accumulate_contact_repulsion(
            world,
            index,
            grid.large_items[slot],
            position,
            repulsion);
    }
}

__device__ __forceinline__ void accumulate_unique_contact_pair(
    const DeviceWorld& world,
    int32_t first,
    int32_t second,
    float2 first_position,
    float first_size,
    float2 second_position,
    float second_size)
{
    if (first == second || world.alive[first] != 1 || world.alive[second] != 1) return;
    const float2 direction = wrapped_delta(
        first_position,
        second_position,
        world.world_half_extent);
    const float distance_squared = direction.x * direction.x + direction.y * direction.y;
    if (distance_squared <= 1.0e-10f) return;
    const float contact_distance = first_size + second_size;
    if (distance_squared >= contact_distance * contact_distance) return;
    const float scale = (contact_distance * rsqrtf(distance_squared)) - 1.0f;
    const float2 contribution{direction.x * scale, direction.y * scale};
    atomicAdd(&world.repulsion[first].x, -contribution.x);
    atomicAdd(&world.repulsion[first].y, -contribution.y);
    atomicAdd(&world.repulsion[second].x, contribution.x);
    atomicAdd(&world.repulsion[second].y, contribution.y);
    const float2 relative{
        world.velocities[first].x - world.velocities[second].x,
        world.velocities[first].y - world.velocities[second].y};
    const float impact_speed = hypotf(relative.x, relative.y);
    if (impact_speed > world.collision_damage_threshold) {
        const float damage = (impact_speed - world.collision_damage_threshold) *
            world.collision_damage_constant * fminf(first_size, second_size);
        atomicAdd(&world.pending_damage[first], damage);
        atomicAdd(&world.pending_damage[second], damage);
    }
}

// A warp owns one crowded-cell tile, stages 32 candidates, and reuses
// their geometry for a tile of unique pairs. Serial overflow pointer chasing
// is done once per tile instead of once for every nearby Bibite.
__device__ int stage_contact_chunk(const DeviceWorld& world,
    const ContactGridView& grid, int cell, int begin, int count, int cursor,
    int* ids, float2* positions, float* sizes)
{
    const int lane=threadIdx.x&31;
    if (lane==0) {
        const int fixed=min(count,kContactBucketCapacity);
        for (int item=0;item<32;++item) {
            int id=-1;
            const int ordinal=begin+item;
            if (ordinal<count) {
                if (ordinal<fixed) id=grid.cell_items[cell*kContactBucketCapacity+ordinal];
                else if (cursor>=0 && cursor<world.max_bibites) {
                    id=cursor;
                    const int next=grid.overflow_next[cursor];
                    cursor=next==cursor ? -1 : next;
                }
            }
            ids[item]=id;
        }
    }
    cursor=__shfl_sync(0xffffffffu,cursor,0);
    __syncwarp();
    const int id=ids[lane];
    positions[lane]=id>=0 && id<world.max_bibites ? world.positions[id] : float2{0,0};
    sizes[lane]=id>=0 && id<world.max_bibites ? world.size[id] : 0.0f;
    __syncwarp();
    return cursor;
}

// Expand crowded cells into independent home-tile tasks. This prevents a
// handful of dense cells from monopolizing only a handful of GPU warps. The
// initial linked-list cursor is walked once per cell, not once per task.
__device__ void build_contact_tile_tasks(const DeviceWorld& world,
    const ContactGridView& contact,cg::grid_group grid,int thread,int stride)
{
    if (thread==0) *world.contact_tile_count=0;
    grid.sync();
    for (int occupied=thread;occupied<*contact.occupied_count;occupied+=stride) {
        const int cell=contact.occupied_cells[occupied];
        const int count=max(0,min(world.max_bibites,contact.cell_counts[cell]));
        const int tiles=(count+31)/32;
        const int start=atomicAdd(world.contact_tile_count,tiles);
        int cursor=contact.overflow_heads[cell];
        const int fixed=min(count,kContactBucketCapacity);
        for (int begin=0;begin<count;begin+=32) {
            const int slot=start+begin/32;
            if (slot<world.max_bibites)
                world.contact_tile_tasks[slot]={cell,begin,cursor};
            for (int item=max(begin,fixed);item<min(count,begin+32);++item) {
                if (cursor<0 || cursor>=world.max_bibites) break;
                const int next=contact.overflow_next[cursor];
                cursor=next==cursor ? -1 : next;
            }
        }
    }
    grid.sync();
}

__device__ void calculate_overflow_contact_tiles(
    const DeviceWorld& world,const ContactGridView& grid)
{
    constexpr int warps=kThreads/32;
    __shared__ int home_ids[warps][32],other_ids[warps][32];
    __shared__ float2 home_positions[warps][32],other_positions[warps][32];
    __shared__ float home_sizes[warps][32],other_sizes[warps][32];
    const int warp=threadIdx.x/32,lane=threadIdx.x&31;
    const int width=world.contact_grid_width;
    constexpr int ox[5]={0,1,0,1,-1},oy[5]={0,0,1,1,1};
    for (int task_slot=blockIdx.x*warps+warp;
         task_slot<min(*world.contact_tile_count,world.max_bibites);task_slot+=gridDim.x*warps) {
        const ContactTileTask task=world.contact_tile_tasks[task_slot];
        const int home=task.cell;
        const int home_count=max(0,min(world.max_bibites,grid.cell_counts[home]));
        int home_cursor=task.cursor;
        for (int home_begin=task.begin;home_begin<min(home_count,task.begin+32);home_begin+=32) {
            home_cursor=stage_contact_chunk(world,grid,home,home_begin,home_count,home_cursor,
                home_ids[warp],home_positions[warp],home_sizes[warp]);
            for (int offset=0;offset<5;++offset) {
                const int other=wrap_spatial_index(home/width+oy[offset],width)*width+
                    wrap_spatial_index(home%width+ox[offset],width);
                const int other_count=max(0,min(world.max_bibites,grid.cell_counts[other]));
                if (home_count<=kContactBucketCapacity && other_count<=kContactBucketCapacity) continue;
                int other_cursor=grid.overflow_heads[other];
                for (int other_begin=0;other_begin<other_count;other_begin+=32) {
                    other_cursor=stage_contact_chunk(world,grid,other,other_begin,other_count,
                        other_cursor,other_ids[warp],other_positions[warp],other_sizes[warp]);
                    const int first_count=min(32,home_count-home_begin);
                    const int second_count=min(32,other_count-other_begin);
                    for (int pair=lane;pair<first_count*second_count;pair+=32) {
                        const int first=pair/second_count,second=pair%second_count;
                        const int first_ordinal=home_begin+first,second_ordinal=other_begin+second;
                        // Fixed/fixed pairs were handled by the ordinary cell tiles.
                        if (first_ordinal<kContactBucketCapacity && second_ordinal<kContactBucketCapacity) continue;
                        if (offset==0 && second_ordinal<=first_ordinal) continue;
                        const int a=home_ids[warp][first],b=other_ids[warp][second];
                        if (a<0 || a>=world.max_bibites || b<0 || b>=world.max_bibites) continue;
                        accumulate_unique_contact_pair(world,a,b,
                            home_positions[warp][first],home_sizes[warp][first],
                            other_positions[warp][second],other_sizes[warp][second]);
                    }
                    __syncwarp();
                }
            }
            __syncwarp();
        }
    }
}

// Oversized bodies are not inserted into fixed/overflow cell buckets.
// Normal/large and large/large pairs each retain exactly one owner.
__device__ void calculate_supplemental_contact_pairs(
    const DeviceWorld& world, const ContactGridView& grid)
{
    if (*grid.overflow_count>0) calculate_overflow_contact_tiles(world,grid);
    const int count=max(0,min(*grid.large_count,world.max_bibites));
    if (count==0) return;
    for (int work=blockIdx.x*blockDim.x+threadIdx.x;work<*world.live_count;
         work+=blockDim.x*gridDim.x) {
        const int index=world.live_items[work];
        if (world.alive[index]!=1) continue;
        const bool large=world.size[index]>kRegularMaximumBodySize;
        for (int slot=0;slot<count;++slot) {
            const int other=grid.large_items[slot];
            if (index!=other && (!large || index<other))
                accumulate_unique_contact_pair(world,index,other,
                    world.positions[index],world.size[index],world.positions[other],world.size[other]);
        }
    }
}

// Each warp owns one occupied home cell. The five canonical neighbor
// offsets below ensure that fixed-bucket pairs are evaluated exactly once.
__device__ void calculate_unique_contact_pairs_tiled(
    const DeviceWorld& world,
    const ContactGridView& grid)
{
    constexpr int32_t kWarpsPerBlock = kThreads / 32;
    __shared__ int32_t home_ids[kWarpsPerBlock][kContactBucketCapacity];
    __shared__ int32_t neighbour_ids[kWarpsPerBlock][kContactBucketCapacity];
    __shared__ float2 home_positions[kWarpsPerBlock][kContactBucketCapacity];
    __shared__ float2 neighbour_positions[kWarpsPerBlock][kContactBucketCapacity];
    __shared__ float home_sizes[kWarpsPerBlock][kContactBucketCapacity];
    __shared__ float neighbour_sizes[kWarpsPerBlock][kContactBucketCapacity];

    const int32_t warp = static_cast<int32_t>(threadIdx.x) >> 5;
    const int32_t lane = static_cast<int32_t>(threadIdx.x) & 31;
    const int32_t warp_index = static_cast<int32_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const int32_t warp_stride = static_cast<int32_t>(gridDim.x) * kWarpsPerBlock;
    const int32_t occupied_count = *grid.occupied_count;
    const int32_t width = world.contact_grid_width;
    constexpr int32_t offsets_x[5] = {0, 1, 0, 1, -1};
    constexpr int32_t offsets_y[5] = {0, 0, 1, 1, 1};

    for (int32_t occupied_slot = warp_index;
         occupied_slot < occupied_count;
         occupied_slot += warp_stride) {
        const int32_t home_cell = grid.occupied_cells[occupied_slot];
        const int32_t home_count = min(
            grid.cell_counts[home_cell],
            kContactBucketCapacity);
        if (lane < home_count) {
            const int32_t item = grid.cell_items[
                home_cell * kContactBucketCapacity + lane];
            home_ids[warp][lane] = item;
            home_positions[warp][lane] = world.positions[item];
            home_sizes[warp][lane] = world.size[item];
        }
        __syncwarp();

        const int32_t home_x = home_cell % width;
        const int32_t home_y = home_cell / width;
        for (int32_t offset = 0; offset < 5; ++offset) {
            const int32_t neighbour_x = wrap_spatial_index(
                home_x + offsets_x[offset],
                width);
            const int32_t neighbour_y = wrap_spatial_index(
                home_y + offsets_y[offset],
                width);
            const int32_t neighbour_cell = neighbour_y * width + neighbour_x;
            if (!occupied_cell(grid.occupied, neighbour_cell)) continue;
            const int32_t neighbour_count = min(
                grid.cell_counts[neighbour_cell],
                kContactBucketCapacity);
            if (lane < neighbour_count) {
                const int32_t item = grid.cell_items[
                    neighbour_cell * kContactBucketCapacity + lane];
                neighbour_ids[warp][lane] = item;
                neighbour_positions[warp][lane] = world.positions[item];
                neighbour_sizes[warp][lane] = world.size[item];
            }
            __syncwarp();

            if (offset == 0) {
                int32_t pair = 0;
                for (int32_t first_slot = 0; first_slot < home_count; ++first_slot) {
                    for (int32_t second_slot = first_slot + 1;
                         second_slot < home_count;
                         ++second_slot, ++pair) {
                        if ((pair & 31) != lane) continue;
                        accumulate_unique_contact_pair(
                            world,
                            home_ids[warp][first_slot],
                            home_ids[warp][second_slot],
                            home_positions[warp][first_slot],
                            home_sizes[warp][first_slot],
                            home_positions[warp][second_slot],
                            home_sizes[warp][second_slot]);
                    }
                }
            } else {
                const int32_t pair_count = home_count * neighbour_count;
                for (int32_t pair = lane; pair < pair_count; pair += 32) {
                    const int32_t first_slot = pair / neighbour_count;
                    const int32_t second_slot = pair - first_slot * neighbour_count;
                    accumulate_unique_contact_pair(
                        world,
                        home_ids[warp][first_slot],
                        neighbour_ids[warp][second_slot],
                        home_positions[warp][first_slot],
                        home_sizes[warp][first_slot],
                        neighbour_positions[warp][second_slot],
                        neighbour_sizes[warp][second_slot]);
                }
            }
            __syncwarp();
        }
    }
}

// Four independent eight-lane groups share each physical warp.  Contact cells
// are deliberately sparse, so this retains shared-memory reuse without
// dedicating 32 lanes to the common one-to-four-body case.
__device__ void calculate_unique_contact_pairs_subwarp_tiled(
    const DeviceWorld& world,
    const ContactGridView& grid)
{
    constexpr int32_t kGroupSize = 8;
    constexpr int32_t kGroupsPerBlock = kThreads / kGroupSize;
    __shared__ int32_t home_ids[kGroupsPerBlock][kContactBucketCapacity];
    __shared__ int32_t neighbour_ids[kGroupsPerBlock][kContactBucketCapacity];
    __shared__ float2 home_positions[kGroupsPerBlock][kContactBucketCapacity];
    __shared__ float2 neighbour_positions[kGroupsPerBlock][kContactBucketCapacity];
    __shared__ float home_sizes[kGroupsPerBlock][kContactBucketCapacity];
    __shared__ float neighbour_sizes[kGroupsPerBlock][kContactBucketCapacity];

    const int32_t local_group = static_cast<int32_t>(threadIdx.x) / kGroupSize;
    const int32_t lane = static_cast<int32_t>(threadIdx.x) & (kGroupSize - 1);
    const int32_t group_in_warp =
        (static_cast<int32_t>(threadIdx.x) & 31) / kGroupSize;
    const unsigned int group_mask =
        ((1u << kGroupSize) - 1u) << (group_in_warp * kGroupSize);
    const int32_t group_index =
        static_cast<int32_t>(blockIdx.x) * kGroupsPerBlock + local_group;
    const int32_t group_stride = static_cast<int32_t>(gridDim.x) * kGroupsPerBlock;
    const int32_t occupied_count = *grid.occupied_count;
    const int32_t width = world.contact_grid_width;
    constexpr int32_t offsets_x[5] = {0, 1, 0, 1, -1};
    constexpr int32_t offsets_y[5] = {0, 0, 1, 1, 1};

    for (int32_t occupied_slot = group_index;
         occupied_slot < occupied_count;
         occupied_slot += group_stride) {
        const int32_t home_cell = grid.occupied_cells[occupied_slot];
        const int32_t home_count = min(
            grid.cell_counts[home_cell],
            kContactBucketCapacity);
        for (int32_t slot = lane; slot < home_count; slot += kGroupSize) {
            const int32_t item = grid.cell_items[
                home_cell * kContactBucketCapacity + slot];
            home_ids[local_group][slot] = item;
            home_positions[local_group][slot] = world.positions[item];
            home_sizes[local_group][slot] = world.size[item];
        }
        __syncwarp(group_mask);

        const int32_t home_x = home_cell % width;
        const int32_t home_y = home_cell / width;
        for (int32_t offset = 0; offset < 5; ++offset) {
            const int32_t neighbour_x = wrap_spatial_index(
                home_x + offsets_x[offset],
                width);
            const int32_t neighbour_y = wrap_spatial_index(
                home_y + offsets_y[offset],
                width);
            const int32_t neighbour_cell = neighbour_y * width + neighbour_x;
            if (!occupied_cell(grid.occupied, neighbour_cell)) continue;
            const int32_t neighbour_count = min(
                grid.cell_counts[neighbour_cell],
                kContactBucketCapacity);
            for (int32_t slot = lane; slot < neighbour_count; slot += kGroupSize) {
                const int32_t item = grid.cell_items[
                    neighbour_cell * kContactBucketCapacity + slot];
                neighbour_ids[local_group][slot] = item;
                neighbour_positions[local_group][slot] = world.positions[item];
                neighbour_sizes[local_group][slot] = world.size[item];
            }
            __syncwarp(group_mask);

            if (offset == 0) {
                int32_t pair = 0;
                for (int32_t first_slot = 0; first_slot < home_count; ++first_slot) {
                    for (int32_t second_slot = first_slot + 1;
                         second_slot < home_count;
                         ++second_slot, ++pair) {
                        if ((pair & (kGroupSize - 1)) != lane) continue;
                        accumulate_unique_contact_pair(
                            world,
                            home_ids[local_group][first_slot],
                            home_ids[local_group][second_slot],
                            home_positions[local_group][first_slot],
                            home_sizes[local_group][first_slot],
                            home_positions[local_group][second_slot],
                            home_sizes[local_group][second_slot]);
                    }
                }
            } else {
                const int32_t pair_count = home_count * neighbour_count;
                for (int32_t pair = lane; pair < pair_count; pair += kGroupSize) {
                    const int32_t first_slot = pair / neighbour_count;
                    const int32_t second_slot = pair - first_slot * neighbour_count;
                    accumulate_unique_contact_pair(
                        world,
                        home_ids[local_group][first_slot],
                        neighbour_ids[local_group][second_slot],
                        home_positions[local_group][first_slot],
                        home_sizes[local_group][first_slot],
                        neighbour_positions[local_group][second_slot],
                        neighbour_sizes[local_group][second_slot]);
                }
            }
            __syncwarp(group_mask);
        }
    }
}

// Sparse worlds commonly have only one or two Bibites in each occupied cell.
// Giving a whole warp to such a cell leaves almost every lane idle and spends
// more time in warp barriers than in distance tests.  This path assigns one
// occupied cell to each CUDA thread while retaining the same canonical
// half-neighbourhood and exact one-evaluation-per-pair behaviour as the tiled
// path.  Dense cells still use the shared-memory warp implementation above.
__device__ void calculate_unique_contact_pairs_sparse_cells(
    const DeviceWorld& world,
    const ContactGridView& grid)
{
    const int32_t thread = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t stride = static_cast<int32_t>(blockDim.x * gridDim.x);
    const int32_t occupied_count = *grid.occupied_count;
    const int32_t width = world.contact_grid_width;
    constexpr int32_t offsets_x[5] = {0, 1, 0, 1, -1};
    constexpr int32_t offsets_y[5] = {0, 0, 1, 1, 1};

    for (int32_t occupied_slot = thread;
         occupied_slot < occupied_count;
         occupied_slot += stride) {
        const int32_t home_cell = grid.occupied_cells[occupied_slot];
        const int32_t home_count = min(
            grid.cell_counts[home_cell],
            kContactBucketCapacity);
        const int32_t home_offset = home_cell * kContactBucketCapacity;
        const int32_t home_x = home_cell % width;
        const int32_t home_y = home_cell / width;

        for (int32_t offset = 0; offset < 5; ++offset) {
            const int32_t neighbour_x = wrap_spatial_index(
                home_x + offsets_x[offset],
                width);
            const int32_t neighbour_y = wrap_spatial_index(
                home_y + offsets_y[offset],
                width);
            const int32_t neighbour_cell = neighbour_y * width + neighbour_x;
            if (!occupied_cell(grid.occupied, neighbour_cell)) continue;
            const int32_t neighbour_count = min(
                grid.cell_counts[neighbour_cell],
                kContactBucketCapacity);
            const int32_t neighbour_offset = neighbour_cell * kContactBucketCapacity;

            if (offset == 0) {
                for (int32_t first_slot = 0; first_slot < home_count; ++first_slot) {
                    const int32_t first = grid.cell_items[home_offset + first_slot];
                    const float2 first_position = world.positions[first];
                    const float first_size = world.size[first];
                    for (int32_t second_slot = first_slot + 1;
                         second_slot < home_count;
                         ++second_slot) {
                        const int32_t second =
                            grid.cell_items[home_offset + second_slot];
                        accumulate_unique_contact_pair(
                            world,
                            first,
                            second,
                            first_position,
                            first_size,
                            world.positions[second],
                            world.size[second]);
                    }
                }
            } else {
                for (int32_t first_slot = 0; first_slot < home_count; ++first_slot) {
                    const int32_t first = grid.cell_items[home_offset + first_slot];
                    const float2 first_position = world.positions[first];
                    const float first_size = world.size[first];
                    for (int32_t second_slot = 0;
                         second_slot < neighbour_count;
                         ++second_slot) {
                        const int32_t second =
                            grid.cell_items[neighbour_offset + second_slot];
                        accumulate_unique_contact_pair(
                            world,
                            first,
                            second,
                            first_position,
                            first_size,
                            world.positions[second],
                            world.size[second]);
                    }
                }
            }
        }
    }
}

__device__ __forceinline__ void consider_pellet(
    const DeviceWorld& world,
    int32_t index,
    float2 position,
    int32_t pellet,
    int32_t& nearest_food,
    float& nearest_distance_squared,
    float2& nearest_direction)
{
    if (pellet < 0 || pellet >= world.pellet_count ||
        (world.pellet_active[pellet] != 1 && world.pellet_active[pellet] != 2) ||
        (world.pellet_held_by[pellet] != -1 &&
            world.pellet_held_by[pellet] != index)) return;
    const float2 direction = wrapped_delta(
        position,
        world.pellet_positions[pellet],
        world.world_half_extent);
    const float distance_squared = direction.x * direction.x + direction.y * direction.y;
    if (distance_squared < nearest_distance_squared) {
        nearest_food = pellet;
        nearest_distance_squared = distance_squared;
        nearest_direction = direction;
    }
}

__device__ __forceinline__ void consider_neighbour(
    const DeviceWorld& world,
    int32_t index,
    float2 position,
    int32_t other,
    float sense_squared,
    int32_t& nearest_neighbour,
    int32_t& neighbours_seen,
    float& nearest_distance_squared,
    float2& nearest_direction)
{
    if (other == index || other < 0 || other >= world.max_bibites ||
        world.alive[other] != 1) {
        return;
    }
    const float2 direction = wrapped_delta(
        position,
        world.positions[other],
        world.world_half_extent);
    const float distance_squared = direction.x * direction.x + direction.y * direction.y;
    if (distance_squared > sense_squared || distance_squared <= 1.0e-10f) return;
    ++neighbours_seen;
    if (distance_squared < nearest_distance_squared) {
        nearest_neighbour = other;
        nearest_distance_squared = distance_squared;
        nearest_direction = direction;
    }
}

__device__ void lookup_perception_from_cell(
    const DeviceWorld& world,
    int32_t index,
    int32_t origin_cell,
    bool skip_food = false)
{
    const float2 position = world.positions[index];
    const float sense_squared = world.sense_radius * world.sense_radius;
    float nearest_food_squared = sense_squared;
    float nearest_neighbour_squared = sense_squared;
    float2 nearest_food_direction{0.0f, 0.0f};
    float2 nearest_neighbour_direction{0.0f, 0.0f};
    int32_t nearest_food = -1;
    int32_t nearest_neighbour = -1;
    int32_t neighbours_seen = 0;

    // In high-throughput mode, keep following a still-valid food target instead
    // of repeating the much more expensive all-pellet nearest-neighbour query.
    // The exact lookup still runs immediately when the target is eaten, leaves
    // sensor range, or has never been acquired.
    bool scan_food = !skip_food && (world.diagnostic_mask & 8192u) == 0u;
    const bool scan_neighbours = (world.diagnostic_mask & 16384u) == 0u;
    if (scan_food &&
        (world.lock_food_target != 0 ||
            (world.diagnostic_mask & 4096u) != 0u)) {
        const int32_t cached = world.cached_food[index];
        if (cached >= 0 && cached < world.pellet_count &&
            (world.pellet_active[cached] == 1 || world.pellet_active[cached] == 2) &&
            (world.pellet_held_by[cached] == -1 ||
                world.pellet_held_by[cached] == index)) {
            const float2 direction = wrapped_delta(
                position,
                world.pellet_positions[cached],
                world.world_half_extent);
            const float distance_squared =
                direction.x * direction.x + direction.y * direction.y;
            if (distance_squared <= sense_squared) {
                nearest_food = cached;
                nearest_food_squared = distance_squared;
                nearest_food_direction = direction;
                scan_food = false;
            }
        }
    }

    const int32_t width = world.sense_grid_width;
    const int32_t origin_x = origin_cell % width;
    const int32_t origin_y = origin_cell / width;
    const int32_t radius = world.sense_cell_radius;
    const bool scan_all = radius * 2 + 1 >= width;
    const int32_t span = scan_all ? width : radius * 2 + 1;
    for (int32_t scan_y = 0; scan_y < span; ++scan_y) {
        const int32_t cell_y = scan_all
            ? scan_y
            : wrap_spatial_index(origin_y + scan_y - radius, width);
        for (int32_t scan_x = 0; scan_x < span; ++scan_x) {
            const int32_t cell_x = scan_all
                ? scan_x
                : wrap_spatial_index(origin_x + scan_x - radius, width);
            const int32_t cell = cell_y * width + cell_x;

            if (scan_food && occupied_cell(world.pellet_occupied, cell)) {
                uint32_t mask = world.pellet_cell_masks[cell];
                const int32_t item_offset = cell * kPelletBucketCapacity;
                while (mask != 0u) {
                    const int32_t slot = __ffs(static_cast<int32_t>(mask)) - 1;
                    mask &= mask - 1u;
                    consider_pellet(
                        world,
                        index,
                        position,
                        world.pellet_cell_items[item_offset + slot],
                        nearest_food,
                        nearest_food_squared,
                        nearest_food_direction);
                }
            }

            if (scan_food) {
                const int32_t start=world.pellet_overflow_heads[cell];
                const int32_t count=start<0 || start>=world.pellet_count ? 0 :
                    max(0,min(world.pellet_overflow_counts[cell],world.pellet_count-start));
                for (int32_t item=0;item<count;++item)
                    consider_pellet(world,index,position,world.pellet_overflow_next[start+item],
                        nearest_food,nearest_food_squared,nearest_food_direction);
            }

            if (scan_neighbours &&
                occupied_cell(world.sense_bibite_occupied, cell)) {
                const int32_t fixed_count = min(
                    world.sense_bibite_cell_counts[cell],
                    kSenseBucketCapacity);
                const int32_t item_offset = cell * kSenseBucketCapacity;
                for (int32_t slot = 0; slot < fixed_count; ++slot) {
                    consider_neighbour(
                        world,
                        index,
                        position,
                        world.sense_bibite_cell_items[item_offset + slot],
                        sense_squared,
                        nearest_neighbour,
                        neighbours_seen,
                        nearest_neighbour_squared,
                        nearest_neighbour_direction);
                }
                const int32_t overflow_limit = max(0, min(world.max_bibites,
                    world.sense_bibite_cell_counts[cell] - kSenseBucketCapacity));
                int32_t visited = 0;
                for (int32_t other = world.sense_bibite_overflow_heads[cell];
                     other >= 0 && other < world.max_bibites && visited < overflow_limit;
                     ++visited) {
                    consider_neighbour(
                        world,
                        index,
                        position,
                        other,
                        sense_squared,
                        nearest_neighbour,
                        neighbours_seen,
                        nearest_neighbour_squared,
                        nearest_neighbour_direction);
                    const int32_t next = world.sense_bibite_overflow_next[other];
                    if (next == other) break;
                    other = next;
                }
            }
        }
    }

    // A consolidated full-brain query will update food itself. Do not erase
    // an existing lock before that query gets the chance to validate it.
    if (!skip_food) {
        world.cached_food[index]=nearest_food;
        world.cached_food_distance_squared[index]=nearest_food_squared;
        world.cached_food_direction[index]=nearest_food>=0
            ? relative_direction(safe_normalize(nearest_food_direction),world.headings[index])
            : float2{0.0f,0.0f};
    }
    world.cached_neighbour[index] = nearest_neighbour;
    world.cached_neighbour_count[index] = neighbours_seen;
    world.cached_neighbour_distance_squared[index] = nearest_neighbour_squared;
    world.cached_neighbour_direction[index] = nearest_neighbour >= 0
        ? relative_direction(safe_normalize(nearest_neighbour_direction), world.headings[index])
        : float2{0.0f, 0.0f};
    world.cached_neighbour_color[index] = nearest_neighbour >= 0
        ? world.colors[nearest_neighbour]
        : float3{0.0f, 0.0f, 0.0f};
}

__device__ void refresh_perception(const DeviceWorld& world, int32_t index)
{
    const float2 position = world.positions[index];
    const float sense_squared = world.sense_radius * world.sense_radius;
    const float heading = world.headings[index];

    const int32_t food = world.cached_food[index];
    if (food >= 0 && food < world.pellet_count &&
        (world.pellet_active[food] == 1 || world.pellet_active[food] == 2) &&
        (world.pellet_held_by[food] == -1 ||
            world.pellet_held_by[food] == index)) {
        const float2 direction = wrapped_delta(
            position,
            world.pellet_positions[food],
            world.world_half_extent);
        const float distance_squared = direction.x * direction.x + direction.y * direction.y;
        if (distance_squared <= sense_squared) {
            world.cached_food_distance_squared[index] = distance_squared;
            world.cached_food_direction[index] = relative_direction(
                safe_normalize(direction),
                heading);
        } else {
            world.cached_food[index] = -1;
            world.cached_food_distance_squared[index] = sense_squared;
            world.cached_food_direction[index] = {0.0f, 0.0f};
        }
    } else {
        world.cached_food[index] = -1;
        world.cached_food_distance_squared[index] = sense_squared;
        world.cached_food_direction[index] = {0.0f, 0.0f};
    }

    const int32_t neighbour = world.cached_neighbour[index];
    if (neighbour >= 0 && neighbour < world.max_bibites && neighbour != index &&
        world.alive[neighbour] == 1) {
        const float2 direction = wrapped_delta(
            position,
            world.positions[neighbour],
            world.world_half_extent);
        const float distance_squared = direction.x * direction.x + direction.y * direction.y;
        if (distance_squared <= sense_squared && distance_squared > 1.0e-10f) {
            world.cached_neighbour_distance_squared[index] = distance_squared;
            world.cached_neighbour_direction[index] = relative_direction(
                safe_normalize(direction),
                heading);
            world.cached_neighbour_color[index] = world.colors[neighbour];
        } else {
            world.cached_neighbour[index] = -1;
            world.cached_neighbour_distance_squared[index] = sense_squared;
            world.cached_neighbour_direction[index] = {0.0f, 0.0f};
            world.cached_neighbour_color[index] = {0.0f, 0.0f, 0.0f};
        }
    } else {
        world.cached_neighbour[index] = -1;
        world.cached_neighbour_distance_squared[index] = sense_squared;
        world.cached_neighbour_direction[index] = {0.0f, 0.0f};
        world.cached_neighbour_color[index] = {0.0f, 0.0f, 0.0f};
    }
}


__device__ __forceinline__ void consider_template_food(
    const DeviceWorld& world,
    int32_t index,
    float2 position,
    float heading,
    int32_t pellet,
    TemplateFoodSenses& senses)
{
    if (pellet < 0 || pellet >= world.pellet_count) return;
    if (world.pellet_held_by[pellet] != -1 &&
        world.pellet_held_by[pellet] != index) return;
    const int32_t material = world.pellet_active[pellet];
    if (material != 1 && material != 2) return;
    const float2 delta = wrapped_delta(position,
        world.pellet_positions[pellet], world.world_half_extent);
    const float squared = delta.x * delta.x + delta.y * delta.y;
    const float radius_squared = world.sense_radius * world.sense_radius;
    if (squared > radius_squared) return;
    if (material == 1) {
        ++senses.plants_seen;
        if (squared < senses.plant_distance_squared ||
            (squared==senses.plant_distance_squared && senses.plant>=0 && pellet<senses.plant)) {
            senses.plant = pellet;
            senses.plant_distance_squared = squared;
            senses.plant_direction = delta;
        }
    } else {
        ++senses.meats_seen;
        if (squared < senses.meat_distance_squared ||
            (squared==senses.meat_distance_squared && senses.meat>=0 && pellet<senses.meat)) {
            senses.meat = pellet;
            senses.meat_distance_squared = squared;
            senses.meat_direction = delta;
        }
    }
}

__device__ TemplateFoodSenses lookup_template_food_senses(
    const DeviceWorld& world, int32_t index)
{
    TemplateFoodSenses senses{};
    const float radius_squared = world.sense_radius * world.sense_radius;
    senses.plant_distance_squared = radius_squared;
    senses.meat_distance_squared = radius_squared;
    const float2 position = world.positions[index];
    const float heading = world.headings[index];
    const int32_t width = world.sense_grid_width;
    const int32_t origin = grid_cell(world, position, width);
    const int32_t origin_x = origin % width;
    const int32_t origin_y = origin / width;
    const int32_t radius = world.sense_cell_radius;
    const bool scan_all = radius * 2 + 1 >= width;
    const int32_t span = scan_all ? width : radius * 2 + 1;
    for (int32_t scan_y = 0; scan_y < span; ++scan_y) {
        const int32_t cell_y = scan_all ? scan_y :
            wrap_spatial_index(origin_y + scan_y - radius, width);
        for (int32_t scan_x = 0; scan_x < span; ++scan_x) {
            const int32_t cell_x = scan_all ? scan_x :
                wrap_spatial_index(origin_x + scan_x - radius, width);
            const int32_t cell = cell_y * width + cell_x;
            uint32_t mask = world.pellet_cell_masks[cell];
            const int32_t offset = cell * kPelletBucketCapacity;
            while (mask != 0u) {
                const int32_t slot = __ffs(static_cast<int32_t>(mask)) - 1;
                mask &= mask - 1u;
                consider_template_food(world, index, position, heading,
                    world.pellet_cell_items[offset + slot], senses);
            }
            const int32_t start=world.pellet_overflow_heads[cell];
            const int32_t count=start<0 || start>=world.pellet_count ? 0 :
                max(0,min(world.pellet_overflow_counts[cell],world.pellet_count-start));
            for (int32_t item=0;item<count;++item)
                consider_template_food(world,index,position,heading,
                    world.pellet_overflow_next[start+item],senses);
        }
    }
    float sn, cs;
    sincosf(heading, &sn, &cs);
    if (senses.plant >= 0) {
        const float2 d = safe_normalize(senses.plant_direction);
        senses.plant_direction = {d.x * cs - d.y * sn, d.x * sn + d.y * cs};
    }
    if (senses.meat >= 0) {
        const float2 d = safe_normalize(senses.meat_direction);
        senses.meat_direction = {d.x * cs - d.y * sn, d.x * sn + d.y * cs};
    }
    return senses;
}

#include "bgf_dense_food_query.cuh"

__device__ __forceinline__ float template_sensor_value(
    const DeviceWorld& world,
    int32_t index,
    int32_t sensor,
    const TemplateFoodSenses& food_senses,
    int32_t nearest_neighbour,
    float nearest_neighbour_squared,
    float2 local_neighbour,
    int32_t neighbours_seen,
    float3 pheromone,
    float3 pheromone_angle,
    float3 pheromone_heading,
    float speed)
{
    const float4 life = world.life_state[index];
    switch (sensor) {
        case 0: return 1.0f;
        case 1:
            return clampf(world.energy[index] /
                (world.reproduction_energy * 3.0f), 0.0f, 1.0f);
        case 2: return world.size[index] / fmaxf(life.x, 0.01f);
        case 3: return clampf(world.health[index] /
            fmaxf(100.0f * world.size[index], 1.0f), 0.0f, 1.0f);
        case 4:
            return clampf((world.food_gain[index] + world.meat_gain[index]) /
                fmaxf(world.pellet_energy,
                    world.initial_energy * world.size[index] * 0.4f),
                0.0f, 1.0f);
        case 5: return clampf(speed / world.traits[index].x, 0.0f, 1.5f);
        case 6: return world.actions_a[index].y * world.traits[index].y;
        case 7: return world.held_target[index] != 0 ? 1.0f : 0.0f;
        case 8: return clampf(2.0f * world.last_damage[index] /
            fmaxf(world.health[index], 1.0f), 0.0f, 1.0f);
        case 9: return floorf(life.y);
        case 10:
            return nearest_neighbour >= 0
                ? 1.0f - sqrtf(nearest_neighbour_squared) / world.sense_radius
                : 0.0f;
        case 11:
            return nearest_neighbour >= 0
                ? atan2f(local_neighbour.x, local_neighbour.y) *
                    (1.0f / 3.141592653589793f)
                : 0.0f;
        case 12:
            return static_cast<float>(neighbours_seen) * 0.25f;
        case 13:
            return food_senses.plant >= 0
                ? 1.0f - sqrtf(food_senses.plant_distance_squared) /
                    world.sense_radius
                : 0.0f;
        case 14:
            return food_senses.plant >= 0
                ? atan2f(food_senses.plant_direction.x,
                    food_senses.plant_direction.y) *
                    (1.0f / 3.141592653589793f)
                : 0.0f;
        case 15: return static_cast<float>(food_senses.plants_seen) * 0.25f;
        case 16:
            return food_senses.meat >= 0
                ? 1.0f - sqrtf(food_senses.meat_distance_squared) /
                    world.sense_radius
                : 0.0f;
        case 17:
            return food_senses.meat >= 0
                ? atan2f(food_senses.meat_direction.x,
                    food_senses.meat_direction.y) *
                    (1.0f / 3.141592653589793f)
                : 0.0f;
        case 18: return static_cast<float>(food_senses.meats_seen) * 0.25f;
        case 19: return world.cached_neighbour_color[index].x;
        case 20: return world.cached_neighbour_color[index].y;
        case 21: return world.cached_neighbour_color[index].z;
        case 22: return static_cast<int32_t>(floorf(world.age[index] /
            fmaxf(life.w, 0.05f))) & 1;
        case 23: return life.z * (1.0f / 60.0f);
        case 24: return world.age[index] * (1.0f / 60.0f);
        case 25: return tanhf(pheromone.x * 0.2f);
        case 26: return tanhf(pheromone.y * 0.2f);
        case 27: return tanhf(pheromone.z * 0.2f);
        case 28: return pheromone_angle.x;
        case 29: return pheromone_angle.y;
        case 30: return pheromone_angle.z;
        case 31: return pheromone_heading.x;
        case 32: return pheromone_heading.y;
        case 33: return pheromone_heading.z;
        default: return 0.0f;
    }
}

__device__ __forceinline__ float field_angle(float x, float y, float heading)
{
    if (x * x + y * y < 1.0e-12f) return 0.0f;
    const float2 local = relative_direction(safe_normalize({x, y}), heading);
    return atan2f(local.x, local.y) * (1.0f / 3.141592653589793f);
}

__global__ void pack_native_sensor_values_kernel(
    DeviceWorld world, int32_t index, BgfWorldBrainNodeState* nodes)
{
    if (blockIdx.x != 0 || threadIdx.x != 0 || index < 0 ||
        index >= world.max_bibites || world.alive[index] != 1 ||
        world.template_brain[index] != 0 ||
        world.native_brain_version[index] == 0) return;
    const TemplateFoodSenses food_senses = lookup_template_food_senses(world, index);
    const int32_t nearest = world.cached_neighbour[index];
    const float2 neighbour_direction = world.cached_neighbour_direction[index];
    const float neighbour_distance = world.cached_neighbour_distance_squared[index];
    const int32_t seen = world.cached_neighbour_count[index];
    const int32_t cell = pheromone_cell(world, world.positions[index]);
    const unsigned long long last_step = world.counters->completed_steps > 0ull
        ? world.counters->completed_steps - 1ull : 0ull;
    const unsigned long long period = last_step /
        static_cast<unsigned long long>(kPheromoneUpdateFactor);
    const bool field_is_b = (period & 1ull) == 0ull;
    const float3* field = field_is_b ? world.pheromone_b : world.pheromone_a;
    const float3 pheromone = field[cell];
    const int32_t width = world.pheromone_grid_width;
    const int32_t height = world.pheromone_grid_height;
    const int32_t x = cell % width;
    const int32_t y = cell / width;
    const float3 left = field[y * width + wrap_index(x - 1, width)];
    const float3 right = field[y * width + wrap_index(x + 1, width)];
    const float3 below = field[wrap_index(y - 1, height) * width + x];
    const float3 above = field[wrap_index(y + 1, height) * width + x];
    const float heading = world.headings[index];
    const float3 angles{
        field_angle(right.x - left.x, above.x - below.x, heading),
        field_angle(right.y - left.y, above.y - below.y, heading),
        field_angle(right.z - left.z, above.z - below.z, heading)};
    const float3* heading_x = field_is_b
        ? world.pheromone_heading_x_b : world.pheromone_heading_x_a;
    const float3* heading_y = field_is_b
        ? world.pheromone_heading_y_b : world.pheromone_heading_y_a;
    const float3 moment_x = heading_x[cell];
    const float3 moment_y = heading_y[cell];
    const float3 headings{
        field_angle(moment_x.x, moment_y.x, heading),
        field_angle(moment_x.y, moment_y.y, heading),
        field_angle(moment_x.z, moment_y.z, heading)};
    const float2 velocity = world.velocities[index];
    const float speed = hypotf(velocity.x, velocity.y);
    for (int32_t sensor = 0; sensor < kStockSensorCount; ++sensor)
        nodes[sensor].last_output = template_sensor_value(
            world, index, sensor, food_senses, nearest, neighbour_distance,
            neighbour_direction, seen, pheromone, angles, headings, speed);
}

__device__ void apply_brain_actions(
    const DeviceWorld& world, int32_t index, const float* outputs,
    bool full_native_brain, uint32_t random_state)
{
    const int32_t nearest_food = world.cached_food[index];
    const float2 local_food = world.cached_food_direction[index];
    float acceleration = world.template_brain[index] != 0
        ? clampf(outputs[0], 0.0f, 1.0f)
        : (outputs[0] + 1.0f) * 0.5f;
    float turn = outputs[1];
    const float hunger = clampf(
        (world.initial_energy * 1.5f - world.energy[index]) /
            fmaxf(world.initial_energy * 0.75f, 1.0f),
        0.0f, 1.0f);
    if (nearest_food >= 0) {
        // Native starter brains are deliberately diverse and initially random.
        // A hunger-only reflex supplies the same basic viability that stock
        // starter species get from curated brains, then fades away after a meal
        // so neural behaviour and evolution remain in control.
        if (hunger > 0.0f) {
            const float desired_turn = clampf(
                atan2f(local_food.x, local_food.y) *
                    (2.0f / 3.141592653589793f),
                -1.0f,
                1.0f);
            turn = turn * (1.0f - hunger) + desired_turn * hunger;
            acceleration = fmaxf(acceleration, 0.35f + hunger * 0.65f);
        }
    }
    world.actions_a[index] = {
        acceleration,
        turn,
        fmaxf(outputs[2], 0.0f),
        fmaxf(outputs[3], 0.0f)};
    world.actions_b[index] = {fmaxf(outputs[4], 0.0f), outputs[5]};
    const bool stock_brain = world.template_brain[index] != 0 || full_native_brain;
    const float eating_drive = full_native_brain
        ? fmaxf(outputs[8], hunger) : (stock_brain ? outputs[8] : 1.0f);
    const float digestion_drive = full_native_brain
        ? fmaxf(outputs[9], fmaxf(0.55f, hunger))
        : (stock_brain ? outputs[9] : 1.0f);
    world.actions_c[index] = {
        eating_drive,
        digestion_drive,
        outputs[10], outputs[14]};
    world.actions_d[index] = {
        outputs[6], stock_brain ? outputs[7] : 1.0f,
        stock_brain ? outputs[12] : 1.0f, outputs[13]};
    world.clock_reset_output[index] = outputs[11];
    world.rng[index] = random_state;
}

template <bool PrepareOnly = false>
__device__ void evaluate_brain(
    const DeviceWorld& world,
    int32_t index,
    const float3* pheromone_field)
{
    uint32_t random_state = world.rng[index];
    const bool full_native_brain = world.template_brain[index] == 0 &&
        world.native_brain_version[index] != 0;
    TemplateFoodSenses food_senses{};
    if (world.template_brain[index] != 0 || full_native_brain) {
        food_senses = *world.food_senses_cache_active!=0
            ? world.food_senses_cache[index] : lookup_template_food_senses(world,index);
        const bool prefer_meat = world.diet[index] >= 0.5f;
        int32_t target = prefer_meat
            ? (food_senses.meat >= 0 ? food_senses.meat : food_senses.plant)
            : (food_senses.plant >= 0 ? food_senses.plant : food_senses.meat);
        bool keep_cached = false;
        float2 cached_direction{0.0f, 0.0f};
        float cached_distance_squared = world.sense_radius * world.sense_radius;
        if (full_native_brain && world.lock_food_target != 0) {
            const int32_t cached = world.cached_food[index];
            if (cached >= 0 && cached < world.pellet_count &&
                (world.pellet_active[cached] == 1 ||
                    world.pellet_active[cached] == 2) &&
                (world.pellet_held_by[cached] == -1 ||
                    world.pellet_held_by[cached] == index)) {
                const float2 displacement = wrapped_delta(
                    world.positions[index], world.pellet_positions[cached],
                    world.world_half_extent);
                cached_distance_squared = displacement.x * displacement.x +
                    displacement.y * displacement.y;
                if (cached_distance_squared <=
                    world.sense_radius * world.sense_radius) {
                    target = cached;
                    keep_cached = true;
                    cached_direction = relative_direction(
                        safe_normalize(displacement), world.headings[index]);
                }
            }
        }
        world.cached_food[index] = target;
        const bool meat_target = target >= 0 && target == food_senses.meat;
        world.cached_food_direction[index] = keep_cached ? cached_direction
            : target < 0 ? float2{0.0f, 0.0f}
            : (meat_target ? food_senses.meat_direction :
                food_senses.plant_direction);
        world.cached_food_distance_squared[index] = keep_cached
            ? cached_distance_squared : target < 0
            ? world.sense_radius * world.sense_radius
            : (meat_target ? food_senses.meat_distance_squared :
                food_senses.plant_distance_squared);
    }
    const int32_t nearest_food = world.cached_food[index];
    const int32_t nearest_neighbour = world.cached_neighbour[index];
    const float nearest_food_squared = world.cached_food_distance_squared[index];
    const float nearest_neighbour_squared = world.cached_neighbour_distance_squared[index];
    const float2 local_food = world.cached_food_direction[index];
    const float2 local_neighbour = world.cached_neighbour_direction[index];
    const int32_t neighbours_seen = world.cached_neighbour_count[index];
    const int32_t field_cell = pheromone_cell(world, world.positions[index]);
    const float3 pheromone = pheromone_field[field_cell];
    float3 pheromone_angles{0.0f, 0.0f, 0.0f};
    float3 pheromone_headings{0.0f, 0.0f, 0.0f};
    if (world.template_brain[index] != 0 || full_native_brain) {
        const int32_t width = world.pheromone_grid_width;
        const int32_t height = world.pheromone_grid_height;
        const int32_t x = field_cell % width;
        const int32_t y = field_cell / width;
        const float3 left = pheromone_field[y * width + wrap_index(x - 1, width)];
        const float3 right = pheromone_field[y * width + wrap_index(x + 1, width)];
        const float3 below = pheromone_field[wrap_index(y - 1, height) * width + x];
        const float3 above = pheromone_field[wrap_index(y + 1, height) * width + x];
        const float heading = world.headings[index];
        pheromone_angles = {
            field_angle(right.x - left.x, above.x - below.x, heading),
            field_angle(right.y - left.y, above.y - below.y, heading),
            field_angle(right.z - left.z, above.z - below.z, heading)};
        const float3* heading_x_field = pheromone_field == world.pheromone_a
            ? world.pheromone_heading_x_a : world.pheromone_heading_x_b;
        const float3* heading_y_field = pheromone_field == world.pheromone_a
            ? world.pheromone_heading_y_a : world.pheromone_heading_y_b;
        const float3 moment_x = heading_x_field[field_cell];
        const float3 moment_y = heading_y_field[field_cell];
        pheromone_headings = {
            field_angle(moment_x.x, moment_y.x, heading),
            field_angle(moment_x.y, moment_y.y, heading),
            field_angle(moment_x.z, moment_y.z, heading)};
    }
    const float speed = hypotf(world.velocities[index].x, world.velocities[index].y);
    float outputs[kTemplateOutputCount]{};

    if (world.template_brain[index] != 0) {
        const int32_t topology = world.template_brain[index] - 1;
        const int32_t node_count = static_cast<int32_t>(
            world.template_topology_node_counts[topology]);
        const int32_t synapse_count = static_cast<int32_t>(
            world.template_topology_synapse_counts[topology]);
        const int32_t active_node_count = static_cast<int32_t>(
            world.template_topology_active_node_counts[topology]);
        for (int32_t active = 0; active < active_node_count; ++active) {
            const int32_t node = static_cast<int32_t>(
                world.template_active_nodes[topology_node_slot(topology, active)]);
            const size_t target = template_node_slot(world, index, node);
            const uint32_t descriptor = world.template_node_descriptors[
                topology_node_slot(topology, node)];
            const int32_t type = descriptor_node_type(descriptor);
            world.template_node_accum[target] = type == 10 ? 1.0f : 0.0f;
            if (type == 0) {
                const int32_t sensor = descriptor_sensor(descriptor);
                world.template_node_last_output[target] = template_sensor_value(
                    world,
                    index,
                    sensor,
                    food_senses,
                    nearest_neighbour,
                    nearest_neighbour_squared,
                    local_neighbour,
                    neighbours_seen,
                    pheromone,
                    pheromone_angles,
                    pheromone_headings,
                    speed);
            }
        }
        for (int32_t synapse = 0; synapse < synapse_count; ++synapse) {
            const size_t connection = template_synapse_slot(world, index, synapse);
            const uint16_t edge = world.template_synapse_edges[
                topology_synapse_slot(topology, synapse)];
            const int32_t source_node = edge_source(edge);
            const int32_t destination_node = edge_destination(edge);
            if (source_node < 0 || source_node >= node_count ||
                destination_node < 0 || destination_node >= node_count) {
                continue;
            }
            const float contribution =
                world.template_node_last_output[template_node_slot(world, index, source_node)] *
                __half2float(world.template_synapse_weights[connection]);
            const size_t destination = template_node_slot(world, index, destination_node);
            const uint32_t destination_descriptor = world.template_node_descriptors[
                topology_node_slot(topology, destination_node)];
            if (descriptor_node_type(destination_descriptor) == 10) {
                world.template_node_accum[destination] *= contribution;
            } else {
                world.template_node_accum[destination] += contribution;
            }
        }
        bool action_seen[kTemplateOutputCount]{};
        for (int32_t active = 0; active < active_node_count; ++active) {
            const int32_t node = static_cast<int32_t>(
                world.template_active_nodes[topology_node_slot(topology, active)]);
            const size_t target = template_node_slot(world, index, node);
            const uint32_t descriptor = world.template_node_descriptors[
                topology_node_slot(topology, node)];
            const int32_t type = descriptor_node_type(descriptor);
            if (type == 0) continue;
            const float input = world.template_node_accum[target];
            const float result = execute_template_neuron(
                type,
                input,
                __half2float(world.template_node_biases[target]),
                world.template_node_last_input[target],
                world.template_node_last_output[target],
                world.fixed_delta_time * static_cast<float>(
                    (world.diagnostic_mask & 1048576u) != 0u
                        ? 8
                        : ((world.diagnostic_mask & 524288u) != 0u
                            ? 4
                            : world.brain_update_factor)));
            world.template_node_last_input[target] = input;
            world.template_node_last_output[target] = result;
            const int32_t action = descriptor_action(descriptor);
            if (action < 0 || action >= kTemplateOutputCount) continue;
            if (!action_seen[action] || action == 5) {
                outputs[action] = action == 5 && action_seen[action]
                    ? fmaxf(outputs[action], result)
                    : result;
                action_seen[action] = true;
            }
        }
    } else {
        float sensors[kSensorCount] = {
            1.0f,
            clampf(world.energy[index] / world.reproduction_energy, 0.0f, 2.0f),
            clampf(world.age[index] / world.traits[index].w, 0.0f, 2.0f),
            clampf(speed / world.traits[index].x, 0.0f, 1.5f),
            local_food.x,
            local_food.y,
            nearest_food >= 0 ? 1.0f - sqrtf(nearest_food_squared) / world.sense_radius : 0.0f,
            local_neighbour.x,
            local_neighbour.y,
            clampf(static_cast<float>(neighbours_seen) / 12.0f, 0.0f, 1.0f),
            pheromone.x / (1.0f + pheromone.x),
            pheromone.y / (1.0f + pheromone.y),
            pheromone.z / (1.0f + pheromone.z),
            signed_uniform(random_state),
            sinf(world.headings[index]),
            cosf(world.headings[index])};
        float extra_sensors[kExtraSensorCount]{};
        if (full_native_brain) {
            for (int32_t sensor = 0; sensor < kStockSensorCount; ++sensor) {
                const float value = template_sensor_value(
                    world, index, sensor, food_senses,
                    nearest_neighbour, nearest_neighbour_squared,
                    local_neighbour, neighbours_seen, pheromone,
                    pheromone_angles, pheromone_headings, speed);
                if (sensor < kSensorCount) sensors[sensor] = value;
                else extra_sensors[sensor - kSensorCount] = value;
            }
        }
        if constexpr (PrepareOnly) {
            for (int32_t sensor = 0; sensor < kStockSensorCount; ++sensor)
                world.brain_inputs[static_cast<size_t>(sensor) * world.max_bibites + index] =
                    sensor < kSensorCount ? sensors[sensor] : extra_sensors[sensor - kSensorCount];
            world.rng[index] = random_state;
            return;
        }
        float first_hidden[kHiddenWidth]{};
        float second_hidden[kHiddenWidth]{};
        float hidden_outputs[kTemplateOutputCount]{};
        for (int32_t word = 0; word < kHiddenFirstEnd / 32; ++word) {
            uint32_t bits = world.hidden_masks[hidden_column_slot(world, index, word)];
            while (bits != 0u) {
                const int32_t edge = word * 32 + __ffs(bits) - 1;
                const int32_t destination = edge / kSensorCount;
                const int32_t source = edge % kSensorCount;
                first_hidden[destination] = fmaf(sensors[source],
                    __half2float(world.hidden_weights[
                        dense_weight_slot(world, index, edge)]),
                    first_hidden[destination]);
                bits &= bits - 1u;
            }
        }
        if (full_native_brain) {
            for (int32_t word = 0; word < kExtensionMaskWords; ++word) {
                uint32_t bits = world.extension_masks[
                    hidden_column_slot(world, index, word)];
                while (bits != 0u) {
                    const int32_t edge = word * 32 + __ffs(bits) - 1;
                    if (edge < kExtensionInputEnd) {
                        const int32_t destination = edge / kExtraSensorCount;
                        const int32_t source = edge % kExtraSensorCount;
                        first_hidden[destination] = fmaf(extra_sensors[source],
                            __half2float(world.extension_weights[
                                dense_weight_slot(world, index, edge)]),
                            first_hidden[destination]);
                    }
                    bits &= bits - 1u;
                }
            }
        }
        for (int32_t node = 0; node < kHiddenWidth; ++node) {
            const size_t slot = hidden_column_slot(world, index, node);
            const float input = first_hidden[node];
            first_hidden[node] = tanhf(input +
                __half2float(world.hidden_biases[slot]));
            world.hidden_last_input[slot] = __float2half_rn(input);
            world.hidden_last_output[slot] = __float2half_rn(first_hidden[node]);
        }
        for (int32_t word = kHiddenFirstEnd / 32;
             word <= (kHiddenSecondEnd - 1) / 32; ++word) {
            uint32_t bits = world.hidden_masks[hidden_column_slot(world, index, word)];
            if (word == kHiddenSecondEnd / 32) bits &= 0x0000ffffu;
            while (bits != 0u) {
                const int32_t edge = word * 32 + __ffs(bits) - 1 - kHiddenFirstEnd;
                const int32_t destination = edge / kHiddenWidth;
                const int32_t source = edge % kHiddenWidth;
                second_hidden[destination] = fmaf(first_hidden[source],
                    __half2float(world.hidden_weights[
                        dense_weight_slot(world, index, edge + kHiddenFirstEnd)]),
                    second_hidden[destination]);
                bits &= bits - 1u;
            }
        }
        for (int32_t node = 0; node < kHiddenWidth; ++node) {
            const size_t slot = hidden_column_slot(world, index, kHiddenWidth + node);
            const float input = second_hidden[node];
            second_hidden[node] = tanhf(input +
                __half2float(world.hidden_biases[slot]));
            world.hidden_last_input[slot] = __float2half_rn(input);
            world.hidden_last_output[slot] = __float2half_rn(second_hidden[node]);
        }
        for (int32_t word = kHiddenSecondEnd / 32;
             word < kHiddenMaskWords; ++word) {
            uint32_t bits = world.hidden_masks[hidden_column_slot(world, index, word)];
            if (word == kHiddenSecondEnd / 32) bits &= 0xffff0000u;
            if (word == kHiddenMaskWords - 1) bits &= 0x00ffffffu;
            while (bits != 0u) {
                const int32_t edge = word * 32 + __ffs(bits) - 1 - kHiddenSecondEnd;
                const int32_t destination = edge / kHiddenWidth;
                const int32_t source = edge % kHiddenWidth;
                hidden_outputs[destination] = fmaf(second_hidden[source],
                    __half2float(world.hidden_weights[
                        dense_weight_slot(world, index, edge + kHiddenSecondEnd)]),
                    hidden_outputs[destination]);
                bits &= bits - 1u;
            }
        }
        if (full_native_brain) {
            for (int32_t word = 0; word < kExtensionMaskWords; ++word) {
                uint32_t bits = world.extension_masks[
                    hidden_column_slot(world, index, word)];
                while (bits != 0u) {
                    const int32_t edge = word * 32 + __ffs(bits) - 1;
                    if (edge >= kExtensionInputEnd &&
                        edge < kExtensionConnectionCount) {
                        const int32_t local = edge - kExtensionInputEnd;
                        const int32_t output = kOutputCount + local / kHiddenWidth;
                        const int32_t source = local % kHiddenWidth;
                        hidden_outputs[output] = fmaf(second_hidden[source],
                            __half2float(world.extension_weights[
                                dense_weight_slot(world, index, edge)]),
                            hidden_outputs[output]);
                    }
                    bits &= bits - 1u;
                }
            }
        }
        for (int32_t output = 0; output < kOutputCount; ++output) {
            float accumulated = hidden_outputs[output];
            for (int32_t sensor = 0; sensor < kSensorCount; sensor += 2) {
                const size_t first_slot = dense_weight_slot(
                    world,
                    index,
                    output * kSensorCount + sensor);
                const __half2 packed = *reinterpret_cast<const __half2*>(
                    world.weights + first_slot);
                const float2 pair = __half22float2(packed);
                accumulated = fmaf(sensors[sensor], pair.x, accumulated);
                accumulated = fmaf(sensors[sensor + 1], pair.y, accumulated);
            }
            outputs[output] = tanhf(accumulated);
        }
        if (full_native_brain) {
            for (int32_t output = kOutputCount;
                 output < kTemplateOutputCount; ++output) {
                const float activation = tanhf(hidden_outputs[output]);
                // Stock's continuous drives use a non-negative range. Keep
                // signed herding, grab, clock reset, and attack controls.
                outputs[output] = output == 7 || output == 8 ||
                    output == 9 || output == 12 || output == 13
                    ? (activation + 1.0f) * 0.5f : activation;
            }
        }
    }

    apply_brain_actions(world, index, outputs, full_native_brain, random_state);
}

__device__ __forceinline__ void process_perception(
    const DeviceWorld& world,
    int32_t index,
    bool perform_lookup,
    bool perform_refresh,
    int32_t known_sense_cell,
    bool skip_food = false)
{
    if (perform_lookup) {
        if (known_sense_cell >= 0) {
            lookup_perception_from_cell(world, index, known_sense_cell, skip_food);
        } else {
            lookup_perception_from_cell(world, index,
                grid_cell(world, world.positions[index], world.sense_grid_width), skip_food);
        }
    } else if (perform_refresh) {
        refresh_perception(world, index);
    }
}

__device__ __forceinline__ float effective_turn(const DeviceWorld& world,
    int32_t index, float requested_turn)
{
    const int32_t neighbour = world.cached_neighbour[index];
    if (neighbour < 0 || neighbour >= world.max_bibites ||
        world.alive[neighbour] != 1) return requested_turn;
    const float herd = clampf(world.actions_d[index].x, -1.0f, 1.0f);
    if (fabsf(herd) < 0.01f) return requested_turn;
    const float2 direction = world.cached_neighbour_direction[index];
    const float target_turn = atan2f(direction.x, direction.y) *
        (1.0f / 3.141592653589793f);
    return requested_turn * (1.0f - fabsf(herd)) + target_turn * herd;
}

__device__ __forceinline__ void process_post_decision(
    const DeviceWorld& world,
    int32_t index)
{
    float stomach_energy = world.food_gain[index];
    float stomach_meat = world.meat_gain[index];
    const int32_t nearest_food = world.cached_food[index];
    const float grab_drive = world.actions_c[index].z;
    int32_t held = world.held_target[index];
    if (held > 0) {
        const int32_t other = held - 1;
        if (other >= world.max_bibites || world.alive[other] != 1 ||
            world.instance_id[other] != world.held_instance[index]) {
            world.held_target[index] = 0;
            held = 0;
        } else if (grab_drive > 0.15f) {
            const float2 delta = wrapped_delta(world.positions[index],
                world.positions[other], world.world_half_extent);
            const float reach = world.size[index] + world.size[other] + 0.75f;
            const float squared = delta.x * delta.x + delta.y * delta.y;
            if (squared > 16.0f * reach * reach) {
                world.held_target[index] = 0;
                held = 0;
            } else if (squared > reach * reach) {
                const float strength = fminf(0.2f * grab_drive, 0.4f);
                atomicAdd(&world.repulsion[other].x, -delta.x * strength);
                atomicAdd(&world.repulsion[other].y, -delta.y * strength);
                atomicAdd(&world.repulsion[index].x, delta.x * strength);
                atomicAdd(&world.repulsion[index].y, delta.y * strength);
            }
        }
    } else if (held < 0) {
        const int32_t pellet = -held - 1;
        if (pellet < 0 || pellet >= world.pellet_count ||
            world.pellet_held_by[pellet] != index ||
            (world.pellet_active[pellet] != 1 &&
                world.pellet_active[pellet] != 2)) {
            world.held_target[index] = 0;
            held = 0;
        }
    }
    if (held == 0 && grab_drive > 0.15f) {
        const int32_t neighbour = world.cached_neighbour[index];
        const float reach = (world.size[index] + 0.75f) *
            world.mobility_scale;
        if (nearest_food >= 0 && nearest_food < world.pellet_count &&
            world.cached_food_distance_squared[index] <= reach * reach &&
            (world.pellet_active[nearest_food] == 1 ||
                world.pellet_active[nearest_food] == 2) &&
            atomicCAS(&world.pellet_held_by[nearest_food], -1, index) == -1) {
            world.held_target[index] = -nearest_food - 1;
            held = -nearest_food - 1;
        } else if (neighbour >= 0 && neighbour < world.max_bibites &&
            world.alive[neighbour] == 1 &&
            world.cached_neighbour_distance_squared[index] <= reach * reach) {
            world.held_target[index] = neighbour + 1;
            world.held_instance[index] = world.instance_id[neighbour];
            held = neighbour + 1;
        }
    }
    if (world.actions_c[index].x > fmaxf(0.15f,
            held == -nearest_food - 1 ? grab_drive : 0.15f) &&
        nearest_food >= 0 && nearest_food < world.pellet_count) {
        const int32_t pellet_material = world.pellet_active[nearest_food];
        if ((pellet_material == 1 || pellet_material == 2) &&
            (world.pellet_held_by[nearest_food] == -1 ||
                world.pellet_held_by[nearest_food] == index)) {
            const float2 direction = wrapped_delta(
                world.positions[index],
                world.pellet_positions[nearest_food],
                world.world_half_extent);
            // Test the whole next movement segment instead of only the current
            // point. Large worlds deliberately scale travel speed with map size;
            // a point test could otherwise step over a pellet between ticks.
            const float dt = world.fixed_delta_time;
            const float4 traits = world.traits[index];
            const float4 action = world.actions_a[index];
            float heading = world.headings[index] +
                effective_turn(world, index, action.y) * traits.y * dt;
            if (heading > 3.141592653589793f) heading -= 6.283185307179586f;
            if (heading < -3.141592653589793f) heading += 6.283185307179586f;
            const float2 forward{sinf(heading), cosf(heading)};
            float2 projected_velocity = world.velocities[index];
            const float contact_scale =
                15.0f * static_cast<float>(world.contact_solve_factor);
            projected_velocity.x +=
                (forward.x * action.x * 8.0f * world.mobility_scale +
                    world.repulsion[index].x * contact_scale) * dt;
            projected_velocity.y +=
                (forward.y * action.x * 8.0f * world.mobility_scale +
                    world.repulsion[index].y * contact_scale) * dt;
            const float projected_speed_squared =
                projected_velocity.x * projected_velocity.x +
                projected_velocity.y * projected_velocity.y;
            if (projected_speed_squared > traits.x * traits.x) {
                const float scale = traits.x * rsqrtf(projected_speed_squared);
                projected_velocity.x *= scale;
                projected_velocity.y *= scale;
            }
            const float drag = *world.drag_retention;
            const float2 travel{
                projected_velocity.x * drag * dt,
                projected_velocity.y * drag * dt};
            const float travel_squared = travel.x * travel.x + travel.y * travel.y;
            const float along = travel_squared > 1.0e-12f
                ? clampf(
                    (direction.x * travel.x + direction.y * travel.y) /
                        travel_squared,
                    0.0f,
                    1.0f)
                : 0.0f;
            const float miss_x = direction.x - travel.x * along;
            const float miss_y = direction.y - travel.y * along;
            const float distance_squared = miss_x * miss_x + miss_y * miss_y;
            // The coarse high-warp integrator scales locomotion on very large
            // maps. Scale the capture radius by the same bounded factor so a
            // Bibite that reaches food cannot orbit or step past it forever.
            const float eat_distance =
                (world.size[index] + 0.75f) * world.mobility_scale;
            if (distance_squared <= eat_distance * eat_distance) {
                const float capacity = fmaxf(world.pellet_energy,
                    world.initial_energy * world.size[index] * 0.4f);
                const int32_t room_units = static_cast<int32_t>(floorf(
                    fmaxf(0.0f, capacity - stomach_energy - stomach_meat) *
                    static_cast<float>(kPelletFoodUnits) / world.pellet_energy));
                 // Large maps scale motion and capture radius together. Scale
                 // the bite rate too or Bibites pass through food faster than
                 // their mouth can collect even a small fraction of a pellet.
                const int32_t bite_units = min(room_units, max(1,
                    static_cast<int32_t>(ceilf(
                         kPelletFoodUnits * world.fixed_delta_time *
                         world.mobility_scale /
                        kPelletEatingSeconds))));
                // Claim only the portion that fits. A pellet remains visible and
                // available to others until its last unit has been eaten.
                for (int attempt = 0; bite_units > 0 && attempt < 8; ++attempt) {
                    const int32_t remaining = world.pellet_food_units[nearest_food];
                    if (remaining <= 0) break;
                    const int32_t consumed = min(remaining, bite_units);
                    if (atomicCAS(&world.pellet_food_units[nearest_food],
                            remaining, remaining - consumed) != remaining) continue;
                    const float portion = world.pellet_energy *
                        static_cast<float>(consumed) / kPelletFoodUnits;
                    if (pellet_material == 2) stomach_meat += portion;
                    else stomach_energy += portion;
                    if (remaining == consumed &&
                        atomicCAS(&world.pellet_active[nearest_food],
                            pellet_material, 0) == pellet_material) {
                        remove_pellet(world, nearest_food);
                        const int32_t eaten_slot = atomicAdd(world.eaten_pellet_count, 1);
                        world.eaten_pellet_items[eaten_slot] = nearest_food;
                        atomicAdd(&world.counters->pellets_eaten, 1ull);
                        world.cached_food[index] = -1;
                        world.cached_food_direction[index] = {0.0f, 0.0f};
                        world.cached_food_distance_squared[index] =
                            world.sense_radius * world.sense_radius;
                        if (world.pellet_held_by[nearest_food] == index) {
                            world.pellet_held_by[nearest_food] = -1;
                            world.held_target[index] = 0;
                        }
                    }
                    break;
                }
            }
        } else {
            world.cached_food[index] = -1;
            world.cached_food_direction[index] = {0.0f, 0.0f};
            world.cached_food_distance_squared[index] =
                world.sense_radius * world.sense_radius;
        }
    }
    world.food_gain[index] = stomach_energy;
    world.meat_gain[index] = stomach_meat;
    const float desire_to_attack = world.actions_c[index].w;
    const int32_t victim = world.cached_neighbour[index];
    if (desire_to_attack > 0.15f && world.bite_cooldown[index] <= 0.0f &&
        victim >= 0 && victim < world.max_bibites &&
        world.alive[victim] == 1) {
        const float2 delta = wrapped_delta(world.positions[index],
            world.positions[victim], world.world_half_extent);
        const float distance_squared = delta.x * delta.x + delta.y * delta.y;
        const float reach = world.size[index] + world.size[victim] + 0.75f;
        const float heading = world.headings[index];
        const float facing = delta.x * sinf(heading) + delta.y * cosf(heading);
        if (distance_squared > 1.0e-8f &&
            distance_squared <= reach * reach &&
            facing > sqrtf(distance_squared) * 0.35f) {
            const float damage = world.biting_damage_factor *
                (world.biting_pressure / 80.0f) * world.size[index] *
                clampf(desire_to_attack, 0.0f, 1.0f) * 4.0f;
            atomicAdd(&world.pending_damage[victim], damage);
            const float room = fmaxf(0.0f,
                fmaxf(world.pellet_energy,
                    world.initial_energy * world.size[index] * 0.4f) -
                stomach_energy - stomach_meat);
            world.meat_gain[index] += fminf(room, damage * 0.2f);
            world.bite_cooldown[index] = 0.5f;
            atomicAdd(&world.repulsion[victim].x, delta.x *
                rsqrtf(distance_squared) * damage * 0.01f);
            atomicAdd(&world.repulsion[victim].y, delta.y *
                rsqrtf(distance_squared) * damage * 0.01f);
        }
    }
    world.wants_reproduction[index] =
        world.actions_b[index].y > 0.25f &&
        world.life_state[index].y >= 1.0f &&
        world.reproduction_cooldown[index] <= 0.0f ? 1 : 0;
    if (world.wants_reproduction[index] != 0) {
        const int32_t reproduction_slot = atomicAdd(world.reproduction_count, 1);
        world.reproduction_items[reproduction_slot] = index;
    }
}

template <bool PrepareOnly = false>
__device__ void process_decision(
    const DeviceWorld& world,
    int32_t index,
    bool perform_lookup,
    bool perform_refresh,
    int32_t known_sense_cell,
    bool brain_due,
    const float3* pheromone_active)
{
    process_perception(
        world,
        index,
        perform_lookup,
        perform_refresh,
        known_sense_cell,
        brain_due && (world.template_brain[index] != 0 ||
            world.native_brain_version[index] != 0));
    if (brain_due) {
        evaluate_brain<PrepareOnly>(world, index, pheromone_active);
    }
    if constexpr (!PrepareOnly) process_post_decision(world, index);
}

// The full decision pass follows cell order on perception-refresh ticks.  The
// eight-lane groups keep nearby query lanes together while avoiding the severe
// under-occupancy of one full warp per sparse cell.
template <bool PrepareOnly = false>
__device__ void process_spatially_ordered_decisions(
    const DeviceWorld& world,
    bool brain_due,
    const float3* pheromone_active)
{
    constexpr int32_t kGroupSize = 8;
    constexpr int32_t kGroupsPerBlock = kThreads / kGroupSize;
    const int32_t group_in_block = static_cast<int32_t>(threadIdx.x) / kGroupSize;
    const int32_t lane = static_cast<int32_t>(threadIdx.x) & (kGroupSize - 1);
    const int32_t group = static_cast<int32_t>(blockIdx.x) * kGroupsPerBlock +
        group_in_block;
    const int32_t group_stride = static_cast<int32_t>(gridDim.x) * kGroupsPerBlock;
    const int32_t occupied_count = *world.sense_bibite_occupied_count;
    for (int32_t occupied_slot = group;
         occupied_slot < occupied_count;
         occupied_slot += group_stride) {
        const int32_t cell = world.sense_bibite_occupied_cells[occupied_slot];
        const int32_t count = min(
            world.sense_bibite_cell_counts[cell],
            kSenseBucketCapacity);
        for (int32_t item_slot = lane;
             item_slot < count;
             item_slot += kGroupSize) {
            const int32_t index = world.sense_bibite_cell_items[
                cell * kSenseBucketCapacity + item_slot];
            if (world.alive[index] == 1) {
                process_decision<PrepareOnly>(
                    world,
                    index,
                    true,
                    false,
                    cell,
                    brain_due,
                    pheromone_active);
            }
        }
    }
}


__device__ __forceinline__ float3 diffuse_pheromone_cell(
    const float3* source, int32_t cell, int32_t left, int32_t right,
    int32_t up, int32_t down, float diffusion, float decay)
{
    const float3 center = source[cell];
    const float3 neighbours{
        (source[left].x + source[right].x + source[up].x + source[down].x) * 0.25f,
        (source[left].y + source[right].y + source[up].y + source[down].y) * 0.25f,
        (source[left].z + source[right].z + source[up].z + source[down].z) * 0.25f};
    return {
        (center.x + (neighbours.x - center.x) * diffusion) * decay,
        (center.y + (neighbours.y - center.y) * diffusion) * decay,
        (center.z + (neighbours.z - center.z) * diffusion) * decay};
}

__device__ __forceinline__ uint32_t masked_connections(const uint32_t* masks,
    const DeviceWorld& world, int index, int first, int count)
{
    const int shift = first & 31;
    uint32_t bits = masks[hidden_column_slot(world,index,first>>5)] >> shift;
    if (shift + count > 32)
        bits |= masks[hidden_column_slot(world,index,(first>>5)+1)] << (32-shift);
    return bits & ((1u<<count)-1u);
}

// A lane evaluates one evolving brain; adjacent lanes fetch the same
// synapse column. Fixed destination loops avoid dynamic local-array writes
// in the monolithic kernel. FP16 genetic values, FP32 fused accumulation.
template <bool Dense = false>
__global__ __launch_bounds__(128, 6) void native_brain_kernel(DeviceWorld world)
{
    if (blockIdx.x==0 && threadIdx.x==0) {
        const auto before=world.counters->phase_decision_cycles;
        finish_profile_phase(world.counters,&world.counters->phase_decision_cycles);
        world.timing->food_cycles+=world.counters->phase_decision_cycles-before;
    }
    const unsigned long long step = world.counters->completed_steps;
    const int factor = (world.diagnostic_mask & 1048576u) ? 8 :
        (world.diagnostic_mask & 524288u) ? 4 : world.brain_update_factor;
    if (step % factor == 0 && (world.diagnostic_mask & 4u) == 0u) {
        for (int32_t work = blockIdx.x * blockDim.x + threadIdx.x;
             work < *world.live_count; work += blockDim.x * gridDim.x) {
            const int32_t index = world.live_items[work];
            if (world.alive[index] != 1 || world.template_brain[index] != 0) continue;
            const bool full = world.native_brain_version[index] != 0;
            float first[kHiddenWidth], second[kHiddenWidth];
#pragma unroll
            for (int32_t node = 0; node < kHiddenWidth; ++node) {
                float value = 0.0f;
                uint32_t bits = masked_connections(world.hidden_masks, world,
                    index, node*kSensorCount, kSensorCount);
                if constexpr (Dense) {
#pragma unroll
                    for (int input=0;input<kSensorCount;++input)
                        if (bits&(1u<<input))
                            value=fmaf(world.brain_inputs[static_cast<size_t>(input)*world.max_bibites+index],
                                __half2float(world.hidden_weights[
                                    dense_weight_slot(world,index,node*kSensorCount+input)]),value);
                    bits=0;
                }
                while (bits) {
                    const int input = __ffs(bits)-1;
                    const int edge = node*kSensorCount+input;
                    value = fmaf(world.brain_inputs[static_cast<size_t>(input)*world.max_bibites+index],
                        __half2float(world.hidden_weights[dense_weight_slot(world,index,edge)]),value);
                    bits &= bits-1;
                }
                if (full) {
                    bits = masked_connections(world.extension_masks,world,index,
                        node*kExtraSensorCount,kExtraSensorCount);
                    if constexpr (Dense) {
#pragma unroll
                        for (int input=0;input<kExtraSensorCount;++input)
                            if (bits&(1u<<input))
                                value=fmaf(world.brain_inputs[
                                    static_cast<size_t>(input+kSensorCount)*world.max_bibites+index],
                                    __half2float(world.extension_weights[
                                        dense_weight_slot(world,index,node*kExtraSensorCount+input)]),value);
                        bits=0;
                    }
                    while (bits) {
                        const int input = __ffs(bits)-1;
                        const int edge = node*kExtraSensorCount+input;
                        value = fmaf(world.brain_inputs[static_cast<size_t>(input+kSensorCount)*
                            world.max_bibites+index],
                            __half2float(world.extension_weights[dense_weight_slot(world,index,edge)]),value);
                        bits &= bits-1;
                    }
                }
                const size_t slot = hidden_column_slot(world,index,node);
                first[node] = tanhf(value + __half2float(world.hidden_biases[slot]));
                world.hidden_last_input[slot] = __float2half_rn(value);
                world.hidden_last_output[slot] = __float2half_rn(first[node]);
            }
#pragma unroll
            for (int32_t node = 0; node < kHiddenWidth; ++node) {
                float value = 0.0f;
#pragma unroll
                for (int32_t input = 0; input < kHiddenWidth; ++input) {
                    const int edge = kHiddenFirstEnd + node*kHiddenWidth + input;
                    if ((world.hidden_masks[hidden_column_slot(world,index,edge>>5)] &
                            (1u << (edge&31))) != 0)
                        value = fmaf(first[input],
                            __half2float(world.hidden_weights[dense_weight_slot(world,index,edge)]),
                            value);
                }
                const size_t slot = hidden_column_slot(world,index,node+kHiddenWidth);
                second[node] = tanhf(value + __half2float(world.hidden_biases[slot]));
                world.hidden_last_input[slot] = __float2half_rn(value);
                world.hidden_last_output[slot] = __float2half_rn(second[node]);
            }
#pragma unroll 1
            for (int32_t output = 0; output < kTemplateOutputCount; ++output) {
                float value = 0.0f;
#pragma unroll
                for (int32_t input = 0; input < kHiddenWidth; ++input) {
                    const int edge = output < kOutputCount
                        ? kHiddenSecondEnd + output*kHiddenWidth+input
                        : kExtensionInputEnd+(output-kOutputCount)*kHiddenWidth+input;
                    const uint32_t bits = output < kOutputCount
                        ? world.hidden_masks[hidden_column_slot(world,index,edge>>5)]
                        : world.extension_masks[hidden_column_slot(world,index,edge>>5)];
                    if ((output < kOutputCount || full) && (bits & (1u<<(edge&31))) != 0) {
                        const __half weight = output < kOutputCount
                            ? world.hidden_weights[dense_weight_slot(world,index,edge)]
                            : world.extension_weights[dense_weight_slot(world,index,edge)];
                        value = fmaf(second[input], __half2float(weight), value);
                    }
                }
                if (output < kOutputCount) {
#pragma unroll
                    for (int32_t input = 0; input < kSensorCount; ++input)
                        value = fmaf(world.brain_inputs[static_cast<size_t>(input) *
                            world.max_bibites + index], __half2float(world.weights[
                                dense_weight_slot(world,index,output*kSensorCount+input)]), value);
                }
                float activation = tanhf(value);
                if (full && (output == 7 || output == 8 || output == 9 ||
                        output == 12 || output == 13)) activation = (activation+1.0f)*0.5f;
                world.brain_outputs[static_cast<size_t>(output)*world.max_bibites+index] =
                    output < kOutputCount || full ? activation : 0.0f;
            }
        }
    }

}

// Kept opt-in until comparative timing/rounding tests pass on each GPU family.
#include "bgf_tensor_brain.cuh"

template <int Stage = 0,int MinimumBlocks = 2>
__global__ __launch_bounds__(kThreads,(Stage==0?MinimumBlocks:1)) void world_stage_kernel(DeviceWorld world, int32_t step_count)
{
    cg::grid_group grid = cg::this_grid();
    [[maybe_unused]] const int32_t thread = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    [[maybe_unused]] const int32_t stride = static_cast<int32_t>(blockDim.x * gridDim.x);
    // Small populations otherwise use only a handful of blocks for the most
    // expensive per-body stages. Keep cooperative barriers uniform, but spread
    // body work across 32/64 lanes per block in opt-in comparison variants.
    [[maybe_unused]] const int32_t body_threads=Stage==0 && (world.diagnostic_mask&67108864u)
        ? 32 : Stage==0 && (world.diagnostic_mask&134217728u) ? 64 : kThreads;
    [[maybe_unused]] const int32_t live_thread=threadIdx.x<static_cast<unsigned>(body_threads)
        ? static_cast<int32_t>(blockIdx.x)*body_threads+static_cast<int32_t>(threadIdx.x)
        : world.max_bibites;
    [[maybe_unused]] const int32_t live_stride=body_threads*static_cast<int32_t>(gridDim.x);
    [[maybe_unused]] const int32_t pheromone_cells = world.pheromone_grid_width * world.pheromone_grid_height;
    [[maybe_unused]] const unsigned long long base_step = world.counters->completed_steps;
    if constexpr (Stage==0 || Stage==1 || Stage==7) grid.sync();
    if constexpr (Stage == 0) if (thread == 0) {
        world.counters->phase_prepare_cycles = 0ull;
        world.counters->phase_spatial_index_cycles = 0ull;
        world.counters->phase_contact_cycles = 0ull;
        world.counters->phase_decision_cycles = 0ull;
        world.counters->phase_motion_cycles = 0ull;
        world.counters->phase_lifecycle_cycles = 0ull;
        asm volatile("mov.u64 %0, %%globaltimer;" :
            "=l"(world.counters->phase_started_cycle));
    }

    for (int32_t local_step = 0; local_step < step_count; ++local_step) {
        [[maybe_unused]] const unsigned long long absolute_step = base_step + static_cast<unsigned long long>(local_step);
        [[maybe_unused]] const int32_t vision_lookup_factor =
            (world.diagnostic_mask & 262144u) != 0u
                ? 160
                : ((world.diagnostic_mask & 131072u) != 0u
                    ? 80
                    : ((world.diagnostic_mask & 65536u) != 0u
                        ? 40
                        : world.vision_lookup_factor));
        [[maybe_unused]] const bool lookup_due = absolute_step %
            static_cast<unsigned long long>(vision_lookup_factor) == 0ull;
        [[maybe_unused]] const bool sense_index_due = lookup_due || *world.sense_grid_dirty != 0;
        [[maybe_unused]] const bool sense_due = absolute_step % kSenseUpdateFactor == 0ull;
        [[maybe_unused]] const int32_t brain_update_factor =
            (world.diagnostic_mask & 1048576u) != 0u
                ? 8
                : ((world.diagnostic_mask & 524288u) != 0u
                    ? 4
                    : world.brain_update_factor);
        [[maybe_unused]] const bool brain_due = absolute_step %
            static_cast<unsigned long long>(brain_update_factor) == 0ull;
        [[maybe_unused]] const bool pheromone_due = absolute_step % kPheromoneUpdateFactor == 0ull;
        [[maybe_unused]] const unsigned long long pheromone_period =
            absolute_step / static_cast<unsigned long long>(kPheromoneUpdateFactor);
        [[maybe_unused]] float3* pheromone_source = (pheromone_period & 1ull) == 0ull
            ? world.pheromone_a : world.pheromone_b;
        [[maybe_unused]] float3* pheromone_destination = (pheromone_period & 1ull) == 0ull
            ? world.pheromone_b : world.pheromone_a;
        [[maybe_unused]] float3* pheromone_active = pheromone_due
            ? pheromone_destination
            : ((pheromone_period & 1ull) == 0ull
                ? world.pheromone_b : world.pheromone_a);
        [[maybe_unused]] float3* heading_x_source = (pheromone_period & 1ull) == 0ull
            ? world.pheromone_heading_x_a : world.pheromone_heading_x_b;
        [[maybe_unused]] float3* heading_x_destination = (pheromone_period & 1ull) == 0ull
            ? world.pheromone_heading_x_b : world.pheromone_heading_x_a;
        [[maybe_unused]] float3* heading_y_source = (pheromone_period & 1ull) == 0ull
            ? world.pheromone_heading_y_a : world.pheromone_heading_y_b;
        [[maybe_unused]] float3* heading_y_destination = (pheromone_period & 1ull) == 0ull
            ? world.pheromone_heading_y_b : world.pheromone_heading_y_a;
        [[maybe_unused]] float3* heading_x_active = pheromone_due ? heading_x_destination
            : ((pheromone_period & 1ull) == 0ull
                ? world.pheromone_heading_x_b : world.pheromone_heading_x_a);
        [[maybe_unused]] float3* heading_y_active = pheromone_due ? heading_y_destination
            : ((pheromone_period & 1ull) == 0ull
                ? world.pheromone_heading_y_b : world.pheromone_heading_y_a);
        [[maybe_unused]] const int32_t contact_grid_factor = (world.diagnostic_mask & 512u) != 0u
            ? 4
            : ((world.diagnostic_mask & 256u) != 0u
                ? 2
                : world.contact_grid_update_factor);
        [[maybe_unused]] const unsigned long long contact_grid_epoch =
            absolute_step / static_cast<unsigned long long>(contact_grid_factor);
        [[maybe_unused]] const int32_t current_contact_buffer =
            static_cast<int32_t>(contact_grid_epoch & 1ull);
        [[maybe_unused]] const int32_t next_contact_buffer = current_contact_buffer ^ 1;
        [[maybe_unused]] const bool rebuild_next_contact_grid =
            absolute_step % static_cast<unsigned long long>(contact_grid_factor) ==
            static_cast<unsigned long long>(contact_grid_factor - 1);
        [[maybe_unused]] const int32_t contact_solve_factor = (world.diagnostic_mask & 2048u) != 0u
            ? 4
            : ((world.diagnostic_mask & 1024u) != 0u
                ? 2
                : world.contact_solve_factor);
        [[maybe_unused]] const bool contact_due =
            absolute_step % static_cast<unsigned long long>(contact_solve_factor) == 0ull;
        [[maybe_unused]] const bool contact_grid_dirty = *world.contact_grid_dirty != 0;

        if constexpr (Stage == 0 || Stage == 1) {
        const ContactGridView next_contact_grid =
            contact_grid_view(world, next_contact_buffer);
        const ContactGridView current_contact_grid =
            contact_grid_view(world, current_contact_buffer);
        const int32_t reusable_contact_cell_count = rebuild_next_contact_grid
            ? *next_contact_grid.occupied_count
            : 0;
        const int32_t dirty_contact_cell_count = contact_grid_dirty
            ? *current_contact_grid.occupied_count
            : 0;
        const int32_t previous_sense_cell_count = sense_index_due
            ? *world.sense_bibite_occupied_count
            : 0;
        // Every block must capture the old sparse-list lengths before thread
        // zero resets them below. Otherwise late blocks see zero, fail to clear
        // old cells/overflow heads, and reinsertion can build cyclic lists.
        if (rebuild_next_contact_grid || contact_grid_dirty || sense_index_due) grid.sync();
        for (int32_t slot = thread; slot < reusable_contact_cell_count; slot += stride) {
            const int32_t cell = next_contact_grid.occupied_cells[slot];
            next_contact_grid.cell_counts[cell] = 0;
            next_contact_grid.overflow_heads[cell] = -1;
            atomicAnd(
                &next_contact_grid.occupied[cell >> 5],
                ~(1u << (cell & 31)));
        }
        for (int32_t slot = thread; slot < dirty_contact_cell_count; slot += stride) {
            const int32_t cell = current_contact_grid.occupied_cells[slot];
            current_contact_grid.cell_counts[cell] = 0;
            current_contact_grid.overflow_heads[cell] = -1;
            atomicAnd(
                &current_contact_grid.occupied[cell >> 5],
                ~(1u << (cell & 31)));
        }
        if (thread == 0) {
            if (rebuild_next_contact_grid) {
                *next_contact_grid.occupied_count = 0;
                *next_contact_grid.overflow_count = 0;
                *next_contact_grid.large_count = 0;
            }
            if (contact_grid_dirty) {
                *current_contact_grid.occupied_count = 0;
                *current_contact_grid.overflow_count = 0;
                *current_contact_grid.large_count = 0;
            }
            *world.eaten_pellet_count = 0;
            *world.reproduction_count = 0;
            *world.dead_bibite_count = 0;
            if (sense_index_due) {
                *world.sense_bibite_occupied_count = 0;
                *world.sense_bibite_overflow_count = 0;
            }
        }
        if (sense_index_due) {
            for (int32_t slot = thread; slot < previous_sense_cell_count; slot += stride) {
                const int32_t cell = world.sense_bibite_occupied_cells[slot];
                world.sense_bibite_cell_counts[cell] = 0;
                world.sense_bibite_overflow_heads[cell] = -1;
                atomicAnd(
                    &world.sense_bibite_occupied[cell >> 5],
                    ~(1u << (cell & 31)));
            }
        }
        if (brain_due || sense_index_due) {
            for (int32_t cell = thread;
                 cell < world.sense_grid_width * world.sense_grid_width; cell += stride)
                world.pellet_overflow_counts[cell] = 0;
        }
        for (int32_t cell = thread;
             pheromone_due && cell < pheromone_cells;
             cell += stride) {
            const int32_t x = cell % world.pheromone_grid_width;
            const int32_t y = cell / world.pheromone_grid_width;
            const int32_t left = y * world.pheromone_grid_width +
                wrap_index(x - 1, world.pheromone_grid_width);
            const int32_t right = y * world.pheromone_grid_width +
                wrap_index(x + 1, world.pheromone_grid_width);
            const int32_t up = wrap_index(y - 1, world.pheromone_grid_height) *
                world.pheromone_grid_width + x;
            const int32_t down = wrap_index(y + 1, world.pheromone_grid_height) *
                world.pheromone_grid_width + x;
            const float retained = 1.0f - world.pheromone_diffusion;
            const float retained_squared = retained * retained;
            const float diffusion = 1.0f - retained_squared * retained_squared;
            const float decay_squared = world.pheromone_decay * world.pheromone_decay;
            const float decay = decay_squared * decay_squared;
            pheromone_destination[cell] = diffuse_pheromone_cell(
                pheromone_source, cell, left, right, up, down, diffusion, decay);
            heading_x_destination[cell] = diffuse_pheromone_cell(
                heading_x_source, cell, left, right, up, down, diffusion, decay);
            heading_y_destination[cell] = diffuse_pheromone_cell(
                heading_y_source, cell, left, right, up, down, diffusion, decay);
        }
        grid.sync();
        if (thread == 0) {
            finish_profile_phase(
                world.counters,
                &world.counters->phase_prepare_cycles);
        }

        for (int32_t work = live_thread;
             (contact_grid_dirty || sense_index_due) && work < *world.live_count;
             work += live_stride) {
            const int32_t index = world.live_items[work];
            if (world.alive[index] != 1) continue;
            if (contact_grid_dirty && world.size[index] <= kRegularMaximumBodySize) {
                const int32_t contact_cell = grid_cell(
                    world,
                    world.positions[index],
                    world.contact_grid_width);
                insert_contact_item(current_contact_grid, contact_cell, index);
            } else if (contact_grid_dirty) {
                const int32_t slot = atomicAdd(current_contact_grid.large_count, 1);
                current_contact_grid.large_items[slot] = index;
            }
            if (sense_index_due) {
                const int32_t sense_cell = grid_cell(
                    world,
                    world.positions[index],
                    world.sense_grid_width);
                insert_sense_item(world, sense_cell, index);
            }
        }
        if ((brain_due || sense_index_due) && *world.pellet_overflow_count>0)
            build_pellet_overflow_segments(world,grid,thread,stride);
        if (thread==0) *world.food_senses_cache_active=
            brain_due && (world.diagnostic_mask&(4u|536870912u))==0u &&
            *world.pellet_overflow_count>kParallelFoodOverflowThreshold ? 1 : 0;
        if (contact_grid_dirty || sense_index_due || brain_due) grid.sync();
        if (contact_due && (world.diagnostic_mask&(1u|8u))==0u &&
            *current_contact_grid.overflow_count>0 &&
            world.world_half_extent*2.0f/world.contact_grid_width>=kRegularMaximumBodySize*2.0f)
            build_contact_tile_tasks(world,current_contact_grid,grid,thread,stride);
        if (thread == 0) {
            if (contact_grid_dirty) *world.contact_grid_dirty = 0;
            finish_profile_phase(
                world.counters,
                &world.counters->phase_spatial_index_cycles);
        }
        if (brain_due && *world.food_senses_cache_active!=0) {
            grid.sync();
            cache_dense_food_senses(world);
            grid.sync();
            if (thread==0) {
                const auto before=world.counters->phase_decision_cycles;
                finish_profile_phase(world.counters,&world.counters->phase_decision_cycles);
                world.timing->food_cycles+=world.counters->phase_decision_cycles-before;
            }
            grid.sync();
        }
        }

        if constexpr (Stage == 0 || Stage == 2) {
        const ContactGridView current_contact_grid =
            contact_grid_view(world, current_contact_buffer);
        const bool run_contact = contact_due && (world.diagnostic_mask & 1u) == 0u;
        if (run_contact) {
            const float contact_cell_width = world.world_half_extent * 2.0f /
                static_cast<float>(world.contact_grid_width);
            const bool use_pair_tiles =
                contact_cell_width >= kRegularMaximumBodySize * 2.0f &&
                (world.diagnostic_mask & 8u) == 0u;
            if (use_pair_tiles) {
                if ((world.diagnostic_mask & 32u) != 0u) {
                    calculate_unique_contact_pairs_sparse_cells(world, current_contact_grid);
                } else if ((world.diagnostic_mask & 16u) != 0u) {
                    calculate_unique_contact_pairs_tiled(world, current_contact_grid);
                } else {
                    calculate_unique_contact_pairs_subwarp_tiled(world, current_contact_grid);
                }
                if (*current_contact_grid.overflow_count > 0 ||
                    *current_contact_grid.large_count > 0)
                    calculate_supplemental_contact_pairs(world,current_contact_grid);
            } else {
                for (int32_t work = live_thread; work < *world.live_count; work += live_stride) {
            const int32_t index = world.live_items[work];
                    if (world.alive[index] != 1) continue;
                    float2 contact_repulsion{};
                    calculate_contact_repulsion(
                        world,
                        current_contact_grid,
                        index,
                        contact_repulsion);
                    world.repulsion[index] = contact_repulsion;
                }
            }
        }
        if constexpr (Stage==0) {
            if (run_contact) grid.sync();
            if (thread==0) finish_profile_phase(world.counters,&world.counters->phase_contact_cycles);
        }
        }

        if constexpr (Stage == 0 || Stage == 3) {
        if constexpr (Stage==3) if (thread==0)
            finish_profile_phase(world.counters,&world.counters->phase_contact_cycles);
        const bool ordered_decision =
            lookup_due &&
            (world.diagnostic_mask & 2u) == 0u &&
            (world.diagnostic_mask & 32768u) == 0u &&
            *world.sense_bibite_overflow_count == 0;
        const bool run_brain =
            brain_due && (world.diagnostic_mask & 4u) == 0u;
        if (ordered_decision) {
            process_spatially_ordered_decisions<(Stage == 3)>(
                world,
                run_brain,
                pheromone_active);
        } else {
            const bool run_lookup =
                lookup_due && (world.diagnostic_mask & 2u) == 0u;
            const bool run_refresh =
                !run_lookup && sense_due && (world.diagnostic_mask & 2u) == 0u;
            for (int32_t work = live_thread; work < *world.live_count; work += live_stride) {
            const int32_t index = world.live_items[work];
                if (world.alive[index] != 1) continue;
                process_decision<(Stage == 3)>(
                    world,
                    index,
                    run_lookup,
                    run_refresh,
                    -1,
                    run_brain,
                    pheromone_active);
            }
        }
        if constexpr (Stage==0) {
            grid.sync();
            if (thread==0) finish_profile_phase(world.counters,&world.counters->phase_decision_cycles);
        }

        }
        if constexpr (Stage == 5) {
            if (thread==0) {
                if (brain_due && (world.diagnostic_mask&4u)==0u) {
                    const auto before=world.counters->phase_decision_cycles;
                    finish_profile_phase(world.counters,&world.counters->phase_decision_cycles);
                    world.timing->brain_cycles+=world.counters->phase_decision_cycles-before;
                } else if (lookup_due || sense_due) {
                    const auto before=world.counters->phase_decision_cycles;
                    finish_profile_phase(world.counters,&world.counters->phase_decision_cycles);
                    world.timing->food_cycles+=world.counters->phase_decision_cycles-before;
                } else finish_profile_phase(world.counters,&world.counters->phase_contact_cycles);
            }
            for (int work = thread; work < *world.live_count; work += live_stride) {
                const int index = world.live_items[work];
                if (world.alive[index] != 1) continue;
                if (brain_due && (world.diagnostic_mask & 4u) == 0u &&
                    world.template_brain[index] == 0) {
                    float outputs[kTemplateOutputCount];
#pragma unroll
                    for (int output = 0; output < kTemplateOutputCount; ++output)
                        outputs[output] = world.brain_outputs[
                            static_cast<size_t>(output)*world.max_bibites+index];
                    apply_brain_actions(world,index,outputs,
                        world.native_brain_version[index] != 0,world.rng[index]);
                }
                process_post_decision(world,index);
            }

        }
        if constexpr (Stage == 0 || Stage == 6) {
            if constexpr (Stage==6) if (thread==0) {
                const auto before=world.counters->phase_decision_cycles;
                finish_profile_phase(world.counters,&world.counters->phase_decision_cycles);
                world.timing->post_cycles+=world.counters->phase_decision_cycles-before;
            }
        const int32_t write_contact_buffer = rebuild_next_contact_grid
            ? next_contact_buffer
            : current_contact_buffer;
        const ContactGridView write_contact_grid =
            contact_grid_view(world, write_contact_buffer);
        for (int32_t work = live_thread; work < *world.live_count; work += live_stride) {
            const int32_t index = world.live_items[work];
            if (world.alive[index] != 1) continue;
            const float dt = world.fixed_delta_time;
            const float4 traits = world.traits[index];
            const float4 action = world.actions_a[index];
            const float2 secretion = world.actions_b[index];
            float heading = world.headings[index] +
                effective_turn(world, index, action.y) * traits.y * dt;
            if (heading > 3.141592653589793f) heading -= 6.283185307179586f;
            if (heading < -3.141592653589793f) heading += 6.283185307179586f;
            const float2 forward{sinf(heading), cosf(heading)};
            float2 velocity = world.velocities[index];
            const float contact_scale = 15.0f * static_cast<float>(contact_solve_factor);
            velocity.x += (forward.x * action.x * 8.0f * world.mobility_scale +
                world.repulsion[index].x * contact_scale) * dt;
            velocity.y += (forward.y * action.x * 8.0f * world.mobility_scale +
                world.repulsion[index].y * contact_scale) * dt;
            const float speed_squared = velocity.x * velocity.x + velocity.y * velocity.y;
            if (speed_squared > traits.x * traits.x) {
                const float scale = traits.x * rsqrtf(speed_squared);
                velocity.x *= scale;
                velocity.y *= scale;
            }
            const float drag = *world.drag_retention;
            velocity.x *= drag;
            velocity.y *= drag;
            float2 position = world.positions[index];
            position.x = wrap_coordinate(position.x + velocity.x * dt, world.world_half_extent);
            position.y = wrap_coordinate(position.y + velocity.y * dt, world.world_half_extent);

            const float digestion_drive = world.actions_c[index].y;
            const float digestion_capacity = world.pellet_energy *
                world.size[index] * dt / kPelletDigestionSeconds *
                clampf(digestion_drive, 0.0f, 1.0f);
            const float digestion = fminf(world.food_gain[index],
                digestion_capacity);
            world.food_gain[index] -= digestion;
            const float meat_digestion = fminf(world.meat_gain[index],
                fmaxf(0.0f, digestion_capacity - digestion));
            world.meat_gain[index] -= meat_digestion;
            const float2 efficiency = world.digestion_efficiency[index];
            const float plant_efficiency = efficiency.x;
            const float meat_efficiency = efficiency.y;
            float energy = world.energy[index] +
                digestion * plant_efficiency +
                meat_digestion * meat_efficiency;
            energy -= (traits.z + action.x * action.x * 0.22f +
                (action.z + action.w + secretion.x) * 0.03f) * dt;
            const float damage = atomicExch(&world.pending_damage[index], 0.0f);
            world.last_damage[index] = damage;
            float health = world.health[index] - damage;
            float4 life = world.life_state[index];
            const float2 reproductive = world.reproductive_traits[index];
            float body_size = world.size[index];
            const float growth_drive = clampf(world.actions_d[index].z, 0.0f, 1.0f);
            const float growth = fminf(fmaxf(0.0f, life.x - body_size),
                growth_drive * dt * 0.02f);
            if (growth > 0.0f && energy > growth * 25.0f + 1.0f) {
                body_size += growth;
                health += growth * 100.0f;
                energy -= growth * 25.0f;
            }
            const float egg_energy = world.reproduction_energy * 0.32f;
            const float production = world.actions_d[index].y;
            if (body_size >= life.x * 0.98f &&
                (production < 0.0f || energy > world.reproduction_energy * 0.65f) &&
                fabsf(production) >= 0.15f) {
                const float remapped = copysignf(
                    (fabsf(production) - 0.15f) * (1.0f / 0.85f), production);
                float egg_delta = remapped * dt /
                    fmaxf(reproductive.x, 0.25f);
                egg_delta = clampf(egg_delta, -life.y,
                    fmaxf(0.0f, reproductive.y - life.y));
                if (egg_delta > 0.0f) {
                    egg_delta = fminf(egg_delta,
                        fmaxf(0.0f, energy - 1.0f) / egg_energy);
                    energy -= egg_delta * egg_energy;
                } else {
                    energy += -egg_delta * egg_energy * 0.5f;
                }
                life.y += egg_delta;
            }
            life.z = world.clock_reset_output[index] > 0.75f
                ? 0.0f : life.z + dt;
            const float heal_drive = clampf(world.actions_d[index].w, 0.0f, 1.0f);
            const float healing = fminf(fmaxf(0.0f,
                100.0f * body_size - health),
                heal_drive * 5.0f * dt);
            health += healing;
            energy -= healing * 0.5f;
            energy = fminf(energy, world.reproduction_energy * 3.0f);
            const float age = world.age[index] + dt;
            const float cooldown = fmaxf(world.reproduction_cooldown[index] - dt, 0.0f);
            const bool invalid_state = !isfinite(position.x) || !isfinite(position.y) ||
                !isfinite(energy) || !isfinite(age) || !isfinite(traits.w);
            const bool starved = !invalid_state && energy <= 0.0f;
            const bool aged_out = !invalid_state && !starved && age >= traits.w;
            const bool killed = !invalid_state && !starved && health <= 0.0f;
            const int32_t held = world.held_target[index];
            if (held < 0) {
                const int32_t pellet = -held - 1;
                if (pellet >= 0 && pellet < world.pellet_count &&
                    world.pellet_held_by[pellet] == index) {
                    const bool release = invalid_state || starved || aged_out ||
                        killed || world.actions_c[index].z <= -0.15f ||
                        (world.pellet_active[pellet] != 1 &&
                            world.pellet_active[pellet] != 2);
                    if (release) {
                        world.pellet_held_by[pellet] = -1;
                        world.held_target[index] = 0;
                    } else {
                        const float2 held_position{
                            wrap_coordinate(position.x + forward.x * body_size,
                                world.world_half_extent),
                            wrap_coordinate(position.y + forward.y * body_size,
                                world.world_half_extent)};
                        const int32_t new_cell = grid_cell(world, held_position,
                            world.sense_grid_width);
                        if (world.pellet_hash_cell[pellet] != new_cell)
                            remove_pellet(world, pellet);
                        world.pellet_positions[pellet] = held_position;
                        if (world.pellet_hash_cell[pellet] < 0)
                            insert_pellet(world, pellet);
                    }
                } else {
                    world.held_target[index] = 0;
                }
            } else if (held > 0 && (invalid_state || starved || aged_out ||
                killed || world.actions_c[index].z <= -0.15f)) {
                world.held_target[index] = 0;
            }
            if (invalid_state || starved || aged_out || killed) {
                world.repulsion[index] = {0.0f, 0.0f};
                if (!invalid_state) {
                    world.positions[index] = position;
                    world.energy[index] = fmaxf(energy, 0.0f);
                    const int32_t dead_slot = atomicAdd(world.dead_bibite_count, 1);
                    world.dead_bibite_items[dead_slot] = index;
                }
                release_template_instance(world,index);
                world.alive[index] = 0;
                world.wants_reproduction[index] = 0;
                const int32_t free_slot = atomicAdd(world.free_bibite_count, 1);
                world.free_bibite_items[free_slot] = index;
                atomicAdd(&world.counters->deaths, 1ull);
                if (invalid_state) {
                    atomicAdd(&world.counters->invalid_state_deaths, 1ull);
                } else if (starved) {
                    atomicAdd(&world.counters->starvation_deaths, 1ull);
                } else if (aged_out) {
                    atomicAdd(&world.counters->age_deaths, 1ull);
                }
                atomicSub(&world.counters->living_bibites, 1);
                continue;
            }

            world.positions[index] = position;
            world.velocities[index] = velocity;
            world.headings[index] = heading;
            world.energy[index] = energy;
            world.health[index] = health;
            world.size[index] = body_size;
            world.life_state[index] = life;
            world.age[index] = age;
            world.reproduction_cooldown[index] = cooldown;
            world.bite_cooldown[index] = fmaxf(
                world.bite_cooldown[index] - dt, 0.0f);
            world.repulsion[index] = {0.0f, 0.0f};
            if (rebuild_next_contact_grid &&
                world.size[index] <= kRegularMaximumBodySize) {
                const int32_t contact_cell = grid_cell(
                    world,
                    position,
                    world.contact_grid_width);
                insert_contact_item(write_contact_grid, contact_cell, index);
            } else if (rebuild_next_contact_grid) {
                const int32_t slot = atomicAdd(write_contact_grid.large_count, 1);
                write_contact_grid.large_items[slot] = index;
            }
            const int32_t field_cell = pheromone_cell(world, position);
            if (action.z > 0.0f) {
                const float amount = action.z * 0.08f;
                atomicAdd(&pheromone_active[field_cell].x, amount);
                atomicAdd(&heading_x_active[field_cell].x, amount * forward.x);
                atomicAdd(&heading_y_active[field_cell].x, amount * forward.y);
            }
            if (action.w > 0.0f) {
                const float amount = action.w * 0.08f;
                atomicAdd(&pheromone_active[field_cell].y, amount);
                atomicAdd(&heading_x_active[field_cell].y, amount * forward.x);
                atomicAdd(&heading_y_active[field_cell].y, amount * forward.y);
            }
            if (secretion.x > 0.0f) {
                const float amount = secretion.x * 0.08f;
                atomicAdd(&pheromone_active[field_cell].z, amount);
                atomicAdd(&heading_x_active[field_cell].z, amount * forward.x);
                atomicAdd(&heading_y_active[field_cell].z, amount * forward.y);
            }
        }
        if constexpr (Stage==0) {
            grid.sync();
            if (thread==0) finish_profile_phase(world.counters,&world.counters->phase_motion_cycles);
        }

        }
        if constexpr (Stage == 0 || Stage == 7) {
        if constexpr (Stage==7) if (thread==0)
            finish_profile_phase(world.counters,&world.counters->phase_motion_cycles);
        const int32_t write_contact_buffer = rebuild_next_contact_grid
            ? next_contact_buffer : current_contact_buffer;
        const ContactGridView write_contact_grid = contact_grid_view(world,write_contact_buffer);
        const int32_t dead_bibites = *world.dead_bibite_count;
        for (int32_t dead_slot = thread; dead_slot < dead_bibites;
             dead_slot += stride) {
            const int32_t dead = world.dead_bibite_items[dead_slot];
            int32_t pellet = -1;
            const int32_t reserve = world.pellet_count - world.pellet_limit;
            for (int32_t attempt = 0; attempt < min(reserve, 128); ++attempt) {
                const int32_t candidate = world.pellet_limit +
                    (atomicAdd(world.meat_cursor, 1) % reserve);
                if (world.pellet_pending[candidate] == 0 &&
                    atomicCAS(&world.pellet_active[candidate], 0, 3) == 0) {
                    world.pellet_held_by[candidate] = -1;
                    pellet = candidate;
                    break;
                }
            }
            if (pellet < 0 && world.pellet_limit > 0) {
                for (int32_t attempt = 0;
                     attempt < min(world.pellet_limit, 128); ++attempt) {
                    const int32_t candidate = atomicAdd(world.meat_cursor, 1) %
                        world.pellet_limit;
                    if (atomicCAS(&world.pellet_active[candidate], 1, 3) == 1) {
                        remove_pellet(world, candidate);
                        world.pellet_held_by[candidate] = -1;
                        pellet = candidate;
                        break;
                    }
                }
            }
            if (pellet < 0) continue;
            const float body_energy = fmaxf(0.0f, world.energy[dead]) * 0.5f +
                world.size[dead] * 10.0f;
            const int32_t units = max(1, min(kMaximumPelletFoodUnits,
                __float2int_rn(body_energy / world.pellet_energy *
                    kPelletFoodUnits)));
            world.pellet_positions[pellet] = world.positions[dead];
            world.pellet_nominal_units[pellet] = units;
            world.pellet_food_units[pellet] = units;
            insert_pellet(world, pellet);
            __threadfence();
            world.pellet_active[pellet] = 2;
        }
        grid.sync();

        // Compact dead work before appending offspring. Public slot IDs never move.
        if (*world.live_count != world.counters->living_bibites) {
            if (thread == 0) *world.live_scratch_count = 0;
            grid.sync();
            const int32_t old_count = min(*world.live_count, world.max_bibites);
            for (int32_t work = live_thread; work < old_count; work += live_stride) {
                const int32_t index = world.live_items[work];
                if (world.alive[index] == 1) {
                    const int32_t slot = atomicAdd(world.live_scratch_count, 1);
                    world.live_scratch[slot] = index;
                }
            }
            grid.sync();
            const int32_t count = *world.live_scratch_count;
            for (int32_t work = live_thread; work < count; work += live_stride)
                world.live_items[work] = world.live_scratch[work];
            grid.sync();
            if (thread == 0) *world.live_count = count;
            grid.sync();
        }
        const int32_t reproduction_candidates = *world.reproduction_count;
        for (int32_t candidate_slot = thread;
             candidate_slot < reproduction_candidates;
             candidate_slot += stride) {
            const int32_t parent = world.reproduction_items[candidate_slot];
            if (world.alive[parent] != 1 || world.wants_reproduction[parent] == 0 ||
                world.life_state[parent].y < 1.0f) {
                continue;
            }
            const int32_t free_slot = atomicSub(world.free_bibite_count, 1) - 1;
            if (free_slot < 0) {
                atomicAdd(world.free_bibite_count, 1);
                world.wants_reproduction[parent] = 0;
                continue;
            }
            uint32_t random_state = world.rng[parent];
            const int32_t child = world.free_bibite_items[free_slot];
            if (atomicCAS(&world.alive[child], 0, 2) != 0) {
                atomicAdd(world.free_bibite_count, 1);
                world.wants_reproduction[parent] = 0;
                world.rng[parent] = random_state;
                continue;
            }
            atomicAdd(&world.counters->living_bibites, 1);

            const float child_energy = world.reproduction_energy * 0.32f;
            const float angle = uniform01(random_state) * 6.283185307179586f;
            float2 child_position = world.positions[parent];
            child_position.x = wrap_coordinate(
                child_position.x + sinf(angle) * world.size[parent] * 2.2f,
                world.world_half_extent);
            child_position.y = wrap_coordinate(
                child_position.y + cosf(angle) * world.size[parent] * 2.2f,
                world.world_half_extent);
            initialise_bibite(
                world,
                child,
                mix_bits(random_state ^ static_cast<uint32_t>(absolute_step) ^
                    static_cast<uint32_t>(child * 0x9e3779b9u)),
                child_position,
                child_energy,
                world.generation[parent] + 1,
                parent);
            __threadfence();
            world.alive[child] = 1;
            if (world.size[child] <= kRegularMaximumBodySize) {
                const int32_t contact_cell = grid_cell(
                    world,
                    world.positions[child],
                    world.contact_grid_width);
                insert_contact_item(write_contact_grid, contact_cell, child);
            } else {
                const int32_t slot = atomicAdd(write_contact_grid.large_count, 1);
                write_contact_grid.large_items[slot] = child;
            }
            float4 parent_life = world.life_state[parent];
            parent_life.y -= 1.0f;
            world.life_state[parent] = parent_life;
            world.reproduction_cooldown[parent] = 1.0f;
            world.wants_reproduction[parent] = 0;
            world.rng[parent] = random_state;
            atomicAdd(&world.counters->births, 1ull);
        }
        const int32_t eaten_pellets = *world.eaten_pellet_count;
        for (int32_t slot = thread; slot < eaten_pellets; slot += stride) {
            const int32_t pellet = world.eaten_pellet_items[slot];
            if (pellet >= world.pellet_limit) continue; // Meat reserve does not regrow plants.
            const int32_t delay_steps = max(
                1,
                min(
                    kPelletRespawnWheelSize - 1,
                    static_cast<int32_t>(ceilf(
                        kPelletRespawnSeconds /
                        (world.fixed_delta_time * fmaxf(1.0f, world.food_growth_factor))))));
            const int32_t respawn_bucket = static_cast<int32_t>(
                (absolute_step + static_cast<unsigned long long>(delay_steps)) &
                static_cast<unsigned long long>(kPelletRespawnWheelSize - 1));
            const int32_t respawn_slot = atomicAdd(
                &world.pellet_respawn_counts[respawn_bucket],
                1);
            if (respawn_slot < world.pellet_count) {
                world.pellet_respawn_items[
                    static_cast<size_t>(respawn_bucket) * world.pellet_count +
                    respawn_slot] = pellet;
                world.pellet_pending[pellet] = 1;
            }
        }
        grid.sync();

        // Food remains absent for a short amount of simulated time after it is
        // eaten. A timing wheel makes regrowth proportional to the eaten queue,
        // without returning to a full pellet scan every tick.
        const int32_t due_bucket = static_cast<int32_t>(
            absolute_step &
            static_cast<unsigned long long>(kPelletRespawnWheelSize - 1));
        const int32_t due_pellets = min(
            world.pellet_respawn_counts[due_bucket],
            world.pellet_count);
        for (int32_t slot = thread; slot < due_pellets; slot += stride) {
            const int32_t pellet = world.pellet_respawn_items[
                static_cast<size_t>(due_bucket) * world.pellet_count + slot];
            if (pellet < 0 || pellet >= world.pellet_count) {
                continue;
            }
            if (pellet >= world.pellet_limit || world.pellet_active[pellet] != 0) {
                world.pellet_pending[pellet] = 0;
                continue;
            }
            uint32_t random_state = world.pellet_rng[pellet] ^
                static_cast<uint32_t>(absolute_step * 0x9e3779b97f4a7c15ull);
            if (world.food_growth_factor < 1.0f &&
                uniform01(random_state) >= world.food_growth_factor) {
                world.pellet_rng[pellet] = random_state;
                const int32_t retry_steps = max(
                    1,
                    min(kPelletRespawnWheelSize - 1,
                        static_cast<int32_t>(ceilf(
                            kPelletRespawnSeconds / world.fixed_delta_time))));
                const int32_t retry_bucket = static_cast<int32_t>(
                    (absolute_step + static_cast<unsigned long long>(retry_steps)) &
                    static_cast<unsigned long long>(kPelletRespawnWheelSize - 1));
                const int32_t retry_slot = atomicAdd(
                    &world.pellet_respawn_counts[retry_bucket], 1);
                if (retry_slot < world.pellet_count) {
                    world.pellet_respawn_items[
                        static_cast<size_t>(retry_bucket) * world.pellet_count +
                        retry_slot] = pellet;
                } else {
                    world.pellet_pending[pellet] = 0;
                }
                continue;
            }
            float pellet_size = 1.0f;
            world.pellet_positions[pellet] = sample_food_zone_position(
                world, random_state, true, &pellet_size);
            world.pellet_rng[pellet] = random_state;
            world.pellet_nominal_units[pellet] = max(1, __float2int_rn(
                pellet_size * kPelletFoodUnits));
            world.pellet_food_units[pellet] = world.pellet_nominal_units[pellet];
            insert_pellet(world, pellet);
            __threadfence();
            world.pellet_active[pellet] = 1;
            world.pellet_held_by[pellet] = -1;
            world.pellet_pending[pellet] = 0;
        }
        grid.sync();
        if (thread == 0) {
            world.pellet_respawn_counts[due_bucket] = 0;
            *world.sense_grid_dirty = 0;
        }
        grid.sync();
        if (thread == 0) {
            finish_profile_phase(
                world.counters,
                &world.counters->phase_lifecycle_cycles);
        }
        }
    }

    if constexpr (Stage == 0 || Stage == 7) if (thread == 0) {
        world.counters->completed_steps = base_step + static_cast<unsigned long long>(step_count);
    }
    if constexpr (Stage==0 || Stage==1 || Stage==7) grid.sync();
}

__device__ __forceinline__ void set_render_vertex(
    RenderVertex* vertices,
    int32_t index,
    float x,
    float y,
    float z,
    float r,
    float g,
    float b,
    float a,
    float u = 0.0f,
    float v = 0.0f)
{
    RenderVertex& vertex = vertices[index];
    vertex.position_x = x;
    vertex.position_y = y;
    vertex.position_z = z;
    vertex.color_r = r;
    vertex.color_g = g;
    vertex.color_b = b;
    vertex.color_a = a;
    vertex.uv_x = u;
    vertex.uv_y = v;
}

__device__ void write_bibite_render_vertices(
    const DeviceWorld& world,
    int32_t source,
    int32_t destination,
    RenderVertex* vertices)
{
    constexpr int32_t kVerticesPerBibite = 9;
    const int32_t start = destination * kVerticesPerBibite;
    const float2 center = world.positions[source];
    const float heading = world.headings[source];
    float forward_x = 0.0f;
    float forward_y = 0.0f;
    sincosf(heading, &forward_x, &forward_y);
    const float right_x = forward_y;
    const float right_y = -forward_x;
    const float length = 3.8f + 3.2f * clampf(world.size[source], 0.45f, 2.2f);
    const float width = length * 0.48f;
    const float3 raw_color = world.colors[source];
    const float r = clampf(raw_color.x, 0.0f, 1.0f);
    const float g = clampf(raw_color.y, 0.0f, 1.0f);
    const float b = clampf(raw_color.z, 0.0f, 1.0f);

    const float nose_r = r * 0.8f + 0.2f;
    const float nose_g = g * 0.8f + 0.2f;
    const float nose_b = b * 0.8f + 0.2f;
    const float tail_r = r * 0.75f;
    const float tail_g = g * 0.75f;
    const float tail_b = b * 0.75f;
    set_render_vertex(vertices, start + 0, center.x, center.y, -0.05f, r, g, b, 1.0f);
    set_render_vertex(vertices, start + 1,
        center.x + forward_x * length,
        center.y + forward_y * length,
        -0.05f, nose_r, nose_g, nose_b, 1.0f);
    set_render_vertex(vertices, start + 2,
        center.x + forward_x * length * 0.55f - right_x * width * 0.75f,
        center.y + forward_y * length * 0.55f - right_y * width * 0.75f,
        -0.05f, r, g, b, 1.0f);
    set_render_vertex(vertices, start + 3,
        center.x - forward_x * length * 0.20f - right_x * width,
        center.y - forward_y * length * 0.20f - right_y * width,
        -0.05f, r, g, b, 1.0f);
    set_render_vertex(vertices, start + 4,
        center.x - forward_x * length * 0.80f - right_x * width * 0.55f,
        center.y - forward_y * length * 0.80f - right_y * width * 0.55f,
        -0.05f, tail_r, tail_g, tail_b, 1.0f);
    set_render_vertex(vertices, start + 5,
        center.x - forward_x * length * 0.95f,
        center.y - forward_y * length * 0.95f,
        -0.05f, tail_r, tail_g, tail_b, 1.0f);
    set_render_vertex(vertices, start + 6,
        center.x - forward_x * length * 0.80f + right_x * width * 0.55f,
        center.y - forward_y * length * 0.80f + right_y * width * 0.55f,
        -0.05f, tail_r, tail_g, tail_b, 1.0f);
    set_render_vertex(vertices, start + 7,
        center.x - forward_x * length * 0.20f + right_x * width,
        center.y - forward_y * length * 0.20f + right_y * width,
        -0.05f, r, g, b, 1.0f);
    set_render_vertex(vertices, start + 8,
        center.x + forward_x * length * 0.55f + right_x * width * 0.75f,
        center.y + forward_y * length * 0.55f + right_y * width * 0.75f,
        -0.05f, r, g, b, 1.0f);
}

__device__ void write_pellet_render_vertices(
    const DeviceWorld& world,
    int32_t source,
    int32_t destination,
    RenderVertex* vertices,
    const BgfWorldD3D11RenderConfig& config)
{
    constexpr int32_t kVerticesPerPellet = 4;
    const int32_t start = destination * kVerticesPerPellet;
    const float2 center = world.pellet_positions[source];
    const float fraction = static_cast<float>(world.pellet_food_units[source]) /
        kPelletFoodUnits;
    const float scale = sqrtf(fmaxf(fraction, 0.0f));
    const bool meat = world.pellet_active[source] == 2;
    const float red = meat ? 0.8f : config.pellet_color_r;
    const float green = meat ? 0.18f : config.pellet_color_g;
    const float blue = meat ? 0.24f : config.pellet_color_b;
    const float left = center.x - config.pellet_half_width * scale;
    const float right = center.x + config.pellet_half_width * scale;
    const float bottom = center.y - config.pellet_half_height * scale;
    const float top = center.y + config.pellet_half_height * scale;
    set_render_vertex(vertices, start + 0, left, bottom, 0.05f,
        red, green, blue,
        config.pellet_color_a, config.pellet_uv_min_x, config.pellet_uv_min_y);
    set_render_vertex(vertices, start + 1, left, top, 0.05f,
        red, green, blue,
        config.pellet_color_a, config.pellet_uv_min_x, config.pellet_uv_max_y);
    set_render_vertex(vertices, start + 2, right, top, 0.05f,
        red, green, blue,
        config.pellet_color_a, config.pellet_uv_max_x, config.pellet_uv_max_y);
    set_render_vertex(vertices, start + 3, right, bottom, 0.05f,
        red, green, blue,
        config.pellet_color_a, config.pellet_uv_max_x, config.pellet_uv_min_y);
}

__device__ __forceinline__ int warp_compact(bool keep, int* count)
{
    const unsigned bits=__ballot_sync(0xffffffffu,keep);
    int base=0;
    if ((threadIdx.x&31)==0 && bits) base=atomicAdd(count,__popc(bits));
    base=__shfl_sync(0xffffffffu,base,0);
    const unsigned lower=(1u<<(threadIdx.x&31))-1u;
    return keep ? base+__popc(bits&lower) : -1;
}

__device__ __forceinline__ void warp_add(float value, float* total)
{
    for (int shift=16;shift;shift>>=1)
        value+=__shfl_down_sync(0xffffffffu,value,shift);
    if ((threadIdx.x&31)==0 && value!=0.0f) atomicAdd(total,value);
}

__global__ void pack_snapshot_kernel(
    DeviceWorld world,
    BgfWorldBibite* bibites,
    int32_t bibite_capacity,
    BgfWorldBibite* visible_bibites,
    int32_t visible_capacity,
    float4 view_bounds,
    BgfWorldPellet* pellets,
    int32_t pellet_capacity,
    SnapshotCounters* snapshot,
    RenderVertex* render_bibites,
    RenderVertex* render_pellets,
    BgfWorldD3D11RenderConfig render_config)
{
    const int32_t thread = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t stride = static_cast<int32_t>(blockDim.x * gridDim.x);
    const int live_count = *world.live_count;
    const int rounded_live = (live_count+31)&~31;
    for (int work=thread; work<rounded_live; work+=stride) {
        const int index=work<live_count ? world.live_items[work] : 0;
        const bool alive=work<live_count && world.alive[index]==1;
        const int destination=warp_compact(alive,&snapshot->bibites);
        if (bibites) warp_add(alive ? world.energy[index] : 0.0f,&snapshot->total_energy);
        const float2 view_position=world.positions[index];
        const bool visible=visible_bibites && alive &&
            view_position.x>=view_bounds.x && view_position.x<=view_bounds.z &&
            view_position.y>=view_bounds.y && view_position.y<=view_bounds.w;
        const int visible_destination=warp_compact(visible,&snapshot->visible_bibites);
        if (!alive) continue;
        if (render_bibites && destination < render_config.bibite_capacity) {
            write_bibite_render_vertices(world, index, destination, render_bibites);
        }
        const bool write_world_sample = bibites && destination < bibite_capacity;
        const bool write_visible = visible_destination >= 0 &&
            visible_destination < visible_capacity;
        if (!write_world_sample && !write_visible) continue;
        const float2 position = world.positions[index];
        const float2 velocity = world.velocities[index];
        const float3 color = world.colors[index];
        BgfWorldBibite result{};
        result.slot = index;
        result.generation = world.generation[index];
        result.lineage_id = world.lineage_id[index];
        result.tag_id = world.tag_id[index];
        const int32_t topology = world.template_brain[index] - 1;
        const bool full_native = topology < 0 &&
            world.native_brain_version[index] != 0;
        result.brain_nodes = topology >= 0
            ? static_cast<int32_t>(world.template_topology_node_counts[topology])
            : (full_native ? kFullNativeBrainNodes : kLegacyNativeBrainNodes);
        result.brain_synapses = topology >= 0
            ? static_cast<int32_t>(world.template_topology_synapse_counts[topology])
            : kWeightCount + world.hidden_synapse_count[index] +
                (full_native ? world.extension_synapse_count[index] : 0);
        result.position_x = position.x;
        result.position_y = position.y;
        result.velocity_x = velocity.x;
        result.velocity_y = velocity.y;
        result.heading = world.headings[index];
        result.energy = world.energy[index];
        result.age = world.age[index];
        result.size = world.size[index];
        result.color_r = color.x;
        result.color_g = color.y;
        result.color_b = color.z;
        if (write_world_sample) bibites[destination] = result;
        if (write_visible) visible_bibites[visible_destination] = result;
    }
    const int rounded_pellets=(world.pellet_count+31)&~31;
    for (int index=thread; index<rounded_pellets; index+=stride) {
        const int material=index<world.pellet_count ? world.pellet_active[index] : 0;
        const bool alive=material==1 || material==2;
        const float pellet_energy=alive ? world.pellet_energy*
            static_cast<float>(world.pellet_food_units[index])/kPelletFoodUnits : 0.0f;
        warp_compact(material==1,&snapshot->plants);
        warp_compact(material==2,&snapshot->meats);
        warp_add(material==1 ? pellet_energy : 0.0f,&snapshot->plant_energy);
        warp_add(material==2 ? pellet_energy : 0.0f,&snapshot->meat_energy);
        const int destination=warp_compact(alive,&snapshot->pellets);
        if (!alive) continue;
        if (render_pellets && destination < render_config.pellet_capacity) {
            write_pellet_render_vertices(
                world,
                index,
                destination,
                render_pellets,
                render_config);
        }
        if (!pellets || destination >= pellet_capacity) continue;
        const float2 position = world.pellet_positions[index];
        BgfWorldPellet result{};
        result.slot = index;
        result.position_x = position.x;
        result.position_y = position.y;
        result.energy = pellet_energy;
        result.material = material == 2 ? 1 : 0;
        pellets[destination] = result;
    }
}

cudaError_t capture_render_buffers(GpuWorldContext& context, RenderBufferSet& buffers)
{
    if (buffers.faulted || !buffers.render_bibite_vertices || !buffers.render_pellet_vertices)
        return cudaErrorInvalidResourceHandle;
    const size_t body_bytes=sizeof(RenderVertex)*9u*buffers.render_config.bibite_capacity;
    const size_t food_bytes=sizeof(RenderVertex)*4u*buffers.render_config.pellet_capacity;
    cudaError_t status=cudaSuccess;
    if (!buffers.captured_bibites) status=cudaMalloc(
        reinterpret_cast<void**>(&buffers.captured_bibites),body_bytes);
    if (status==cudaSuccess && !buffers.captured_pellets) status=cudaMalloc(
        reinterpret_cast<void**>(&buffers.captured_pellets),food_bytes);
    if (status==cudaSuccess && !buffers.captured_counters) status=cudaMalloc(
        reinterpret_cast<void**>(&buffers.captured_counters),sizeof(SnapshotCounters));
    if (status==cudaSuccess && !buffers.render_stream)
        status=cudaStreamCreateWithFlags(&buffers.render_stream,cudaStreamNonBlocking);
    if (status!=cudaSuccess) return status;
    status=cudaMemset(buffers.captured_counters,0,sizeof(SnapshotCounters));
    if (status!=cudaSuccess) return status;
    pack_snapshot_kernel<<<256,kThreads>>>(context.world,nullptr,0,nullptr,0,
        make_float4(0,0,0,0),nullptr,0,buffers.captured_counters,
        buffers.captured_bibites,buffers.captured_pellets,buffers.render_config);
    status=cudaGetLastError();
    SnapshotCounters counts{};
    if (status==cudaSuccess) status=cudaMemcpy(&counts,buffers.captured_counters,
        sizeof(counts),cudaMemcpyDeviceToHost);
    if (status==cudaSuccess) {
        buffers.captured_bibite_count=std::min(counts.bibites,buffers.render_config.bibite_capacity);
        buffers.captured_pellet_count=std::min(counts.pellets,buffers.render_config.pellet_capacity);
        buffers.capture_ready=true;
    }
    return status;
}

cudaError_t copy_captured_render_buffers(RenderBufferSet& buffers)
{
    if (!buffers.capture_ready) return cudaErrorInvalidValue;
    cudaGraphicsResource_t resources[2]={buffers.render_bibite_vertices,buffers.render_pellet_vertices};
    cudaStream_t stream=buffers.render_stream;
    cudaError_t status=cudaGraphicsMapResources(2,resources,stream);
    if (status!=cudaSuccess) return status;
    void *bodies=nullptr,*food=nullptr;size_t body_bytes=0,food_bytes=0;
    status=cudaGraphicsResourceGetMappedPointer(&bodies,&body_bytes,resources[0]);
    if (status==cudaSuccess) status=cudaGraphicsResourceGetMappedPointer(&food,&food_bytes,resources[1]);
    const size_t required_body=sizeof(RenderVertex)*9u*buffers.render_config.bibite_capacity;
    const size_t required_food=sizeof(RenderVertex)*4u*buffers.render_config.pellet_capacity;
    if (status==cudaSuccess && (body_bytes<required_body || food_bytes<required_food))
        status=cudaErrorInvalidValue;
    if (status==cudaSuccess && !buffers.render_buffers_initialized)
        status=cudaMemsetAsync(bodies,0,required_body,stream);
    if (status==cudaSuccess && !buffers.render_buffers_initialized)
        status=cudaMemsetAsync(food,0,required_food,stream);
    if (status==cudaSuccess) status=cudaMemcpyAsync(bodies,buffers.captured_bibites,
        sizeof(RenderVertex)*9u*buffers.captured_bibite_count,cudaMemcpyDeviceToDevice,stream);
    if (status==cudaSuccess) status=cudaMemcpyAsync(food,buffers.captured_pellets,
        sizeof(RenderVertex)*4u*buffers.captured_pellet_count,cudaMemcpyDeviceToDevice,stream);
    const cudaError_t unmap=cudaGraphicsUnmapResources(2,resources,stream);
    if (status==cudaSuccess) status=unmap;
    if (status==cudaSuccess) status=cudaStreamSynchronize(stream);
    if (status==cudaSuccess) {
        buffers.rendered_bibites=buffers.captured_bibite_count;
        buffers.rendered_pellets=buffers.captured_pellet_count;
        buffers.render_buffers_initialized=true;
    } else buffers.faulted=true;
    return status;
}

cudaError_t update_render_buffers(GpuWorldContext& context, RenderBufferSet& buffers)
{
    if (buffers.capture_ready) return copy_captured_render_buffers(buffers);
    if (buffers.faulted || !buffers.render_bibite_vertices ||
        !buffers.render_pellet_vertices) {
        return cudaErrorInvalidResourceHandle;
    }
    cudaGraphicsResource_t resources[2] = {
        buffers.render_bibite_vertices,
        buffers.render_pellet_vertices};
    cudaError_t status = cudaGraphicsMapResources(2, resources, nullptr);
    if (status != cudaSuccess) {
        buffers.faulted = true;
        return status;
    }

    RenderVertex* render_bibites = nullptr;
    RenderVertex* render_pellets = nullptr;
    size_t bibite_bytes = 0;
    size_t pellet_bytes = 0;
    status = cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&render_bibites),
        &bibite_bytes,
        resources[0]);
    if (status == cudaSuccess) {
        status = cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&render_pellets),
            &pellet_bytes,
            resources[1]);
    }
    const size_t required_bibite_bytes = sizeof(RenderVertex) * 9u *
        static_cast<size_t>(buffers.render_config.bibite_capacity);
    const size_t required_pellet_bytes = sizeof(RenderVertex) * 4u *
        static_cast<size_t>(buffers.render_config.pellet_capacity);
    if (status == cudaSuccess &&
        (bibite_bytes < required_bibite_bytes || pellet_bytes < required_pellet_bytes)) {
        status = cudaErrorInvalidValue;
    }
    if (status == cudaSuccess && !buffers.render_buffers_initialized) {
        status = cudaMemset(render_bibites, 0, required_bibite_bytes);
    }
    if (status == cudaSuccess && !buffers.render_buffers_initialized) {
        status = cudaMemset(render_pellets, 0, required_pellet_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMemset(context.snapshot_counters, 0, sizeof(SnapshotCounters));
    }
    if (status == cudaSuccess) {
        const int32_t maximum_items = std::max(
            context.config.max_bibites,
            context.config.pellet_count);
        const int32_t blocks = std::max(
            1,
            std::min(256, (maximum_items + kThreads - 1) / kThreads));
        pack_snapshot_kernel<<<blocks, kThreads>>>(
            context.world,
            nullptr,
            0,
            nullptr,
            0,
            make_float4(0.0f, 0.0f, 0.0f, 0.0f),
            nullptr,
            0,
            context.snapshot_counters,
            render_bibites,
            render_pellets,
            buffers.render_config);
        status = cudaGetLastError();
    }
    SnapshotCounters rendered{};
    if (status == cudaSuccess) {
        status = cudaMemcpy(
            &rendered,
            context.snapshot_counters,
            sizeof(rendered),
            cudaMemcpyDeviceToHost);
    }
    if (status == cudaSuccess) {
        const int32_t new_bibites = std::max(
            0,
            std::min(rendered.bibites, buffers.render_config.bibite_capacity));
        const int32_t new_pellets = std::max(
            0,
            std::min(rendered.pellets, buffers.render_config.pellet_capacity));
        if (new_bibites < buffers.rendered_bibites) {
            status = cudaMemset(
                render_bibites + static_cast<size_t>(new_bibites) * 9u,
                0,
                sizeof(RenderVertex) * 9u *
                    static_cast<size_t>(buffers.rendered_bibites - new_bibites));
        }
        if (status == cudaSuccess && new_pellets < buffers.rendered_pellets) {
            status = cudaMemset(
                render_pellets + static_cast<size_t>(new_pellets) * 4u,
                0,
                sizeof(RenderVertex) * 4u *
                    static_cast<size_t>(buffers.rendered_pellets - new_pellets));
        }
        if (status == cudaSuccess) {
            buffers.rendered_bibites = new_bibites;
            buffers.rendered_pellets = new_pellets;
            buffers.render_buffers_initialized = true;
        }
    }
    const cudaError_t unmap_status = cudaGraphicsUnmapResources(2, resources, nullptr);
    const cudaError_t result = status == cudaSuccess ? unmap_status : status;
    buffers.faulted = result != cudaSuccess;
    return result;
}

cudaError_t rebuild_runtime(GpuWorldContext& context)
{
    cudaError_t status = cudaMemset(context.world.live_count, 0, sizeof(int32_t));
    if (status != cudaSuccess) return status;
    rebuild_runtime_kernel<<<(context.world.max_bibites + kThreads - 1) / kThreads,
        kThreads>>>(context.world);
    context.live_list_dirty = false;
    return cudaGetLastError();
}

cudaError_t refresh_pellet_overflow(GpuWorldContext& context)
{
    clear_pellet_overflow_kernel<<<
        (context.world.sense_grid_width * context.world.sense_grid_width + kThreads - 1) /
            kThreads, kThreads>>>(context.world);
    count_pellet_overflow_kernel<<<
        (context.world.pellet_count + kThreads - 1) / kThreads, kThreads>>>(context.world);
    prefix_pellet_overflow_kernel<<<1,1>>>(context.world);
    index_pellet_overflow_kernel<<<
        (context.world.pellet_count + kThreads - 1) / kThreads, kThreads>>>(context.world);
    return cudaGetLastError();
}

template <typename T>
cudaError_t allocate_device(T** pointer, size_t count)
{
    *pointer = nullptr;
    if (count == 0) return cudaSuccess;
    return cudaMalloc(reinterpret_cast<void**>(pointer), sizeof(T) * count);
}


cudaError_t ensure_brain_workspace(GpuWorldContext& context)
{
    auto& world=context.world;
    if (world.brain_inputs && world.brain_outputs) return cudaSuccess;
    float *inputs=nullptr,*outputs=nullptr;
    cudaError_t status=allocate_device(&inputs,static_cast<size_t>(world.max_bibites)*kStockSensorCount);
    if (status==cudaSuccess) status=allocate_device(&outputs,
        static_cast<size_t>(world.max_bibites)*kTemplateOutputCount);
    if (status!=cudaSuccess) { cudaFree(inputs);cudaFree(outputs);return status; }
    world.brain_inputs=inputs;world.brain_outputs=outputs;
    return cudaSuccess;
}

cudaError_t ensure_template_instance_storage(GpuWorldContext& context, int required = 1)
{
    DeviceWorld& world = context.world;
    if (required < 0 || required > world.max_bibites) return cudaErrorMemoryAllocation;
    if (required <= world.template_instance_capacity) return cudaSuccess;
    int capacity = std::min(world.max_bibites, std::max(32, world.template_instance_capacity));
    while (capacity < required) capacity = std::min(world.max_bibites,capacity*2);
    const int old_capacity = world.template_instance_capacity;
    // Prepare throwing CPU allocations before acquiring temporary GPU buffers.
    std::vector<int> extra(capacity-old_capacity);
    for (int i = old_capacity; i < capacity; ++i) extra[i-old_capacity]=i;
    __half* bias = nullptr; float *accum=nullptr,*input=nullptr,*output=nullptr;
    __half* weights = nullptr;
    cudaError_t status = allocate_device(&bias,static_cast<size_t>(capacity)*kMaxTemplateNodes);
    if (status == cudaSuccess) status = allocate_device(&accum,static_cast<size_t>(capacity)*kMaxTemplateNodes);
    if (status == cudaSuccess) status = allocate_device(&input,static_cast<size_t>(capacity)*kMaxTemplateNodes);
    if (status == cudaSuccess) status = allocate_device(&output,static_cast<size_t>(capacity)*kMaxTemplateNodes);
    if (status == cudaSuccess) status = allocate_device(&weights,static_cast<size_t>(capacity)*kMaxTemplateSynapses);
#define BGF_POOL_COPY(dst,src,type,cols) \
    if (status == cudaSuccess && old_capacity) status = cudaMemcpy2D(dst,sizeof(type)*capacity,src, \
        sizeof(type)*old_capacity,sizeof(type)*old_capacity,cols,cudaMemcpyDeviceToDevice)
    BGF_POOL_COPY(bias,world.template_node_biases,__half,kMaxTemplateNodes);
    BGF_POOL_COPY(accum,world.template_node_accum,float,kMaxTemplateNodes);
    BGF_POOL_COPY(input,world.template_node_last_input,float,kMaxTemplateNodes);
    BGF_POOL_COPY(output,world.template_node_last_output,float,kMaxTemplateNodes);
    BGF_POOL_COPY(weights,world.template_synapse_weights,__half,kMaxTemplateSynapses);
#undef BGF_POOL_COPY
    int free_count = 0;
    if (status == cudaSuccess) status = cudaMemcpy(&free_count,world.template_free_count,sizeof(int),cudaMemcpyDeviceToHost);
    if (status == cudaSuccess) status = cudaMemcpy(world.template_free_items+free_count,
        extra.data(),extra.size()*sizeof(int),cudaMemcpyHostToDevice);
    free_count += capacity-old_capacity;
    if (status == cudaSuccess) status = cudaMemcpy(world.template_free_count,&free_count,sizeof(int),cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        cudaFree(bias);cudaFree(accum);cudaFree(input);cudaFree(output);cudaFree(weights);
        return status;
    }
    cudaFree(world.template_node_biases);cudaFree(world.template_node_accum);
    cudaFree(world.template_node_last_input);cudaFree(world.template_node_last_output);
    cudaFree(world.template_synapse_weights);
    world.template_node_biases=bias;world.template_node_accum=accum;
    world.template_node_last_input=input;world.template_node_last_output=output;
    world.template_synapse_weights=weights;world.template_instance_capacity=capacity;
    return cudaSuccess;
}

cudaError_t reserve_template_births(GpuWorldContext& context, int steps)
{
    if (!context.world.template_instance_capacity) return cudaSuccess;
    int count = 0;
    cudaError_t status = cudaMemcpy(&count,context.world.template_living_count,
        sizeof(int),cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return status;
    // A native newborn needs at least 7.5 simulated seconds to reach adult size.
    // Callers cap a chunk at six seconds, so only existing parents can lay.
    const int64_t possible = static_cast<int64_t>(count) *
        (2 + static_cast<int64_t>(std::ceil(steps*context.world.fixed_delta_time)));
    return ensure_template_instance_storage(context,
        static_cast<int>(std::min<int64_t>(context.world.max_bibites, count+possible)));
}

cudaError_t ensure_template_topology_capacity(
    GpuWorldContext& context,
    int32_t required_capacity)
{
    DeviceWorld& world = context.world;
    if (required_capacity <= world.template_topology_capacity) return cudaSuccess;

    int32_t capacity = std::max(kInitialTemplateTopologyCapacity, world.template_topology_capacity);
    while (capacity < required_capacity) {
        if (capacity > INT32_MAX / 2) return cudaErrorMemoryAllocation;
        capacity *= 2;
    }

    uint16_t* node_counts = nullptr;
    uint16_t* synapse_counts = nullptr;
    uint16_t* active_counts = nullptr;
    uint8_t* active_nodes = nullptr;
    uint32_t* node_descriptors = nullptr;
    uint16_t* synapse_edges = nullptr;
    cudaError_t status = allocate_device(&node_counts, static_cast<size_t>(capacity));
    if (status == cudaSuccess) {
        status = allocate_device(&synapse_counts, static_cast<size_t>(capacity));
    }
    if (status == cudaSuccess) {
        status = allocate_device(&active_counts, static_cast<size_t>(capacity));
    }
    if (status == cudaSuccess) {
        status = allocate_device(
            &active_nodes,
            static_cast<size_t>(capacity) * kMaxTemplateNodes);
    }
    if (status == cudaSuccess) {
        status = allocate_device(
            &node_descriptors,
            static_cast<size_t>(capacity) * kMaxTemplateNodes);
    }
    if (status == cudaSuccess) {
        status = allocate_device(
            &synapse_edges,
            static_cast<size_t>(capacity) * kMaxTemplateSynapses);
    }

    const size_t topology_count = context.template_topologies.size();
    if (status == cudaSuccess && topology_count > 0) {
        status = cudaMemcpy(
            node_counts,
            world.template_topology_node_counts,
            sizeof(uint16_t) * topology_count,
            cudaMemcpyDeviceToDevice);
    }
    if (status == cudaSuccess && topology_count > 0) {
        status = cudaMemcpy(
            synapse_counts,
            world.template_topology_synapse_counts,
            sizeof(uint16_t) * topology_count,
            cudaMemcpyDeviceToDevice);
    }
    if (status == cudaSuccess && topology_count > 0) {
        status = cudaMemcpy(
            active_counts,
            world.template_topology_active_node_counts,
            sizeof(uint16_t) * topology_count,
            cudaMemcpyDeviceToDevice);
    }
    if (status == cudaSuccess && topology_count > 0) {
        status = cudaMemcpy(
            active_nodes,
            world.template_active_nodes,
            sizeof(uint8_t) * topology_count * kMaxTemplateNodes,
            cudaMemcpyDeviceToDevice);
    }
    if (status == cudaSuccess && topology_count > 0) {
        status = cudaMemcpy(
            node_descriptors,
            world.template_node_descriptors,
            sizeof(uint32_t) * topology_count * kMaxTemplateNodes,
            cudaMemcpyDeviceToDevice);
    }
    if (status == cudaSuccess && topology_count > 0) {
        status = cudaMemcpy(
            synapse_edges,
            world.template_synapse_edges,
            sizeof(uint16_t) * topology_count * kMaxTemplateSynapses,
            cudaMemcpyDeviceToDevice);
    }
    if (status != cudaSuccess) {
        cudaFree(synapse_edges);
        cudaFree(node_descriptors);
        cudaFree(active_nodes);
        cudaFree(active_counts);
        cudaFree(synapse_counts);
        cudaFree(node_counts);
        return status;
    }

    cudaFree(world.template_synapse_edges);
    cudaFree(world.template_node_descriptors);
    cudaFree(world.template_active_nodes);
    cudaFree(world.template_topology_active_node_counts);
    cudaFree(world.template_topology_synapse_counts);
    cudaFree(world.template_topology_node_counts);
    world.template_topology_node_counts = node_counts;
    world.template_topology_synapse_counts = synapse_counts;
    world.template_topology_active_node_counts = active_counts;
    world.template_active_nodes = active_nodes;
    world.template_node_descriptors = node_descriptors;
    world.template_synapse_edges = synapse_edges;
    world.template_topology_capacity = capacity;
    return cudaSuccess;
}

cudaError_t find_or_register_template_topology(
    GpuWorldContext& context,
    const TemplateTopologyHost& topology,
    int32_t* topology_index)
{
    for (size_t index = 0; index < context.template_topologies.size(); ++index) {
        const TemplateTopologyHost& existing = context.template_topologies[index];
        if (existing.node_descriptors == topology.node_descriptors &&
            existing.active_nodes == topology.active_nodes &&
            existing.synapse_edges == topology.synapse_edges) {
            *topology_index = static_cast<int32_t>(index);
            return cudaSuccess;
        }
    }

    const int32_t index = static_cast<int32_t>(context.template_topologies.size());
    cudaError_t status = ensure_template_topology_capacity(context, index + 1);
    if (status != cudaSuccess) return status;

    const uint16_t node_count = static_cast<uint16_t>(topology.node_descriptors.size());
    const uint16_t synapse_count = static_cast<uint16_t>(topology.synapse_edges.size());
    const uint16_t active_count = static_cast<uint16_t>(topology.active_nodes.size());
    status = cudaMemcpy(
        context.world.template_topology_node_counts + index,
        &node_count,
        sizeof(node_count),
        cudaMemcpyHostToDevice);
    if (status == cudaSuccess) {
        status = cudaMemcpy(
            context.world.template_topology_synapse_counts + index,
            &synapse_count,
            sizeof(synapse_count),
            cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(
            context.world.template_topology_active_node_counts + index,
            &active_count,
            sizeof(active_count),
            cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess && active_count > 0) {
        status = cudaMemcpy(
            context.world.template_active_nodes + topology_node_slot(index, 0),
            topology.active_nodes.data(),
            sizeof(uint8_t) * active_count,
            cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess && node_count > 0) {
        status = cudaMemcpy(
            context.world.template_node_descriptors + topology_node_slot(index, 0),
            topology.node_descriptors.data(),
            sizeof(uint32_t) * node_count,
            cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess && synapse_count > 0) {
        status = cudaMemcpy(
            context.world.template_synapse_edges + topology_synapse_slot(index, 0),
            topology.synapse_edges.data(),
            sizeof(uint16_t) * synapse_count,
            cudaMemcpyHostToDevice);
    }
    if (status != cudaSuccess) return status;

    context.template_topologies.push_back(topology);
    *topology_index = index;
    return cudaSuccess;
}

struct CheckpointCapture { std::vector<uint8_t> bytes; };
thread_local CheckpointCapture* active_checkpoint_capture = nullptr;
thread_local uint32_t checkpoint_write_format = kCheckpointFormatVersion;

bool write_host_bytes(std::FILE* file, const void* source, size_t bytes)
{
    if (active_checkpoint_capture) {
        auto& data=active_checkpoint_capture->bytes;
        constexpr size_t maximum=1536ull*1024ull*1024ull;
        if (bytes>maximum-data.size()) return false;
        if (!bytes) return true;
        const auto* first=static_cast<const uint8_t*>(source);
        data.insert(data.end(),first,first+bytes);
        return true;
    }
    return bytes == 0 || std::fwrite(source, 1, bytes, file) == bytes;
}

// Never truncate the last usable checkpoint on disk-full, CUDA download, or
// process-interruption failures. The temporary is on the destination volume,
// created exclusively and closed on every path before an atomic replacement.
class CheckpointTransaction {
public:
    explicit CheckpointTransaction(const char* destination) : destination_(destination)
    {
        static std::atomic<uint32_t> sequence{0};
        for (int attempt = 0; attempt < 32; ++attempt) {
            temporary_ = destination_ + ".writing-" +
                std::to_string(GetCurrentProcessId()) + "-" +
                std::to_string(++sequence);
            int descriptor = -1;
            if (_sopen_s(&descriptor, temporary_.c_str(),
                    _O_CREAT | _O_EXCL | _O_WRONLY | _O_BINARY,
                    _SH_DENYRW, _S_IREAD | _S_IWRITE) == 0) {
                file_ = _fdopen(descriptor, "wb");
                if (!file_) _close(descriptor);
                return;
            }
        }
        temporary_.clear();
    }

    ~CheckpointTransaction()
    {
        if (file_) std::fclose(file_);
        if (!temporary_.empty()) std::remove(temporary_.c_str());
    }

    std::FILE* file() const { return file_; }

    bool commit()
    {
        if (!file_) return false;
        bool success = std::fflush(file_) == 0;
        if (success) success = _commit(_fileno(file_)) == 0;
        // fclose is deliberately unconditional, including flush failures.
        const bool closed = std::fclose(file_) == 0;
        file_ = nullptr;
        if (!success || !closed) return false;
        if (!MoveFileExA(temporary_.c_str(), destination_.c_str(),
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) return false;
        temporary_.clear();
        return true;
    }

private:
    std::string destination_;
    std::string temporary_;
    std::FILE* file_ = nullptr;
};

bool read_host_bytes(std::FILE* file, void* destination, size_t bytes)
{
    return bytes == 0 || std::fread(destination, 1, bytes, file) == bytes;
}

template <typename T>
bool write_host_values(std::FILE* file, const T* source, size_t count)
{
    return write_host_bytes(file, source, sizeof(T) * count);
}

template <typename T>
bool read_host_values(std::FILE* file, T* destination, size_t count)
{
    return read_host_bytes(file, destination, sizeof(T) * count);
}

cudaError_t write_device_bytes(
    std::FILE* file,
    const void* source,
    size_t bytes)
{
    constexpr size_t chunk_bytes = 16u * 1024u * 1024u;
    std::vector<uint8_t> staging(std::min(chunk_bytes, std::max<size_t>(bytes, 1u)));
    const uint8_t* device = static_cast<const uint8_t*>(source);
    size_t offset = 0;
    while (offset < bytes) {
        const size_t count = std::min(staging.size(), bytes - offset);
        cudaError_t status = cudaMemcpy(
            staging.data(),
            device + offset,
            count,
            cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) return status;
        if (!write_host_bytes(file, staging.data(), count)) return cudaErrorUnknown;
        offset += count;
    }
    return cudaSuccess;
}

cudaError_t read_device_bytes(
    std::FILE* file,
    void* destination,
    size_t bytes)
{
    constexpr size_t chunk_bytes = 16u * 1024u * 1024u;
    std::vector<uint8_t> staging(std::min(chunk_bytes, std::max<size_t>(bytes, 1u)));
    uint8_t* device = static_cast<uint8_t*>(destination);
    size_t offset = 0;
    while (offset < bytes) {
        const size_t count = std::min(staging.size(), bytes - offset);
        if (!read_host_bytes(file, staging.data(), count)) return cudaErrorInvalidValue;
        cudaError_t status = cudaMemcpy(
            device + offset,
            staging.data(),
            count,
            cudaMemcpyHostToDevice);
        if (status != cudaSuccess) return status;
        offset += count;
    }
    return cudaSuccess;
}

template <typename T>
cudaError_t write_device_values(std::FILE* file, const T* source, size_t count)
{
    return write_device_bytes(file, source, sizeof(T) * count);
}

template <typename T>
cudaError_t read_device_values(std::FILE* file, T* destination, size_t count)
{
    return read_device_bytes(file, destination, sizeof(T) * count);
}

__global__ void upgrade_legacy_food_kernel(DeviceWorld world)
{
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < world.max_bibites) world.food_gain[index] = 0.0f;
    if (index < world.pellet_count) {
        world.pellet_nominal_units[index] = kPelletFoodUnits;
        world.pellet_food_units[index] = world.pellet_active[index] == 1
            ? kPelletFoodUnits : 0;
    }
}

__global__ void upgrade_nominal_pellet_units_kernel(DeviceWorld world)
{
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < world.pellet_count)
        world.pellet_nominal_units[index] = max(
            kPelletFoodUnits, world.pellet_food_units[index]);
}

__global__ void upgrade_legacy_biology_kernel(DeviceWorld world)
{
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= world.max_bibites) return;
    world.diet[index] = 0.0f;
    world.health[index] = 100.0f * world.size[index];
    world.bite_cooldown[index] = 0.0f;
    world.pending_damage[index] = 0.0f;
    world.meat_gain[index] = 0.0f;
    world.actions_c[index] = {1.0f, 1.0f, 0.0f, 0.0f};
    world.actions_d[index] = {0.0f, 0.0f, 0.0f, 0.0f};
}

__global__ void upgrade_legacy_brain_io_kernel(DeviceWorld world)
{
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < world.max_bibites) {
        const float size = world.size[index];
        world.life_state[index] = {size, 0.0f,
            world.age[index], 1.0f};
        world.reproductive_traits[index] = {15.0f, 2.0f};
        world.last_damage[index] = 0.0f;
        world.held_target[index] = 0;
        world.held_instance[index] = 0u;
        world.clock_reset_output[index] = 0.0f;
    }
    if (index < world.pellet_count)
        world.pellet_held_by[index] = -1;
}

__global__ void upgrade_legacy_brains_kernel(DeviceWorld world)
{
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= world.max_bibites) return;
    if (world.alive[index] != 1 || world.template_brain[index] != 0) {
        world.hidden_synapse_count[index] = 0;
        for (int32_t word = 0; word < kHiddenMaskWords; ++word)
            world.hidden_masks[hidden_column_slot(world, index, word)] = 0u;
        return;
    }
    uint32_t random_state = mix_bits(world.rng[index] ^
        (0x9e3779b9u * static_cast<uint32_t>(index + 1)));
    // Existing worlds keep their original direct-brain decisions on load.
    // Descendants may later gain hidden-to-output links by mutation.
    initialise_hidden_brain(world, index, random_state, false);
}

__global__ void upgrade_v4_dormant_weights_kernel(DeviceWorld world)
{
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= world.max_bibites || world.alive[index] != 1 ||
        world.template_brain[index] != 0) return;
    // v4 used zero for untouched hidden edges. Active zeros must remain valid;
    // only untouched inactive zeros become the new never-initialised sentinel.
    for (int32_t edge = 0; edge < kHiddenConnectionCount; ++edge) {
        const uint32_t word = world.hidden_masks[
            hidden_column_slot(world, index, edge >> 5)];
        if ((word & (1u << (edge & 31))) != 0u) continue;
        const size_t slot = dense_weight_slot(world, index, edge);
        if (__half2float(world.hidden_weights[slot]) == 0.0f)
            world.hidden_weights[slot] = __float2half_rn(nanf(""));
    }
}

bool valid_config(const BgfWorldConfig& config);

bool valid_checkpoint_header(const CheckpointHeader& header)
{
    return std::memcmp(header.magic, kCheckpointMagic, sizeof(kCheckpointMagic)) == 0 &&
        (header.format_version >= 1u &&
            header.format_version <= kCheckpointFormatVersion) &&
        header.header_bytes == sizeof(CheckpointHeader) &&
        valid_config(header.config) &&
        header.topology_count <= static_cast<uint32_t>(header.config.max_bibites) &&
        header.has_template_instance_storage <= 1u &&
        (header.topology_count == 0u || header.has_template_instance_storage == 1u);
}

bool valid_checkpoint_topology(const TemplateTopologyHost& topology)
{
    const size_t count = topology.node_descriptors.size();
    bool active[kMaxTemplateNodes]{};
    for (uint32_t descriptor : topology.node_descriptors) {
        if (descriptor_node_type(descriptor) > 13 ||
            descriptor_sensor(descriptor) > 33 ||
            descriptor_action(descriptor) >= kTemplateOutputCount ||
            (descriptor & 0xff000000u) != 0u) return false;
    }
    for (uint8_t node : topology.active_nodes) {
        if (node >= count || active[node]) return false;
        active[node] = true;
    }
    for (uint16_t edge : topology.synapse_edges) {
        if (edge_source(edge) >= count || edge_destination(edge) >= count ||
            !active[edge_source(edge)] || !active[edge_destination(edge)]) return false;
    }
    return true;
}

// Check file-provided indices before a simulation/render kernel can follow
// them. This is a load-only pass; it adds no work to the simulation tick.
__global__ void validate_checkpoint_state_kernel(
    DeviceWorld world, int32_t topology_count, int32_t free_count,
    int32_t reproduction_count, int32_t eaten_count,
    int32_t* invalid, SnapshotCounters* counts, bool preserve_identity,
    int32_t* free_seen)
{
    const int32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;
    for (int32_t index = thread; index < world.max_bibites; index += stride) {
        const int32_t alive = world.alive[index];
        const int32_t brain = world.template_brain[index];
        if ((alive != 0 && alive != 1) || brain < 0 || brain > topology_count) {
            atomicExch(invalid, 1);
            continue;
        }
        if (alive == 0) continue;
        atomicAdd(&counts->bibites, 1);
        if (!preserve_identity) world.instance_id[index] = 1u;
        else if (world.instance_id[index]==0u) atomicExch(invalid,1);
        const float2 position = world.positions[index];
        const float2 velocity = world.velocities[index];
        const float4 traits = world.traits[index];
        const float3 color = world.colors[index];
        const float4 actions = world.actions_a[index];
        const float2 other_actions = world.actions_b[index];
        const float4 mouth_actions = world.actions_c[index];
        const float4 body_actions = world.actions_d[index];
        const float4 life = world.life_state[index];
        const float2 reproduction = world.reproductive_traits[index];
        const int32_t held = world.held_target[index];
        if (!isfinite(position.x) || !isfinite(position.y) ||
            position.x < -world.world_half_extent || position.x >= world.world_half_extent ||
            position.y < -world.world_half_extent || position.y >= world.world_half_extent ||
            !isfinite(velocity.x) || !isfinite(velocity.y) ||
            !isfinite(world.headings[index]) ||
            !isfinite(world.energy[index]) || world.energy[index] <= 0.0f ||
            !isfinite(world.age[index]) || world.age[index] < 0.0f ||
            !isfinite(world.size[index]) || world.size[index] <= 0.0f ||
            !isfinite(traits.x) || traits.x <= 0.0f ||
            !isfinite(traits.y) || traits.y <= 0.0f ||
            !isfinite(traits.z) || traits.z <= 0.0f ||
            !isfinite(traits.w) || traits.w <= 0.0f ||
            !isfinite(color.x) || !isfinite(color.y) || !isfinite(color.z) ||
            !isfinite(actions.x) || !isfinite(actions.y) ||
            !isfinite(actions.z) || !isfinite(actions.w) ||
            !isfinite(other_actions.x) || !isfinite(other_actions.y) ||
            !isfinite(world.reproduction_cooldown[index]) ||
            !isfinite(world.gene_mutation_strength[index]) ||
            !isfinite(world.brain_mutation_strength[index]) ||
            !isfinite(world.food_gain[index]) || world.food_gain[index] < 0.0f ||
            !isfinite(world.meat_gain[index]) || world.meat_gain[index] < 0.0f ||
            !isfinite(world.diet[index]) || world.diet[index] < 0.0f ||
            world.diet[index] > 1.0f ||
            !isfinite(world.health[index]) || world.health[index] <= 0.0f ||
            !isfinite(world.bite_cooldown[index]) ||
            world.bite_cooldown[index] < 0.0f ||
            !isfinite(mouth_actions.x) || !isfinite(mouth_actions.y) ||
            !isfinite(mouth_actions.z) || !isfinite(mouth_actions.w) ||
            !isfinite(body_actions.x) || !isfinite(body_actions.y) ||
            !isfinite(body_actions.z) || !isfinite(body_actions.w) ||
            !isfinite(life.x) || life.x < world.size[index] * 0.99f ||
            !isfinite(life.y) || life.y < 0.0f ||
            !isfinite(life.z) || life.z < 0.0f ||
            !isfinite(life.w) || life.w < 0.05f ||
            !isfinite(reproduction.x) || reproduction.x < 0.25f ||
            !isfinite(reproduction.y) || reproduction.y < 1.0f ||
            life.y > reproduction.y + 0.01f ||
            !isfinite(world.last_damage[index]) ||
            world.last_damage[index] < 0.0f ||
            !isfinite(world.clock_reset_output[index]) ||
            held > world.max_bibites || held < -world.pellet_count ||
            world.cached_food[index] < -1 || world.cached_food[index] >= world.pellet_count ||
            world.cached_neighbour[index] < -1 || world.cached_neighbour[index] >= world.max_bibites) {
            atomicExch(invalid, 1);
        }
        if (held > 0 && held <= world.max_bibites &&
            world.alive[held - 1] != 1)
            atomicExch(invalid, 1);
        if (held < 0 && held >= -world.pellet_count &&
            (world.pellet_held_by[-held - 1] != index ||
                world.pellet_active[-held - 1] == 0))
            atomicExch(invalid, 1);
        if (held > 0 && held <= world.max_bibites) {
            if (preserve_identity) {
                if (world.held_instance[index] != world.instance_id[held - 1])
                    atomicExch(invalid, 1);
            } else world.held_instance[index] = 1u;
        }
        if (brain == 0) {
            const uint8_t native_version = world.native_brain_version[index];
            if (native_version > 1u) atomicExch(invalid, 1);
            for (int32_t weight = 0; weight < kWeightCount; ++weight) {
                if (!isfinite(__half2float(world.weights[
                        dense_weight_slot(world, index, weight)]))) atomicExch(invalid, 1);
            }
            int32_t active_links = 0;
            for (int32_t word = 0; word < kHiddenMaskWords; ++word) {
                uint32_t bits = world.hidden_masks[
                    hidden_column_slot(world, index, word)];
                if (word == kHiddenMaskWords - 1 && (bits & 0xff000000u) != 0u) {
                    atomicExch(invalid, 1);
                    bits &= 0x00ffffffu;
                }
                active_links += __popc(bits);
                while (bits != 0u) {
                    const int32_t edge = word * 32 + __ffs(bits) - 1;
                    if (!isfinite(__half2float(world.hidden_weights[
                            dense_weight_slot(world, index, edge)])))
                        atomicExch(invalid, 1);
                    bits &= bits - 1u;
                }
            }
            if (active_links != world.hidden_synapse_count[index])
                atomicExch(invalid, 1);
            int32_t extension_links = 0;
            for (int32_t word = 0; word < kExtensionMaskWords; ++word) {
                uint32_t bits = world.extension_masks[
                    hidden_column_slot(world, index, word)];
                if (word == kExtensionMaskWords - 1 &&
                    (bits & 0xfffffff0u) != 0u) {
                    atomicExch(invalid, 1);
                    bits &= 0x0000000fu;
                }
                extension_links += __popc(bits);
                while (bits != 0u) {
                    const int32_t edge = word * 32 + __ffs(bits) - 1;
                    if (!isfinite(__half2float(world.extension_weights[
                            dense_weight_slot(world, index, edge)])))
                        atomicExch(invalid, 1);
                    bits &= bits - 1u;
                }
            }
            if (extension_links != world.extension_synapse_count[index] ||
                (native_version == 0u && extension_links != 0))
                atomicExch(invalid, 1);
            for (int32_t node = 0; node < kHiddenNodeCount; ++node) {
                const size_t slot = hidden_column_slot(world, index, node);
                if (!isfinite(__half2float(world.hidden_biases[slot])) ||
                    !isfinite(__half2float(world.hidden_last_input[slot])) ||
                    !isfinite(__half2float(world.hidden_last_output[slot])))
                    atomicExch(invalid, 1);
            }
        } else {
            const int pool = world.template_instance[index];
            if (pool < 0 || pool >= world.template_instance_capacity) {
                atomicExch(invalid,1); continue;
            }
            const int32_t node_count = world.template_topology_node_counts[brain - 1];
            const int32_t synapse_count = world.template_topology_synapse_counts[brain - 1];
            for (int32_t node = 0; node < node_count; ++node) {
                const size_t slot = template_node_slot(world, index, node);
                if (!isfinite(__half2float(world.template_node_biases[slot])) ||
                    !isfinite(world.template_node_last_input[slot]) ||
                    !isfinite(world.template_node_last_output[slot])) atomicExch(invalid, 1);
            }
            for (int32_t synapse = 0; synapse < synapse_count; ++synapse) {
                if (!isfinite(__half2float(world.template_synapse_weights[
                        template_synapse_slot(world, index, synapse)]))) atomicExch(invalid, 1);
            }
        }
    }
    for (int32_t index = thread; index < free_count; index += stride) {
        const int32_t slot = world.free_bibite_items[index];
        if (slot < 0 || slot >= world.max_bibites) {
            atomicExch(invalid, 1);
        } else if (world.alive[slot] != 0 ||
            atomicCAS(&free_seen[slot], 0, 1) != 0) {
            atomicExch(invalid, 1);
        }
    }
    for (int32_t index = thread; index < reproduction_count; index += stride) {
        const int32_t slot = world.reproduction_items[index];
        if (slot < 0 || slot >= world.max_bibites) atomicExch(invalid, 1);
    }
    for (int32_t index = thread; index < eaten_count; index += stride) {
        const int32_t slot = world.eaten_pellet_items[index];
        if (slot < 0 || slot >= world.pellet_count) atomicExch(invalid, 1);
    }
    for (int32_t index = thread; index < world.pellet_count; index += stride) {
        const float2 position = world.pellet_positions[index];
        const int32_t owner = world.pellet_held_by[index];
        if ((world.pellet_active[index] != 0 &&
                world.pellet_active[index] != 1 &&
                world.pellet_active[index] != 2) ||
            world.pellet_food_units[index] < 0 ||
            world.pellet_food_units[index] > kMaximumPelletFoodUnits ||
            world.pellet_nominal_units[index] < 1 ||
            world.pellet_nominal_units[index] > kMaximumPelletFoodUnits ||
            world.pellet_food_units[index] > world.pellet_nominal_units[index] ||
            (world.pellet_active[index] != 0 &&
                world.pellet_food_units[index] == 0) ||
            !isfinite(position.x) || !isfinite(position.y) ||
            position.x < -world.world_half_extent || position.x >= world.world_half_extent ||
            position.y < -world.world_half_extent || position.y >= world.world_half_extent ||
            (index >= world.pellet_limit && world.pellet_active[index] == 1) ||
            owner < -1 || owner >= world.max_bibites ||
            (owner >= 0 && (world.alive[owner] != 1 ||
                world.held_target[owner] != -index - 1))) {
            atomicExch(invalid, 1);
        }
    }
    const int32_t field_cells = world.pheromone_grid_width * world.pheromone_grid_height;
    for (int32_t index = thread; index < field_cells; index += stride) {
        const float3 a = world.pheromone_a[index];
        const float3 b = world.pheromone_b[index];
        const float3 xa = world.pheromone_heading_x_a[index];
        const float3 xb = world.pheromone_heading_x_b[index];
        const float3 ya = world.pheromone_heading_y_a[index];
        const float3 yb = world.pheromone_heading_y_b[index];
        if (!isfinite(a.x) || !isfinite(a.y) || !isfinite(a.z) ||
            !isfinite(b.x) || !isfinite(b.y) || !isfinite(b.z) ||
            !isfinite(xa.x) || !isfinite(xa.y) || !isfinite(xa.z) ||
            !isfinite(xb.x) || !isfinite(xb.y) || !isfinite(xb.z) ||
            !isfinite(ya.x) || !isfinite(ya.y) || !isfinite(ya.z) ||
            !isfinite(yb.x) || !isfinite(yb.y) || !isfinite(yb.z))
            atomicExch(invalid, 1);
    }
}

__global__ void rebuild_checkpoint_pellet_index_kernel(DeviceWorld world)
{
    for (int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < world.pellet_count; index += blockDim.x * gridDim.x) {
        if (world.pellet_active[index] == 1 || world.pellet_active[index] == 2)
            insert_pellet(world, index);
    }
}

cudaError_t validate_checkpoint_state(GpuWorldContext& context, bool preserve_identity)
{
    DeviceWorld& world = context.world;
    int32_t free_count = 0, reproduction_count = 0, eaten_count = 0;
    WorldCounters counters{};
    cudaError_t status = cudaMemcpy(&free_count, world.free_bibite_count,
        sizeof(free_count), cudaMemcpyDeviceToHost);
    if (status == cudaSuccess) status = cudaMemcpy(&reproduction_count,
        world.reproduction_count, sizeof(reproduction_count), cudaMemcpyDeviceToHost);
    if (status == cudaSuccess) status = cudaMemcpy(&eaten_count,
        world.eaten_pellet_count, sizeof(eaten_count), cudaMemcpyDeviceToHost);
    if (status == cudaSuccess) status = cudaMemcpy(&counters,
        world.counters, sizeof(counters), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return status;
    if (free_count < 0 || free_count > world.max_bibites ||
        reproduction_count < 0 || reproduction_count > world.max_bibites ||
        eaten_count < 0 || eaten_count > world.pellet_count ||
        counters.living_bibites < 0 || counters.living_bibites > world.max_bibites ||
        counters.living_bibites + free_count != world.max_bibites) return cudaErrorInvalidValue;
    // Slot-list scattering has finished. Reuse its staging index as a
    // duplicate detector without corrupting saved incarnation counters.
    if (!context.checkpoint_slots) return cudaErrorInvalidValue;
    status = cudaMemset(context.checkpoint_slots, 0,
        sizeof(int32_t) * static_cast<size_t>(world.max_bibites));
    if (status == cudaSuccess && !preserve_identity) status = cudaMemset(world.instance_id, 0,
        sizeof(uint32_t) * static_cast<size_t>(world.max_bibites));
    if (status == cudaSuccess) status = cudaMemset(context.command_result, 0, sizeof(int32_t));
    if (status == cudaSuccess) status = cudaMemset(context.snapshot_counters, 0, sizeof(SnapshotCounters));
    if (status != cudaSuccess) return status;
    validate_checkpoint_state_kernel<<<256, kThreads>>>(world,
        static_cast<int32_t>(context.template_topologies.size()),
        free_count, reproduction_count, eaten_count,
        context.command_result, context.snapshot_counters, preserve_identity,
        context.checkpoint_slots);
    status = cudaGetLastError();
    int32_t invalid = 0;
    SnapshotCounters counts{};
    if (status == cudaSuccess) status = cudaMemcpy(&invalid, context.command_result,
        sizeof(invalid), cudaMemcpyDeviceToHost);
    if (status == cudaSuccess) status = cudaMemcpy(&counts, context.snapshot_counters,
        sizeof(counts), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return status;
    if (invalid != 0 || counts.bibites != counters.living_bibites) return cudaErrorInvalidValue;

    const size_t cells = static_cast<size_t>(world.sense_grid_width) * world.sense_grid_width;
    status = cudaMemset(world.pellet_cell_masks, 0, sizeof(uint32_t) * cells);
    if (status == cudaSuccess) status = cudaMemset(world.pellet_cell_items, 0xff,
        sizeof(int32_t) * cells * kPelletBucketCapacity);
    if (status == cudaSuccess) status = cudaMemset(world.pellet_hash_cell, 0xff,
        sizeof(int32_t) * world.pellet_count);
    if (status == cudaSuccess) status = cudaMemset(world.pellet_hash_slot, 0xff,
        sizeof(int32_t) * world.pellet_count);
    if (status == cudaSuccess) status = cudaMemset(world.pellet_occupied, 0,
        sizeof(uint32_t) * ((cells + 31u) / 32u));
    if (status == cudaSuccess) status = cudaMemset(world.pellet_overflow_count, 0, sizeof(int32_t));
    if (status == cudaSuccess) {
        rebuild_checkpoint_pellet_index_kernel<<<128, kThreads>>>(world);
        status = cudaGetLastError();
    }
    return status;
}

cudaError_t reset_derived_spatial_state(GpuWorldContext& context)
{
    DeviceWorld& world = context.world;
    const size_t bibites = static_cast<size_t>(world.max_bibites);
    const size_t contact_cells = static_cast<size_t>(world.contact_grid_width) *
        world.contact_grid_width;
    const size_t sense_cells = static_cast<size_t>(world.sense_grid_width) *
        world.sense_grid_width;
    const size_t contact_bitmap_words = (contact_cells + 31u) / 32u;
    const size_t sense_bitmap_words = (sense_cells + 31u) / 32u;
    cudaError_t status = cudaMemset(
        world.contact_cell_counts,
        0,
        sizeof(int32_t) * contact_cells * 2u);
    if (status == cudaSuccess) status = cudaMemset(
        world.contact_overflow_heads,
        0xff,
        sizeof(int32_t) * contact_cells * 2u);
    if (status == cudaSuccess) status = cudaMemset(
        world.contact_overflow_next,
        0xff,
        sizeof(int32_t) * bibites * 2u);
    if (status == cudaSuccess) status = cudaMemset(
        world.contact_occupied,
        0,
        sizeof(uint32_t) * contact_bitmap_words * 2u);
    if (status == cudaSuccess) status = cudaMemset(
        world.contact_occupied_count,
        0,
        sizeof(int32_t) * 2u);
    if (status == cudaSuccess) status = cudaMemset(
        world.contact_overflow_count,
        0,
        sizeof(int32_t) * 2u);
    if (status == cudaSuccess) status = cudaMemset(
        world.large_contact_count,
        0,
        sizeof(int32_t) * 2u);
    const int32_t dirty = 1;
    if (status == cudaSuccess) status = cudaMemcpy(
        world.contact_grid_dirty,
        &dirty,
        sizeof(dirty),
        cudaMemcpyHostToDevice);
    if (status == cudaSuccess) status = cudaMemset(
        world.sense_bibite_cell_counts,
        0,
        sizeof(int32_t) * sense_cells);
    if (status == cudaSuccess) status = cudaMemset(
        world.sense_bibite_overflow_heads,
        0xff,
        sizeof(int32_t) * sense_cells);
    if (status == cudaSuccess) status = cudaMemset(
        world.sense_bibite_overflow_next,
        0xff,
        sizeof(int32_t) * bibites);
    if (status == cudaSuccess) status = cudaMemset(
        world.sense_bibite_occupied,
        0,
        sizeof(uint32_t) * sense_bitmap_words);
    if (status == cudaSuccess) status = cudaMemset(
        world.sense_bibite_occupied_count,
        0,
        sizeof(int32_t));
    if (status == cudaSuccess) status = cudaMemset(
        world.sense_bibite_overflow_count,
        0,
        sizeof(int32_t));
    if (status == cudaSuccess) status = cudaMemset(
        context.snapshot_counters,
        0,
        sizeof(SnapshotCounters));
    return status;
}


cudaError_t reconfigure_runtime_grids(GpuWorldContext& context, int contact_width, int sense_width)
{
    DeviceWorld& world=context.world;
    world.contact_grid_width=contact_width;
    world.sense_grid_width=sense_width;
    world.sense_cell_radius=std::max(1,std::min(sense_width/2,
        static_cast<int>(std::ceil(world.sense_radius*sense_width/(world.world_half_extent*2.0f)))));
    cudaError_t status=reset_derived_spatial_state(context);
    const size_t cells=static_cast<size_t>(sense_width)*sense_width;
    if (status==cudaSuccess) status=cudaMemset(world.pellet_cell_masks,0,sizeof(uint32_t)*cells);
    if (status==cudaSuccess) status=cudaMemset(world.pellet_cell_items,0xff,
        sizeof(int32_t)*cells*kPelletBucketCapacity);
    if (status==cudaSuccess) status=cudaMemset(world.pellet_hash_cell,0xff,
        sizeof(int32_t)*world.pellet_count);
    if (status==cudaSuccess) status=cudaMemset(world.pellet_hash_slot,0xff,
        sizeof(int32_t)*world.pellet_count);
    if (status==cudaSuccess) status=cudaMemset(world.pellet_occupied,0,sizeof(uint32_t)*((cells+31)/32));
    if (status==cudaSuccess) status=cudaMemset(world.pellet_overflow_count,0,sizeof(int32_t));
    const int dirty=1;
    if (status==cudaSuccess) status=cudaMemcpy(world.sense_grid_dirty,&dirty,sizeof(int),cudaMemcpyHostToDevice);
    if (status==cudaSuccess) {
        rebuild_checkpoint_pellet_index_kernel<<<128,kThreads>>>(world);
        status=cudaGetLastError();
    }
    if (status==cudaSuccess) status=refresh_pellet_overflow(context);
    return status;
}

int32_t choose_density_grid_width(int64_t population, float target_occupancy,
    int32_t minimum_width, int32_t maximum_width);

cudaError_t adapt_runtime_grids(GpuWorldContext& context)
{
    const int living=std::max(0,context.latest_living);
    const int contact=std::min(context.allocated_contact_width,
        choose_density_grid_width(living,0.25f,32,256));
    const int sense=std::min(context.allocated_sense_width,
        choose_density_grid_width(static_cast<int64_t>(living)+context.latest_active_pellets,4.0f,32,256));
    if (contact==context.world.contact_grid_width && sense==context.world.sense_grid_width)
        return cudaSuccess;
    return reconfigure_runtime_grids(context,contact,sense);
}

void free_world(DeviceWorld& world)
{
    cudaFree(world.live_items); cudaFree(world.live_scratch);
    cudaFree(world.live_count); cudaFree(world.live_scratch_count);
    cudaFree(world.digestion_efficiency); cudaFree(world.drag_retention);
    cudaFree(world.pellet_overflow_heads); cudaFree(world.pellet_overflow_next);
    cudaFree(world.pellet_overflow_counts); cudaFree(world.pellet_overflow_tile_prefix);
    cudaFree(world.food_senses_cache); cudaFree(world.food_senses_cache_active);
    cudaFree(world.contact_tile_tasks); cudaFree(world.contact_tile_count);
    cudaFree(world.brain_inputs); cudaFree(world.brain_outputs); cudaFree(world.timing);
    cudaFree(world.template_instance); cudaFree(world.template_free_items);
    cudaFree(world.template_free_count); cudaFree(world.template_living_count);
    cudaFree(world.instance_id);
    cudaFree(world.counters);
    cudaFree(world.pheromone_b);
    cudaFree(world.pheromone_a);
    cudaFree(world.pheromone_heading_x_a);
    cudaFree(world.pheromone_heading_x_b);
    cudaFree(world.pheromone_heading_y_a);
    cudaFree(world.pheromone_heading_y_b);
    cudaFree(world.food_zones);
    cudaFree(world.pellet_rng);
    cudaFree(world.pellet_positions);
    cudaFree(world.pellet_food_units);
    cudaFree(world.pellet_nominal_units);
    cudaFree(world.pellet_held_by);
    cudaFree(world.pellet_active);
    cudaFree(world.cached_neighbour_color);
    cudaFree(world.cached_neighbour_distance_squared);
    cudaFree(world.cached_food_distance_squared);
    cudaFree(world.cached_neighbour_direction);
    cudaFree(world.cached_food_direction);
    cudaFree(world.cached_neighbour_count);
    cudaFree(world.cached_neighbour);
    cudaFree(world.cached_food);
    cudaFree(world.pellet_overflow_count);
    cudaFree(world.pellet_occupied);
    cudaFree(world.pellet_hash_slot);
    cudaFree(world.pellet_hash_cell);
    cudaFree(world.pellet_cell_items);
    cudaFree(world.pellet_cell_masks);
    cudaFree(world.eaten_pellet_count);
    cudaFree(world.eaten_pellet_items);
    cudaFree(world.pellet_respawn_counts);
    cudaFree(world.pellet_respawn_items);
    cudaFree(world.pellet_pending);
    cudaFree(world.sense_bibite_overflow_count);
    cudaFree(world.sense_bibite_occupied_count);
    cudaFree(world.sense_bibite_occupied_cells);
    cudaFree(world.sense_bibite_occupied);
    cudaFree(world.sense_bibite_overflow_next);
    cudaFree(world.sense_bibite_overflow_heads);
    cudaFree(world.sense_bibite_cell_items);
    cudaFree(world.sense_bibite_cell_counts);
    cudaFree(world.contact_grid_dirty);
    cudaFree(world.sense_grid_dirty);
    cudaFree(world.contact_home_cell); cudaFree(world.contact_is_overflow);
    cudaFree(world.large_contact_count);
    cudaFree(world.large_contact_items);
    cudaFree(world.contact_occupied);
    cudaFree(world.contact_overflow_count);
    cudaFree(world.contact_overflow_next);
    cudaFree(world.contact_overflow_heads);
    cudaFree(world.contact_cell_items);
    cudaFree(world.contact_cell_counts);
    cudaFree(world.contact_occupied_count);
    cudaFree(world.contact_occupied_cells);
    cudaFree(world.wants_reproduction);
    cudaFree(world.dead_bibite_items);
    cudaFree(world.dead_bibite_count);
    cudaFree(world.meat_cursor);
    cudaFree(world.free_bibite_count);
    cudaFree(world.free_bibite_items);
    cudaFree(world.reproduction_count);
    cudaFree(world.reproduction_items);
    cudaFree(world.food_gain);
    cudaFree(world.meat_gain);
    cudaFree(world.repulsion);
    cudaFree(world.actions_b);
    cudaFree(world.actions_c);
    cudaFree(world.actions_d);
    cudaFree(world.clock_reset_output);
    cudaFree(world.actions_a);
    cudaFree(world.template_synapse_weights);
    cudaFree(world.template_synapse_edges);
    cudaFree(world.template_node_last_output);
    cudaFree(world.template_node_last_input);
    cudaFree(world.template_node_accum);
    cudaFree(world.template_node_biases);
    cudaFree(world.template_node_descriptors);
    cudaFree(world.template_active_nodes);
    cudaFree(world.template_topology_active_node_counts);
    cudaFree(world.template_topology_synapse_counts);
    cudaFree(world.template_topology_node_counts);
    cudaFree(world.template_brain);
    cudaFree(world.extension_synapse_count);
    cudaFree(world.extension_weights);
    cudaFree(world.extension_masks);
    cudaFree(world.native_brain_version);
    cudaFree(world.hidden_synapse_count);
    cudaFree(world.hidden_last_output);
    cudaFree(world.hidden_last_input);
    cudaFree(world.hidden_biases);
    cudaFree(world.hidden_weights);
    cudaFree(world.hidden_masks);
    cudaFree(world.weights);
    cudaFree(world.rng);
    cudaFree(world.tag_id);
    cudaFree(world.lineage_id);
    cudaFree(world.brain_mutation_strength);
    cudaFree(world.diet);
    cudaFree(world.health);
    cudaFree(world.bite_cooldown);
    cudaFree(world.pending_damage);
    cudaFree(world.last_damage);
    cudaFree(world.life_state);
    cudaFree(world.reproductive_traits);
    cudaFree(world.held_target);
    cudaFree(world.held_instance);
    cudaFree(world.gene_mutation_strength);
    cudaFree(world.colors);
    cudaFree(world.traits);
    cudaFree(world.reproduction_cooldown);
    cudaFree(world.size);
    cudaFree(world.age);
    cudaFree(world.energy);
    cudaFree(world.headings);
    cudaFree(world.velocities);
    cudaFree(world.positions);
    cudaFree(world.generation);
    cudaFree(world.alive);
    world = {};
}

cudaError_t unregister_render_buffer_set(RenderBufferSet& buffers)
{
    cudaError_t result = cudaSuccess;
    if (buffers.render_pellet_vertices) {
        const cudaError_t status = cudaGraphicsUnregisterResource(
            buffers.render_pellet_vertices);
        if (result == cudaSuccess) result = status;
        if (status == cudaSuccess) buffers.render_pellet_vertices = nullptr;
    }
    if (buffers.render_bibite_vertices) {
        const cudaError_t status = cudaGraphicsUnregisterResource(
            buffers.render_bibite_vertices);
        if (result == cudaSuccess) result = status;
        if (status == cudaSuccess) buffers.render_bibite_vertices = nullptr;
    }
    if (result == cudaSuccess) {
        if (buffers.render_stream) cudaStreamDestroy(buffers.render_stream);
        cudaFree(buffers.captured_bibites);cudaFree(buffers.captured_pellets);
        cudaFree(buffers.captured_counters);
        buffers={};
    }
    else buffers.faulted = true;
    return result;
}

cudaError_t unregister_render_buffers(GpuWorldContext& context)
{
    const cudaError_t first = unregister_render_buffer_set(context.render_buffers[0]);
    const cudaError_t second = unregister_render_buffer_set(context.render_buffers[1]);
    return first == cudaSuccess ? second : first;
}

int cuda_result(cudaError_t status)
{
    return status == cudaSuccess ? 0 : static_cast<int>(status);
}

int32_t choose_density_grid_width(
    int64_t item_count,
    float target_items_per_cell,
    int32_t minimum,
    int32_t maximum)
{
    const int32_t requested = std::max(
        minimum,
        static_cast<int32_t>(std::ceil(std::sqrt(
            static_cast<double>(std::max<int64_t>(item_count, 1)) /
            std::max(target_items_per_cell, 0.01f)))));
    int32_t width = 1;
    while (width < requested && width < maximum) width <<= 1;
    return std::max(minimum, std::min(width, maximum));
}

bool valid_config(const BgfWorldConfig& config)
{
    if (config.device_index < 0 || config.max_bibites <= 0 ||
        config.max_bibites > kMaximumBibites ||
        config.initial_bibites <= 0 || config.initial_bibites > config.max_bibites ||
        config.pellet_count <= 0 || config.pellet_count > kMaximumPellets ||
        config.spatial_grid_width < 8 ||
        config.pheromone_grid_width < 8 || config.pheromone_grid_height < 8 ||
        !std::isfinite(config.world_half_extent) ||
        !std::isfinite(config.fixed_delta_time) ||
        !std::isfinite(config.initial_energy) ||
        !std::isfinite(config.pellet_energy) ||
        !std::isfinite(config.reproduction_energy) ||
        !std::isfinite(config.sense_radius) ||
        !std::isfinite(config.pheromone_diffusion) ||
        !std::isfinite(config.pheromone_decay) ||
        !std::isfinite(config.mutation_strength) ||
        config.world_half_extent <= 1.0f || config.fixed_delta_time <= 0.0f ||
        config.initial_energy <= 0.0f || config.pellet_energy <= 0.0f ||
        config.reproduction_energy <= config.initial_energy || config.sense_radius <= 0.0f ||
        config.pheromone_diffusion < 0.0f || config.pheromone_diffusion > 1.0f ||
        config.pheromone_decay <= 0.0f || config.pheromone_decay > 1.0f ||
        config.mutation_strength < 0.0f ||
        config.contact_grid_update_factor <= 0 ||
        config.contact_grid_update_factor > 16 ||
        config.contact_solve_factor <= 0 ||
        config.contact_solve_factor > 16 ||
        config.vision_lookup_factor <= 0 ||
        config.vision_lookup_factor > 1024 ||
        config.brain_update_factor <= 0 ||
        config.brain_update_factor > 64 ||
        (config.lock_food_target != 0 && config.lock_food_target != 1)) {
        return false;
    }
    const int64_t spatial_cells = static_cast<int64_t>(config.spatial_grid_width) *
        config.spatial_grid_width;
    const int64_t pheromone_cells = static_cast<int64_t>(config.pheromone_grid_width) *
        config.pheromone_grid_height;
    return spatial_cells <= 16 * 1024 * 1024 && pheromone_cells <= 16 * 1024 * 1024;
}

#include "bgf_checkpoint_runtime.cuh"
#include "bgf_world_pipeline.cuh"

} // namespace

extern "C" BGF_EXPORT void bgf_world_default_config(BgfWorldConfig* config)
{
    if (!config) return;
    *config = {};
    config->device_index = 0;
    config->max_bibites = 8192;
    config->initial_bibites = 2048;
    config->pellet_count = 8192;
    config->spatial_grid_width = 128;
    config->pheromone_grid_width = 128;
    config->pheromone_grid_height = 128;
    config->seed = 0x00c0ffeeu;
    config->world_half_extent = 500.0f;
    config->fixed_delta_time = 0.025f;
    config->initial_energy = 90.0f;
    config->pellet_energy = 28.0f;
    config->reproduction_energy = 180.0f;
    config->sense_radius = 32.0f;
    config->pheromone_diffusion = 0.12f;
    config->pheromone_decay = 0.9975f;
    config->mutation_strength = 0.08f;
    config->diagnostic_mask = 0u;
    config->contact_grid_update_factor = 1;
    config->contact_solve_factor = 1;
    config->vision_lookup_factor = kVisionLookupFactor;
    config->brain_update_factor = kBrainUpdateFactor;
    config->lock_food_target = 0;
}

extern "C" BGF_EXPORT int bgf_world_create(
    const BgfWorldConfig* config,
    BgfWorldHandle* world_handle)
{
    if (!config || !world_handle || !valid_config(*config)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    *world_handle = nullptr;
    cudaError_t status = cudaSetDevice(config->device_index);
    if (status != cudaSuccess) return cuda_result(status);

    int cooperative = 0;
    status = cudaDeviceGetAttribute(
        &cooperative,
        cudaDevAttrCooperativeLaunch,
        config->device_index);
    if (status != cudaSuccess) return cuda_result(status);
    if (cooperative == 0) return static_cast<int>(cudaErrorNotSupported);

    GpuWorldContext* context = new (std::nothrow) GpuWorldContext{};
    if (!context) return static_cast<int>(cudaErrorMemoryAllocation);
    context->device_index = config->device_index;
    context->config = *config;
    context->latest_living = config->initial_bibites;
    context->latest_active_pellets = config->pellet_count;
    status = cudaGetDeviceProperties(&context->properties, config->device_index);
    if (status != cudaSuccess) goto cleanup;

    {
        DeviceWorld& value = context->world;
        value.max_bibites = config->max_bibites;
        value.initial_bibites = config->initial_bibites;
        value.pellet_count = config->pellet_count;
        value.pellet_limit = config->pellet_count;
        value.spatial_grid_width = config->spatial_grid_width;
        value.contact_grid_width = choose_density_grid_width(
            config->max_bibites,
            0.25f,
            32,
            256);
        while (value.contact_grid_width > 32 &&
            config->world_half_extent * 2.0f /
                static_cast<float>(value.contact_grid_width) <
                kRegularMaximumBodySize * 2.0f) {
            value.contact_grid_width >>= 1;
        }
        value.sense_grid_width = choose_density_grid_width(
            static_cast<int64_t>(config->max_bibites) + config->pellet_count,
            4.0f,
            32,
            256);
        value.pheromone_grid_width = config->pheromone_grid_width;
        value.pheromone_grid_height = config->pheromone_grid_height;
        value.seed = config->seed;
        value.world_half_extent = config->world_half_extent;
        value.fixed_delta_time = config->fixed_delta_time;
        value.initial_energy = config->initial_energy;
        value.pellet_energy = config->pellet_energy;
        value.food_growth_factor = 1.0f;
        value.linear_drag = 0.8f;
        value.collision_damage_constant = 0.1f;
        value.collision_damage_threshold = 50.0f;
        value.biting_damage_factor = 5.0f;
        value.biting_pressure = 80.0f;
        value.digestion = {1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f};
        value.reproduction_energy = config->reproduction_energy;
        const float ecology_scale =
            (config->world_half_extent / kEcologyReferenceHalfExtent) *
            std::sqrt(kEcologyReferencePellets /
                static_cast<float>(config->pellet_count));
        value.ecology_scale = std::max(
            1.0f,
            std::min(kMaximumEcologyScale, ecology_scale));
        value.mobility_scale = std::min(value.ecology_scale, 8.0f);
        value.sense_radius = config->sense_radius * value.ecology_scale;
        value.pheromone_diffusion = config->pheromone_diffusion;
        value.pheromone_decay = config->pheromone_decay;
        value.mutation_strength = config->mutation_strength;
        value.diagnostic_mask = config->diagnostic_mask;
        value.contact_grid_update_factor = config->contact_grid_update_factor;
        value.contact_solve_factor = config->contact_solve_factor;
        value.vision_lookup_factor = config->vision_lookup_factor;
        value.brain_update_factor = config->brain_update_factor;
        value.lock_food_target = config->lock_food_target;
        const float sense_cell_width = config->world_half_extent * 2.0f /
            static_cast<float>(value.sense_grid_width);
        value.sense_cell_radius = std::max(
            1,
            std::min(
                value.sense_grid_width / 2,
                static_cast<int32_t>(std::ceil(value.sense_radius / sense_cell_width))));

        const size_t bibites = static_cast<size_t>(config->max_bibites);
        const size_t pellets = static_cast<size_t>(config->pellet_count);
        context->allocated_contact_width = value.contact_grid_width;
        context->allocated_sense_width = value.sense_grid_width;
        const size_t contact_cells = static_cast<size_t>(value.contact_grid_width) *
            value.contact_grid_width;
        const size_t sense_cells = static_cast<size_t>(value.sense_grid_width) *
            value.sense_grid_width;
        const size_t contact_bitmap_words = (contact_cells + 31u) / 32u;
        const size_t sense_bitmap_words = (sense_cells + 31u) / 32u;
        const size_t pheromone_cells = static_cast<size_t>(config->pheromone_grid_width) *
            config->pheromone_grid_height;

#define BGF_WORLD_ALLOC(field, count) do { \
    status = allocate_device(&value.field, (count)); \
    if (status != cudaSuccess) goto cleanup; \
} while (0)
        BGF_WORLD_ALLOC(live_items, bibites);
        BGF_WORLD_ALLOC(live_scratch, bibites);
        BGF_WORLD_ALLOC(live_count, 1);
        BGF_WORLD_ALLOC(live_scratch_count, 1);
        BGF_WORLD_ALLOC(digestion_efficiency, bibites);
        BGF_WORLD_ALLOC(drag_retention, 1);
        BGF_WORLD_ALLOC(pellet_overflow_heads, sense_cells);
        BGF_WORLD_ALLOC(pellet_overflow_next, pellets);
        BGF_WORLD_ALLOC(pellet_overflow_counts, sense_cells);
        BGF_WORLD_ALLOC(pellet_overflow_tile_prefix, (sense_cells+kThreads-1)/kThreads);
        BGF_WORLD_ALLOC(food_senses_cache,bibites);
        BGF_WORLD_ALLOC(food_senses_cache_active,1);
        BGF_WORLD_ALLOC(contact_tile_tasks,bibites);
        BGF_WORLD_ALLOC(contact_tile_count,1);
        BGF_WORLD_ALLOC(timing, 1);
        BGF_WORLD_ALLOC(alive, bibites);
        BGF_WORLD_ALLOC(instance_id, bibites);
        BGF_WORLD_ALLOC(generation, bibites);
        BGF_WORLD_ALLOC(positions, bibites);
        BGF_WORLD_ALLOC(velocities, bibites);
        BGF_WORLD_ALLOC(headings, bibites);
        BGF_WORLD_ALLOC(energy, bibites);
        BGF_WORLD_ALLOC(age, bibites);
        BGF_WORLD_ALLOC(size, bibites);
        BGF_WORLD_ALLOC(reproduction_cooldown, bibites);
        BGF_WORLD_ALLOC(traits, bibites);
        BGF_WORLD_ALLOC(colors, bibites);
        BGF_WORLD_ALLOC(gene_mutation_strength, bibites);
        BGF_WORLD_ALLOC(brain_mutation_strength, bibites);
        BGF_WORLD_ALLOC(diet, bibites);
        BGF_WORLD_ALLOC(health, bibites);
        BGF_WORLD_ALLOC(bite_cooldown, bibites);
        BGF_WORLD_ALLOC(pending_damage, bibites);
        BGF_WORLD_ALLOC(last_damage, bibites);
        BGF_WORLD_ALLOC(life_state, bibites);
        BGF_WORLD_ALLOC(reproductive_traits, bibites);
        BGF_WORLD_ALLOC(held_target, bibites);
        BGF_WORLD_ALLOC(held_instance, bibites);
        BGF_WORLD_ALLOC(lineage_id, bibites);
        BGF_WORLD_ALLOC(tag_id, bibites);
        BGF_WORLD_ALLOC(rng, bibites);
        BGF_WORLD_ALLOC(weights, bibites * kWeightCount);
        BGF_WORLD_ALLOC(hidden_masks, bibites * kHiddenMaskWords);
        BGF_WORLD_ALLOC(hidden_weights, bibites * kHiddenConnectionCount);
        BGF_WORLD_ALLOC(hidden_biases, bibites * kHiddenNodeCount);
        BGF_WORLD_ALLOC(hidden_last_input, bibites * kHiddenNodeCount);
        BGF_WORLD_ALLOC(hidden_last_output, bibites * kHiddenNodeCount);
        BGF_WORLD_ALLOC(hidden_synapse_count, bibites);
        BGF_WORLD_ALLOC(native_brain_version, bibites);
        BGF_WORLD_ALLOC(extension_masks, bibites * kExtensionMaskWords);
        BGF_WORLD_ALLOC(extension_weights, bibites * kExtensionConnectionCount);
        BGF_WORLD_ALLOC(extension_synapse_count, bibites);
        BGF_WORLD_ALLOC(template_brain, bibites);
        BGF_WORLD_ALLOC(template_instance, bibites);
        BGF_WORLD_ALLOC(template_free_items, bibites);
        BGF_WORLD_ALLOC(template_free_count, 1);
        BGF_WORLD_ALLOC(template_living_count, 1);
        BGF_WORLD_ALLOC(actions_a, bibites);
        BGF_WORLD_ALLOC(actions_b, bibites);
        BGF_WORLD_ALLOC(actions_c, bibites);
        BGF_WORLD_ALLOC(actions_d, bibites);
        BGF_WORLD_ALLOC(clock_reset_output, bibites);
        BGF_WORLD_ALLOC(repulsion, bibites);
        BGF_WORLD_ALLOC(food_gain, bibites);
        BGF_WORLD_ALLOC(meat_gain, bibites);
        BGF_WORLD_ALLOC(wants_reproduction, bibites);
        BGF_WORLD_ALLOC(reproduction_items, bibites);
        BGF_WORLD_ALLOC(reproduction_count, 1);
        BGF_WORLD_ALLOC(free_bibite_items, bibites);
        BGF_WORLD_ALLOC(free_bibite_count, 1);
        BGF_WORLD_ALLOC(dead_bibite_items, bibites);
        BGF_WORLD_ALLOC(dead_bibite_count, 1);
        BGF_WORLD_ALLOC(meat_cursor, 1);
        BGF_WORLD_ALLOC(contact_cell_counts, contact_cells * 2u);
        BGF_WORLD_ALLOC(contact_cell_items, contact_cells * kContactBucketCapacity * 2u);
        BGF_WORLD_ALLOC(contact_overflow_heads, contact_cells * 2u);
        BGF_WORLD_ALLOC(contact_overflow_next, bibites * 2u);
        BGF_WORLD_ALLOC(contact_occupied, contact_bitmap_words * 2u);
        BGF_WORLD_ALLOC(contact_occupied_cells, contact_cells * 2u);
        BGF_WORLD_ALLOC(contact_occupied_count, 2);
        BGF_WORLD_ALLOC(contact_overflow_count, 2);
        BGF_WORLD_ALLOC(large_contact_items, bibites * 2u);
        BGF_WORLD_ALLOC(large_contact_count, 2);
        BGF_WORLD_ALLOC(contact_grid_dirty, 1);
        BGF_WORLD_ALLOC(sense_grid_dirty, 1);
        BGF_WORLD_ALLOC(contact_home_cell, bibites * 2u);
        BGF_WORLD_ALLOC(contact_is_overflow, bibites * 2u);
        BGF_WORLD_ALLOC(sense_bibite_cell_counts, sense_cells);
        BGF_WORLD_ALLOC(sense_bibite_cell_items, sense_cells * kSenseBucketCapacity);
        BGF_WORLD_ALLOC(sense_bibite_overflow_heads, sense_cells);
        BGF_WORLD_ALLOC(sense_bibite_overflow_next, bibites);
        BGF_WORLD_ALLOC(sense_bibite_occupied, sense_bitmap_words);
        BGF_WORLD_ALLOC(sense_bibite_occupied_cells, sense_cells);
        BGF_WORLD_ALLOC(sense_bibite_occupied_count, 1);
        BGF_WORLD_ALLOC(sense_bibite_overflow_count, 1);
        BGF_WORLD_ALLOC(pellet_cell_masks, sense_cells);
        BGF_WORLD_ALLOC(pellet_cell_items, sense_cells * kPelletBucketCapacity);
        BGF_WORLD_ALLOC(pellet_hash_cell, pellets);
        BGF_WORLD_ALLOC(pellet_hash_slot, pellets);
        BGF_WORLD_ALLOC(pellet_occupied, sense_bitmap_words);
        BGF_WORLD_ALLOC(pellet_overflow_count, 1);
        BGF_WORLD_ALLOC(eaten_pellet_items, pellets);
        BGF_WORLD_ALLOC(eaten_pellet_count, 1);
        BGF_WORLD_ALLOC(
            pellet_respawn_items,
            pellets * static_cast<size_t>(kPelletRespawnWheelSize));
        BGF_WORLD_ALLOC(pellet_respawn_counts, kPelletRespawnWheelSize);
        BGF_WORLD_ALLOC(pellet_pending, pellets);
        BGF_WORLD_ALLOC(cached_food, bibites);
        BGF_WORLD_ALLOC(cached_neighbour, bibites);
        BGF_WORLD_ALLOC(cached_neighbour_count, bibites);
        BGF_WORLD_ALLOC(cached_food_direction, bibites);
        BGF_WORLD_ALLOC(cached_neighbour_direction, bibites);
        BGF_WORLD_ALLOC(cached_food_distance_squared, bibites);
        BGF_WORLD_ALLOC(cached_neighbour_distance_squared, bibites);
        BGF_WORLD_ALLOC(cached_neighbour_color, bibites);
        BGF_WORLD_ALLOC(pellet_active, pellets);
        BGF_WORLD_ALLOC(pellet_food_units, pellets);
        BGF_WORLD_ALLOC(pellet_nominal_units, pellets);
        BGF_WORLD_ALLOC(pellet_held_by, pellets);
        BGF_WORLD_ALLOC(pellet_positions, pellets);
        BGF_WORLD_ALLOC(pellet_rng, pellets);
        BGF_WORLD_ALLOC(food_zones, kMaximumFoodZones);
        BGF_WORLD_ALLOC(pheromone_a, pheromone_cells);
        BGF_WORLD_ALLOC(pheromone_b, pheromone_cells);
        BGF_WORLD_ALLOC(pheromone_heading_x_a, pheromone_cells);
        BGF_WORLD_ALLOC(pheromone_heading_x_b, pheromone_cells);
        BGF_WORLD_ALLOC(pheromone_heading_y_a, pheromone_cells);
        BGF_WORLD_ALLOC(pheromone_heading_y_b, pheromone_cells);
        BGF_WORLD_ALLOC(counters, 1);
#undef BGF_WORLD_ALLOC

        context->snapshot_bibite_capacity = std::min(
            config->max_bibites,
            kMaximumSnapshotBibites);
        status = allocate_device(
            &context->snapshot_bibites,
            static_cast<size_t>(context->snapshot_bibite_capacity));
        if (status != cudaSuccess) goto cleanup;
        context->snapshot_visible_bibite_capacity = std::min(
            config->max_bibites,
            kMaximumVisibleBibites);
        status = allocate_device(
            &context->snapshot_visible_bibites,
            static_cast<size_t>(context->snapshot_visible_bibite_capacity));
        if (status != cudaSuccess) goto cleanup;
        status = allocate_device(&context->snapshot_pellets, pellets);
        if (status != cudaSuccess) goto cleanup;
        status = allocate_device(&context->snapshot_counters, 1);
        if (status != cudaSuccess) goto cleanup;
        status = allocate_device(&context->selected_detail, 1);
        if (status != cudaSuccess) goto cleanup;
        status = allocate_device(
            &context->selected_nodes,
            static_cast<size_t>(kMaxTemplateNodes));
        if (status != cudaSuccess) goto cleanup;
        status = allocate_device(
            &context->selected_synapses,
            static_cast<size_t>(kMaximumSelectedSynapses));
        if (status != cudaSuccess) goto cleanup;
        status = allocate_device(&context->command_result, 1);
        if (status != cudaSuccess) goto cleanup;

        status = cudaMemset(value.instance_id, 0, sizeof(uint32_t) * bibites);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.pending_damage, 0, sizeof(float) * bibites);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.last_damage, 0, sizeof(float) * bibites);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.life_state, 0, sizeof(float4) * bibites);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.reproductive_traits, 0, sizeof(float2) * bibites);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.held_target, 0, sizeof(int32_t) * bibites);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.held_instance, 0, sizeof(uint32_t) * bibites);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.clock_reset_output, 0, sizeof(float) * bibites);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.dead_bibite_count, 0, sizeof(int32_t));
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.meat_cursor, 0, sizeof(int32_t));
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.hidden_masks, 0,
            sizeof(uint32_t) * bibites * kHiddenMaskWords);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.hidden_weights, 0,
            sizeof(__half) * bibites * kHiddenConnectionCount);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.hidden_biases, 0,
            sizeof(__half) * bibites * kHiddenNodeCount);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.hidden_last_input, 0,
            sizeof(__half) * bibites * kHiddenNodeCount);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.hidden_last_output, 0,
            sizeof(__half) * bibites * kHiddenNodeCount);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.hidden_synapse_count, 0,
            sizeof(uint16_t) * bibites);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.native_brain_version, 0, sizeof(uint8_t) * bibites);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.extension_masks, 0,
            sizeof(uint32_t) * bibites * kExtensionMaskWords);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.extension_weights, 0,
            sizeof(__half) * bibites * kExtensionConnectionCount);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.extension_synapse_count, 0,
            sizeof(uint16_t) * bibites);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.contact_cell_counts, 0, sizeof(int32_t) * contact_cells * 2u);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(
            value.contact_overflow_heads,
            0xff,
            sizeof(int32_t) * contact_cells * 2u);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(
            value.contact_occupied,
            0,
            sizeof(uint32_t) * contact_bitmap_words * 2u);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.contact_occupied_count, 0, sizeof(int32_t) * 2u);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.contact_overflow_count, 0, sizeof(int32_t) * 2u);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.large_contact_count, 0, sizeof(int32_t) * 2u);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.sense_grid_dirty, 0, sizeof(int32_t));
        if (status == cudaSuccess) status = cudaMemset(value.contact_grid_dirty, 0, sizeof(int32_t));
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.sense_bibite_cell_counts, 0, sizeof(int32_t) * sense_cells);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.sense_bibite_overflow_heads, 0xff, sizeof(int32_t) * sense_cells);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.sense_bibite_occupied, 0, sizeof(uint32_t) * sense_bitmap_words);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.sense_bibite_occupied_count, 0, sizeof(int32_t));
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.sense_bibite_overflow_count, 0, sizeof(int32_t));
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.pellet_cell_masks, 0, sizeof(uint32_t) * sense_cells);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(
            value.pellet_cell_items,
            0xff,
            sizeof(int32_t) * sense_cells * kPelletBucketCapacity);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.pellet_hash_cell, 0xff, sizeof(int32_t) * pellets);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.pellet_hash_slot, 0xff, sizeof(int32_t) * pellets);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.pellet_occupied, 0, sizeof(uint32_t) * sense_bitmap_words);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.pellet_overflow_count, 0, sizeof(int32_t));
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.eaten_pellet_count, 0, sizeof(int32_t));
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(
            value.pellet_respawn_counts,
            0,
            sizeof(int32_t) * kPelletRespawnWheelSize);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.pellet_pending, 0, sizeof(int32_t) * pellets);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.pellet_held_by, 0xff, sizeof(int32_t) * pellets);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.pheromone_a, 0, sizeof(float3) * pheromone_cells);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.pheromone_b, 0, sizeof(float3) * pheromone_cells);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.pheromone_heading_x_a, 0, sizeof(float3) * pheromone_cells);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.pheromone_heading_x_b, 0, sizeof(float3) * pheromone_cells);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.pheromone_heading_y_a, 0, sizeof(float3) * pheromone_cells);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.pheromone_heading_y_b, 0, sizeof(float3) * pheromone_cells);
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.counters, 0, sizeof(WorldCounters));
        if (status != cudaSuccess) goto cleanup;

        const int32_t initial_blocks = std::max(
            1,
            static_cast<int32_t>((std::max(bibites, pellets) + kThreads - 1) / kThreads));
        status = cudaMemset(value.template_instance, 0xff, sizeof(int32_t)*bibites);
        if (status == cudaSuccess) status = cudaMemset(value.template_free_count,0,sizeof(int32_t));
        if (status == cudaSuccess) status = cudaMemset(value.template_living_count,0,sizeof(int32_t));
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.live_count, 0, sizeof(int32_t));
        if (status != cudaSuccess) goto cleanup;
        status = cudaMemset(value.pellet_overflow_heads, 0, sizeof(int32_t) * sense_cells);
        if (status==cudaSuccess) status=cudaMemset(value.pellet_overflow_counts,0,sizeof(int32_t)*sense_cells);
        if (status==cudaSuccess) status=cudaMemset(value.food_senses_cache_active,0,sizeof(int32_t));
        if (status==cudaSuccess) status=cudaMemset(value.contact_tile_count,0,sizeof(int32_t));
        if (status != cudaSuccess) goto cleanup;
        value.contact_grid_width = std::min(context->allocated_contact_width,
            choose_density_grid_width(config->initial_bibites,0.25f,32,256));
        value.sense_grid_width = std::min(context->allocated_sense_width,
            choose_density_grid_width(static_cast<int64_t>(config->initial_bibites)+
                config->pellet_count,4.0f,32,256));
        value.sense_cell_radius = std::max(1,std::min(value.sense_grid_width/2,
            static_cast<int>(std::ceil(value.sense_radius*value.sense_grid_width/
                (value.world_half_extent*2.0f)))));
        const float retention = std::exp(-value.linear_drag * value.fixed_delta_time);
        status = cudaMemcpy(value.drag_retention, &retention, sizeof(float), cudaMemcpyHostToDevice);
        if (status != cudaSuccess) goto cleanup;
        initialise_world_kernel<<<std::min(initial_blocks, 256), kThreads>>>(value);
        status = cudaGetLastError();
        if (status != cudaSuccess) goto cleanup;
        status = cudaDeviceSynchronize();
        if (status != cudaSuccess) goto cleanup;

        int32_t blocks_per_sm = 0;
        status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm,
            ((value.diagnostic_mask&2097152u) || !(value.diagnostic_mask&268435456u)) ? reinterpret_cast<void*>(world_stage_kernel<0,1>) :
                reinterpret_cast<void*>(world_stage_kernel<0,2>),
            kThreads,
            0);
        if (status != cudaSuccess) goto cleanup;
        const size_t maximum_work = std::max({
            bibites,
            pellets,
            contact_cells,
            sense_cells,
            pheromone_cells});
        const int32_t desired_blocks = std::max(
            1,
            static_cast<int32_t>((maximum_work + kThreads - 1) / kThreads));
        context->grid_blocks = std::min(
            desired_blocks,
            blocks_per_sm * context->properties.multiProcessorCount);
        context->maximum_grid_blocks =
            blocks_per_sm * context->properties.multiProcessorCount;
        if (context->grid_blocks <= 0) {
            status = cudaErrorNotSupported;
            goto cleanup;
        }
    }

    status = cudaEventCreate(&context->step_started);
    if (status != cudaSuccess) goto cleanup;
    status = cudaEventCreate(&context->step_finished);
    if (status != cudaSuccess) goto cleanup;
    *world_handle = context;
    return 0;

cleanup:
    if (context) {
        unregister_render_buffers(*context);
        if (context->step_finished) cudaEventDestroy(context->step_finished);
        if (context->step_started) cudaEventDestroy(context->step_started);
        cudaFree(context->command_result);
        cudaFree(context->selected_synapses);
        cudaFree(context->selected_nodes);
        cudaFree(context->selected_detail);
        cudaFree(context->snapshot_counters);
        cudaFree(context->snapshot_pellets);
        cudaFree(context->snapshot_visible_bibites);
        cudaFree(context->snapshot_bibites);
        free_world(context->world);
        delete context;
    }
    return cuda_result(status);
}

extern "C" BGF_EXPORT int bgf_world_destroy(BgfWorldHandle world_handle)
{
    if (!world_handle) return 0;
    GpuWorldContext* context = static_cast<GpuWorldContext*>(world_handle);
    cudaError_t status = cudaSetDevice(context->device_index);
    if (status != cudaSuccess) return cuda_result(status);
    const cudaError_t render_status = unregister_render_buffers(*context);
    for (auto& entry:context->graph_cache) if (entry.graph) cudaGraphExecDestroy(entry.graph);
    if (context->step_finished) cudaEventDestroy(context->step_finished);
    if (context->step_started) cudaEventDestroy(context->step_started);
    cudaFree(context->checkpoint_device_staging);
    cudaFreeHost(context->checkpoint_staging);
    cudaFree(context->checkpoint_slots);cudaFree(context->checkpoint_template_slots);
    cudaFree(context->command_result);
    cudaFree(context->selected_synapses);
    cudaFree(context->selected_nodes);
    cudaFree(context->selected_detail);
    cudaFree(context->snapshot_counters);
    cudaFree(context->snapshot_pellets);
    cudaFree(context->snapshot_visible_bibites);
    cudaFree(context->snapshot_bibites);
    free_world(context->world);
    delete context;
    return cuda_result(render_status);
}

extern "C" BGF_EXPORT int bgf_world_step(
    BgfWorldHandle world_handle,
    int32_t steps,
    BgfWorldStepMetrics* metrics)
try
{
    if (!world_handle || steps <= 0) return static_cast<int>(cudaErrorInvalidValue);
    const auto wall_started = std::chrono::steady_clock::now();
    GpuWorldContext* context = static_cast<GpuWorldContext*>(world_handle);
    cudaError_t status = cudaSetDevice(context->device_index);
    if (status != cudaSuccess) return cuda_result(status);

    if (context->live_list_dirty) {
        status = rebuild_runtime(*context);
        if (status != cudaSuccess) return cuda_result(status);
        status=cudaMemcpy(&context->latest_living,context->world.live_count,sizeof(int),cudaMemcpyDeviceToHost);
        if (status!=cudaSuccess) return cuda_result(status);
    }
    status=adapt_runtime_grids(*context);
    if (status!=cudaSuccess) return cuda_result(status);
    DeviceWorld world = context->world;
    const int body_threads=(world.diagnostic_mask&67108864u)?32:
        (world.diagnostic_mask&134217728u)?64:kThreads;
    const int body_blocks=(std::max(context->latest_living,1)+body_threads-1)/body_threads;
    const int bulk_work=std::max({world.pellet_count,
        world.sense_grid_width*world.sense_grid_width,
        world.pheromone_grid_width*world.pheromone_grid_height});
    context->grid_blocks=std::max(1,std::min(context->maximum_grid_blocks,
        std::max(body_blocks,(bulk_work+kThreads-1)/kThreads)));
    context->graph_build_milliseconds=0.0f;
    context->graph_cache_hits=0;
    reset_profile_kernel<<<1, 1>>>(world);
    status = cudaGetLastError();
    if (status != cudaSuccess) return cuda_result(status);
    const bool tensor_mode=(world.diagnostic_mask&16777216u)!=0u;
    if (tensor_mode && (context->properties.major<7 ||
        std::strstr(context->properties.name,"GTX")!=nullptr))
        return static_cast<int>(cudaErrorNotSupported);
    int imported_living = 0;
    if (world.template_instance_capacity) {
        status=cudaMemcpy(&imported_living,world.template_living_count,sizeof(int),
            cudaMemcpyDeviceToHost);
        if (status!=cudaSuccess) return cuda_result(status);
    }
    const bool use_graph = (world.diagnostic_mask & (8388608u|16777216u|33554432u)) != 0u ||
        imported_living>0;
    if (use_graph) {
        status=ensure_brain_workspace(*context);
        if (status!=cudaSuccess) return cuda_result(status);
        world=context->world;
    }
    float gpu_milliseconds = 0.0f;
    const auto finish_chunk = [&]() {
        cudaError_t error=cudaEventRecord(context->step_finished);
        if (error==cudaSuccess) error=cudaEventSynchronize(context->step_finished);
        float elapsed=0.0f;
        if (error==cudaSuccess) error=cudaEventElapsedTime(&elapsed,
            context->step_started,context->step_finished);
        gpu_milliseconds+=elapsed;
        return error;
    };
    if (!use_graph) {
        status=cudaEventRecord(context->step_started);
        void* arguments[] = {&world, &steps};
        if (status==cudaSuccess) status = cudaLaunchCooperativeKernel(
            ((world.diagnostic_mask&2097152u) || !(world.diagnostic_mask&268435456u)) ? reinterpret_cast<void*>(world_stage_kernel<0,1>) :
                reinterpret_cast<void*>(world_stage_kernel<0,2>), context->grid_blocks,
            kThreads, arguments, 0, nullptr);
        if (status==cudaSuccess) status=finish_chunk();
    } else {
        int remaining = steps;
        while (remaining > 0 && status == cudaSuccess) {
            const int max_chunk = imported_living>0
                ? std::max(1,std::min(128,static_cast<int>(6.0f/world.fixed_delta_time))) : 128;
            const int chunk = std::min(remaining,max_chunk);
            status = reserve_template_births(*context,chunk);
            world = context->world;
            // No GPU start event until allocations/graph building have finished.
            if (status == cudaSuccess) status = ensure_step_graph(*context, chunk);
            if (status == cudaSuccess) status=cudaEventRecord(context->step_started);
            if (status == cudaSuccess) {
                restart_profile_phase_kernel<<<1,1>>>(world);
                status=cudaGetLastError();
            }
            if (status == cudaSuccess) status = cudaGraphLaunch(context->step_graph, nullptr);
            if (status == cudaSuccess) status=finish_chunk();
            if (status == cudaSuccess) context->host_steps+=chunk;
            remaining -= chunk;
        }
    }
    if (status != cudaSuccess) return cuda_result(status);
    context->has_stepped = true;
    if (!use_graph) context->host_steps+=steps;
    WorldCounters phase_counters{};
    if (metrics) {
        status = cudaMemcpy(
            &phase_counters,
            context->world.counters,
            sizeof(phase_counters),
            cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) return cuda_result(status);
        context->latest_living=phase_counters.living_bibites;
    }

    RuntimeTiming detail_timing{};
    if (metrics) {
        status = cudaMemcpy(&detail_timing, context->world.timing, sizeof(detail_timing),
            cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) return cuda_result(status);
    }
    const auto wall_finished = std::chrono::steady_clock::now();
    const float wall_milliseconds = static_cast<float>(
        std::chrono::duration<double, std::milli>(wall_finished - wall_started).count());
    const float host_overhead = fmaxf(wall_milliseconds - gpu_milliseconds, 0.0f);
    const float denominator = gpu_milliseconds + host_overhead;


    if (metrics) {
        metrics->requested_steps = steps;
        metrics->gpu_milliseconds = gpu_milliseconds;
        metrics->wall_milliseconds = wall_milliseconds;
        metrics->host_overhead_milliseconds = host_overhead;
        metrics->gpu_offload_percent = denominator > 0.0f
            ? gpu_milliseconds / denominator * 100.0f : 0.0f;
        metrics->simulated_seconds = static_cast<double>(steps) * context->config.fixed_delta_time;
        metrics->realtime_multiplier = wall_milliseconds > 0.0f
            ? metrics->simulated_seconds / (wall_milliseconds / 1000.0) : 0.0;
        const unsigned long long measured_cycles =
            phase_counters.phase_prepare_cycles +
            phase_counters.phase_spatial_index_cycles +
            phase_counters.phase_contact_cycles +
            phase_counters.phase_decision_cycles +
            phase_counters.phase_motion_cycles +
            phase_counters.phase_lifecycle_cycles;
        const float milliseconds_per_cycle = measured_cycles > 0ull
            ? gpu_milliseconds / static_cast<float>(measured_cycles)
            : 0.0f;
        metrics->prepare_milliseconds = static_cast<float>(
            phase_counters.phase_prepare_cycles) * milliseconds_per_cycle;
        metrics->spatial_index_milliseconds = static_cast<float>(
            phase_counters.phase_spatial_index_cycles) * milliseconds_per_cycle;
        metrics->contact_milliseconds = static_cast<float>(
            phase_counters.phase_contact_cycles) * milliseconds_per_cycle;
        metrics->decision_milliseconds = static_cast<float>(
            phase_counters.phase_decision_cycles) * milliseconds_per_cycle;
        metrics->motion_milliseconds = static_cast<float>(
            phase_counters.phase_motion_cycles) * milliseconds_per_cycle;
        metrics->lifecycle_milliseconds = static_cast<float>(
            phase_counters.phase_lifecycle_cycles) * milliseconds_per_cycle;
        metrics->food_milliseconds = detail_timing.food_cycles * milliseconds_per_cycle;
        metrics->brain_milliseconds = detail_timing.brain_cycles * milliseconds_per_cycle;
        metrics->post_decision_milliseconds = detail_timing.post_cycles * milliseconds_per_cycle;
        metrics->pipeline = tensor_mode ? 2 : (world.diagnostic_mask&33554432u) ? 3 :
            use_graph ? 1 : 0;
        metrics->graph_build_milliseconds=context->graph_build_milliseconds;
        metrics->graph_cache_hits=context->graph_cache_hits;
    }
    return 0;
}
catch (const std::bad_alloc&) { return static_cast<int>(cudaErrorMemoryAllocation); }
catch (...) { return static_cast<int>(cudaErrorUnknown); }

extern "C" BGF_EXPORT int bgf_world_set_food_settings(
    BgfWorldHandle world_handle,
    int32_t target,
    float growth_factor,
    float pellet_energy)
{
    if (!world_handle || !std::isfinite(growth_factor) ||
        !std::isfinite(pellet_energy) || growth_factor < 0.0f ||
        growth_factor > 50.0f || pellet_energy <= 0.0f) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    GpuWorldContext* context = static_cast<GpuWorldContext*>(world_handle);
    DeviceWorld& world = context->world;
    if (target < 0 || target > world.pellet_count) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t status = cudaSetDevice(context->device_index);
    if (status != cudaSuccess) return cuda_result(status);
    const int32_t old_target = world.pellet_limit;
    if (target != old_target) {
        const int32_t changed = std::abs(target - old_target);
        const int32_t blocks = (changed + kThreads - 1) / kThreads;
        update_food_target_kernel<<<blocks, kThreads>>>(world, old_target, target);
        status = cudaGetLastError();
        if (status != cudaSuccess) return cuda_result(status);
        status = cudaDeviceSynchronize();
        if (status != cudaSuccess) return cuda_result(status);
        world.pellet_limit = target;
    }
    const int32_t maximum_work = std::max({
        world.max_bibites,
        target,
        world.contact_grid_width * world.contact_grid_width,
        world.sense_grid_width * world.sense_grid_width,
        world.pheromone_grid_width * world.pheromone_grid_height});
    context->grid_blocks = std::max(1,
        std::min(context->maximum_grid_blocks,
            (maximum_work + kThreads - 1) / kThreads));
    // A newly created managed world reserves food slots above its starting
    // count. Keep its original movement/sensing balance based on the starting
    // food density, not the reserved capacity.
    if (!context->has_stepped) {
        const float ecology_scale =
            (world.world_half_extent / kEcologyReferenceHalfExtent) *
            std::sqrt(kEcologyReferencePellets /
                static_cast<float>(std::max(target, 1)));
        world.ecology_scale = std::max(
            1.0f, std::min(kMaximumEcologyScale, ecology_scale));
        world.mobility_scale = std::min(world.ecology_scale, 8.0f);
        world.sense_radius = context->config.sense_radius * world.ecology_scale;
        const float cell_width = world.world_half_extent * 2.0f /
            static_cast<float>(world.sense_grid_width);
        world.sense_cell_radius = std::max(
            1, std::min(world.sense_grid_width / 2,
                static_cast<int32_t>(std::ceil(world.sense_radius / cell_width))));
    }
    world.food_growth_factor = growth_factor;
    world.pellet_energy = pellet_energy;
    context->config.pellet_energy = pellet_energy;
    return 0;
}

extern "C" BGF_EXPORT int bgf_world_get_food_settings(
    BgfWorldHandle world_handle,
    int32_t* target,
    float* growth_factor,
    float* pellet_energy)
{
    if (!world_handle || !target || !growth_factor || !pellet_energy) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const GpuWorldContext* context = static_cast<const GpuWorldContext*>(world_handle);
    *target = context->world.pellet_limit;
    *growth_factor = context->world.food_growth_factor;
    *pellet_energy = context->world.pellet_energy;
    return 0;
}

extern "C" BGF_EXPORT int bgf_world_set_linear_drag(
    BgfWorldHandle world_handle, float drag)
{
    if (!world_handle || !std::isfinite(drag) || drag < 0.0f || drag > 4.0f)
        return static_cast<int>(cudaErrorInvalidValue);
    GpuWorldContext* context = static_cast<GpuWorldContext*>(world_handle);
    context->world.linear_drag = drag;
    context->live_list_dirty = true;
    return 0;
}

extern "C" BGF_EXPORT int bgf_world_get_linear_drag(
    BgfWorldHandle world_handle, float* drag)
{
    if (!world_handle || !drag) return static_cast<int>(cudaErrorInvalidValue);
    *drag = static_cast<GpuWorldContext*>(world_handle)->world.linear_drag;
    return 0;
}

extern "C" BGF_EXPORT int bgf_world_set_combat_settings(
    BgfWorldHandle world_handle, float collision_damage_constant,
    float collision_damage_threshold, float biting_damage_factor,
    float biting_pressure)
{
    if (!world_handle || !std::isfinite(collision_damage_constant) ||
        !std::isfinite(collision_damage_threshold) ||
        !std::isfinite(biting_damage_factor) || !std::isfinite(biting_pressure) ||
        collision_damage_constant < 0.0f || collision_damage_constant > 2.0f ||
        collision_damage_threshold < 0.0f || collision_damage_threshold > 200.0f ||
        biting_damage_factor < 0.0f || biting_damage_factor > 20.0f ||
        biting_pressure < 1.0f || biting_pressure > 500.0f)
        return static_cast<int>(cudaErrorInvalidValue);
    DeviceWorld& world = static_cast<GpuWorldContext*>(world_handle)->world;
    world.collision_damage_constant = collision_damage_constant;
    world.collision_damage_threshold = collision_damage_threshold;
    world.biting_damage_factor = biting_damage_factor;
    world.biting_pressure = biting_pressure;
    return 0;
}

extern "C" BGF_EXPORT int bgf_world_set_digestion_settings(
    BgfWorldHandle world_handle, const BgfWorldDigestionSettings* settings)
{
    if (!world_handle || !settings ||
        !std::isfinite(settings->plant_affinity_power) ||
        !std::isfinite(settings->meat_affinity_power) ||
        !std::isfinite(settings->plant_min_efficiency) ||
        !std::isfinite(settings->plant_max_efficiency) ||
        !std::isfinite(settings->meat_min_efficiency) ||
        !std::isfinite(settings->meat_max_efficiency) ||
        settings->plant_affinity_power < 0.1f ||
        settings->plant_affinity_power > 3.0f ||
        settings->meat_affinity_power < 0.1f ||
        settings->meat_affinity_power > 3.0f ||
        settings->plant_min_efficiency < -1.0f ||
        settings->plant_min_efficiency > 1.0f ||
        settings->plant_max_efficiency < -1.0f ||
        settings->plant_max_efficiency > 1.0f ||
        settings->meat_min_efficiency < -1.0f ||
        settings->meat_min_efficiency > 1.0f ||
        settings->meat_max_efficiency < -1.0f ||
        settings->meat_max_efficiency > 1.0f)
        return static_cast<int>(cudaErrorInvalidValue);
    GpuWorldContext* context = static_cast<GpuWorldContext*>(world_handle);
    context->world.digestion = *settings;
    context->live_list_dirty = true;
    return 0;
}

extern "C" BGF_EXPORT int bgf_world_get_digestion_settings(
    BgfWorldHandle world_handle, BgfWorldDigestionSettings* settings)
{
    if (!world_handle || !settings)
        return static_cast<int>(cudaErrorInvalidValue);
    *settings = static_cast<GpuWorldContext*>(world_handle)->world.digestion;
    return 0;
}

extern "C" BGF_EXPORT int bgf_world_set_food_zones(
    BgfWorldHandle world_handle,
    const BgfWorldFoodZone* zones,
    int32_t count,
    int32_t reseed_existing)
{
    if (!world_handle || count < 0 || count > kMaximumFoodZones ||
        (count > 0 && !zones) ||
        (reseed_existing != 0 && reseed_existing != 1)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    GpuWorldContext* context = static_cast<GpuWorldContext*>(world_handle);
    DeviceWorld& world = context->world;
    for (int32_t i = 0; i < count; ++i) {
        const BgfWorldFoodZone& zone = zones[i];
        if (zone.distribution < 0 || zone.distribution > 5 ||
            !std::isfinite(zone.center_x) || !std::isfinite(zone.center_y) ||
            !std::isfinite(zone.radius) || !std::isfinite(zone.inner_radius) ||
            !std::isfinite(zone.half_width) || !std::isfinite(zone.half_height) ||
            !std::isfinite(zone.seed_weight) || !std::isfinite(zone.growth_weight) ||
            !std::isfinite(zone.pellet_size) ||
            zone.pellet_size < 0.0f || zone.pellet_size > 50.0f ||
            std::abs(zone.center_x) > world.world_half_extent ||
            std::abs(zone.center_y) > world.world_half_extent ||
            zone.inner_radius < 0.0f || zone.inner_radius >= 1.0f ||
            zone.seed_weight < 0.0f || zone.growth_weight < 0.0f ||
            (zone.distribution == 5
                ? zone.half_width <= 0.0f || zone.half_height <= 0.0f
                : zone.radius <= 0.0f)) {
            return static_cast<int>(cudaErrorInvalidValue);
        }
    }
    cudaError_t status = cudaSetDevice(context->device_index);
    if (status != cudaSuccess) return cuda_result(status);
    if (count > 0) {
        status = cudaMemcpy(world.food_zones, zones,
            sizeof(BgfWorldFoodZone) * static_cast<size_t>(count),
            cudaMemcpyHostToDevice);
        if (status != cudaSuccess) return cuda_result(status);
    }
    world.food_zone_count = count;
    reseed_food_zone_pellets_kernel<<<128, kThreads>>>(world, reseed_existing);
    status = cudaGetLastError();
    if (status != cudaSuccess) return cuda_result(status);
    // The world constructor runs before Unity can project scenario zones.
    // On a multi-island scenario its uniform starter population would otherwise
    // begin in the empty ocean, far from every plant. Only move fresh starters;
    // editing zones or loading a checkpoint must never teleport living Bibites.
    if (!context->has_stepped && count > 0 && reseed_existing != 0) {
        move_initial_bibites_into_food_zones_kernel<<<128, kThreads>>>(world);
        status = cudaGetLastError();
        if (status != cudaSuccess) return cuda_result(status);
        status = reset_derived_spatial_state(*context);
        if (status != cudaSuccess) return cuda_result(status);
    }
    if (reseed_existing != 0) {
        const size_t cells = static_cast<size_t>(world.sense_grid_width) *
            world.sense_grid_width;
        status = cudaMemset(world.pellet_cell_masks, 0,
            sizeof(uint32_t) * cells);
        if (status == cudaSuccess) status = cudaMemset(world.pellet_cell_items,
            0xff, sizeof(int32_t) * cells * kPelletBucketCapacity);
        if (status == cudaSuccess) status = cudaMemset(world.pellet_hash_cell,
            0xff, sizeof(int32_t) * world.pellet_count);
        if (status == cudaSuccess) status = cudaMemset(world.pellet_hash_slot,
            0xff, sizeof(int32_t) * world.pellet_count);
        if (status == cudaSuccess) status = cudaMemset(world.pellet_occupied,
            0, sizeof(uint32_t) * ((cells + 31u) / 32u));
        if (status == cudaSuccess) status = cudaMemset(world.pellet_overflow_count,
            0, sizeof(int32_t));
        if (status == cudaSuccess) status = cudaMemset(world.cached_food,
            0xff, sizeof(int32_t) * world.max_bibites);
        if (status == cudaSuccess) {
            rebuild_checkpoint_pellet_index_kernel<<<128, kThreads>>>(world);
            status = cudaGetLastError();
        }
    }
    if (status == cudaSuccess) status = cudaDeviceSynchronize();
    return cuda_result(status);
}


extern "C" BGF_EXPORT int bgf_world_get_runtime_info(BgfWorldHandle handle,BgfWorldRuntimeInfo* info)
{
    if (!handle || !info) return static_cast<int>(cudaErrorInvalidValue);
    *info={};
    auto& context=*static_cast<GpuWorldContext*>(handle);
    cudaError_t status=cudaSetDevice(context.device_index);
    if (status==cudaSuccess) status=rebuild_runtime(context);
    if (status==cudaSuccess) status=cudaMemcpy(&info->living_work_items,context.world.live_count,sizeof(int),cudaMemcpyDeviceToHost);
    if (status==cudaSuccess) status=cudaMemcpy(&info->imported_brains,context.world.template_living_count,sizeof(int),cudaMemcpyDeviceToHost);
    if (status!=cudaSuccess) return cuda_result(status);
    info->imported_pool_capacity=context.world.template_instance_capacity;
    info->contact_grid_width=context.world.contact_grid_width;
    info->sense_grid_width=context.world.sense_grid_width;
    info->imported_pool_bytes=static_cast<uint64_t>(info->imported_pool_capacity)*
        (kMaxTemplateNodes*(sizeof(__half)+3*sizeof(float))+kMaxTemplateSynapses*sizeof(__half));
    info->brain_workspace_bytes=context.world.brain_inputs ? static_cast<uint64_t>(context.world.max_bibites)*
        (kStockSensorCount+kTemplateOutputCount)*sizeof(float) : 0;
    return 0;
}

extern "C" BGF_EXPORT int bgf_world_get_stats(BgfWorldHandle world, BgfWorldStats* stats)
{
    int bodies=0,pellets=0,visible=0;
    return bgf_world_download_snapshot_with_view(world,nullptr,0,&bodies,nullptr,0,&pellets,
        nullptr,0,&visible,0,0,0,0,stats);
}

extern "C" BGF_EXPORT int bgf_world_spawn_bibite(
    BgfWorldHandle world_handle,
    float position_x,
    float position_y,
    float heading,
    int32_t* spawned_slot)
{
    if (!world_handle || !spawned_slot || !std::isfinite(position_x) ||
        !std::isfinite(position_y) || !std::isfinite(heading)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    GpuWorldContext* context = static_cast<GpuWorldContext*>(world_handle);
    context->live_list_dirty = true;
    cudaError_t status = cudaSetDevice(context->device_index);
    if (status != cudaSuccess) return cuda_result(status);

    int32_t* device_slot = nullptr;
    status = cudaMalloc(reinterpret_cast<void**>(&device_slot), sizeof(int32_t));
    if (status != cudaSuccess) return cuda_result(status);
    status = cudaMemset(device_slot, 0xff, sizeof(int32_t));
    if (status == cudaSuccess) {
        const uint32_t seed = mix_bits(
            context->config.seed ^ 0x51ed270bu ^ ++context->manual_spawn_sequence);
        spawn_bibite_kernel<<<1, 1>>>(
            context->world,
            float2{position_x, position_y},
            heading,
            seed,
            device_slot);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) status = cudaDeviceSynchronize();
    if (status == cudaSuccess) {
        status = cudaMemcpy(
            spawned_slot,
            device_slot,
            sizeof(int32_t),
            cudaMemcpyDeviceToHost);
    }
    cudaFree(device_slot);
    return cuda_result(status);
}

extern "C" BGF_EXPORT int bgf_world_spawn_template_bibite(
    BgfWorldHandle world_handle,
    const BgfWorldTemplateSpawn* spawn,
    const BgfWorldBrainNode* nodes,
    int32_t node_count,
    const BgfWorldBrainSynapse* synapses,
    int32_t synapse_count,
    int32_t* spawned_slot)
try
{
    if (!world_handle || !spawn || !spawned_slot || !nodes ||
        node_count <= 0 || node_count > kMaxTemplateNodes ||
        synapse_count < 0 || synapse_count > kMaxTemplateSynapses ||
        (synapse_count > 0 && !synapses) ||
        !std::isfinite(spawn->position_x) || !std::isfinite(spawn->position_y) ||
        !std::isfinite(spawn->heading) || !std::isfinite(spawn->energy) ||
        !std::isfinite(spawn->size) || !std::isfinite(spawn->maximum_speed) ||
        !std::isfinite(spawn->turn_speed) || !std::isfinite(spawn->metabolism) ||
        !std::isfinite(spawn->lifespan) || !std::isfinite(spawn->color_r) ||
        !std::isfinite(spawn->color_g) || !std::isfinite(spawn->color_b) ||
        !std::isfinite(spawn->gene_mutation_strength) ||
        !std::isfinite(spawn->brain_mutation_strength) ||
        !std::isfinite(spawn->diet) ||
        !std::isfinite(spawn->adult_size) ||
        !std::isfinite(spawn->clock_period) ||
        !std::isfinite(spawn->lay_period) ||
        !std::isfinite(spawn->womb_capacity) ||
        spawn->energy <= 0.0f || spawn->size <= 0.0f ||
        spawn->maximum_speed <= 0.0f || spawn->turn_speed <= 0.0f ||
        spawn->metabolism <= 0.0f || spawn->lifespan <= 0.0f ||
        spawn->gene_mutation_strength < 0.0f ||
        spawn->brain_mutation_strength < 0.0f) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    for (int32_t index = 0; index < node_count; ++index) {
        if (nodes[index].type < 0 || nodes[index].type > 13 ||
            nodes[index].sensor < -1 || nodes[index].sensor > 33 ||
            nodes[index].action < -1 || nodes[index].action >= kTemplateOutputCount ||
            !std::isfinite(nodes[index].base_activation)) {
            return static_cast<int>(cudaErrorInvalidValue);
        }
    }
    for (int32_t index = 0; index < synapse_count; ++index) {
        if (synapses[index].node_in < 0 || synapses[index].node_in >= node_count ||
            synapses[index].node_out < 0 || synapses[index].node_out >= node_count ||
            !std::isfinite(synapses[index].weight)) {
            return static_cast<int>(cudaErrorInvalidValue);
        }
    }

    GpuWorldContext* context = static_cast<GpuWorldContext*>(world_handle);
    context->live_list_dirty = true;
    cudaError_t status = cudaSetDevice(context->device_index);
    if (status != cudaSuccess) return cuda_result(status);
    const float4 traits{
        clampf(spawn->maximum_speed, 0.1f, 100.0f) * context->world.mobility_scale,
        clampf(spawn->turn_speed, 0.1f, 50.0f),
        clampf(spawn->metabolism, 0.001f, 10.0f),
        clampf(spawn->lifespan, 1.0f, 1000000.0f)};
    const float3 color{
        clampf(spawn->color_r, 0.0f, 1.0f),
        clampf(spawn->color_g, 0.0f, 1.0f),
        clampf(spawn->color_b, 0.0f, 1.0f)};
    const float size = clampf(spawn->size, 0.1f, 10.0f);
    const float adult_size = clampf(fmaxf(size, spawn->adult_size), 0.1f, 10.0f);
    const float4 life_state{adult_size, 0.0f, 0.0f,
        spawn->clock_period > 0.0f ? clampf(spawn->clock_period, 0.05f, 3600.0f) : 1.0f};
    const float2 reproductive_traits{
        spawn->lay_period > 0.0f ? clampf(spawn->lay_period, 0.25f, 3600.0f) : 15.0f,
        spawn->womb_capacity > 0.0f ? clampf(spawn->womb_capacity, 1.0f, 8.0f) : 2.0f};
    const float energy = fmaxf(spawn->energy, 0.001f);
    const float gene_mutation = clampf(spawn->gene_mutation_strength, 0.0f, 4.0f);
    const float brain_mutation = clampf(spawn->brain_mutation_strength, 0.0f, 4.0f);
    std::vector<uint8_t> active_flags(static_cast<size_t>(node_count), 0u);
    for (int32_t index = 0; index < node_count; ++index) {
        if (nodes[index].action >= 0) active_flags[index] = 1u;
    }
    for (int32_t index = 0; index < synapse_count; ++index) {
        active_flags[synapses[index].node_in] = 1u;
        active_flags[synapses[index].node_out] = 1u;
    }
    TemplateTopologyHost topology;
    topology.node_descriptors.resize(static_cast<size_t>(node_count));
    topology.active_nodes.reserve(static_cast<size_t>(node_count));
    topology.synapse_edges.resize(static_cast<size_t>(synapse_count));
    std::vector<__half> node_biases(static_cast<size_t>(node_count));
    for (int32_t index = 0; index < node_count; ++index) {
        topology.node_descriptors[index] = pack_node_descriptor(
            nodes[index].type,
            nodes[index].sensor,
            nodes[index].action);
        node_biases[index] = __float2half_rn(clampf(
            nodes[index].base_activation,
            -16.0f,
            16.0f));
        if (active_flags[index] != 0u) {
            topology.active_nodes.push_back(static_cast<uint8_t>(index));
        }
    }
    std::vector<__half> synapse_weights(static_cast<size_t>(synapse_count));
    for (int32_t index = 0; index < synapse_count; ++index) {
        topology.synapse_edges[index] = pack_synapse_edge(
            synapses[index].node_in,
            synapses[index].node_out);
        synapse_weights[index] = __float2half_rn(clampf(
            synapses[index].weight,
            -16.0f,
            16.0f));
    }

    int32_t topology_index = -1;
    status = find_or_register_template_topology(*context, topology, &topology_index);
    if (status != cudaSuccess) return cuda_result(status);
    int imported_count = 0;
    status = cudaMemcpy(&imported_count,context->world.template_living_count,sizeof(int),
        cudaMemcpyDeviceToHost);
    if (status == cudaSuccess) status = ensure_template_instance_storage(*context,imported_count+1);
    if (status != cudaSuccess) return cuda_result(status);

    int status_code = bgf_world_spawn_bibite(
        world_handle,
        spawn->position_x,
        spawn->position_y,
        spawn->heading,
        spawned_slot);
    if (status_code != 0 || *spawned_slot < 0) return status_code;

    const int32_t slot = *spawned_slot;
    assign_template_instance_kernel<<<1,1>>>(context->world,slot);
    int pool_slot = -1;
    status = cudaMemcpy(&pool_slot,context->world.template_instance+slot,sizeof(int),cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return cuda_result(status);
    const int32_t one_based_topology = topology_index + 1;

#define BGF_COPY_SCALAR(field, value, type) do { \
    const type copy_value = (value); \
    status = cudaMemcpy(context->world.field + slot, &copy_value, sizeof(type), cudaMemcpyHostToDevice); \
    if (status != cudaSuccess) return cuda_result(status); \
} while (0)
    BGF_COPY_SCALAR(template_brain, one_based_topology, int32_t);
    BGF_COPY_SCALAR(hidden_synapse_count, 0, uint16_t);
    BGF_COPY_SCALAR(generation, std::max(0, spawn->generation), int32_t);
    BGF_COPY_SCALAR(energy, energy, float);
    BGF_COPY_SCALAR(size, size, float);
    BGF_COPY_SCALAR(life_state, life_state, float4);
    BGF_COPY_SCALAR(reproductive_traits, reproductive_traits, float2);
    BGF_COPY_SCALAR(traits, traits, float4);
    BGF_COPY_SCALAR(colors, color, float3);
    BGF_COPY_SCALAR(gene_mutation_strength, gene_mutation, float);
    BGF_COPY_SCALAR(brain_mutation_strength, brain_mutation, float);
    BGF_COPY_SCALAR(diet, clampf(spawn->diet, 0.0f, 1.0f), float);
    BGF_COPY_SCALAR(health, 100.0f * size, float);
    BGF_COPY_SCALAR(lineage_id, spawn->lineage_id, uint64_t);
    BGF_COPY_SCALAR(tag_id, spawn->tag_id, uint64_t);
#undef BGF_COPY_SCALAR

#define BGF_COPY_COLUMN(field, source, type, count) do { \
    if ((count) > 0) { \
        status = cudaMemcpy2D( \
            context->world.field + pool_slot, \
            sizeof(type) * static_cast<size_t>(context->world.template_instance_capacity), \
            (source).data(), \
            sizeof(type), \
            sizeof(type), \
            static_cast<size_t>(count), \
            cudaMemcpyHostToDevice); \
        if (status != cudaSuccess) return cuda_result(status); \
    } \
} while (0)
    BGF_COPY_COLUMN(template_node_biases, node_biases, __half, node_count);
    BGF_COPY_COLUMN(template_synapse_weights, synapse_weights, __half, synapse_count);
#undef BGF_COPY_COLUMN
    const size_t node_pitch = sizeof(float) *
        static_cast<size_t>(context->world.template_instance_capacity);
    status = cudaMemset2D(
        context->world.template_node_accum + pool_slot,
        node_pitch,
        0,
        sizeof(float),
        static_cast<size_t>(node_count));
    if (status == cudaSuccess) {
        status = cudaMemset2D(
            context->world.template_node_last_input + pool_slot,
            node_pitch,
            0,
            sizeof(float),
            static_cast<size_t>(node_count));
    }
    if (status == cudaSuccess) {
        status = cudaMemset2D(
            context->world.template_node_last_output + pool_slot,
            node_pitch,
            0,
            sizeof(float),
            static_cast<size_t>(node_count));
    }
    if (status == cudaSuccess) {
        finish_template_spawn_kernel<<<1, 1>>>(context->world);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) status = cudaDeviceSynchronize();
    return cuda_result(status);
}
catch (const std::bad_alloc&) { return static_cast<int>(cudaErrorMemoryAllocation); }
catch (...) { return static_cast<int>(cudaErrorUnknown); }

extern "C" BGF_EXPORT int bgf_world_download_bibites(
    BgfWorldHandle world_handle,
    BgfWorldBibite* bibites,
    int32_t capacity,
    int32_t* written)
{
    if (!world_handle || !written || capacity < 0 || (capacity > 0 && !bibites)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    *written = 0;
    GpuWorldContext* context = static_cast<GpuWorldContext*>(world_handle);
    cudaError_t status = cudaSetDevice(context->device_index);
    if (status != cudaSuccess) return cuda_result(status);
    const size_t count = static_cast<size_t>(context->config.max_bibites);
    std::vector<int32_t> alive(count);
    std::vector<int32_t> generation(count);
    std::vector<uint64_t> lineage_id(count);
    std::vector<uint64_t> tag_id(count);
    std::vector<int32_t> brain_topologies(count);
    std::vector<uint8_t> native_brain_versions(count);
    std::vector<float2> positions(count);
    std::vector<float2> velocities(count);
    std::vector<float> headings(count);
    std::vector<float> energy(count);
    std::vector<float> age(count);
    std::vector<float> size(count);
    std::vector<float3> colors(count);
    std::vector<uint16_t> hidden_synapse_counts(count);
    std::vector<uint16_t> extension_synapse_counts(count);

#define BGF_WORLD_DOWNLOAD(host, field, type) do { \
    status = cudaMemcpy((host).data(), context->world.field, sizeof(type) * count, cudaMemcpyDeviceToHost); \
    if (status != cudaSuccess) return cuda_result(status); \
} while (0)
    BGF_WORLD_DOWNLOAD(alive, alive, int32_t);
    BGF_WORLD_DOWNLOAD(generation, generation, int32_t);
    BGF_WORLD_DOWNLOAD(lineage_id, lineage_id, uint64_t);
    BGF_WORLD_DOWNLOAD(tag_id, tag_id, uint64_t);
    BGF_WORLD_DOWNLOAD(brain_topologies, template_brain, int32_t);
    BGF_WORLD_DOWNLOAD(native_brain_versions, native_brain_version, uint8_t);
    BGF_WORLD_DOWNLOAD(positions, positions, float2);
    BGF_WORLD_DOWNLOAD(velocities, velocities, float2);
    BGF_WORLD_DOWNLOAD(headings, headings, float);
    BGF_WORLD_DOWNLOAD(energy, energy, float);
    BGF_WORLD_DOWNLOAD(age, age, float);
    BGF_WORLD_DOWNLOAD(size, size, float);
    BGF_WORLD_DOWNLOAD(colors, colors, float3);
    BGF_WORLD_DOWNLOAD(hidden_synapse_counts, hidden_synapse_count, uint16_t);
    BGF_WORLD_DOWNLOAD(extension_synapse_counts, extension_synapse_count, uint16_t);
#undef BGF_WORLD_DOWNLOAD

    for (size_t index = 0; index < count && *written < capacity; ++index) {
        if (alive[index] != 1) continue;
        BgfWorldBibite& result = bibites[*written];
        result.slot = static_cast<int32_t>(index);
        result.generation = generation[index];
        result.lineage_id = lineage_id[index];
        result.tag_id = tag_id[index];
        const int32_t topology = brain_topologies[index] - 1;
        if (topology >= 0 &&
            topology < static_cast<int32_t>(context->template_topologies.size())) {
            const TemplateTopologyHost& shared = context->template_topologies[topology];
            result.brain_nodes = static_cast<int32_t>(shared.node_descriptors.size());
            result.brain_synapses = static_cast<int32_t>(shared.synapse_edges.size());
        } else {
            const bool full_native = native_brain_versions[index] != 0;
            result.brain_nodes = full_native
                ? kFullNativeBrainNodes : kLegacyNativeBrainNodes;
            result.brain_synapses = kWeightCount + hidden_synapse_counts[index] +
                (full_native ? extension_synapse_counts[index] : 0);
        }
        result.position_x = positions[index].x;
        result.position_y = positions[index].y;
        result.velocity_x = velocities[index].x;
        result.velocity_y = velocities[index].y;
        result.heading = headings[index];
        result.energy = energy[index];
        result.age = age[index];
        result.size = size[index];
        result.color_r = colors[index].x;
        result.color_g = colors[index].y;
        result.color_b = colors[index].z;
        ++(*written);
    }
    return 0;
}

extern "C" BGF_EXPORT int bgf_world_download_pellets(
    BgfWorldHandle world_handle,
    BgfWorldPellet* pellets,
    int32_t capacity,
    int32_t* written)
{
    if (!world_handle || !written || capacity < 0 || (capacity > 0 && !pellets)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    *written = 0;
    GpuWorldContext* context = static_cast<GpuWorldContext*>(world_handle);
    cudaError_t status = cudaSetDevice(context->device_index);
    if (status != cudaSuccess) return cuda_result(status);
    const size_t count = static_cast<size_t>(context->config.pellet_count);
    std::vector<int32_t> active(count);
    std::vector<int32_t> food_units(count);
    std::vector<float2> positions(count);
    status = cudaMemcpy(
        active.data(),
        context->world.pellet_active,
        sizeof(int32_t) * count,
        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return cuda_result(status);
    status = cudaMemcpy(
        food_units.data(),
        context->world.pellet_food_units,
        sizeof(int32_t) * count,
        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return cuda_result(status);
    status = cudaMemcpy(
        positions.data(),
        context->world.pellet_positions,
        sizeof(float2) * count,
        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return cuda_result(status);

    for (size_t index = 0; index < count && *written < capacity; ++index) {
        if (active[index] != 1 && active[index] != 2) continue;
        BgfWorldPellet& result = pellets[*written];
        result.slot = static_cast<int32_t>(index);
        result.position_x = positions[index].x;
        result.position_y = positions[index].y;
        result.energy = context->world.pellet_energy *
            static_cast<float>(food_units[index]) / kPelletFoodUnits;
        result.material = active[index] == 2 ? 1 : 0;
        ++(*written);
    }
    return 0;
}


extern "C" BGF_EXPORT int bgf_world_download_visible_bibites(BgfWorldHandle handle,
    BgfWorldBibite* output,int32_t capacity,int32_t* written,
    float min_x,float min_y,float max_x,float max_y)
{
    if (!handle || !written || capacity<0 || (capacity>0 && !output) ||
        !std::isfinite(min_x) || !std::isfinite(min_y) || !std::isfinite(max_x) ||
        !std::isfinite(max_y) || min_x>max_x || min_y>max_y)
        return static_cast<int>(cudaErrorInvalidValue);
    *written=0;
    auto& context=*static_cast<GpuWorldContext*>(handle);
    cudaError_t status=cudaSetDevice(context.device_index);
    if (status==cudaSuccess) status=rebuild_runtime(context);
    if (status==cudaSuccess) status=cudaMemset(context.snapshot_counters,0,sizeof(SnapshotCounters));
    if (status!=cudaSuccess) return cuda_result(status);
    DeviceWorld view=context.world;
    view.pellet_count=0; // View detail does not need the food pass or chart sums.
    const int limit=std::min(capacity,context.snapshot_visible_bibite_capacity);
    pack_snapshot_kernel<<<128,kThreads>>>(view,nullptr,0,
        limit>0 ? context.snapshot_visible_bibites : nullptr,limit,
        make_float4(min_x,min_y,max_x,max_y),nullptr,0,context.snapshot_counters,
        nullptr,nullptr,BgfWorldD3D11RenderConfig{});
    status=cudaGetLastError();
    SnapshotCounters counts{};
    if (status==cudaSuccess) status=cudaMemcpy(&counts,context.snapshot_counters,
        sizeof(counts),cudaMemcpyDeviceToHost);
    const int count=std::min(limit,counts.visible_bibites);
    if (status==cudaSuccess && count>0) status=cudaMemcpy(output,context.snapshot_visible_bibites,
        sizeof(BgfWorldBibite)*count,cudaMemcpyDeviceToHost);
    if (status==cudaSuccess) *written=count;
    return cuda_result(status);
}

extern "C" BGF_EXPORT int bgf_world_download_snapshot_with_view(
    BgfWorldHandle world_handle,
    BgfWorldBibite* bibites,
    int32_t bibite_capacity,
    int32_t* bibites_written,
    BgfWorldPellet* pellets,
    int32_t pellet_capacity,
    int32_t* pellets_written,
    BgfWorldBibite* visible_bibites,
    int32_t visible_capacity,
    int32_t* visible_written,
    float view_min_x,
    float view_min_y,
    float view_max_x,
    float view_max_y,
    BgfWorldStats* stats)
{
    if (!world_handle || !bibites_written || !pellets_written ||
        !visible_written || !stats ||
        bibite_capacity < 0 || pellet_capacity < 0 || visible_capacity < 0 ||
        (bibite_capacity > 0 && !bibites) ||
        (pellet_capacity > 0 && !pellets) ||
        (visible_capacity > 0 && (!visible_bibites ||
            !std::isfinite(view_min_x) || !std::isfinite(view_min_y) ||
            !std::isfinite(view_max_x) || !std::isfinite(view_max_y) ||
            view_min_x > view_max_x || view_min_y > view_max_y))) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    *bibites_written = 0;
    *pellets_written = 0;
    *visible_written = 0;
    *stats = {};

    GpuWorldContext* context = static_cast<GpuWorldContext*>(world_handle);
    cudaError_t status = cudaSetDevice(context->device_index);
    if (status != cudaSuccess) return cuda_result(status);
    if (context->live_list_dirty) {
        status=rebuild_runtime(*context);
        if (status!=cudaSuccess) return cuda_result(status);
    }
    status = cudaMemset(context->snapshot_counters, 0, sizeof(SnapshotCounters));
    if (status != cudaSuccess) return cuda_result(status);

    const int32_t maximum_items = std::max(
        context->config.max_bibites,
        context->config.pellet_count);
    const int32_t blocks = std::max(
        1,
        std::min(256, (maximum_items + kThreads - 1) / kThreads));
    const int32_t native_bibite_capacity = std::min(
        bibite_capacity,
        context->snapshot_bibite_capacity);
    const int32_t native_visible_capacity = std::min(
        visible_capacity,
        context->snapshot_visible_bibite_capacity);
    pack_snapshot_kernel<<<blocks, kThreads>>>(
        context->world,
        context->snapshot_bibites,
        native_bibite_capacity,
        native_visible_capacity > 0 ? context->snapshot_visible_bibites : nullptr,
        native_visible_capacity,
        make_float4(view_min_x, view_min_y, view_max_x, view_max_y),
        pellet_capacity > 0 ? context->snapshot_pellets : nullptr,
        std::min(pellet_capacity,context->config.pellet_count),
        context->snapshot_counters,
        nullptr,
        nullptr,
        BgfWorldD3D11RenderConfig{});
    status = cudaGetLastError();
    if (status != cudaSuccess) return cuda_result(status);

    SnapshotCounters snapshot{};
    status = cudaMemcpy(
        &snapshot,
        context->snapshot_counters,
        sizeof(snapshot),
        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return cuda_result(status);
    const int32_t bibite_count = std::min(snapshot.bibites, native_bibite_capacity);
    const int32_t visible_count = std::min(
        snapshot.visible_bibites, native_visible_capacity);
    const int32_t pellet_count = std::min(snapshot.pellets, pellet_capacity);
    if (bibite_count > 0) {
        status = cudaMemcpy(
            bibites,
            context->snapshot_bibites,
            sizeof(BgfWorldBibite) * static_cast<size_t>(bibite_count),
            cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) return cuda_result(status);
    }
    if (visible_count > 0) {
        status = cudaMemcpy(
            visible_bibites,
            context->snapshot_visible_bibites,
            sizeof(BgfWorldBibite) * static_cast<size_t>(visible_count),
            cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) return cuda_result(status);
    }
    if (pellet_count > 0) {
        status = cudaMemcpy(
            pellets,
            context->snapshot_pellets,
            sizeof(BgfWorldPellet) * static_cast<size_t>(pellet_count),
            cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) return cuda_result(status);
    }

    WorldCounters counters{};
    status = cudaMemcpy(
        &counters,
        context->world.counters,
        sizeof(counters),
        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return cuda_result(status);
    *bibites_written = bibite_count;
    *visible_written = visible_count;
    *pellets_written = pellet_count;
    stats->completed_steps = counters.completed_steps;
    stats->births = counters.births;
    stats->deaths = counters.deaths;
    stats->pellets_eaten = counters.pellets_eaten;
    stats->living_bibites = snapshot.bibites;
    stats->active_pellets = snapshot.plants;
    stats->total_energy = snapshot.total_energy;
    stats->average_energy = snapshot.bibites > 0
        ? snapshot.total_energy / static_cast<float>(snapshot.bibites)
        : 0.0f;
    stats->simulated_seconds = static_cast<double>(counters.completed_steps) *
        context->config.fixed_delta_time;
    stats->starvation_deaths = counters.starvation_deaths;
    stats->age_deaths = counters.age_deaths;
    stats->invalid_state_deaths = counters.invalid_state_deaths;
    stats->plant_energy = snapshot.plant_energy;
    stats->active_meat = snapshot.meats;
    stats->meat_energy = snapshot.meat_energy;
    context->latest_living=snapshot.bibites;
    context->latest_active_pellets=snapshot.pellets;
    return 0;
}

extern "C" BGF_EXPORT int bgf_world_download_snapshot(
    BgfWorldHandle world_handle,
    BgfWorldBibite* bibites,
    int32_t bibite_capacity,
    int32_t* bibites_written,
    BgfWorldPellet* pellets,
    int32_t pellet_capacity,
    int32_t* pellets_written,
    BgfWorldStats* stats)
{
    int32_t visible_written = 0;
    return bgf_world_download_snapshot_with_view(
        world_handle, bibites, bibite_capacity, bibites_written,
        pellets, pellet_capacity, pellets_written,
        nullptr, 0, &visible_written,
        0.0f, 0.0f, 0.0f, 0.0f, stats);
}

extern "C" BGF_EXPORT int bgf_world_get_bibite_detail(
    BgfWorldHandle world_handle,
    int32_t slot,
    BgfWorldBibiteDetail* detail,
    BgfWorldBrainNodeState* nodes,
    int32_t node_capacity,
    int32_t* nodes_written,
    BgfWorldBrainSynapseState* synapses,
    int32_t synapse_capacity,
    int32_t* synapses_written)
{
    if (!world_handle || !detail || !nodes_written || !synapses_written ||
        slot < 0 || node_capacity < 0 || synapse_capacity < 0 ||
        (node_capacity > 0 && !nodes) ||
        (synapse_capacity > 0 && !synapses)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    *detail = {};
    detail->slot = slot;
    *nodes_written = 0;
    *synapses_written = 0;

    GpuWorldContext* context = static_cast<GpuWorldContext*>(world_handle);
    if (slot >= context->config.max_bibites) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t status = cudaSetDevice(context->device_index);
    if (status != cudaSuccess) return cuda_result(status);
    status = refresh_pellet_overflow(*context);
    if (status != cudaSuccess) return cuda_result(status);
    pack_bibite_detail_kernel<<<1, kThreads>>>(
        context->world,
        slot,
        context->selected_detail,
        context->selected_nodes,
        context->selected_synapses);
    status = cudaGetLastError();
    if (status != cudaSuccess) return cuda_result(status);
    pack_native_sensor_values_kernel<<<1, 1>>>(
        context->world, slot, context->selected_nodes);
    status = cudaGetLastError();
    if (status != cudaSuccess) return cuda_result(status);
    status = cudaMemcpy(
        detail,
        context->selected_detail,
        sizeof(*detail),
        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return cuda_result(status);
    if (detail->alive != 1) return 0;

    const int32_t node_count = std::min(detail->brain_nodes, node_capacity);
    const int32_t synapse_count = std::min(
        std::min(detail->brain_synapses, synapse_capacity),
        kMaximumSelectedSynapses);
    if (node_count > 0) {
        status = cudaMemcpy(
            nodes,
            context->selected_nodes,
            sizeof(BgfWorldBrainNodeState) * static_cast<size_t>(node_count),
            cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) return cuda_result(status);
    }
    if (synapse_count > 0) {
        status = cudaMemcpy(
            synapses,
            context->selected_synapses,
            sizeof(BgfWorldBrainSynapseState) * static_cast<size_t>(synapse_count),
            cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) return cuda_result(status);
    }
    *nodes_written = node_count;
    *synapses_written = synapse_count;
    return 0;
}

extern "C" BGF_EXPORT int bgf_world_set_hidden_link_enabled(
    BgfWorldHandle world_handle, int32_t slot, int32_t edge, int32_t enabled)
{
    if (!world_handle || slot < 0 || edge < 0 ||
        edge >= kHiddenConnectionCount + kExtensionConnectionCount ||
        (enabled != 0 && enabled != 1))
        return static_cast<int>(cudaErrorInvalidValue);
    GpuWorldContext* context = static_cast<GpuWorldContext*>(world_handle);
    if (slot >= context->config.max_bibites)
        return static_cast<int>(cudaErrorInvalidValue);
    cudaError_t status = cudaSetDevice(context->device_index);
    if (status != cudaSuccess) return cuda_result(status);
    set_hidden_link_enabled_kernel<<<1, 1>>>(
        context->world, slot, edge, enabled, context->command_result);
    status = cudaGetLastError();
    if (status != cudaSuccess) return cuda_result(status);
    int32_t changed = 0;
    status = cudaMemcpy(&changed, context->command_result,
        sizeof(changed), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return cuda_result(status);
    return changed == 1 ? 0 : static_cast<int>(cudaErrorInvalidValue);
}

extern "C" BGF_EXPORT int bgf_world_get_hidden_link(
    BgfWorldHandle world_handle, int32_t slot, int32_t edge,
    int32_t* enabled, float* effective_weight)
{
    if (!world_handle || !enabled || !effective_weight || slot < 0 ||
        edge < 0 || edge >= kHiddenConnectionCount + kExtensionConnectionCount)
        return static_cast<int>(cudaErrorInvalidValue);
    GpuWorldContext* context = static_cast<GpuWorldContext*>(world_handle);
    if (slot >= context->config.max_bibites)
        return static_cast<int>(cudaErrorInvalidValue);
    cudaError_t status = cudaSetDevice(context->device_index);
    if (status != cudaSuccess) return cuda_result(status);
    const DeviceWorld& world = context->world;
    int32_t alive = 0, template_brain = 0;
    uint8_t native_version = 0;
    status = cudaMemcpy(&alive, world.alive + slot,
        sizeof(alive), cudaMemcpyDeviceToHost);
    if (status == cudaSuccess)
        status = cudaMemcpy(&template_brain, world.template_brain + slot,
            sizeof(template_brain), cudaMemcpyDeviceToHost);
    if (status == cudaSuccess)
        status = cudaMemcpy(&native_version, world.native_brain_version + slot,
            sizeof(native_version), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return cuda_result(status);
    const bool extension = edge >= kHiddenConnectionCount;
    if (alive != 1 || template_brain != 0 || (extension && native_version == 0))
        return static_cast<int>(cudaErrorInvalidValue);
    const int32_t local_edge = extension ? edge - kHiddenConnectionCount : edge;
    const uint32_t* masks = extension ? world.extension_masks : world.hidden_masks;
    const __half* weights = extension ? world.extension_weights : world.hidden_weights;
    uint32_t mask = 0;
    status = cudaMemcpy(&mask, masks +
        hidden_column_slot(world, slot, local_edge >> 5),
        sizeof(mask), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return cuda_result(status);
    *enabled = (mask & (1u << (local_edge & 31))) != 0u ? 1 : 0;
    *effective_weight = 0.0f;
    if (*enabled != 0) {
        __half stored{};
        status = cudaMemcpy(&stored, weights +
            dense_weight_slot(world, slot, local_edge),
            sizeof(stored), cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) return cuda_result(status);
        *effective_weight = __half2float(stored);
    }
    return 0;
}

extern "C" BGF_EXPORT int bgf_world_set_bibite_tag(
    BgfWorldHandle world_handle,
    int32_t slot,
    uint64_t tag_id)
{
    if (!world_handle || slot < 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    GpuWorldContext* context = static_cast<GpuWorldContext*>(world_handle);
    if (slot >= context->config.max_bibites) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t status = cudaSetDevice(context->device_index);
    if (status != cudaSuccess) return cuda_result(status);
    int32_t alive = 0;
    status = cudaMemcpy(
        &alive,
        context->world.alive + slot,
        sizeof(alive),
        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return cuda_result(status);
    if (alive != 1) return static_cast<int>(cudaErrorInvalidValue);
    status = cudaMemcpy(
        context->world.tag_id + slot,
        &tag_id,
        sizeof(tag_id),
        cudaMemcpyHostToDevice);
    return cuda_result(status);
}

extern "C" BGF_EXPORT int bgf_world_kill_bibite(
    BgfWorldHandle world_handle,
    int32_t slot)
{
    if (!world_handle || slot < 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    GpuWorldContext* context = static_cast<GpuWorldContext*>(world_handle);
    context->live_list_dirty = true;
    if (slot >= context->config.max_bibites) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t status = cudaSetDevice(context->device_index);
    if (status != cudaSuccess) return cuda_result(status);
    kill_bibite_kernel<<<1, 1>>>(context->world, slot, context->command_result);
    status = cudaGetLastError();
    if (status != cudaSuccess) return cuda_result(status);
    release_dead_bibite_grabs_kernel<<<128, kThreads>>>(context->world, slot);
    status = cudaGetLastError();
    if (status != cudaSuccess) return cuda_result(status);
    int32_t removed = 0;
    status = cudaMemcpy(
        &removed,
        context->command_result,
        sizeof(removed),
        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return cuda_result(status);
    return removed == 1 ? 0 : static_cast<int>(cudaErrorInvalidValue);
}

extern "C" BGF_EXPORT int bgf_world_force_reproduction(
    BgfWorldHandle world_handle,
    int32_t parent_slot,
    int32_t* child_slot)
try
{
    if (!world_handle || !child_slot || parent_slot < 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    *child_slot = -1;
    GpuWorldContext* context = static_cast<GpuWorldContext*>(world_handle);
    context->live_list_dirty = true;
    if (parent_slot >= context->config.max_bibites) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t status = cudaSetDevice(context->device_index);
    if (status != cudaSuccess) return cuda_result(status);
    const uint32_t command_seed = mix_bits(
        context->config.seed ^ 0x6c617965u ^ ++context->manual_spawn_sequence);
    status = reserve_template_births(*context,1);
    if (status != cudaSuccess) return cuda_result(status);
    force_reproduction_kernel<<<1, 1>>>(
        context->world,
        parent_slot,
        command_seed,
        context->command_result);
    status = cudaGetLastError();
    if (status != cudaSuccess) return cuda_result(status);
    status = cudaMemcpy(
        child_slot,
        context->command_result,
        sizeof(*child_slot),
        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return cuda_result(status);
    return *child_slot >= 0 ? 0 : static_cast<int>(cudaErrorInvalidValue);
}
catch (const std::bad_alloc&) { return static_cast<int>(cudaErrorMemoryAllocation); }
catch (...) { return static_cast<int>(cudaErrorUnknown); }

extern "C" BGF_EXPORT int bgf_world_prepare_d3d11_multithreading(
    void* vertex_buffer,
    uint32_t* device_flags,
    int32_t* protection_was_enabled)
{
    if (!vertex_buffer || !device_flags || !protection_was_enabled)
        return static_cast<int>(cudaErrorInvalidValue);
    *device_flags = 0;
    *protection_was_enabled = 0;
    ID3D11Device* device = nullptr;
    reinterpret_cast<ID3D11Resource*>(vertex_buffer)->GetDevice(&device);
    if (!device) return static_cast<int>(cudaErrorInvalidResourceHandle);
    *device_flags = device->GetCreationFlags();
    // A SINGLETHREADED device explicitly forbids the worker's resource calls;
    // changing the context lock cannot change that device-creation contract.
    if ((*device_flags & D3D11_CREATE_DEVICE_SINGLETHREADED) != 0u) {
        device->Release();
        return static_cast<int>(cudaErrorNotSupported);
    }
    ID3D11DeviceContext* immediate = nullptr;
    device->GetImmediateContext(&immediate);
    device->Release();
    if (!immediate) return static_cast<int>(cudaErrorInvalidResourceHandle);
    ID3D11Multithread* multithread = nullptr;
    const HRESULT result = immediate->QueryInterface(
        __uuidof(ID3D11Multithread), reinterpret_cast<void**>(&multithread));
    immediate->Release();
    if (FAILED(result) || !multithread) return static_cast<int>(cudaErrorNotSupported);
    *protection_was_enabled = multithread->GetMultithreadProtected() ? 1 : 0;
    multithread->SetMultithreadProtected(TRUE);
    const bool protected_now = multithread->GetMultithreadProtected() != FALSE;
    multithread->Release();
    // Keep it enabled for the device lifetime: restoring an earlier false
    // value during scene teardown could race a new world's registration.
    return protected_now ? 0 : static_cast<int>(cudaErrorNotSupported);
}

extern "C" BGF_EXPORT int bgf_world_register_d3d11_render_buffers(
    BgfWorldHandle world_handle,
    const BgfWorldD3D11RenderConfig* config)
{
    return bgf_world_register_d3d11_render_buffer_set(world_handle, 0, config);
}

extern "C" BGF_EXPORT int bgf_world_register_d3d11_render_buffer_set(
    BgfWorldHandle world_handle,
    int32_t buffer_index,
    const BgfWorldD3D11RenderConfig* config)
{
    if (!world_handle || buffer_index < 0 || buffer_index >= 2 ||
        !config || !config->bibite_vertex_buffer ||
        !config->pellet_vertex_buffer || config->bibite_capacity <= 0 ||
        config->pellet_capacity <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    GpuWorldContext* context = static_cast<GpuWorldContext*>(world_handle);
    if (config->bibite_capacity > context->config.max_bibites ||
        config->pellet_capacity > context->config.pellet_count) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t status = cudaSetDevice(context->device_index);
    if (status != cudaSuccess) return cuda_result(status);
    RenderBufferSet& buffers = context->render_buffers[buffer_index];
    status = unregister_render_buffer_set(buffers);
    if (status != cudaSuccess) return cuda_result(status);

    status = cudaGraphicsD3D11RegisterResource(
        &buffers.render_bibite_vertices,
        reinterpret_cast<ID3D11Resource*>(config->bibite_vertex_buffer),
        cudaGraphicsRegisterFlagsNone);
    if (status != cudaSuccess) return cuda_result(status);
    status = cudaGraphicsD3D11RegisterResource(
        &buffers.render_pellet_vertices,
        reinterpret_cast<ID3D11Resource*>(config->pellet_vertex_buffer),
        cudaGraphicsRegisterFlagsNone);
    if (status != cudaSuccess) {
        unregister_render_buffer_set(buffers);
        return cuda_result(status);
    }
    buffers.render_config = *config;
    buffers.rendered_bibites = 0;
    buffers.rendered_pellets = 0;
    buffers.render_buffers_initialized = false;
    buffers.faulted = false;
    return 0;
}

extern "C" BGF_EXPORT int bgf_world_unregister_d3d11_render_buffers(
    BgfWorldHandle world_handle)
{
    if (!world_handle) return static_cast<int>(cudaErrorInvalidValue);
    GpuWorldContext* context = static_cast<GpuWorldContext*>(world_handle);
    cudaError_t status = cudaSetDevice(context->device_index);
    if (status != cudaSuccess) return cuda_result(status);
    return cuda_result(unregister_render_buffers(*context));
}

extern "C" BGF_EXPORT int bgf_world_abandon_d3d11_render_buffers(BgfWorldHandle world_handle)
{
    if (!world_handle) return static_cast<int>(cudaErrorInvalidValue);
    GpuWorldContext* context = static_cast<GpuWorldContext*>(world_handle);
    // The caller has already quarantined the matching Unity resources. Do not
    // retry uncertain graphics operations from the simulation/disposal thread.
    context->render_buffers[0] = {};
    context->render_buffers[1] = {};
    return 0;
}

extern "C" BGF_EXPORT int bgf_world_update_d3d11_render_buffers(
    BgfWorldHandle world_handle)
{
    int32_t bibites = 0;
    int32_t pellets = 0;
    return bgf_world_update_d3d11_render_buffer_set(
        world_handle, 0, &bibites, &pellets);
}

extern "C" BGF_EXPORT int bgf_world_capture_d3d11_render_buffer_set(
    BgfWorldHandle world_handle,int buffer_index)
{
    if (!world_handle || buffer_index<0 || buffer_index>=2) return static_cast<int>(cudaErrorInvalidValue);
    auto* context=static_cast<GpuWorldContext*>(world_handle);
    cudaError_t status=cudaSetDevice(context->device_index);
    if (status==cudaSuccess && context->live_list_dirty) status=rebuild_runtime(*context);
    if (status==cudaSuccess) status=capture_render_buffers(*context,context->render_buffers[buffer_index]);
    return cuda_result(status);
}

extern "C" BGF_EXPORT int bgf_world_update_d3d11_render_buffer_set(
    BgfWorldHandle world_handle,
    int32_t buffer_index,
    int32_t* bibite_count,
    int32_t* pellet_count)
{
    if (bibite_count) *bibite_count = 0;
    if (pellet_count) *pellet_count = 0;
    if (!world_handle || buffer_index < 0 || buffer_index >= 2 ||
        !bibite_count || !pellet_count) return static_cast<int>(cudaErrorInvalidValue);
    GpuWorldContext* context = static_cast<GpuWorldContext*>(world_handle);
    cudaError_t status = cudaSetDevice(context->device_index);
    if (status != cudaSuccess) return cuda_result(status);
    RenderBufferSet& buffers = context->render_buffers[buffer_index];
    status = update_render_buffers(*context, buffers);
    if (status == cudaSuccess) {
        *bibite_count = buffers.rendered_bibites;
        *pellet_count = buffers.rendered_pellets;
    }
    return cuda_result(status);
}

extern "C" BGF_EXPORT int bgf_world_get_device_name(
    BgfWorldHandle world_handle,
    char* buffer,
    int32_t buffer_length)
{
    if (!world_handle || !buffer || buffer_length <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    GpuWorldContext* context = static_cast<GpuWorldContext*>(world_handle);
    std::strncpy(buffer, context->properties.name, static_cast<size_t>(buffer_length - 1));
    buffer[buffer_length - 1] = '\0';
    return 0;
}

extern "C" BGF_EXPORT int bgf_world_save_checkpoint(
    BgfWorldHandle world_handle,
    const char* path)
try {
    if (!world_handle || !path || path[0] == '\0') {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    GpuWorldContext* context = static_cast<GpuWorldContext*>(world_handle);
    cudaError_t status = cudaSetDevice(context->device_index);
    if (status != cudaSuccess) return cuda_result(status);
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) return cuda_result(status);

    status = prepare_checkpoint_rows(*context);
    if (status != cudaSuccess) return cuda_result(status);
    std::unique_ptr<CheckpointTransaction> transaction;
    if (!active_checkpoint_capture) transaction.reset(new CheckpointTransaction(path));
    std::FILE* file = transaction ? transaction->file() : nullptr;
    if (!file && !active_checkpoint_capture) {
        return static_cast<int>(cudaErrorUnknown);
    }
    const auto abort_save = [&](cudaError_t error) {
        return cuda_result(error);
    };

    CheckpointHeader header{};
    std::memcpy(header.magic, kCheckpointMagic, sizeof(kCheckpointMagic));
    header.format_version = checkpoint_write_format;
    header.header_bytes = sizeof(CheckpointHeader);
    header.config = context->config;
    header.topology_count = static_cast<uint32_t>(context->template_topologies.size());
    header.has_template_instance_storage =
        context->world.template_node_biases ? 1u : 0u;
    header.manual_spawn_sequence = context->manual_spawn_sequence;
    if (!write_host_values(file, &header, 1)) {
        return abort_save(cudaErrorUnknown);
    }

    for (const TemplateTopologyHost& topology : context->template_topologies) {
        const uint32_t counts[3] = {
            static_cast<uint32_t>(topology.node_descriptors.size()),
            static_cast<uint32_t>(topology.active_nodes.size()),
            static_cast<uint32_t>(topology.synapse_edges.size())};
        if (!write_host_values(file, counts, 3) ||
            !write_host_values(
                file,
                topology.node_descriptors.data(),
                topology.node_descriptors.size()) ||
            !write_host_values(
                file,
                topology.active_nodes.data(),
                topology.active_nodes.size()) ||
            !write_host_values(
                file,
                topology.synapse_edges.data(),
                topology.synapse_edges.size())) {
            return abort_save(cudaErrorUnknown);
        }
    }

    if (checkpoint_write_format >= 12) {
    const int32_t metadata[4] = { context->world.contact_grid_width,
        context->world.sense_grid_width,static_cast<int>(context->checkpoint_live.size()),
        static_cast<int>(context->checkpoint_imported.size()) };
    if (!write_host_values(file,metadata,4) ||
        !write_host_values(file,context->checkpoint_live.data(),context->checkpoint_live.size()) ||
        !write_host_values(file,context->checkpoint_imported.data(),context->checkpoint_imported.size()))
        return abort_save(cudaErrorUnknown);

    } else if (context->world.contact_grid_width!=context->allocated_contact_width ||
        context->world.sense_grid_width!=context->allocated_sense_width)
        return abort_save(cudaErrorNotSupported);

    DeviceWorld& world = context->world;
    const size_t bibites = static_cast<size_t>(world.max_bibites);
    const size_t pellets = static_cast<size_t>(world.pellet_count);
    const size_t sense_cells = static_cast<size_t>(world.sense_grid_width) *
        world.sense_grid_width;
    const size_t sense_bitmap_words = (sense_cells + 31u) / 32u;
    const size_t pheromone_cells = static_cast<size_t>(world.pheromone_grid_width) *
        world.pheromone_grid_height;

#define BGF_CHECKPOINT_WRITE(field, count) do { \
    status = write_checkpoint_values(*context, file, world.field, (count), #field); \
    if (status != cudaSuccess) return abort_save(status); \
} while (0)
    BGF_CHECKPOINT_WRITE(counters, 1);
    BGF_CHECKPOINT_WRITE(alive, bibites);
    BGF_CHECKPOINT_WRITE(generation, bibites);
    BGF_CHECKPOINT_WRITE(positions, bibites);
    BGF_CHECKPOINT_WRITE(velocities, bibites);
    BGF_CHECKPOINT_WRITE(headings, bibites);
    BGF_CHECKPOINT_WRITE(energy, bibites);
    BGF_CHECKPOINT_WRITE(age, bibites);
    BGF_CHECKPOINT_WRITE(size, bibites);
    BGF_CHECKPOINT_WRITE(reproduction_cooldown, bibites);
    BGF_CHECKPOINT_WRITE(traits, bibites);
    BGF_CHECKPOINT_WRITE(colors, bibites);
    BGF_CHECKPOINT_WRITE(gene_mutation_strength, bibites);
    BGF_CHECKPOINT_WRITE(brain_mutation_strength, bibites);
    BGF_CHECKPOINT_WRITE(lineage_id, bibites);
    BGF_CHECKPOINT_WRITE(tag_id, bibites);
    BGF_CHECKPOINT_WRITE(rng, bibites);
    BGF_CHECKPOINT_WRITE(weights, bibites * kWeightCount);
    BGF_CHECKPOINT_WRITE(template_brain, bibites);
    BGF_CHECKPOINT_WRITE(actions_a, bibites);
    BGF_CHECKPOINT_WRITE(actions_b, bibites);
    BGF_CHECKPOINT_WRITE(repulsion, bibites);
    BGF_CHECKPOINT_WRITE(food_gain, bibites);
    BGF_CHECKPOINT_WRITE(wants_reproduction, bibites);
    BGF_CHECKPOINT_WRITE(reproduction_items, bibites);
    BGF_CHECKPOINT_WRITE(reproduction_count, 1);
    BGF_CHECKPOINT_WRITE(free_bibite_items, bibites);
    BGF_CHECKPOINT_WRITE(free_bibite_count, 1);
    BGF_CHECKPOINT_WRITE(cached_food, bibites);
    BGF_CHECKPOINT_WRITE(cached_neighbour, bibites);
    BGF_CHECKPOINT_WRITE(cached_neighbour_count, bibites);
    BGF_CHECKPOINT_WRITE(cached_food_direction, bibites);
    BGF_CHECKPOINT_WRITE(cached_neighbour_direction, bibites);
    BGF_CHECKPOINT_WRITE(cached_food_distance_squared, bibites);
    BGF_CHECKPOINT_WRITE(cached_neighbour_distance_squared, bibites);
    BGF_CHECKPOINT_WRITE(cached_neighbour_color, bibites);
    BGF_CHECKPOINT_WRITE(pellet_active, pellets);
    BGF_CHECKPOINT_WRITE(pellet_positions, pellets);
    BGF_CHECKPOINT_WRITE(pellet_rng, pellets);
    BGF_CHECKPOINT_WRITE(pellet_cell_masks, sense_cells);
    BGF_CHECKPOINT_WRITE(
        pellet_cell_items,
        sense_cells * static_cast<size_t>(kPelletBucketCapacity));
    BGF_CHECKPOINT_WRITE(pellet_hash_cell, pellets);
    BGF_CHECKPOINT_WRITE(pellet_hash_slot, pellets);
    BGF_CHECKPOINT_WRITE(pellet_occupied, sense_bitmap_words);
    BGF_CHECKPOINT_WRITE(pellet_overflow_count, 1);
    BGF_CHECKPOINT_WRITE(eaten_pellet_items, pellets);
    BGF_CHECKPOINT_WRITE(eaten_pellet_count, 1);

    std::vector<int32_t> respawn_counts(kPelletRespawnWheelSize);
    status = cudaMemcpy(
        respawn_counts.data(),
        world.pellet_respawn_counts,
        sizeof(int32_t) * respawn_counts.size(),
        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return abort_save(status);
    if (!write_host_values(file, respawn_counts.data(), respawn_counts.size())) {
        return abort_save(cudaErrorUnknown);
    }
    for (int32_t bucket = 0; bucket < kPelletRespawnWheelSize; ++bucket) {
        const int32_t count = std::max(
            0,
            std::min(world.pellet_count, respawn_counts[bucket]));
        status = write_device_values(
            file,
            world.pellet_respawn_items + static_cast<size_t>(bucket) * pellets,
            static_cast<size_t>(count));
        if (status != cudaSuccess) return abort_save(status);
    }

    BGF_CHECKPOINT_WRITE(pheromone_a, pheromone_cells);
    BGF_CHECKPOINT_WRITE(pheromone_b, pheromone_cells);
    if (header.has_template_instance_storage != 0u) {
        BGF_CHECKPOINT_WRITE(
            template_node_biases,
            bibites * static_cast<size_t>(kMaxTemplateNodes));
        BGF_CHECKPOINT_WRITE(
            template_node_accum,
            bibites * static_cast<size_t>(kMaxTemplateNodes));
        BGF_CHECKPOINT_WRITE(
            template_node_last_input,
            bibites * static_cast<size_t>(kMaxTemplateNodes));
        BGF_CHECKPOINT_WRITE(
            template_node_last_output,
            bibites * static_cast<size_t>(kMaxTemplateNodes));
        BGF_CHECKPOINT_WRITE(
            template_synapse_weights,
            bibites * static_cast<size_t>(kMaxTemplateSynapses));
    }
#undef BGF_CHECKPOINT_WRITE

    const FoodCheckpointState food_state{
        world.pellet_limit,
        world.food_growth_factor,
        world.pellet_energy,
        world.ecology_scale,
        world.mobility_scale,
        world.sense_radius,
        world.sense_cell_radius};
    if (!write_host_values(file, &food_state, 1)) {
        return static_cast<int>(cudaErrorUnknown);
    }
    status = write_checkpoint_values(*context, file, world.pellet_food_units, pellets, "pellet_food_units");
    if (status != cudaSuccess) return abort_save(status);
    status = write_checkpoint_values(*context, file, world.hidden_masks, bibites * kHiddenMaskWords, "hidden_masks");
    if (status != cudaSuccess) return abort_save(status);
    status = write_checkpoint_values(*context, file, world.hidden_weights, bibites * kHiddenConnectionCount, "hidden_weights");
    if (status != cudaSuccess) return abort_save(status);
    status = write_checkpoint_values(*context, file, world.hidden_biases, bibites * kHiddenNodeCount, "hidden_biases");
    if (status != cudaSuccess) return abort_save(status);
    status = write_checkpoint_values(*context, file, world.hidden_last_input, bibites * kHiddenNodeCount, "hidden_last_input");
    if (status != cudaSuccess) return abort_save(status);
    status = write_checkpoint_values(*context, file, world.hidden_last_output, bibites * kHiddenNodeCount, "hidden_last_output");
    if (status != cudaSuccess) return abort_save(status);
    status = write_checkpoint_values(*context, file, world.hidden_synapse_count, bibites, "hidden_synapse_count");
    if (status != cudaSuccess) return abort_save(status);
    status = write_checkpoint_values(*context, file, world.pellet_nominal_units, pellets, "pellet_nominal_units");
    if (status != cudaSuccess) return abort_save(status);
    const RuntimeCheckpointState runtime_state{world.linear_drag};
    if (!write_host_values(file, &runtime_state, 1)) return abort_save(cudaErrorUnknown);
    status = write_checkpoint_values(*context, file, world.diet, bibites, "diet");
    if (status != cudaSuccess) return abort_save(status);
    status = write_checkpoint_values(*context, file, world.health, bibites, "health");
    if (status != cudaSuccess) return abort_save(status);
    status = write_checkpoint_values(*context, file, world.bite_cooldown, bibites, "bite_cooldown");
    if (status != cudaSuccess) return abort_save(status);
    status = write_checkpoint_values(*context, file, world.meat_gain, bibites, "meat_gain");
    if (status != cudaSuccess) return abort_save(status);
    status = write_checkpoint_values(*context, file, world.actions_c, bibites, "actions_c");
    if (status != cudaSuccess) return abort_save(status);
    status = write_checkpoint_values(*context, file, world.actions_d, bibites, "actions_d");
    if (status != cudaSuccess) return abort_save(status);
    const CombatCheckpointState combat_state{world.collision_damage_constant,
        world.collision_damage_threshold, world.biting_damage_factor,
        world.biting_pressure};
    if (!write_host_values(file, &combat_state, 1))
        return abort_save(cudaErrorUnknown);
    status = write_checkpoint_values(*context, file, world.life_state, bibites, "life_state");
    if (status != cudaSuccess) return abort_save(status);
    status = write_checkpoint_values(*context, file, world.reproductive_traits, bibites, "reproductive_traits");
    if (status != cudaSuccess) return abort_save(status);
    status = write_checkpoint_values(*context, file, world.last_damage, bibites, "last_damage");
    if (status != cudaSuccess) return abort_save(status);
    status = write_checkpoint_values(*context, file, world.held_target, bibites, "held_target");
    if (status != cudaSuccess) return abort_save(status);
    status = write_checkpoint_values(*context, file, world.held_instance, bibites, "held_instance");
    if (status != cudaSuccess) return abort_save(status);
    status = write_checkpoint_values(*context, file, world.clock_reset_output, bibites, "clock_reset_output");
    if (status != cudaSuccess) return abort_save(status);
    status = write_checkpoint_values(*context, file, world.pellet_held_by, pellets, "pellet_held_by");
    if (status != cudaSuccess) return abort_save(status);
    status = write_checkpoint_values(*context, file, world.pheromone_heading_x_a, pheromone_cells, "pheromone_heading_x_a");
    if (status != cudaSuccess) return abort_save(status);
    status = write_checkpoint_values(*context, file, world.pheromone_heading_x_b, pheromone_cells, "pheromone_heading_x_b");
    if (status != cudaSuccess) return abort_save(status);
    status = write_checkpoint_values(*context, file, world.pheromone_heading_y_a, pheromone_cells, "pheromone_heading_y_a");
    if (status != cudaSuccess) return abort_save(status);
    status = write_checkpoint_values(*context, file, world.pheromone_heading_y_b, pheromone_cells, "pheromone_heading_y_b");
    if (status != cudaSuccess) return abort_save(status);
    status = write_checkpoint_values(*context, file, world.native_brain_version, bibites, "native_brain_version");
    if (status != cudaSuccess) return abort_save(status);
    status = write_checkpoint_values(*context, file, world.extension_masks, bibites * kExtensionMaskWords, "extension_masks");
    if (status != cudaSuccess) return abort_save(status);
    status = write_checkpoint_values(*context, file, world.extension_weights, bibites * kExtensionConnectionCount, "extension_weights");
    if (status != cudaSuccess) return abort_save(status);
    status = write_checkpoint_values(*context, file, world.extension_synapse_count, bibites, "extension_synapse_count");
    if (status != cudaSuccess) return abort_save(status);
    if (!write_host_values(file, &world.digestion, 1))
        return abort_save(cudaErrorUnknown);
    if (checkpoint_write_format>=12) {
        status = write_checkpoint_values(*context,file,world.instance_id,bibites,"instance_id");
        if (status != cudaSuccess) return abort_save(status);
    }
    if (!write_host_values(file, &kCheckpointFooter, 1) ||
        (transaction && !transaction->commit())) {
        return static_cast<int>(cudaErrorUnknown);
    }
    return 0;
}
catch (const std::bad_alloc&) {
    return static_cast<int>(cudaErrorMemoryAllocation);
}
catch (...) {
    return static_cast<int>(cudaErrorUnknown);
}

extern "C" BGF_EXPORT int bgf_world_save_checkpoint_legacy_v11(BgfWorldHandle world,const char* path)
{
    if (!world || !path || !path[0]) return static_cast<int>(cudaErrorInvalidValue);
    auto& context=*static_cast<GpuWorldContext*>(world);
    cudaError_t status=cudaSetDevice(context.device_index);
    if (status!=cudaSuccess) return cuda_result(status);
    const int contact=context.world.contact_grid_width,sense=context.world.sense_grid_width;
    status=reconfigure_runtime_grids(context,context.allocated_contact_width,context.allocated_sense_width);
    if (status!=cudaSuccess) return cuda_result(status);
    struct Guard { uint32_t previous; ~Guard(){checkpoint_write_format=previous;} } guard{checkpoint_write_format};
    checkpoint_write_format=11;
    const int saved=bgf_world_save_checkpoint(world,path);
    status=reconfigure_runtime_grids(context,contact,sense);
    return saved ? saved : cuda_result(status);
}

extern "C" BGF_EXPORT int bgf_world_capture_checkpoint(
    BgfWorldHandle world, BgfCheckpointCaptureHandle* handle)
try {
    if (!world || !handle || active_checkpoint_capture) return static_cast<int>(cudaErrorInvalidValue);
    *handle=nullptr;
    std::unique_ptr<CheckpointCapture> capture(new CheckpointCapture);
    // Reset the thread-local sink on every exit, including allocation failure.
    struct Guard { ~Guard(){ active_checkpoint_capture=nullptr; } } guard;
    active_checkpoint_capture=capture.get();
    const int status=bgf_world_save_checkpoint(world,"capture-only");
    if (status != 0) return status;
    *handle=capture.release();
    return 0;
} catch (const std::bad_alloc&) { return static_cast<int>(cudaErrorMemoryAllocation); }
catch (...) { return static_cast<int>(cudaErrorUnknown); }

extern "C" BGF_EXPORT int bgf_world_write_captured_checkpoint(
    BgfCheckpointCaptureHandle handle, const char* path)
try {
    if (!handle || !path || !path[0] || active_checkpoint_capture)
        return static_cast<int>(cudaErrorInvalidValue);
    const auto& bytes=static_cast<CheckpointCapture*>(handle)->bytes;
    CheckpointTransaction transaction(path);
    if (!transaction.file()) return static_cast<int>(cudaErrorUnknown);
    for (size_t offset=0;offset<bytes.size();offset+=kCheckpointTransferBytes) {
        const size_t count=std::min(kCheckpointTransferBytes,bytes.size()-offset);
        if (!write_host_bytes(transaction.file(),bytes.data()+offset,count))
            return static_cast<int>(cudaErrorUnknown);
    }
    return transaction.commit() ? 0 : static_cast<int>(cudaErrorUnknown);
} catch (const std::bad_alloc&) { return static_cast<int>(cudaErrorMemoryAllocation); }
catch (...) { return static_cast<int>(cudaErrorUnknown); }

extern "C" BGF_EXPORT int bgf_world_release_checkpoint_capture(BgfCheckpointCaptureHandle handle)
{
    delete static_cast<CheckpointCapture*>(handle);
    return 0;
}

extern "C" BGF_EXPORT int bgf_world_checkpoint_config(
    const char* path,
    int32_t device_index,
    BgfWorldConfig* config)
{
    if (!path || path[0] == '\0' || device_index < 0 || !config) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    std::FILE* file = nullptr;
    if (fopen_s(&file, path, "rb") != 0 || !file) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    CheckpointHeader header{};
    const bool read = read_host_values(file, &header, 1);
    uint64_t footer = 0;
    const bool complete = _fseeki64(file, 0, SEEK_END) == 0 &&
        _ftelli64(file) >= static_cast<__int64>(sizeof(CheckpointHeader) +
            sizeof(WorldCounters) + sizeof(kCheckpointFooter)) &&
        _fseeki64(file, -static_cast<__int64>(sizeof(footer)), SEEK_END) == 0 &&
        read_host_values(file, &footer, 1) && footer == kCheckpointFooter;
    std::fclose(file);
    if (!read || !complete || !valid_checkpoint_header(header)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    header.config.device_index = device_index;
    *config = header.config;
    return 0;
}

extern "C" BGF_EXPORT int bgf_world_load_checkpoint(
    const char* path,
    int32_t device_index,
    BgfWorldHandle* world_handle)
try {
    if (!path || path[0] == '\0' || !world_handle) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    *world_handle = nullptr;
    BgfWorldConfig config{};
    int result = bgf_world_checkpoint_config(path, device_index, &config);
    if (result != 0) return result;
    result = bgf_world_create(&config, world_handle);
    if (result != 0) return result;

    GpuWorldContext* context = static_cast<GpuWorldContext*>(*world_handle);
    const auto cleanup_world = [&](GpuWorldContext* pending) {
        if (pending) {
            bgf_world_destroy(pending);
            *world_handle = nullptr;
        }
    };
    std::unique_ptr<GpuWorldContext, decltype(cleanup_world)> pending_world(
        context, cleanup_world);
    std::FILE* file = nullptr;
    if (fopen_s(&file, path, "rb") != 0 || !file) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    std::unique_ptr<std::FILE, decltype(&std::fclose)> source_file(file, &std::fclose);
    const auto abort_load = [&](cudaError_t error) {
        return cuda_result(error);
    };

    CheckpointHeader header{};
    if (!read_host_values(file, &header, 1) || !valid_checkpoint_header(header)) {
            return abort_load(cudaErrorInvalidValue);
    }
    BgfWorldConfig loaded_config = header.config;
    loaded_config.device_index = device_index;
    if (std::memcmp(&loaded_config, &config, sizeof(config)) != 0) {
        return abort_load(cudaErrorInvalidValue);
    }
    context->manual_spawn_sequence = header.manual_spawn_sequence;
    context->template_topologies.clear();
    for (uint32_t index = 0; index < header.topology_count; ++index) {
        uint32_t counts[3]{};
        if (!read_host_values(file, counts, 3) ||
            counts[0] == 0u || counts[0] > static_cast<uint32_t>(kMaxTemplateNodes) ||
            counts[1] > counts[0] ||
            counts[2] > static_cast<uint32_t>(kMaxTemplateSynapses)) {
            return abort_load(cudaErrorInvalidValue);
        }
        TemplateTopologyHost topology;
        topology.node_descriptors.resize(counts[0]);
        topology.active_nodes.resize(counts[1]);
        topology.synapse_edges.resize(counts[2]);
        if (!read_host_values(
                file,
                topology.node_descriptors.data(),
                topology.node_descriptors.size()) ||
            !read_host_values(
                file,
                topology.active_nodes.data(),
                topology.active_nodes.size()) ||
            !read_host_values(
                file,
                topology.synapse_edges.data(),
                topology.synapse_edges.size()) || !valid_checkpoint_topology(topology)) {
        return abort_load(cudaErrorInvalidValue);
        }
        int32_t registered = -1;
        cudaError_t status = find_or_register_template_topology(
            *context,
            topology,
            &registered);
        if (status != cudaSuccess || registered != static_cast<int32_t>(index)) {
            return abort_load(status == cudaSuccess ? cudaErrorInvalidValue : status);
        }
    }
    cudaError_t staging_status = ensure_checkpoint_staging(*context);
    if (staging_status != cudaSuccess) return abort_load(staging_status);
    if (header.format_version >= 12u) {
        int metadata[4]{};
        if (!read_host_values(file,metadata,4) || metadata[2]<0 ||
            metadata[2]>context->world.max_bibites || metadata[3]<0 || metadata[3]>metadata[2] ||
            metadata[0]<32 || metadata[0]>context->allocated_contact_width ||
            metadata[1]<32 || metadata[1]>context->allocated_sense_width ||
            (metadata[0]&(metadata[0]-1)) || (metadata[1]&(metadata[1]-1)))
            return abort_load(cudaErrorInvalidValue);
        context->world.contact_grid_width=metadata[0];context->world.sense_grid_width=metadata[1];
        context->checkpoint_live.resize(metadata[2]);context->checkpoint_imported.resize(metadata[3]);
        if (!read_host_values(file,context->checkpoint_live.data(),context->checkpoint_live.size()) ||
            !read_host_values(file,context->checkpoint_imported.data(),context->checkpoint_imported.size()))
            return abort_load(cudaErrorInvalidValue);
        int previous=-1;
        for (int slot : context->checkpoint_live) {
            if (slot<=previous || slot>=context->world.max_bibites) return abort_load(cudaErrorInvalidValue);
            previous=slot;
        }
        previous=-1;
        for (int slot : context->checkpoint_imported) {
            if (slot<=previous || !std::binary_search(context->checkpoint_live.begin(),
                context->checkpoint_live.end(),slot)) return abort_load(cudaErrorInvalidValue);
            previous=slot;
        }
        staging_status=cudaMemcpy(context->checkpoint_slots,context->checkpoint_live.data(),
            sizeof(int)*context->checkpoint_live.size(),cudaMemcpyHostToDevice);
        if (staging_status==cudaSuccess) staging_status=cudaMemset(context->world.alive,0,
            sizeof(int)*context->world.max_bibites);
        if (staging_status==cudaSuccess && header.has_template_instance_storage)
            staging_status=restore_checkpoint_pool(*context);
        if (staging_status!=cudaSuccess) return abort_load(staging_status);
    } else {
        // Versions 1-11 wrote the capacity-sized grids.
        context->world.contact_grid_width=context->allocated_contact_width;
        context->world.sense_grid_width=context->allocated_sense_width;
    }

    DeviceWorld& world = context->world;
    const size_t bibites = static_cast<size_t>(world.max_bibites);
    const size_t pellets = static_cast<size_t>(world.pellet_count);
    const size_t sense_cells = static_cast<size_t>(world.sense_grid_width) *
        world.sense_grid_width;
    const size_t sense_bitmap_words = (sense_cells + 31u) / 32u;
    const size_t pheromone_cells = static_cast<size_t>(world.pheromone_grid_width) *
        world.pheromone_grid_height;
    cudaError_t status = cudaSuccess;

#define BGF_CHECKPOINT_READ(field, count) do { \
    status = read_checkpoint_values(*context, file, world.field, (count), #field, header.format_version); \
    if (status != cudaSuccess) return abort_load(status); \
} while (0)
    BGF_CHECKPOINT_READ(counters, 1);
    BGF_CHECKPOINT_READ(alive, bibites);
    BGF_CHECKPOINT_READ(generation, bibites);
    BGF_CHECKPOINT_READ(positions, bibites);
    BGF_CHECKPOINT_READ(velocities, bibites);
    BGF_CHECKPOINT_READ(headings, bibites);
    BGF_CHECKPOINT_READ(energy, bibites);
    BGF_CHECKPOINT_READ(age, bibites);
    BGF_CHECKPOINT_READ(size, bibites);
    BGF_CHECKPOINT_READ(reproduction_cooldown, bibites);
    BGF_CHECKPOINT_READ(traits, bibites);
    BGF_CHECKPOINT_READ(colors, bibites);
    BGF_CHECKPOINT_READ(gene_mutation_strength, bibites);
    BGF_CHECKPOINT_READ(brain_mutation_strength, bibites);
    BGF_CHECKPOINT_READ(lineage_id, bibites);
    BGF_CHECKPOINT_READ(tag_id, bibites);
    BGF_CHECKPOINT_READ(rng, bibites);
    BGF_CHECKPOINT_READ(weights, bibites * kWeightCount);
    BGF_CHECKPOINT_READ(template_brain, bibites);
    BGF_CHECKPOINT_READ(actions_a, bibites);
    BGF_CHECKPOINT_READ(actions_b, bibites);
    BGF_CHECKPOINT_READ(repulsion, bibites);
    BGF_CHECKPOINT_READ(food_gain, bibites);
    BGF_CHECKPOINT_READ(wants_reproduction, bibites);
    BGF_CHECKPOINT_READ(reproduction_items, bibites);
    BGF_CHECKPOINT_READ(reproduction_count, 1);
    BGF_CHECKPOINT_READ(free_bibite_items, bibites);
    BGF_CHECKPOINT_READ(free_bibite_count, 1);
    BGF_CHECKPOINT_READ(cached_food, bibites);
    BGF_CHECKPOINT_READ(cached_neighbour, bibites);
    BGF_CHECKPOINT_READ(cached_neighbour_count, bibites);
    BGF_CHECKPOINT_READ(cached_food_direction, bibites);
    BGF_CHECKPOINT_READ(cached_neighbour_direction, bibites);
    BGF_CHECKPOINT_READ(cached_food_distance_squared, bibites);
    BGF_CHECKPOINT_READ(cached_neighbour_distance_squared, bibites);
    BGF_CHECKPOINT_READ(cached_neighbour_color, bibites);
    BGF_CHECKPOINT_READ(pellet_active, pellets);
    BGF_CHECKPOINT_READ(pellet_positions, pellets);
    BGF_CHECKPOINT_READ(pellet_rng, pellets);
    BGF_CHECKPOINT_READ(pellet_cell_masks, sense_cells);
    BGF_CHECKPOINT_READ(
        pellet_cell_items,
        sense_cells * static_cast<size_t>(kPelletBucketCapacity));
    BGF_CHECKPOINT_READ(pellet_hash_cell, pellets);
    BGF_CHECKPOINT_READ(pellet_hash_slot, pellets);
    BGF_CHECKPOINT_READ(pellet_occupied, sense_bitmap_words);
    BGF_CHECKPOINT_READ(pellet_overflow_count, 1);
    BGF_CHECKPOINT_READ(eaten_pellet_items, pellets);
    BGF_CHECKPOINT_READ(eaten_pellet_count, 1);

    std::vector<int32_t> respawn_counts(kPelletRespawnWheelSize);
    if (!read_host_values(file, respawn_counts.data(), respawn_counts.size())) {
        return abort_load(cudaErrorInvalidValue);
    }
    status = cudaMemset(
        world.pellet_respawn_items,
        0xff,
        sizeof(int32_t) * pellets * static_cast<size_t>(kPelletRespawnWheelSize));
    if (status != cudaSuccess) return abort_load(status);
    std::vector<uint8_t> respawn_seen(pellets, 0u);
    std::vector<int32_t> respawn_items;
    for (int32_t bucket = 0; bucket < kPelletRespawnWheelSize; ++bucket) {
        if (respawn_counts[bucket] < 0 || respawn_counts[bucket] > world.pellet_count) {
            return abort_load(cudaErrorInvalidValue);
        }
        respawn_items.resize(static_cast<size_t>(respawn_counts[bucket]));
        if (!read_host_values(file, respawn_items.data(), respawn_items.size())) {
            return abort_load(cudaErrorInvalidValue);
        }
        for (int32_t pellet : respawn_items) {
            if (pellet < 0 || pellet >= world.pellet_count || respawn_seen[pellet] != 0u) {
                return abort_load(cudaErrorInvalidValue);
            }
            respawn_seen[pellet] = 1u;
        }
        status = respawn_items.empty() ? cudaSuccess : cudaMemcpy(
            world.pellet_respawn_items + static_cast<size_t>(bucket) * pellets,
            respawn_items.data(), sizeof(int32_t) * respawn_items.size(),
            cudaMemcpyHostToDevice);
        if (status != cudaSuccess) return abort_load(status);
    }
    status = cudaMemcpy(
        world.pellet_respawn_counts,
        respawn_counts.data(),
        sizeof(int32_t) * respawn_counts.size(),
        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) return abort_load(status);

    BGF_CHECKPOINT_READ(pheromone_a, pheromone_cells);
    BGF_CHECKPOINT_READ(pheromone_b, pheromone_cells);
    if (header.has_template_instance_storage != 0u) {
        if (header.format_version < 12u) {
            status=prepare_checkpoint_rows(*context);
            if (status==cudaSuccess) status=restore_checkpoint_pool(*context);
            if (status!=cudaSuccess) return abort_load(status);
        }
        BGF_CHECKPOINT_READ(
            template_node_biases,
            bibites * static_cast<size_t>(kMaxTemplateNodes));
        BGF_CHECKPOINT_READ(
            template_node_accum,
            bibites * static_cast<size_t>(kMaxTemplateNodes));
        BGF_CHECKPOINT_READ(
            template_node_last_input,
            bibites * static_cast<size_t>(kMaxTemplateNodes));
        BGF_CHECKPOINT_READ(
            template_node_last_output,
            bibites * static_cast<size_t>(kMaxTemplateNodes));
        BGF_CHECKPOINT_READ(
            template_synapse_weights,
            bibites * static_cast<size_t>(kMaxTemplateSynapses));
    }
#undef BGF_CHECKPOINT_READ

    if (header.format_version >= 2u) {
        FoodCheckpointState food_state{};
        if (!read_host_values(file, &food_state, 1) ||
            food_state.target < 0 || food_state.target > world.pellet_count ||
            !std::isfinite(food_state.growth_factor) ||
            food_state.growth_factor < 0.0f || food_state.growth_factor > 50.0f ||
            !std::isfinite(food_state.pellet_energy) ||
            food_state.pellet_energy <= 0.0f ||
            !std::isfinite(food_state.ecology_scale) ||
            food_state.ecology_scale < 1.0f || food_state.ecology_scale > kMaximumEcologyScale ||
            !std::isfinite(food_state.mobility_scale) ||
            food_state.mobility_scale < 1.0f || food_state.mobility_scale > 8.0f ||
            !std::isfinite(food_state.sense_radius) ||
            food_state.sense_radius <= 0.0f ||
            food_state.sense_cell_radius < 1 ||
            food_state.sense_cell_radius > world.sense_grid_width / 2) {
            return abort_load(cudaErrorInvalidValue);
        }
        world.pellet_limit = food_state.target;
        world.food_growth_factor = food_state.growth_factor;
        world.pellet_energy = food_state.pellet_energy;
        world.ecology_scale = food_state.ecology_scale;
        world.mobility_scale = food_state.mobility_scale;
        world.sense_radius = food_state.sense_radius;
        world.sense_cell_radius = food_state.sense_cell_radius;
        context->config.pellet_energy = food_state.pellet_energy;
    }
    if (header.format_version >= 3u) {
        status = read_checkpoint_values(*context, file, world.pellet_food_units, pellets, "pellet_food_units", header.format_version);
        if (status != cudaSuccess) return abort_load(status);
    } else {
        const int32_t items = std::max(world.max_bibites, world.pellet_count);
        upgrade_legacy_food_kernel<<<(items + kThreads - 1) / kThreads,
            kThreads>>>(world);
        status = cudaGetLastError();
        if (status != cudaSuccess) return abort_load(status);
    }
    if (header.format_version >= 4u) {
        status = read_checkpoint_values(*context, file, world.hidden_masks, bibites * kHiddenMaskWords, "hidden_masks", header.format_version);
        if (status != cudaSuccess) return abort_load(status);
        status = read_checkpoint_values(*context, file, world.hidden_weights, bibites * kHiddenConnectionCount, "hidden_weights", header.format_version);
        if (status != cudaSuccess) return abort_load(status);
        status = read_checkpoint_values(*context, file, world.hidden_biases, bibites * kHiddenNodeCount, "hidden_biases", header.format_version);
        if (status != cudaSuccess) return abort_load(status);
        status = read_checkpoint_values(*context, file, world.hidden_last_input, bibites * kHiddenNodeCount, "hidden_last_input", header.format_version);
        if (status != cudaSuccess) return abort_load(status);
        status = read_checkpoint_values(*context, file, world.hidden_last_output, bibites * kHiddenNodeCount, "hidden_last_output", header.format_version);
        if (status != cudaSuccess) return abort_load(status);
        status = read_checkpoint_values(*context, file, world.hidden_synapse_count, bibites, "hidden_synapse_count", header.format_version);
        if (status != cudaSuccess) return abort_load(status);
        if (header.format_version == 4u) {
            upgrade_v4_dormant_weights_kernel<<<
                (world.max_bibites + kThreads - 1) / kThreads,
                kThreads>>>(world);
            status = cudaGetLastError();
            if (status != cudaSuccess) return abort_load(status);
        }
    } else {
        upgrade_legacy_brains_kernel<<<
            (world.max_bibites + kThreads - 1) / kThreads, kThreads>>>(world);
        status = cudaGetLastError();
        if (status != cudaSuccess) return abort_load(status);
    }
    if (header.format_version >= 6u) {
        status = read_checkpoint_values(*context, file, world.pellet_nominal_units, pellets, "pellet_nominal_units", header.format_version);
        if (status != cudaSuccess) return abort_load(status);
        RuntimeCheckpointState runtime_state{};
        if (!read_host_values(file, &runtime_state, 1) ||
            !std::isfinite(runtime_state.linear_drag) ||
            runtime_state.linear_drag < 0.0f || runtime_state.linear_drag > 4.0f)
            return abort_load(cudaErrorInvalidValue);
        world.linear_drag = runtime_state.linear_drag;
    } else if (header.format_version >= 3u) {
        upgrade_nominal_pellet_units_kernel<<<
            (world.pellet_count + kThreads - 1) / kThreads,
            kThreads>>>(world);
        status = cudaGetLastError();
        if (status != cudaSuccess) return abort_load(status);
    }
    if (header.format_version >= 7u) {
        status = read_checkpoint_values(*context, file, world.diet, bibites, "diet", header.format_version);
        if (status != cudaSuccess) return abort_load(status);
        status = read_checkpoint_values(*context, file, world.health, bibites, "health", header.format_version);
        if (status != cudaSuccess) return abort_load(status);
        status = read_checkpoint_values(*context, file, world.bite_cooldown, bibites, "bite_cooldown", header.format_version);
        if (status != cudaSuccess) return abort_load(status);
        status = read_checkpoint_values(*context, file, world.meat_gain, bibites, "meat_gain", header.format_version);
        if (status != cudaSuccess) return abort_load(status);
        status = read_checkpoint_values(*context, file, world.actions_c, bibites, "actions_c", header.format_version);
        if (status != cudaSuccess) return abort_load(status);
        status = read_checkpoint_values(*context, file, world.actions_d, bibites, "actions_d", header.format_version);
        if (status != cudaSuccess) return abort_load(status);
    } else {
        upgrade_legacy_biology_kernel<<<
            (world.max_bibites + kThreads - 1) / kThreads,
            kThreads>>>(world);
        status = cudaGetLastError();
        if (status != cudaSuccess) return abort_load(status);
    }
    if (header.format_version >= 8u) {
        CombatCheckpointState combat_state{};
        if (!read_host_values(file, &combat_state, 1) ||
            !std::isfinite(combat_state.collision_damage_constant) ||
            !std::isfinite(combat_state.collision_damage_threshold) ||
            !std::isfinite(combat_state.biting_damage_factor) ||
            !std::isfinite(combat_state.biting_pressure) ||
            combat_state.collision_damage_constant < 0.0f ||
            combat_state.collision_damage_constant > 2.0f ||
            combat_state.collision_damage_threshold < 0.0f ||
            combat_state.collision_damage_threshold > 200.0f ||
            combat_state.biting_damage_factor < 0.0f ||
            combat_state.biting_damage_factor > 20.0f ||
            combat_state.biting_pressure < 1.0f ||
            combat_state.biting_pressure > 500.0f)
            return abort_load(cudaErrorInvalidValue);
        world.collision_damage_constant = combat_state.collision_damage_constant;
        world.collision_damage_threshold = combat_state.collision_damage_threshold;
        world.biting_damage_factor = combat_state.biting_damage_factor;
        world.biting_pressure = combat_state.biting_pressure;
    }
    if (header.format_version >= 9u) {
        status = read_checkpoint_values(*context, file, world.life_state, bibites, "life_state", header.format_version);
        if (status != cudaSuccess) return abort_load(status);
        status = read_checkpoint_values(*context, file, world.reproductive_traits, bibites, "reproductive_traits", header.format_version);
        if (status != cudaSuccess) return abort_load(status);
        status = read_checkpoint_values(*context, file, world.last_damage, bibites, "last_damage", header.format_version);
        if (status != cudaSuccess) return abort_load(status);
        status = read_checkpoint_values(*context, file, world.held_target, bibites, "held_target", header.format_version);
        if (status != cudaSuccess) return abort_load(status);
        status = read_checkpoint_values(*context, file, world.held_instance, bibites, "held_instance", header.format_version);
        if (status != cudaSuccess) return abort_load(status);
        status = read_checkpoint_values(*context, file, world.clock_reset_output, bibites, "clock_reset_output", header.format_version);
        if (status != cudaSuccess) return abort_load(status);
        status = read_checkpoint_values(*context, file, world.pellet_held_by, pellets, "pellet_held_by", header.format_version);
        if (status != cudaSuccess) return abort_load(status);
        status = read_checkpoint_values(*context, file, world.pheromone_heading_x_a, pheromone_cells, "pheromone_heading_x_a", header.format_version);
        if (status != cudaSuccess) return abort_load(status);
        status = read_checkpoint_values(*context, file, world.pheromone_heading_x_b, pheromone_cells, "pheromone_heading_x_b", header.format_version);
        if (status != cudaSuccess) return abort_load(status);
        status = read_checkpoint_values(*context, file, world.pheromone_heading_y_a, pheromone_cells, "pheromone_heading_y_a", header.format_version);
        if (status != cudaSuccess) return abort_load(status);
        status = read_checkpoint_values(*context, file, world.pheromone_heading_y_b, pheromone_cells, "pheromone_heading_y_b", header.format_version);
        if (status != cudaSuccess) return abort_load(status);
    } else {
        upgrade_legacy_brain_io_kernel<<<
            (std::max(world.max_bibites, world.pellet_count) + kThreads - 1) /
                kThreads, kThreads>>>(world);
        status = cudaGetLastError();
        if (status != cudaSuccess) return abort_load(status);
    }
    if (header.format_version >= 10u) {
        status = read_checkpoint_values(*context, file, world.native_brain_version, bibites, "native_brain_version", header.format_version);
        if (status != cudaSuccess) return abort_load(status);
        status = read_checkpoint_values(*context, file, world.extension_masks, bibites * kExtensionMaskWords, "extension_masks", header.format_version);
        if (status != cudaSuccess) return abort_load(status);
        status = read_checkpoint_values(*context, file, world.extension_weights, bibites * kExtensionConnectionCount, "extension_weights", header.format_version);
        if (status != cudaSuccess) return abort_load(status);
        status = read_checkpoint_values(*context, file, world.extension_synapse_count, bibites, "extension_synapse_count", header.format_version);
        if (status != cudaSuccess) return abort_load(status);
    }
    if (header.format_version >= 11u) {
        BgfWorldDigestionSettings settings{};
        if (!read_host_values(file, &settings, 1) ||
            bgf_world_set_digestion_settings(context, &settings) != 0)
            return abort_load(cudaErrorInvalidValue);
    }
    if (header.format_version >= 12u) {
        status=read_checkpoint_values(*context,file,world.instance_id,bibites,"instance_id",header.format_version);
        if (status!=cudaSuccess) return abort_load(status);
    }
    uint64_t footer = 0;
    if (!read_host_values(file, &footer, 1) || footer != kCheckpointFooter) {
        return abort_load(cudaErrorInvalidValue);
    }
    if (std::fgetc(file) != EOF) {
        return abort_load(cudaErrorInvalidValue);
    }
    status = validate_checkpoint_state(*context,header.format_version>=12u);
    if (status != cudaSuccess) return abort_load(status);
    source_file.reset();
    status = reset_derived_spatial_state(*context);
    if (status != cudaSuccess) {
        return cuda_result(status);
    }
    status = cudaMemset(world.pellet_pending, 0,
        sizeof(int32_t) * static_cast<size_t>(world.pellet_count));
    if (status == cudaSuccess) {
        rebuild_food_pending_kernel<<<kPelletRespawnWheelSize, kThreads>>>(world);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        return cuda_result(status);
    }
    status = cudaMemset(world.pending_damage, 0,
        sizeof(float) * static_cast<size_t>(world.max_bibites));
    if (status == cudaSuccess)
        status = cudaMemset(world.dead_bibite_count, 0, sizeof(int32_t));
    if (status == cudaSuccess)
        status = cudaMemset(world.meat_cursor, 0, sizeof(int32_t));
    if (status != cudaSuccess) return cuda_result(status);
    context->has_stepped = true;
    const int food_status = bgf_world_set_food_settings(
        *world_handle,
        world.pellet_limit,
        world.food_growth_factor,
        world.pellet_energy);
    if (food_status != 0) {
        return food_status;
    }
    context->live_list_dirty = true;
    status = rebuild_runtime(*context);
    if (status == cudaSuccess) status = refresh_pellet_overflow(*context);
    if (status == cudaSuccess) status = cudaMemcpy(&context->latest_living,world.live_count,sizeof(int),cudaMemcpyDeviceToHost);
    if (status == cudaSuccess) status = cudaMemcpy(&context->host_steps,world.counters,sizeof(uint64_t),cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return cuda_result(status);
    pending_world.release();
    return 0;
}
catch (const std::bad_alloc&) {
    return static_cast<int>(cudaErrorMemoryAllocation);
}
catch (...) {
    return static_cast<int>(cudaErrorUnknown);
}
