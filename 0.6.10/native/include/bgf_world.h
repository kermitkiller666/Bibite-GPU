#pragma once

#include "bgf_api.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* BgfWorldHandle;
typedef void* BgfCheckpointCaptureHandle;

typedef struct BgfWorldConfig {
    int32_t device_index;
    int32_t max_bibites;
    int32_t initial_bibites;
    int32_t pellet_count;
    int32_t spatial_grid_width;
    int32_t pheromone_grid_width;
    int32_t pheromone_grid_height;
    uint32_t seed;
    float world_half_extent;
    float fixed_delta_time;
    float initial_energy;
    float pellet_energy;
    float reproduction_energy;
    float sense_radius;
    float pheromone_diffusion;
    float pheromone_decay;
    float mutation_strength;
    uint32_t diagnostic_mask;
    int32_t contact_grid_update_factor;
    int32_t contact_solve_factor;
    int32_t vision_lookup_factor;
    int32_t brain_update_factor;
    int32_t lock_food_target;
} BgfWorldConfig;

/* Blittable snapshot of one stock plant-spawning zone. Distribution follows
 * SpawnDistribution: flat/centric/exterior circle, ring/flat ring, rectangle.
 * Weights are relative, not pellet counts or energy. */
typedef struct BgfWorldFoodZone {
    int32_t distribution;
    float center_x;
    float center_y;
    float radius;
    float inner_radius;
    float half_width;
    float half_height;
    float seed_weight;
    float growth_weight;
    float pellet_size; /* stock zone multiplier, 0 treated as legacy 1x */
} BgfWorldFoodZone;

typedef struct BgfWorldStepMetrics {
    int32_t requested_steps;
    float gpu_milliseconds;
    float wall_milliseconds;
    float host_overhead_milliseconds;
    float gpu_offload_percent;
    double simulated_seconds;
    double realtime_multiplier;
    float prepare_milliseconds;
    float spatial_index_milliseconds;
    float contact_milliseconds;
    float decision_milliseconds;
    float motion_milliseconds;
    float lifecycle_milliseconds;
    float food_milliseconds;
    float brain_milliseconds;
    float post_decision_milliseconds;
    int32_t pipeline; /* 0 fused, 1 sparse graph, 2 experimental tensor, 3 dense graph. */
    float graph_build_milliseconds; /* Host capture/instantiate time; also in full wall time. */
    int32_t graph_cache_hits;
} BgfWorldStepMetrics;

typedef struct BgfWorldStats {
    uint64_t completed_steps;
    uint64_t births;
    uint64_t deaths;
    uint64_t pellets_eaten;
    int32_t living_bibites;
    int32_t active_pellets;
    float total_energy;
    float average_energy;
    double simulated_seconds;
    uint64_t starvation_deaths;
    uint64_t age_deaths;
    uint64_t invalid_state_deaths;
    float plant_energy; /* Actual remaining energy, including zone pellet sizes. */
    int32_t active_meat;
    float meat_energy;
} BgfWorldStats;

typedef struct BgfWorldBibite {
    int32_t slot;
    int32_t generation;
    uint64_t lineage_id;
    uint64_t tag_id;
    int32_t brain_nodes;
    int32_t brain_synapses;
    float position_x;
    float position_y;
    float velocity_x;
    float velocity_y;
    float heading;
    float energy;
    float age;
    float size;
    float color_r;
    float color_g;
    float color_b;
} BgfWorldBibite;

/*
 * Detailed state for the one Bibite currently selected in the Unity UI.
 * This is intentionally queried on demand instead of being appended to every
 * presentation record: even a 500,000-Bibite world therefore pays for exactly
 * one CPU inspector record.
 */
typedef struct BgfWorldBibiteDetail {
    int32_t slot;
    int32_t alive;
    int32_t generation;
    int32_t brain_nodes;
    int32_t brain_synapses;
    int32_t template_brain;
    int32_t cached_food;
    int32_t cached_neighbour;
    int32_t neighbours_seen;
    int32_t reserved; /* Runtime slot incarnation, nonzero for a live Bibite. */
    uint64_t lineage_id;
    uint64_t tag_id;
    float position_x;
    float position_y;
    float velocity_x;
    float velocity_y;
    float heading;
    float energy;
    float age;
    float size;
    float reproduction_cooldown;
    float maximum_speed;
    float turn_speed;
    float metabolism;
    float lifespan;
    float color_r;
    float color_g;
    float color_b;
    float gene_mutation_strength;
    float brain_mutation_strength;
    float diet;
    float health;
    float plant_stomach;
    float meat_stomach;
    float eat_output;
    float digestion_output;
    float attack_output;
    float acceleration_output;
    float rotation_output;
    float pheromone_1_output;
    float pheromone_2_output;
    float pheromone_3_output;
    float reproduction_output;
    float food_direction_x;
    float food_direction_y;
    float food_distance_squared;
    float neighbour_direction_x;
    float neighbour_direction_y;
    float neighbour_distance_squared;
    float neighbour_color_r;
    float neighbour_color_g;
    float neighbour_color_b;
    float adult_size;
    float egg_progress;
    float clock_time;
    float clock_tic;
    float growth_output;
    float egg_production_output;
    float grab_output;
    float herding_output;
    float clock_reset_output;
    int32_t held_count;
} BgfWorldBibiteDetail;

typedef struct BgfWorldBrainNodeState {
    int32_t type;
    int32_t sensor;
    int32_t action;
    float base_activation;
    float last_input;
    float last_output;
} BgfWorldBrainNodeState;

typedef struct BgfWorldBrainSynapseState {
    int32_t node_in;
    int32_t node_out;
    float weight;
} BgfWorldBrainSynapseState;

/*
 * Stock Bibite templates use a small recurrent NEAT graph.  These records are
 * deliberately plain/blittable so the Unity bridge can copy a selected
 * template into CUDA without constructing a CPU-side Bibite GameObject.
 */
typedef struct BgfWorldTemplateSpawn {
    float position_x;
    float position_y;
    float heading;
    float energy;
    float size;
    float maximum_speed;
    float turn_speed;
    float metabolism;
    float lifespan;
    float color_r;
    float color_g;
    float color_b;
    float gene_mutation_strength;
    float brain_mutation_strength;
    float diet;
    float adult_size;
    float clock_period;
    float lay_period;
    float womb_capacity;
    int32_t generation;
    uint64_t lineage_id;
    uint64_t tag_id;
} BgfWorldTemplateSpawn;

typedef struct BgfWorldBrainNode {
    int32_t type;
    int32_t sensor;
    int32_t action;
    float base_activation;
} BgfWorldBrainNode;

typedef struct BgfWorldBrainSynapse {
    int32_t node_in;
    int32_t node_out;
    float weight;
} BgfWorldBrainSynapse;

typedef struct BgfWorldPellet {
    int32_t slot;
    float position_x;
    float position_y;
    float energy;
    int32_t material; /* 0 plant, 1 meat */
} BgfWorldPellet;

typedef struct BgfWorldD3D11RenderConfig {
    void* bibite_vertex_buffer;
    void* pellet_vertex_buffer;
    int32_t bibite_capacity;
    int32_t pellet_capacity;
    float pellet_half_width;
    float pellet_half_height;
    float pellet_uv_min_x;
    float pellet_uv_min_y;
    float pellet_uv_max_x;
    float pellet_uv_max_y;
    float pellet_color_r;
    float pellet_color_g;
    float pellet_color_b;
    float pellet_color_a;
} BgfWorldD3D11RenderConfig;

BGF_EXPORT void bgf_world_default_config(BgfWorldConfig* config);
BGF_EXPORT int bgf_world_create(const BgfWorldConfig* config, BgfWorldHandle* world);
BGF_EXPORT int bgf_world_destroy(BgfWorldHandle world);
BGF_EXPORT int bgf_world_step(
    BgfWorldHandle world,
    int32_t steps,
    BgfWorldStepMetrics* metrics);
BGF_EXPORT int bgf_world_get_stats(BgfWorldHandle world, BgfWorldStats* stats);
typedef struct BgfWorldRuntimeInfo {
    int32_t living_work_items;
    int32_t imported_brains;
    int32_t imported_pool_capacity;
    int32_t contact_grid_width;
    int32_t sense_grid_width;
    int32_t reserved;
    uint64_t imported_pool_bytes;
    uint64_t brain_workspace_bytes;
} BgfWorldRuntimeInfo;
BGF_EXPORT int bgf_world_get_runtime_info(BgfWorldHandle world, BgfWorldRuntimeInfo* info);
/* Lightweight viewport detail; no food download or global energy summary. */
BGF_EXPORT int bgf_world_download_visible_bibites(BgfWorldHandle world,
    BgfWorldBibite* bibites, int32_t capacity, int32_t* written,
    float min_x, float min_y, float max_x, float max_y);
BGF_EXPORT int bgf_world_spawn_bibite(
    BgfWorldHandle world,
    float position_x,
    float position_y,
    float heading,
    int32_t* spawned_slot);
BGF_EXPORT int bgf_world_spawn_template_bibite(
    BgfWorldHandle world,
    const BgfWorldTemplateSpawn* spawn,
    const BgfWorldBrainNode* nodes,
    int32_t node_count,
    const BgfWorldBrainSynapse* synapses,
    int32_t synapse_count,
    int32_t* spawned_slot);
BGF_EXPORT int bgf_world_download_bibites(
    BgfWorldHandle world,
    BgfWorldBibite* bibites,
    int32_t capacity,
    int32_t* written);
BGF_EXPORT int bgf_world_download_pellets(
    BgfWorldHandle world,
    BgfWorldPellet* pellets,
    int32_t capacity,
    int32_t* written);
BGF_EXPORT int bgf_world_download_snapshot(
    BgfWorldHandle world,
    BgfWorldBibite* bibites,
    int32_t bibite_capacity,
    int32_t* bibites_written,
    BgfWorldPellet* pellets,
    int32_t pellet_capacity,
    int32_t* pellets_written,
    BgfWorldStats* stats);
// Keeps the representative world snapshot for charts/selection while returning
// a separate, camera-filtered set for close-up textured sprites.
BGF_EXPORT int bgf_world_download_snapshot_with_view(
    BgfWorldHandle world,
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
    BgfWorldStats* stats);
BGF_EXPORT int bgf_world_get_bibite_detail(
    BgfWorldHandle world,
    int32_t slot,
    BgfWorldBibiteDetail* detail,
    BgfWorldBrainNodeState* nodes,
    int32_t node_capacity,
    int32_t* nodes_written,
    BgfWorldBrainSynapseState* synapses,
    int32_t synapse_capacity,
    int32_t* synapses_written);
/* GPU-native starter brains only. Disabling makes the effective weight zero
 * while preserving the dormant FP16 value; re-enabling restores that value.
 * Untouched links receive a random initial value when first enabled. Calls
 * must be serialized with stepping on the world worker. */
BGF_EXPORT int bgf_world_set_hidden_link_enabled(
    BgfWorldHandle world,
    int32_t slot,
    int32_t edge,
    int32_t enabled);
BGF_EXPORT int bgf_world_get_hidden_link(
    BgfWorldHandle world,
    int32_t slot,
    int32_t edge,
    int32_t* enabled,
    float* effective_weight);
BGF_EXPORT int bgf_world_set_bibite_tag(
    BgfWorldHandle world,
    int32_t slot,
    uint64_t tag_id);
BGF_EXPORT int bgf_world_kill_bibite(
    BgfWorldHandle world,
    int32_t slot);
BGF_EXPORT int bgf_world_force_reproduction(
    BgfWorldHandle world,
    int32_t parent_slot,
    int32_t* child_slot);
/* Updates the live food population and regrowth without recreating the world.
 * target is bounded by the pellet capacity in BgfWorldConfig. */
BGF_EXPORT int bgf_world_set_food_settings(
    BgfWorldHandle world,
    int32_t target,
    float growth_factor,
    float pellet_energy);
BGF_EXPORT int bgf_world_get_food_settings(
    BgfWorldHandle world,
    int32_t* target,
    float* growth_factor,
    float* pellet_energy);
/* Live physical drag in inverse simulated seconds. 0 disables drag. */
BGF_EXPORT int bgf_world_set_linear_drag(BgfWorldHandle world, float drag);
BGF_EXPORT int bgf_world_get_linear_drag(BgfWorldHandle world, float* drag);
BGF_EXPORT int bgf_world_set_combat_settings(
    BgfWorldHandle world,
    float collision_damage_constant,
    float collision_damage_threshold,
    float biting_damage_factor,
    float biting_pressure);
/* Live stock diet affinity and material conversion settings. Values are
 * applied on the serialized world worker before the next simulation step. */
typedef struct BgfWorldDigestionSettings {
    float plant_affinity_power;
    float meat_affinity_power;
    float plant_min_efficiency;
    float plant_max_efficiency;
    float meat_min_efficiency;
    float meat_max_efficiency;
} BgfWorldDigestionSettings;
BGF_EXPORT int bgf_world_set_digestion_settings(
    BgfWorldHandle world,
    const BgfWorldDigestionSettings* settings);
BGF_EXPORT int bgf_world_get_digestion_settings(
    BgfWorldHandle world,
    BgfWorldDigestionSettings* settings);
/* Updates up to 64 plant zones. When reseed_existing is nonzero, active
 * pellets are repositioned and the food spatial index is rebuilt. The caller
 * must serialize this call with stepping and snapshots on the world worker. */
BGF_EXPORT int bgf_world_set_food_zones(
    BgfWorldHandle world,
    const BgfWorldFoodZone* zones,
    int32_t count,
    int32_t reseed_existing);
BGF_EXPORT int bgf_world_register_d3d11_render_buffers(
    BgfWorldHandle world,
    const BgfWorldD3D11RenderConfig* config);
// Call on Unity's main thread after capturing native pointers, before any
// worker-thread graphics interop. SINGLETHREADED devices are not eligible.
BGF_EXPORT int bgf_world_prepare_d3d11_multithreading(
    void* vertex_buffer,
    uint32_t* device_flags,
    int32_t* protection_was_enabled);
BGF_EXPORT int bgf_world_update_d3d11_render_buffers(BgfWorldHandle world);
// Two persistent surfaces allow Unity to draw one while CUDA owns the other.
// The caller must retire all graphics uses before updating that surface, and
// must not expose it again until this call succeeds (including unmapping).
BGF_EXPORT int bgf_world_register_d3d11_render_buffer_set(
    BgfWorldHandle world,
    int32_t buffer_index,
    const BgfWorldD3D11RenderConfig* config);
BGF_EXPORT int bgf_world_update_d3d11_render_buffer_set(
    BgfWorldHandle world,
    int32_t buffer_index,
    int32_t* bibite_count,
    int32_t* pellet_count);
/* Capture on the simulation worker before submitting an update request.
 * Updating that captured surface then reads no mutable world arrays. */
BGF_EXPORT int bgf_world_capture_d3d11_render_buffer_set(BgfWorldHandle world,
    int32_t buffer_index);
BGF_EXPORT int bgf_world_unregister_d3d11_render_buffers(BgfWorldHandle world);
// Unity render-thread command mailbox. Creating/polling/cancelling a request
// performs no graphics or CUDA work. Registration/unregistration and legacy
// updates require exclusive world ownership until state 2/3. Updates of an
// independently captured surface may overlap stepping; do not recapture or
// retire that surface until the request finishes. Then release the ID.
// Operation: 0 register, 1 update, 2 unregister all. State: 0 pending, 1 running.
BGF_EXPORT void* bgf_world_unity_render_event_func(void);
BGF_EXPORT int bgf_world_create_unity_render_request(
    BgfWorldHandle world, int32_t buffer_index, int32_t operation,
    const BgfWorldD3D11RenderConfig* config, int32_t* event_id);
BGF_EXPORT int bgf_world_poll_unity_render_request(
    int32_t event_id, int32_t* state, int32_t* result,
    int32_t* bibite_count, int32_t* pellet_count);
BGF_EXPORT int bgf_world_cancel_unity_render_request(int32_t event_id);
BGF_EXPORT int bgf_world_release_unity_render_request(int32_t event_id);
// Only after a failed completed unregister callback: retain driver references
// until process exit while allowing the CPU world object itself to retire.
BGF_EXPORT int bgf_world_abandon_d3d11_render_buffers(BgfWorldHandle world);
BGF_EXPORT int bgf_world_get_device_name(
    BgfWorldHandle world,
    char* buffer,
    int32_t buffer_length);
BGF_EXPORT int bgf_world_save_checkpoint(
    BgfWorldHandle world,
    const char* path);
/* Regression/migration export rebuilds the old capacity-sized index layout. */
BGF_EXPORT int bgf_world_save_checkpoint_legacy_v11(BgfWorldHandle world, const char* path);
/* Capture is serialized with the world. Once it returns, the capture owns all
 * its data and may be written/released on a separate CPU thread. */
BGF_EXPORT int bgf_world_capture_checkpoint(BgfWorldHandle world,
    BgfCheckpointCaptureHandle* capture);
BGF_EXPORT int bgf_world_write_captured_checkpoint(BgfCheckpointCaptureHandle capture,
    const char* path);
BGF_EXPORT int bgf_world_release_checkpoint_capture(BgfCheckpointCaptureHandle capture);
BGF_EXPORT int bgf_world_checkpoint_config(
    const char* path,
    int32_t device_index,
    BgfWorldConfig* config);
BGF_EXPORT int bgf_world_load_checkpoint(
    const char* path,
    int32_t device_index,
    BgfWorldHandle* world);

#ifdef __cplusplus
}
#endif
