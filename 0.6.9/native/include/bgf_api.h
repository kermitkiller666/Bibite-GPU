#pragma once

#include <stdint.h>

#ifdef _WIN32
#define BGF_EXPORT __declspec(dllexport)
#else
#define BGF_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BgfBrain {
    int32_t node_offset;
    int32_t target_offset;
    int32_t target_count;
    float period;
} BgfBrain;

typedef struct BgfNode {
    float value;
    float last_output;
    float last_input;
    float bias;
    int32_t type;
} BgfNode;

typedef struct BgfTarget {
    int32_t node_index;
    int32_t edge_offset;
    int32_t edge_count;
} BgfTarget;

typedef struct BgfEdge {
    int32_t source_node;
    uint16_t weight_bits;
    uint16_t reserved;
} BgfEdge;

typedef struct BgfVisionQuery {
    float position_x;
    float position_y;
    float up_x;
    float up_y;
    float view_radius;
    float view_angle_degrees;
    float herd_separation_distance;
    float separation_weight;
    float cohesion_weight;
    float alignment_weight;
    int32_t entity_offset;
    int32_t entity_count;
    int32_t self_entity_id;
    int32_t target_mask;
} BgfVisionQuery;

typedef struct BgfVisionEntity {
    float position_x;
    float position_y;
    float up_x;
    float up_y;
    float radius;
    float size_factor;
    float max_health;
    float health_ratio;
    float color_r;
    float color_g;
    float color_b;
    int32_t type;
    int32_t flags;
    int32_t entity_id;
    int32_t held_by_entity_id;
} BgfVisionEntity;

typedef struct BgfVisionResult {
    int32_t plant_count;
    int32_t meat_count;
    int32_t bibite_count;
    int32_t has_herd;
    float plant_angle;
    float plant_weight;
    float meat_angle;
    float meat_weight;
    float bibite_angle;
    float bibite_weight;
    float target_r;
    float target_g;
    float target_b;
    float herd_direction;
    float herd_separation_projection;
    float max_plant_weight;
    float max_meat_weight;
    float max_bibite_weight;
} BgfVisionResult;

typedef struct BgfPheromoneQuery {
    float position_x;
    float position_y;
    float up_x;
    float up_y;
    float sense_radius;
    float red_death_safe_radius;
    int32_t enable_red_death;
    int32_t reserved;
} BgfPheromoneQuery;

typedef struct BgfPheromoneEntity {
    float position_x;
    float position_y;
    float heading_x;
    float heading_y;
    float red_strength;
    float green_strength;
    float blue_strength;
    float reserved;
} BgfPheromoneEntity;

typedef struct BgfPheromoneResult {
    float red_sum;
    float green_sum;
    float blue_sum;
    float red_angle;
    float green_angle;
    float blue_angle;
    float red_heading_angle;
    float green_heading_angle;
    float blue_heading_angle;
} BgfPheromoneResult;

enum BgfVisionEntityType {
    BGF_VISION_PLANT = 0,
    BGF_VISION_MEAT = 1,
    BGF_VISION_BIBITE = 2,
    BGF_VISION_CORPSE = 3
};

enum BgfVisionEntityFlags {
    BGF_VISION_HELD = 1,
    BGF_VISION_COLOR_READY = 2
};

typedef void* BgfContextHandle;
typedef void* BgfVisionContextHandle;
typedef void* BgfPheromoneContextHandle;

BGF_EXPORT int bgf_get_device_count(int32_t* count);
BGF_EXPORT int bgf_get_device_name(int32_t device_index, char* buffer, int32_t buffer_length);
BGF_EXPORT int bgf_set_device(int32_t device_index);
BGF_EXPORT uint16_t bgf_float_to_half_bits(float value);
BGF_EXPORT float bgf_half_bits_to_float(uint16_t value);

BGF_EXPORT int bgf_create_context(
    int32_t device_index,
    int32_t brain_capacity,
    int32_t node_capacity,
    int32_t target_capacity,
    int32_t edge_capacity,
    BgfContextHandle* context);
BGF_EXPORT int bgf_destroy_context(BgfContextHandle context);
BGF_EXPORT int bgf_upload_brains(
    BgfContextHandle context,
    const BgfBrain* brains,
    int32_t brain_count);
BGF_EXPORT int bgf_upload_nodes(
    BgfContextHandle context,
    const BgfNode* nodes,
    int32_t node_count);
BGF_EXPORT int bgf_upload_topology(
    BgfContextHandle context,
    const BgfTarget* targets,
    int32_t target_count,
    const BgfEdge* edges,
    int32_t edge_count);
BGF_EXPORT int bgf_step(BgfContextHandle context);
BGF_EXPORT int bgf_download_nodes(
    BgfContextHandle context,
    BgfNode* nodes,
    int32_t node_count);

// Reference transfer path retained for correctness testing.
BGF_EXPORT int bgf_evaluate_brains(
    const BgfBrain* brains,
    int32_t brain_count,
    BgfNode* nodes,
    int32_t node_count,
    const BgfTarget* targets,
    int32_t target_count,
    const BgfEdge* edges,
    int32_t edge_count);

BGF_EXPORT int bgf_evaluate_vision(
    const BgfVisionQuery* queries,
    int32_t query_count,
    const BgfVisionEntity* entities,
    int32_t entity_count,
    BgfVisionResult* results);

BGF_EXPORT int bgf_create_vision_context(
    int32_t device_index,
    int32_t query_capacity,
    int32_t entity_capacity,
    BgfVisionContextHandle* context);
BGF_EXPORT int bgf_destroy_vision_context(BgfVisionContextHandle context);
BGF_EXPORT int bgf_evaluate_vision_context(
    BgfVisionContextHandle context,
    const BgfVisionQuery* queries,
    int32_t query_count,
    const BgfVisionEntity* entities,
    int32_t entity_count,
    BgfVisionResult* results);

BGF_EXPORT int bgf_create_pheromone_context(
    int32_t device_index,
    int32_t query_capacity,
    int32_t entity_capacity,
    BgfPheromoneContextHandle* context);
BGF_EXPORT int bgf_destroy_pheromone_context(BgfPheromoneContextHandle context);
BGF_EXPORT int bgf_evaluate_pheromones_context(
    BgfPheromoneContextHandle context,
    const BgfPheromoneQuery* queries,
    int32_t query_count,
    const BgfPheromoneEntity* entities,
    int32_t entity_count,
    BgfPheromoneResult* results);

#ifdef __cplusplus
}
#endif
