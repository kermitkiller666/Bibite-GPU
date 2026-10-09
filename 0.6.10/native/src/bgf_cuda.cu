#include "bgf_api.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>

namespace {

struct GpuContext {
    int32_t device_index;
    int32_t brain_capacity;
    int32_t node_capacity;
    int32_t target_capacity;
    int32_t edge_capacity;
    int32_t brain_count;
    int32_t node_count;
    int32_t target_count;
    int32_t edge_count;
    BgfBrain* brains;
    BgfNode* nodes;
    BgfTarget* targets;
    BgfEdge* edges;
};

struct GpuVisionContext {
    int32_t device_index;
    int32_t query_capacity;
    int32_t entity_capacity;
    BgfVisionQuery* queries;
    BgfVisionEntity* entities;
    BgfVisionResult* results;
};

struct GpuPheromoneContext {
    int32_t device_index;
    int32_t query_capacity;
    int32_t entity_capacity;
    BgfPheromoneQuery* queries;
    BgfPheromoneEntity* entities;
    BgfPheromoneResult* results;
};

enum NodeType : int32_t {
    Input = 0,
    Sigmoid = 1,
    Linear = 2,
    TanH = 3,
    Sine = 4,
    ReLu = 5,
    Gaussian = 6,
    Latch = 7,
    Differential = 8,
    Abs = 9,
    Mult = 10,
    Integrator = 11,
    Inhibitory = 12,
    SoftLatch = 13
};

__host__ __device__ float clamp_output(float value)
{
    return fminf(1024.0f, fmaxf(-1024.0f, value));
}

__device__ float half_bits_to_float(uint16_t bits)
{
    __half_raw raw;
    raw.x = bits;
    __half value = raw;
    return __half2float(value);
}

__device__ bool uses_bias_differently(int32_t type)
{
    return type == Integrator || type == Mult || type == Inhibitory || type == SoftLatch;
}

__device__ float soft_latch(const BgfNode& node, float input)
{
    const float bias = node.bias;
    const float previous_output = node.last_output;
    float previous_input = node.last_input;

    if (input >= 0.999f) return 1.0f;
    if (input <= 0.001f) return 0.0f;
    if (previous_input >= 0.999f) previous_input = 1.0f;
    if (previous_input <= 0.001f) previous_input = 0.0f;

    if (input < previous_input) {
        return previous_output / (1.0f - expf(-bias * previous_input)) *
               (1.0f - expf(-bias * input));
    }

    const float start =
        (previous_output - expf(bias * (previous_input - 1.0f))) /
        (1.0f - expf(bias * (previous_input - 1.0f)));
    return start + (1.0f - start) * expf(bias * (input - 1.0f));
}

__device__ void execute_node(BgfNode& node, float input, float period)
{
    if (!uses_bias_differently(node.type)) {
        input += node.bias;
    } else if (node.type == Mult) {
        input *= node.bias;
    }

    float output = 0.0f;
    switch (node.type) {
        case Input: output = input; break;
        case Sigmoid: output = 1.0f / (1.0f + expf(-input)); break;
        case Linear: output = input; break;
        case TanH: output = tanhf(input); break;
        case Sine: output = sinf(input); break;
        case ReLu: output = fmaxf(input, 0.0f); break;
        case Gaussian: output = 1.0f / (1.0f + input * input); break;
        case Latch:
            output = input >= 1.0f ? 1.0f : (input <= 0.0f ? 0.0f : node.last_output);
            break;
        case Differential: output = (input - node.last_input) / period; break;
        case Abs: output = fabsf(input); break;
        case Mult: output = input; break;
        case Integrator: output = node.last_output + input * period; break;
        case Inhibitory:
            output = input - node.last_input + node.last_output * expf(-node.bias * period);
            break;
        case SoftLatch: output = soft_latch(node, input); break;
        default: output = 0.0f; break;
    }

    output = clamp_output(output);
    node.value = output;
    node.last_output = output;
    node.last_input = input;
}

__global__ void evaluate_brains_kernel(
    const BgfBrain* brains,
    int32_t brain_count,
    BgfNode* nodes,
    const BgfTarget* targets,
    const BgfEdge* edges)
{
    const int32_t brain_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (brain_index >= brain_count) return;

    const BgfBrain brain = brains[brain_index];
    for (int32_t target_index = 0; target_index < brain.target_count; ++target_index) {
        const BgfTarget target = targets[brain.target_offset + target_index];
        BgfNode& node = nodes[brain.node_offset + target.node_index];
        float input = node.type == Mult ? 1.0f : 0.0f;

        for (int32_t edge_index = 0; edge_index < target.edge_count; ++edge_index) {
            const BgfEdge edge = edges[target.edge_offset + edge_index];
            const float weighted =
                half_bits_to_float(edge.weight_bits) *
                nodes[brain.node_offset + edge.source_node].last_output;
            input = node.type == Mult ? input * weighted : input + weighted;
        }

        execute_node(node, input, brain.period);
    }
}

struct Vec2 {
    float x;
    float y;
};

__device__ Vec2 add(Vec2 a, Vec2 b) { return {a.x + b.x, a.y + b.y}; }
__device__ Vec2 scale(Vec2 value, float factor) { return {value.x * factor, value.y * factor}; }
__device__ float length(Vec2 value) { return sqrtf(value.x * value.x + value.y * value.y); }

__device__ Vec2 normalize(Vec2 value)
{
    const float magnitude = length(value);
    return magnitude > 1.0e-12f ? scale(value, 1.0f / magnitude) : Vec2{0.0f, 0.0f};
}

__device__ float signed_angle_degrees(Vec2 from, Vec2 to)
{
    const float from_length = length(from);
    const float to_length = length(to);
    if (from_length < 1.0e-12f || to_length < 1.0e-12f) return 0.0f;
    const float cross = from.x * to.y - from.y * to.x;
    const float dot = from.x * to.x + from.y * to.y;
    return atan2f(cross, dot) * 57.29577951308232f;
}

__device__ float unsigned_angle_degrees(Vec2 from, Vec2 to)
{
    return fabsf(signed_angle_degrees(from, to));
}

__device__ bool target_in_view(
    const BgfVisionQuery& query,
    const BgfVisionEntity& entity,
    Vec2& direction,
    float& distance,
    float& angle)
{
    direction = {entity.position_x - query.position_x, entity.position_y - query.position_y};
    distance = length(direction);
    if (distance - entity.radius > query.view_radius) return false;

    const Vec2 up{query.up_x, query.up_y};
    angle = signed_angle_degrees(up, direction);
    const float half_angle = query.view_angle_degrees * 0.5f;
    if (fabsf(angle) < half_angle) {
        distance = fmaxf(distance - entity.radius, 0.001f);
        direction = scale(normalize(direction), distance);
        return true;
    }

    const float chord = 2.0f * distance * cosf((fabsf(angle) - half_angle) * 0.0174532925199433f);
    const float squared_tangent = distance * distance - entity.radius * entity.radius;
    const float discriminant = chord * chord - 4.0f * squared_tangent;
    if (discriminant <= 0.0f) return false;
    distance = fmaxf(0.001f, chord * 0.5f - sqrtf(discriminant) * 0.5f);
    if (distance > query.view_radius) return false;

    angle = copysignf(half_angle, angle);
    const float radians = angle * 0.0174532925199433f;
    direction = {
        (up.x * cosf(radians) - up.y * sinf(radians)) * distance,
        (up.x * sinf(radians) + up.y * cosf(radians)) * distance};
    return true;
}

__global__ void evaluate_vision_kernel(
    const BgfVisionQuery* queries,
    int32_t query_count,
    const BgfVisionEntity* entities,
    BgfVisionResult* results)
{
    const int32_t query_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_index >= query_count) return;

    const BgfVisionQuery query = queries[query_index];
    BgfVisionResult result{};
    Vec2 plant_sum{};
    Vec2 meat_sum{};
    Vec2 bibite_sum{};
    Vec2 herd_heading{};
    Vec2 herd_center{};
    Vec2 herd_separation{};
    float plant_weight_sum = 0.0f;
    float meat_weight_sum = 0.0f;
    float bibite_weight_sum = 0.0f;

    for (int32_t local_index = 0; local_index < query.entity_count; ++local_index) {
        const BgfVisionEntity entity = entities[query.entity_offset + local_index];
        if ((entity.flags & BGF_VISION_HELD) != 0) continue;
        if (entity.entity_id != 0 && entity.entity_id == query.self_entity_id) continue;
        if (entity.held_by_entity_id != 0 &&
            entity.held_by_entity_id == query.self_entity_id) continue;
        if (entity.type == BGF_VISION_PLANT && (query.target_mask & 2) == 0) continue;
        if (entity.type == BGF_VISION_MEAT && (query.target_mask & 4) == 0) continue;
        if ((entity.type == BGF_VISION_BIBITE || entity.type == BGF_VISION_CORPSE) &&
            (query.target_mask & 1) == 0) continue;

        Vec2 direction{};
        float distance = 0.0f;
        float angle = 0.0f;
        if (!target_in_view(query, entity, direction, distance, angle)) continue;

        const float distance_factor = powf(1.05f - distance / query.view_radius, 2.0f);
        const float angle_factor = 1.05f - fabsf(angle) / (query.view_angle_degrees * 0.5f);

        if (entity.type == BGF_VISION_PLANT) {
            const float weight = distance_factor * entity.size_factor * angle_factor;
            ++result.plant_count;
            result.max_plant_weight = fmaxf(result.max_plant_weight, weight);
            plant_sum = add(plant_sum, scale(direction, weight));
            plant_weight_sum += weight;
        } else if (entity.type == BGF_VISION_MEAT) {
            const float weight = distance_factor * entity.size_factor * angle_factor;
            ++result.meat_count;
            result.max_meat_weight = fmaxf(result.max_meat_weight, weight);
            meat_sum = add(meat_sum, scale(direction, weight));
            meat_weight_sum += weight;
        }

        if (entity.type == BGF_VISION_CORPSE) {
            const float weight = distance_factor * (entity.max_health / 100.0f) *
                (1.0f - entity.health_ratio) * angle_factor;
            ++result.meat_count;
            result.max_meat_weight = fmaxf(result.max_meat_weight, weight);
            meat_sum = add(meat_sum, scale(direction, weight));
            meat_weight_sum += weight;
        }

        if (entity.type == BGF_VISION_BIBITE || entity.type == BGF_VISION_CORPSE) {
            const float weight = entity.max_health / 100.0f * distance_factor * angle_factor;
            ++result.bibite_count;
            result.max_bibite_weight = fmaxf(result.max_bibite_weight, weight);
            bibite_sum = add(bibite_sum, scale(direction, weight));
            bibite_weight_sum += weight;
            herd_heading = add(herd_heading, Vec2{entity.up_x, entity.up_y});
            herd_center = add(herd_center, direction);
            if (distance < query.herd_separation_distance) {
                herd_separation = add(
                    herd_separation,
                    scale(normalize(direction), -(1.0f - distance / query.herd_separation_distance)));
            }
            // The 0.6.3.1 CPU path never updates dist2Bibite, so traversal order
            // makes the last visible Bibite the colour target. Preserve that here.
            if ((entity.flags & BGF_VISION_COLOR_READY) != 0) {
                result.target_r = entity.color_r;
                result.target_g = entity.color_g;
                result.target_b = entity.color_b;
            }
        }
    }

    const Vec2 up{query.up_x, query.up_y};
    if (plant_weight_sum > 0.0f) plant_sum = scale(plant_sum, 1.0f / plant_weight_sum);
    if (meat_weight_sum > 0.0f) meat_sum = scale(meat_sum, 1.0f / meat_weight_sum);
    if (bibite_weight_sum > 0.0f) bibite_sum = scale(bibite_sum, 1.0f / bibite_weight_sum);
    result.plant_angle = 2.0f * signed_angle_degrees(up, plant_sum) / query.view_angle_degrees;
    result.meat_angle = 2.0f * signed_angle_degrees(up, meat_sum) / query.view_angle_degrees;
    result.bibite_angle = 2.0f * signed_angle_degrees(up, bibite_sum) / query.view_angle_degrees;
    result.plant_weight = plant_weight_sum > 0.0f
        ? 1.0f - fminf(length(plant_sum) / query.view_radius, 1.0f) : 0.0f;
    result.meat_weight = meat_weight_sum > 0.0f
        ? 1.0f - fminf(length(meat_sum) / query.view_radius, 1.0f) : 0.0f;
    result.bibite_weight = bibite_weight_sum > 0.0f
        ? 1.0f - fminf(length(bibite_sum) / query.view_radius, 1.0f) : 0.0f;

    if (result.bibite_count > 0) {
        result.has_herd = 1;
        Vec2 direction = add(
            add(
                scale(normalize(herd_separation), query.separation_weight),
                scale(normalize(herd_center), query.cohesion_weight)),
            scale(herd_heading, query.alignment_weight));
        result.herd_direction = signed_angle_degrees(up, direction);
        result.herd_separation_projection = cosf(
            unsigned_angle_degrees(up, herd_separation) * 0.0174532925199433f);
    }
    results[query_index] = result;
}

__global__ void evaluate_pheromones_kernel(
    const BgfPheromoneQuery* queries,
    int32_t query_count,
    const BgfPheromoneEntity* entities,
    int32_t entity_count,
    BgfPheromoneResult* results)
{
    const int32_t query_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_index >= query_count) return;

    const BgfPheromoneQuery query = queries[query_index];
    BgfPheromoneResult result{};
    Vec2 red_direction{};
    Vec2 green_direction{};
    Vec2 blue_direction{};
    Vec2 red_heading{};
    Vec2 green_heading{};
    Vec2 blue_heading{};
    const float radius_squared = query.sense_radius * query.sense_radius;

    for (int32_t entity_index = 0; entity_index < entity_count; ++entity_index) {
        const BgfPheromoneEntity entity = entities[entity_index];
        const Vec2 direction{
            entity.position_x - query.position_x,
            entity.position_y - query.position_y};
        const float distance_squared =
            direction.x * direction.x + direction.y * direction.y;
        if (distance_squared <= 0.0f || distance_squared > radius_squared) continue;
        const float distance = sqrtf(distance_squared);
        if (entity.red_strength > 0.0f) {
            const float weight = entity.red_strength / distance;
            result.red_sum += weight;
            red_direction = add(red_direction, scale(direction, weight));
            red_heading = add(
                red_heading,
                scale(Vec2{entity.heading_x, entity.heading_y}, weight));
        }
        if (entity.green_strength > 0.0f) {
            const float weight = entity.green_strength / distance;
            result.green_sum += weight;
            green_direction = add(green_direction, scale(direction, weight));
            green_heading = add(
                green_heading,
                scale(Vec2{entity.heading_x, entity.heading_y}, weight));
        }
        if (entity.blue_strength > 0.0f) {
            const float weight = entity.blue_strength / distance;
            result.blue_sum += weight;
            blue_direction = add(blue_direction, scale(direction, weight));
            blue_heading = add(
                blue_heading,
                scale(Vec2{entity.heading_x, entity.heading_y}, weight));
        }
    }

    if (query.enable_red_death != 0) {
        const Vec2 position{query.position_x, query.position_y};
        const float position_length = length(position);
        const float distance_to_safe_radius = query.red_death_safe_radius - position_length;
        if (query.sense_radius > distance_to_safe_radius) {
            const float weight = 100.0f / fmaxf(1.0f, distance_to_safe_radius);
            result.red_sum += weight;
            const Vec2 outward = normalize(position);
            red_direction = add(
                red_direction,
                scale(outward, distance_to_safe_radius * weight));
            red_heading = add(red_heading, scale(outward, -weight));
        }
    }

    const Vec2 up{query.up_x, query.up_y};
    result.red_angle = signed_angle_degrees(up, red_direction) / 180.0f;
    result.green_angle = signed_angle_degrees(up, green_direction) / 180.0f;
    result.blue_angle = signed_angle_degrees(up, blue_direction) / 180.0f;
    result.red_heading_angle = signed_angle_degrees(up, red_heading) / 180.0f;
    result.green_heading_angle = signed_angle_degrees(up, green_heading) / 180.0f;
    result.blue_heading_angle = signed_angle_degrees(up, blue_heading) / 180.0f;
    results[query_index] = result;
}

int cuda_status(cudaError_t status)
{
    return status == cudaSuccess ? 0 : static_cast<int>(status);
}

template <typename T>
cudaError_t allocate_if_needed(T** pointer, int32_t capacity)
{
    if (capacity <= 0) {
        *pointer = nullptr;
        return cudaSuccess;
    }
    return cudaMalloc(pointer, sizeof(T) * static_cast<size_t>(capacity));
}

template <typename T>
int upload_array(T* destination, int32_t capacity, const T* source, int32_t count)
{
    if (count < 0 || count > capacity || (count > 0 && (!source || !destination))) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (count == 0) return 0;
    return cuda_status(cudaMemcpy(
        destination,
        source,
        sizeof(T) * static_cast<size_t>(count),
        cudaMemcpyHostToDevice));
}

} // namespace

extern "C" BGF_EXPORT int bgf_get_device_count(int32_t* count)
{
    if (!count) return static_cast<int>(cudaErrorInvalidValue);
    return cuda_status(cudaGetDeviceCount(count));
}

extern "C" BGF_EXPORT int bgf_get_device_name(
    int32_t device_index,
    char* buffer,
    int32_t buffer_length)
{
    if (!buffer || buffer_length <= 0) return static_cast<int>(cudaErrorInvalidValue);
    cudaDeviceProp properties{};
    cudaError_t status = cudaGetDeviceProperties(&properties, device_index);
    if (status != cudaSuccess) return cuda_status(status);
    std::strncpy(buffer, properties.name, static_cast<size_t>(buffer_length - 1));
    buffer[buffer_length - 1] = '\0';
    return 0;
}

extern "C" BGF_EXPORT int bgf_set_device(int32_t device_index)
{
    return cuda_status(cudaSetDevice(device_index));
}

extern "C" BGF_EXPORT uint16_t bgf_float_to_half_bits(float value)
{
    const __half converted = __float2half(value);
    const __half_raw raw = converted;
    return raw.x;
}

extern "C" BGF_EXPORT float bgf_half_bits_to_float(uint16_t value)
{
    __half_raw raw;
    raw.x = value;
    const __half converted = raw;
    return __half2float(converted);
}

extern "C" BGF_EXPORT int bgf_create_context(
    int32_t device_index,
    int32_t brain_capacity,
    int32_t node_capacity,
    int32_t target_capacity,
    int32_t edge_capacity,
    BgfContextHandle* context)
{
    if (!context || brain_capacity <= 0 || node_capacity <= 0 ||
        target_capacity < 0 || edge_capacity < 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    *context = nullptr;
    cudaError_t status = cudaSetDevice(device_index);
    if (status != cudaSuccess) return cuda_status(status);

    GpuContext* value = new GpuContext{};
    value->device_index = device_index;
    value->brain_capacity = brain_capacity;
    value->node_capacity = node_capacity;
    value->target_capacity = target_capacity;
    value->edge_capacity = edge_capacity;

#define BGF_CONTEXT_TRY(call) do { status = (call); if (status != cudaSuccess) goto cleanup; } while (0)
    BGF_CONTEXT_TRY(allocate_if_needed(&value->brains, brain_capacity));
    BGF_CONTEXT_TRY(allocate_if_needed(&value->nodes, node_capacity));
    BGF_CONTEXT_TRY(allocate_if_needed(&value->targets, target_capacity));
    BGF_CONTEXT_TRY(allocate_if_needed(&value->edges, edge_capacity));
    *context = value;
    return 0;

cleanup:
    cudaFree(value->edges);
    cudaFree(value->targets);
    cudaFree(value->nodes);
    cudaFree(value->brains);
    delete value;
    return cuda_status(status);
#undef BGF_CONTEXT_TRY
}

extern "C" BGF_EXPORT int bgf_destroy_context(BgfContextHandle context)
{
    if (!context) return 0;
    GpuContext* value = static_cast<GpuContext*>(context);
    cudaSetDevice(value->device_index);
    cudaFree(value->edges);
    cudaFree(value->targets);
    cudaFree(value->nodes);
    cudaFree(value->brains);
    delete value;
    return 0;
}

extern "C" BGF_EXPORT int bgf_upload_brains(
    BgfContextHandle context,
    const BgfBrain* brains,
    int32_t brain_count)
{
    if (!context) return static_cast<int>(cudaErrorInvalidValue);
    GpuContext* value = static_cast<GpuContext*>(context);
    cudaSetDevice(value->device_index);
    int status = upload_array(value->brains, value->brain_capacity, brains, brain_count);
    if (status == 0) value->brain_count = brain_count;
    return status;
}

extern "C" BGF_EXPORT int bgf_upload_nodes(
    BgfContextHandle context,
    const BgfNode* nodes,
    int32_t node_count)
{
    if (!context) return static_cast<int>(cudaErrorInvalidValue);
    GpuContext* value = static_cast<GpuContext*>(context);
    cudaSetDevice(value->device_index);
    int status = upload_array(value->nodes, value->node_capacity, nodes, node_count);
    if (status == 0) value->node_count = node_count;
    return status;
}

extern "C" BGF_EXPORT int bgf_upload_topology(
    BgfContextHandle context,
    const BgfTarget* targets,
    int32_t target_count,
    const BgfEdge* edges,
    int32_t edge_count)
{
    if (!context) return static_cast<int>(cudaErrorInvalidValue);
    GpuContext* value = static_cast<GpuContext*>(context);
    cudaSetDevice(value->device_index);
    int status = upload_array(value->targets, value->target_capacity, targets, target_count);
    if (status != 0) return status;
    status = upload_array(value->edges, value->edge_capacity, edges, edge_count);
    if (status == 0) {
        value->target_count = target_count;
        value->edge_count = edge_count;
    }
    return status;
}

extern "C" BGF_EXPORT int bgf_step(BgfContextHandle context)
{
    if (!context) return static_cast<int>(cudaErrorInvalidValue);
    GpuContext* value = static_cast<GpuContext*>(context);
    if (value->brain_count <= 0) return 0;
    cudaSetDevice(value->device_index);
    constexpr int threads = 128;
    const int blocks = (value->brain_count + threads - 1) / threads;
    evaluate_brains_kernel<<<blocks, threads>>>(
        value->brains,
        value->brain_count,
        value->nodes,
        value->targets,
        value->edges);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return cuda_status(status);
    return cuda_status(cudaDeviceSynchronize());
}

extern "C" BGF_EXPORT int bgf_download_nodes(
    BgfContextHandle context,
    BgfNode* nodes,
    int32_t node_count)
{
    if (!context || node_count < 0 || !nodes) return static_cast<int>(cudaErrorInvalidValue);
    GpuContext* value = static_cast<GpuContext*>(context);
    if (node_count > value->node_count) return static_cast<int>(cudaErrorInvalidValue);
    cudaSetDevice(value->device_index);
    if (node_count == 0) return 0;
    return cuda_status(cudaMemcpy(
        nodes,
        value->nodes,
        sizeof(BgfNode) * static_cast<size_t>(node_count),
        cudaMemcpyDeviceToHost));
}

extern "C" BGF_EXPORT int bgf_create_vision_context(
    int32_t device_index,
    int32_t query_capacity,
    int32_t entity_capacity,
    BgfVisionContextHandle* context)
{
    if (!context || query_capacity <= 0 || entity_capacity < 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    *context = nullptr;
    cudaError_t status = cudaSetDevice(device_index);
    if (status != cudaSuccess) return cuda_status(status);

    GpuVisionContext* value = new GpuVisionContext{};
    value->device_index = device_index;
    value->query_capacity = query_capacity;
    value->entity_capacity = entity_capacity;

#define BGF_VISION_CONTEXT_TRY(call) do { status = (call); if (status != cudaSuccess) goto cleanup; } while (0)
    BGF_VISION_CONTEXT_TRY(allocate_if_needed(&value->queries, query_capacity));
    BGF_VISION_CONTEXT_TRY(allocate_if_needed(&value->entities, entity_capacity));
    BGF_VISION_CONTEXT_TRY(allocate_if_needed(&value->results, query_capacity));
    *context = value;
    return 0;

cleanup:
    cudaFree(value->results);
    cudaFree(value->entities);
    cudaFree(value->queries);
    delete value;
    return cuda_status(status);
#undef BGF_VISION_CONTEXT_TRY
}

extern "C" BGF_EXPORT int bgf_destroy_vision_context(BgfVisionContextHandle context)
{
    if (!context) return 0;
    GpuVisionContext* value = static_cast<GpuVisionContext*>(context);
    cudaSetDevice(value->device_index);
    cudaFree(value->results);
    cudaFree(value->entities);
    cudaFree(value->queries);
    delete value;
    return 0;
}

extern "C" BGF_EXPORT int bgf_evaluate_vision_context(
    BgfVisionContextHandle context,
    const BgfVisionQuery* queries,
    int32_t query_count,
    const BgfVisionEntity* entities,
    int32_t entity_count,
    BgfVisionResult* results)
{
    if (!context || !queries || query_count <= 0 || !results || entity_count < 0 ||
        (entity_count > 0 && !entities)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    GpuVisionContext* value = static_cast<GpuVisionContext*>(context);
    if (query_count > value->query_capacity || entity_count > value->entity_capacity) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    for (int32_t i = 0; i < query_count; ++i) {
        if (queries[i].entity_offset < 0 || queries[i].entity_count < 0 ||
            queries[i].entity_offset > entity_count - queries[i].entity_count) {
            return static_cast<int>(cudaErrorInvalidValue);
        }
    }

    cudaError_t cuda_result = cudaSetDevice(value->device_index);
    if (cuda_result != cudaSuccess) return cuda_status(cuda_result);
    int status = upload_array(value->queries, value->query_capacity, queries, query_count);
    if (status != 0) return status;
    status = upload_array(value->entities, value->entity_capacity, entities, entity_count);
    if (status != 0) return status;

    constexpr int threads = 128;
    const int blocks = (query_count + threads - 1) / threads;
    evaluate_vision_kernel<<<blocks, threads>>>(
        value->queries,
        query_count,
        value->entities,
        value->results);
    cuda_result = cudaGetLastError();
    if (cuda_result != cudaSuccess) return cuda_status(cuda_result);
    cuda_result = cudaDeviceSynchronize();
    if (cuda_result != cudaSuccess) return cuda_status(cuda_result);
    return cuda_status(cudaMemcpy(
        results,
        value->results,
        sizeof(BgfVisionResult) * static_cast<size_t>(query_count),
        cudaMemcpyDeviceToHost));
}

extern "C" BGF_EXPORT int bgf_create_pheromone_context(
    int32_t device_index,
    int32_t query_capacity,
    int32_t entity_capacity,
    BgfPheromoneContextHandle* context)
{
    if (!context || query_capacity <= 0 || entity_capacity < 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    *context = nullptr;
    cudaError_t status = cudaSetDevice(device_index);
    if (status != cudaSuccess) return cuda_status(status);

    GpuPheromoneContext* value = new GpuPheromoneContext{};
    value->device_index = device_index;
    value->query_capacity = query_capacity;
    value->entity_capacity = entity_capacity;

#define BGF_PHEROMONE_CONTEXT_TRY(call) do { status = (call); if (status != cudaSuccess) goto cleanup; } while (0)
    BGF_PHEROMONE_CONTEXT_TRY(allocate_if_needed(&value->queries, query_capacity));
    BGF_PHEROMONE_CONTEXT_TRY(allocate_if_needed(&value->entities, entity_capacity));
    BGF_PHEROMONE_CONTEXT_TRY(allocate_if_needed(&value->results, query_capacity));
    *context = value;
    return 0;

cleanup:
    cudaFree(value->results);
    cudaFree(value->entities);
    cudaFree(value->queries);
    delete value;
    return cuda_status(status);
#undef BGF_PHEROMONE_CONTEXT_TRY
}

extern "C" BGF_EXPORT int bgf_destroy_pheromone_context(BgfPheromoneContextHandle context)
{
    if (!context) return 0;
    GpuPheromoneContext* value = static_cast<GpuPheromoneContext*>(context);
    cudaSetDevice(value->device_index);
    cudaFree(value->results);
    cudaFree(value->entities);
    cudaFree(value->queries);
    delete value;
    return 0;
}

extern "C" BGF_EXPORT int bgf_evaluate_pheromones_context(
    BgfPheromoneContextHandle context,
    const BgfPheromoneQuery* queries,
    int32_t query_count,
    const BgfPheromoneEntity* entities,
    int32_t entity_count,
    BgfPheromoneResult* results)
{
    if (!context || !queries || query_count <= 0 || !results || entity_count < 0 ||
        (entity_count > 0 && !entities)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    GpuPheromoneContext* value = static_cast<GpuPheromoneContext*>(context);
    if (query_count > value->query_capacity || entity_count > value->entity_capacity) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t cuda_result = cudaSetDevice(value->device_index);
    if (cuda_result != cudaSuccess) return cuda_status(cuda_result);
    int status = upload_array(value->queries, value->query_capacity, queries, query_count);
    if (status != 0) return status;
    status = upload_array(value->entities, value->entity_capacity, entities, entity_count);
    if (status != 0) return status;

    constexpr int threads = 128;
    const int blocks = (query_count + threads - 1) / threads;
    evaluate_pheromones_kernel<<<blocks, threads>>>(
        value->queries,
        query_count,
        value->entities,
        entity_count,
        value->results);
    cuda_result = cudaGetLastError();
    if (cuda_result != cudaSuccess) return cuda_status(cuda_result);
    cuda_result = cudaDeviceSynchronize();
    if (cuda_result != cudaSuccess) return cuda_status(cuda_result);
    return cuda_status(cudaMemcpy(
        results,
        value->results,
        sizeof(BgfPheromoneResult) * static_cast<size_t>(query_count),
        cudaMemcpyDeviceToHost));
}

extern "C" BGF_EXPORT int bgf_evaluate_brains(
    const BgfBrain* brains,
    int32_t brain_count,
    BgfNode* nodes,
    int32_t node_count,
    const BgfTarget* targets,
    int32_t target_count,
    const BgfEdge* edges,
    int32_t edge_count)
{
    if (!brains || brain_count <= 0 || !nodes || node_count <= 0 ||
        !targets || target_count < 0 || !edges || edge_count < 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    BgfBrain* device_brains = nullptr;
    BgfNode* device_nodes = nullptr;
    BgfTarget* device_targets = nullptr;
    BgfEdge* device_edges = nullptr;
    cudaError_t status = cudaSuccess;

#define BGF_CUDA_TRY(call) do { status = (call); if (status != cudaSuccess) goto cleanup; } while (0)

    BGF_CUDA_TRY(cudaMalloc(&device_brains, sizeof(BgfBrain) * brain_count));
    BGF_CUDA_TRY(cudaMalloc(&device_nodes, sizeof(BgfNode) * node_count));
    if (target_count > 0) BGF_CUDA_TRY(cudaMalloc(&device_targets, sizeof(BgfTarget) * target_count));
    if (edge_count > 0) BGF_CUDA_TRY(cudaMalloc(&device_edges, sizeof(BgfEdge) * edge_count));

    BGF_CUDA_TRY(cudaMemcpy(device_brains, brains, sizeof(BgfBrain) * brain_count, cudaMemcpyHostToDevice));
    BGF_CUDA_TRY(cudaMemcpy(device_nodes, nodes, sizeof(BgfNode) * node_count, cudaMemcpyHostToDevice));
    if (target_count > 0) BGF_CUDA_TRY(cudaMemcpy(device_targets, targets, sizeof(BgfTarget) * target_count, cudaMemcpyHostToDevice));
    if (edge_count > 0) BGF_CUDA_TRY(cudaMemcpy(device_edges, edges, sizeof(BgfEdge) * edge_count, cudaMemcpyHostToDevice));

    {
        constexpr int threads = 128;
        const int blocks = (brain_count + threads - 1) / threads;
        evaluate_brains_kernel<<<blocks, threads>>>(
            device_brains, brain_count, device_nodes, device_targets, device_edges);
    }
    BGF_CUDA_TRY(cudaGetLastError());
    BGF_CUDA_TRY(cudaDeviceSynchronize());
    BGF_CUDA_TRY(cudaMemcpy(nodes, device_nodes, sizeof(BgfNode) * node_count, cudaMemcpyDeviceToHost));

cleanup:
    cudaFree(device_edges);
    cudaFree(device_targets);
    cudaFree(device_nodes);
    cudaFree(device_brains);
    return cuda_status(status);

#undef BGF_CUDA_TRY
}

extern "C" BGF_EXPORT int bgf_evaluate_vision(
    const BgfVisionQuery* queries,
    int32_t query_count,
    const BgfVisionEntity* entities,
    int32_t entity_count,
    BgfVisionResult* results)
{
    int32_t device_index = 0;
    cudaError_t status = cudaGetDevice(&device_index);
    if (status != cudaSuccess) return cuda_status(status);
    BgfVisionContextHandle context = nullptr;
    int result = bgf_create_vision_context(
        device_index,
        query_count,
        entity_count,
        &context);
    if (result != 0) return result;
    result = bgf_evaluate_vision_context(
        context,
        queries,
        query_count,
        entities,
        entity_count,
        results);
    bgf_destroy_vision_context(context);
    return result;
}
