#include "bgf_world.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace {

struct RenderRequest {
    BgfWorldHandle world = nullptr;
    int32_t buffer_index = 0;
    int32_t operation = 0;
    BgfWorldD3D11RenderConfig config{};
    // The running callback alone touches result/counts, then publishes state
    // with release ordering. Polling acquires before reading those values.
    std::atomic<int32_t> state{0};
    int32_t result = 0;
    int32_t bibites = 0;
    int32_t pellets = 0;
};

std::mutex requests_mutex;
std::unordered_map<int32_t, std::shared_ptr<RenderRequest>> requests;
uint32_t next_request_id = 1u;

std::shared_ptr<RenderRequest> find_request(int32_t id)
{
    std::lock_guard<std::mutex> guard(requests_mutex);
    const auto found = requests.find(id);
    return found == requests.end() ? nullptr : found->second;
}

// UnityRenderingEvent uses UNITY_INTERFACE_API (__stdcall on Windows). There
// is deliberately no managed callback: Unity's graphics thread must not enter
// Mono or wait for the scripting thread while it owns the D3D context.
void __stdcall render_event(int32_t id)
{
    std::shared_ptr<RenderRequest> request = find_request(id);
    if (!request) return; // A cancelled event can arrive after its ID retired.
    int32_t expected = 0;
    if (!request->state.compare_exchange_strong(expected, 1,
            std::memory_order_acq_rel)) return;
    switch (request->operation) {
    case 0:
        request->result = bgf_world_register_d3d11_render_buffer_set(
            request->world, request->buffer_index, &request->config);
        break;
    case 1:
        request->result = bgf_world_update_d3d11_render_buffer_set(
            request->world, request->buffer_index,
            &request->bibites, &request->pellets);
        break;
    case 2:
        request->result = bgf_world_unregister_d3d11_render_buffers(request->world);
        break;
    default:
        request->result = 1;
        break;
    }
    request->state.store(2, std::memory_order_release);
}

} // namespace

extern "C" BGF_EXPORT void* bgf_world_unity_render_event_func(void)
{
    return reinterpret_cast<void*>(&render_event);
}

extern "C" BGF_EXPORT int bgf_world_create_unity_render_request(
    BgfWorldHandle world, int32_t buffer_index, int32_t operation,
    const BgfWorldD3D11RenderConfig* config, int32_t* event_id)
try {
    if (event_id) *event_id = 0;
    if (!world || !event_id || operation < 0 || operation > 2 ||
        buffer_index < 0 || buffer_index >= 2 || (operation == 0 && !config)) return 1;
    auto request = std::make_shared<RenderRequest>();
    request->world = world;
    request->buffer_index = buffer_index;
    request->operation = operation;
    if (config) request->config = *config;
    std::lock_guard<std::mutex> guard(requests_mutex);
    // One request per live world is expected. The cap also bounds accidental
    // misuse instead of retaining unbounded GPU/world references.
    if (requests.size() >= 128u) return 2;
    int32_t id = 0;
    do {
        id = static_cast<int32_t>(next_request_id++ & 0x7fffffffu);
    } while (id == 0 || requests.find(id) != requests.end());
    requests.emplace(id, std::move(request));
    *event_id = id;
    return 0;
}
catch (...) {
    return 2;
}

extern "C" BGF_EXPORT int bgf_world_poll_unity_render_request(
    int32_t event_id, int32_t* state, int32_t* result,
    int32_t* bibite_count, int32_t* pellet_count)
{
    if (!state || !result || !bibite_count || !pellet_count) return 1;
    auto request = find_request(event_id);
    if (!request) return 1;
    *state = request->state.load(std::memory_order_acquire);
    *result = *bibite_count = *pellet_count = 0;
    if (*state == 2) {
        *result = request->result;
        *bibite_count = request->bibites;
        *pellet_count = request->pellets;
    }
    return 0;
}

extern "C" BGF_EXPORT int bgf_world_cancel_unity_render_request(int32_t event_id)
{
    auto request = find_request(event_id);
    if (!request) return 1;
    int32_t expected = 0;
    request->state.compare_exchange_strong(expected, 3, std::memory_order_acq_rel);
    return 0; // A running callback is never forcibly cancelled.
}

extern "C" BGF_EXPORT int bgf_world_release_unity_render_request(int32_t event_id)
{
    std::lock_guard<std::mutex> guard(requests_mutex);
    const auto found = requests.find(event_id);
    if (found == requests.end()) return 1;
    const int32_t state = found->second->state.load(std::memory_order_acquire);
    if (state != 2 && state != 3) return 1;
    requests.erase(found);
    return 0;
}
