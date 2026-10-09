#include "bgf_world.h"
#include <d3d11.h>
#include <dxgi.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <io.h>
#include <iterator>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>
#include <sys/stat.h>

namespace {

bool require(bool condition, const char* message)
{
    if (condition) return true;
    std::cerr << "FAILED: " << message << "\n";
    return false;
}

bool native_fixture_layout(const std::vector<uint8_t>& bytes, size_t& body, size_t& rows);

bool check(int status, const char* operation)
{
    if (status == 0) return true;
    std::cerr << "FAILED: " << operation << " returned CUDA error " << status << "\n";
    return false;
}

std::vector<uint8_t> read_test_file(const char* path)
{
    std::ifstream file(path, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>());
}

bool write_test_file(const char* path, const std::vector<uint8_t>& bytes)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    file.close();
    return !file.fail();
}

bool graphics_threading_regressions()
{
    // WARP is CPU software rendering: this tests Direct3D resource ownership
    // without touching either physical GPU or the user's running applications.
    for (UINT flags : {0u, static_cast<UINT>(D3D11_CREATE_DEVICE_SINGLETHREADED)}) {
        ID3D11Device* device = nullptr;
        ID3D11DeviceContext* context = nullptr;
        const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_0};
        HRESULT result = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP,
            nullptr, flags, levels, 1, D3D11_SDK_VERSION,
            &device, nullptr, &context);
        if (!require(SUCCEEDED(result), "create software D3D11 threading fixture")) return false;
        D3D11_BUFFER_DESC description{};
        description.ByteWidth = 9u * 9u * sizeof(float);
        description.Usage = D3D11_USAGE_DEFAULT;
        description.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        ID3D11Buffer* buffer = nullptr;
        result = device->CreateBuffer(&description, nullptr, &buffer);
        uint32_t actual_flags = 0;
        int32_t previously_protected = -1;
        bool success = require(SUCCEEDED(result), "create software render buffer");
        if (success) {
            const int status = bgf_world_prepare_d3d11_multithreading(
                buffer, &actual_flags, &previously_protected);
            if (flags != 0u) {
                success = require(status != 0 && (actual_flags & flags) == flags,
                    "SINGLETHREADED graphics devices must reject worker interop");
            } else {
                success = check(status, "enable shared D3D11 context protection") &&
                    check(bgf_world_prepare_d3d11_multithreading(
                        buffer, &actual_flags, &previously_protected),
                        "repeat shared D3D11 context protection") &&
                    require(previously_protected == 1,
                        "D3D11 multithread protection must remain enabled");
            }
        }
        if (buffer) buffer->Release();
        context->Release();
        device->Release();
        if (!success) return false;
    }
    return true;
}

bool render_event_mailbox_regressions(BgfWorldHandle world)
{
    using RenderEvent = void (__stdcall*)(int32_t);
    RenderEvent callback = reinterpret_cast<RenderEvent>(bgf_world_unity_render_event_func());
    if (!require(callback != nullptr, "Unity native render callback must exist")) return false;
    int32_t id = -1, state = -1, result = -1, bibites = -1, pellets = -1;
    if (!require(bgf_world_create_unity_render_request(nullptr, 0, 1, nullptr, &id) != 0 && id == 0,
            "render mailbox must reject a null world") ||
        !check(bgf_world_create_unity_render_request(world, 0, 2, nullptr, &id),
            "queue a render-thread cleanup") ||
        !check(bgf_world_poll_unity_render_request(id, &state, &result, &bibites, &pellets),
            "poll an unissued render event") ||
        !require(state == 0, "unissued render event must stay pending") ||
        !require(bgf_world_release_unity_render_request(id) != 0,
            "a pending render event must retain its world ownership")) return false;
    callback(id); // Empty-resource cleanup exercises completion without a GUI.
    callback(id); // Repeated delivery must not execute the operation twice.
    if (!check(bgf_world_cancel_unity_render_request(id),
            "late stop must leave a completed render event complete")) return false;
    if (!check(bgf_world_poll_unity_render_request(id, &state, &result, &bibites, &pellets),
            "poll completed render event") ||
        !require(state == 2 && result == 0, "completed render callback must publish its result") ||
        !check(bgf_world_release_unity_render_request(id), "release completed render event")) return false;
    callback(id); // A delayed event cannot follow a retired world pointer.
    if (!check(bgf_world_create_unity_render_request(world, 1, 1, nullptr, &id),
            "queue cancelled render update") ||
        !check(bgf_world_cancel_unity_render_request(id), "cancel unissued render update")) return false;
    callback(id);
    if (!check(bgf_world_poll_unity_render_request(id, &state, &result, &bibites, &pellets),
            "poll cancelled render update") ||
        !require(state == 3 && bibites == 0 && pellets == 0,
            "cancelled callback must not touch the world or expose stale counts") ||
        !check(bgf_world_release_unity_render_request(id), "release cancelled render update")) return false;
    callback(id);
    // Race stopping with event delivery. A request may either complete or be
    // cancelled before entry, but must never lose ownership while running.
    for (int attempt = 0; attempt < 32; ++attempt) {
        if (!check(bgf_world_create_unity_render_request(world, 0, 2, nullptr, &id),
                "queue racing render callback")) return false;
        std::thread render_thread([callback, id] { callback(id); });
        const int cancel_status = bgf_world_cancel_unity_render_request(id);
        render_thread.join();
        if (!check(cancel_status, "race stop with native render callback") ||
            !check(bgf_world_poll_unity_render_request(id, &state, &result, &bibites, &pellets),
                "poll raced render callback") ||
            !require((state == 2 && result == 0) || state == 3,
                "raced render request must finish or cancel safely") ||
            !check(bgf_world_release_unity_render_request(id), "release raced render callback")) return false;
        callback(id);
    }
    return true;
}

bool checkpoint_stability_regressions()
{
    BgfWorldConfig config{};
    bgf_world_default_config(&config);
    config.max_bibites = 4;
    config.initial_bibites = 1;
    config.pellet_count = 16;
    config.pheromone_grid_width = config.pheromone_grid_height = 8;
    BgfWorldHandle world = nullptr;
    if (!check(bgf_world_create(&config, &world), "checkpoint safety fixture")) return false;
    if (!render_event_mailbox_regressions(world)) {
        bgf_world_destroy(world);
        return false;
    }
    const char* path = "bgf_checkpoint_safety_test.bgfgpu";
    const char* corrupt_path = "bgf_checkpoint_corrupt_test.bgfgpu";
    const auto finish = [&](bool success) {
        _chmod(path, _S_IREAD | _S_IWRITE);
        std::remove(path);
        std::remove(corrupt_path);
        bgf_world_destroy(world);
        return success;
    };
    int32_t rendered_bibites = 7, rendered_pellets = 9;
    BgfWorldD3D11RenderConfig no_resources{};
    if (!require(bgf_world_register_d3d11_render_buffer_set(world, -1, &no_resources) != 0,
            "negative graphics surface indices must be rejected") ||
        !require(bgf_world_register_d3d11_render_buffer_set(world, 2, &no_resources) != 0,
            "out-of-range graphics surface indices must be rejected") ||
        !require(bgf_world_update_d3d11_render_buffer_set(world, 2,
                &rendered_bibites, &rendered_pellets) != 0 &&
            rendered_bibites == 0 && rendered_pellets == 0,
            "failed graphics update must not expose stale draw counts") ||
        !require(bgf_world_update_d3d11_render_buffer_set(world, 1,
                &rendered_bibites, &rendered_pellets) != 0,
            "an unregistered back buffer must not be mapped")) return finish(false);
    BgfWorldBibiteDetail before{}, after{};
    int32_t nodes = 0, synapses = 0, slot = -1;
    if (!check(bgf_world_get_bibite_detail(world, 0, &before,
            nullptr, 0, &nodes, nullptr, 0, &synapses), "initial slot identity") ||
        !check(bgf_world_kill_bibite(world, 0), "recycle selected slot") ||
        !check(bgf_world_spawn_bibite(world, 0.0f, 0.0f, 0.0f, &slot), "reuse selected slot") ||
        !check(bgf_world_get_bibite_detail(world, slot, &after,
            nullptr, 0, &nodes, nullptr, 0, &synapses), "replacement slot identity") ||
        !require(slot == 0 && before.reserved != 0 && after.reserved != 0 &&
            before.reserved != after.reserved,
            "recycled slots must have a new inspector identity")) return finish(false);
    const int32_t saved_identity=after.reserved;
    int32_t dead_slot = -1, dead_identity = 0;
    if (!check(bgf_world_spawn_bibite(world, 1.0f, 0.0f, 0.0f, &dead_slot),
            "allocate dead-slot identity fixture") ||
        !check(bgf_world_get_bibite_detail(world, dead_slot, &after,
            nullptr, 0, &nodes, nullptr, 0, &synapses), "read dead-slot incarnation"))
        return finish(false);
    dead_identity = after.reserved;
    if (!check(bgf_world_kill_bibite(world, dead_slot),
            "retain dead incarnation before checkpoint")) return finish(false);
    if (!check(bgf_world_save_checkpoint(world, path), "safe checkpoint initial save")) return finish(false);
    const std::vector<uint8_t> original = read_test_file(path);
    if (!require(original.size() > 256u, "checkpoint fixture must contain state")) return finish(false);

    // A failed replace must leave the previous save byte-for-byte unchanged.
    if (!require(_chmod(path, _S_IREAD) == 0, "make checkpoint read-only for failure test") ||
        !require(bgf_world_save_checkpoint(world, path) != 0,
            "read-only checkpoint replacement must report failure") ||
        !require(read_test_file(path) == original,
            "failed save must preserve the prior complete checkpoint")) return finish(false);
    if (!require(_chmod(path, _S_IREAD | _S_IWRITE) == 0, "restore checkpoint write access") ||
        !check(bgf_world_save_checkpoint(world, path), "checkpoint overwrite after failed save")) return finish(false);

    const auto rejects = [&](const std::vector<uint8_t>& bytes, const char* message,
            bool check_header = false) {
        if (!write_test_file(corrupt_path, bytes)) return false;
        BgfWorldConfig header{};
        if (check_header && !require(bgf_world_checkpoint_config(corrupt_path, 0, &header) != 0,
                "truncated checkpoints must fail before GPU allocation")) return false;
        BgfWorldHandle loaded = nullptr;
        const int status = bgf_world_load_checkpoint(corrupt_path, 0, &loaded);
        const bool rejected = require(status != 0 && loaded == nullptr, message);
        if (loaded) bgf_world_destroy(loaded);
        return rejected;
    };
    std::vector<uint8_t> corrupt = original;
    corrupt.resize(corrupt.size() - 3u);
    if (!rejects(corrupt, "truncated checkpoint must be rejected", true)) return finish(false);

    uint32_t header_bytes = 0;
    std::memcpy(&header_bytes, original.data() + 12u, sizeof(header_bytes));
    // The fixture has no template topologies. WorldCounters consists of 14
    // uint64 values, one int32 and trailing alignment (120 bytes, format v1/v2).
    size_t body=0,rows=config.max_bibites;
    if (!native_fixture_layout(original,body,rows)) return finish(false);
    const auto corrupt_int = [&](size_t offset, int32_t value, const char* message) {
        std::vector<uint8_t> changed = original;
        if (offset + sizeof(value) > changed.size()) return false;
        std::memcpy(changed.data() + offset, &value, sizeof(value));
        return rejects(changed, message);
    };
    if (!corrupt_int(body, 2, "checkpoint must reject transient/invalid alive flags") ||
        !corrupt_int(body + 292u * rows, 1,
            "checkpoint must reject out-of-range template topology IDs") ||
        !corrupt_int(body + 336u * rows + 8u * config.max_bibites + sizeof(int32_t), config.max_bibites + 1,
            "checkpoint must reject oversized free-slot queue counts") ||
        !corrupt_int(original.size() - sizeof(uint64_t) - 4u * config.max_bibites,
            0, "checkpoint must reject zero living incarnation IDs") ||
        !corrupt_int(body + 336u * rows + 4u * config.max_bibites + sizeof(int32_t),
            dead_slot, "checkpoint must reject duplicate free-slot queue entries")) return finish(false);
    corrupt = original;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::memcpy(corrupt.data() + body + 8u * rows, &nan, sizeof(nan));
    if (!rejects(corrupt, "checkpoint must reject non-finite living positions")) return finish(false);

    BgfWorldHandle restored = nullptr;
    if (!check(bgf_world_load_checkpoint(path, 0, &restored),
            "valid checkpoint must still load after corruption attempts")) return finish(false);
    bool restored_identity = check(bgf_world_get_bibite_detail(restored, 0, &after,
        nullptr, 0, &nodes, nullptr, 0, &synapses), "restored inspector identity") &&
        require(after.alive == 1 && after.reserved == saved_identity,
            "format 12 must preserve recycled inspector/grab identity across load");
    restored_identity = restored_identity &&
        check(bgf_world_spawn_bibite(restored, 1.0f, 0.0f, 0.0f, &slot),
            "reuse restored dead slot") &&
        check(bgf_world_get_bibite_detail(restored, slot, &after,
            nullptr, 0, &nodes, nullptr, 0, &synapses), "read restored dead-slot incarnation") &&
        require(slot == dead_slot && after.reserved == dead_identity + 1,
            "dead-slot incarnation counters must survive save/load and increment on reuse");
    bgf_world_destroy(restored);
    BgfWorldStepMetrics metrics{};
    if (!restored_identity || !check(bgf_world_step(world, 4, &metrics),
            "corrupt checkpoint rejection must not poison the CUDA context")) return finish(false);
    return finish(true);
}

// The binary-edit fixtures below have no imported topology table. Version 12
// includes grid dimensions and explicit compact live/imported slot lists.
bool native_fixture_layout(const std::vector<uint8_t>& bytes, size_t& body, size_t& rows)
{
    if (bytes.size()<16) return false;
    uint32_t version=0,header=0;
    std::memcpy(&version,bytes.data()+8,4);
    std::memcpy(&header,bytes.data()+12,4);
    if (header+120ull>bytes.size()) return false;
    body=header+120u;
    if (version>=12) {
        int32_t counts[2]{};
        if (header+16ull>bytes.size()) return false;
        std::memcpy(counts,bytes.data()+header+8,8);
        if (counts[0]<0 || counts[1]<0 || counts[1]>counts[0]) return false;
        rows=static_cast<size_t>(counts[0]);
        body+=16u+4u*(rows+static_cast<size_t>(counts[1]));
    }
    return body<=bytes.size();
}

bool evolving_brain_regressions()
{
    BgfWorldConfig config{};
    bgf_world_default_config(&config);
    config.max_bibites = 256;
    config.initial_bibites = 1;
    config.pellet_count = 32;
    config.spatial_grid_width = 32;
    config.pheromone_grid_width = 32;
    config.pheromone_grid_height = 32;
    config.seed = 0x1c0ffeeu;
    BgfWorldHandle world = nullptr;
    if (!check(bgf_world_create(&config, &world), "evolving-brain fixture")) return false;
    const char* path = "bgf_evolving_brain_test.bgfgpu";
    auto finish = [&](bool result) {
        if (world) bgf_world_destroy(world);
        std::remove(path);
        return result;
    };
    std::vector<BgfWorldBrainNodeState> nodes(256);
    std::vector<BgfWorldBrainSynapseState> synapses(1024);
    BgfWorldBibiteDetail parent{};
    int32_t node_count = 0, synapse_count = 0;
    if (!check(bgf_world_get_bibite_detail(world, 0, &parent,
            nodes.data(), static_cast<int32_t>(nodes.size()), &node_count,
            synapses.data(), static_cast<int32_t>(synapses.size()), &synapse_count),
            "starter brain detail") ||
        !require(parent.alive == 1 && parent.template_brain == 2 &&
                parent.brain_nodes == 73 && node_count == 73 &&
                parent.brain_synapses > 96 && synapse_count == parent.brain_synapses,
            "starter must have 34 inputs, two 12-node layers, 15 outputs, and sparse links") ||
        !require(nodes[34].type == 3 && nodes[46].type == 3 &&
                nodes[58].action == 0 && nodes[72].action == 14,
            "inspector must expose both hidden layers and 15 actions")) return finish(false);
    for (int32_t sensor = 0; sensor < 34; ++sensor)
        if (!require(nodes[sensor].sensor == sensor &&
                std::isfinite(nodes[sensor].last_output),
                "native starter exposes every finite stock sensor")) return finish(false);
    if (!require(nodes[0].last_output == 1.0f,
            "native Constant sensor has a live value")) return finish(false);
    for (int32_t index = 0; index < synapse_count; ++index) {
        const auto& synapse = synapses[index];
        const bool valid_edge =
            (synapse.node_in < 16 && synapse.node_out >= 58 && synapse.node_out < 64) ||
            (synapse.node_in < 34 && synapse.node_out >= 34 && synapse.node_out < 46) ||
            (synapse.node_in >= 34 && synapse.node_in < 46 &&
                synapse.node_out >= 46 && synapse.node_out < 58) ||
            (synapse.node_in >= 46 && synapse.node_in < 58 &&
                synapse.node_out >= 58 && synapse.node_out < 73);
        if (!require(valid_edge && std::isfinite(synapse.weight),
                "native synapse must connect adjacent layers or the direct skip path"))
            return finish(false);
    }
    const int32_t initial_synapses = parent.brain_synapses;
    const float parent_first_weight = synapses[0].weight;
    // Give the one test parent ample energy through a checkpoint fixture.
    // Normal world configuration deliberately starts below the birth threshold.
    if (!check(bgf_world_save_checkpoint(world, path),
            "save starter for deterministic birth test")) return finish(false);
    std::vector<uint8_t> boosted = read_test_file(path);
    uint32_t header_bytes = 0;
    if (boosted.size() < 16u) return finish(false);
    std::memcpy(&header_bytes, boosted.data() + 12u, sizeof(header_bytes));
    size_t body=0,rows=config.max_bibites;
    if (!native_fixture_layout(boosted,body,rows)) return finish(false);
    const size_t energy_offset = body+28u*rows;
    if (energy_offset + sizeof(float) > boosted.size()) return finish(false);
    const float boosted_energy = 100000.0f;
    std::memcpy(boosted.data() + energy_offset, &boosted_energy,
        sizeof(boosted_energy));
    if (!write_test_file(path, boosted)) return finish(false);
    bgf_world_destroy(world);
    world = nullptr;
    if (!check(bgf_world_load_checkpoint(path, config.device_index, &world),
            "load boosted birth-test fixture")) return finish(false);
    bool grew = false, changed_weight = false;
    int32_t selected_child = -1;
    for (int32_t birth = 0; birth < 128; ++birth) {
        int32_t child = -1;
        if (!check(bgf_world_force_reproduction(world, 0, &child),
                "structural-mutation birth") || child <= 0) return finish(false);
        BgfWorldBibiteDetail detail{};
        if (!check(bgf_world_get_bibite_detail(world, child, &detail,
                nullptr, 0, &node_count,
                synapses.data(), static_cast<int32_t>(synapses.size()), &synapse_count),
                "offspring brain detail") ||
            !require(detail.generation == 1 && detail.brain_nodes == 73 &&
                    synapse_count == detail.brain_synapses,
                "offspring must inherit the two-layer brain")) return finish(false);
        grew = grew || detail.brain_synapses > initial_synapses;
        changed_weight = changed_weight ||
            std::fabs(synapses[0].weight - parent_first_weight) > 0.0001f;
        if (selected_child < 0 && detail.brain_synapses != initial_synapses)
            selected_child = child;
    }
    if (!require(grew && changed_weight && selected_child > 0,
            "birth mutations must change weights and allow connection counts to increase"))
        return finish(false);
    BgfWorldBibiteDetail before{};
    std::vector<BgfWorldBrainSynapseState> before_synapses(1024);
    if (!check(bgf_world_get_bibite_detail(world, selected_child, &before,
            nullptr, 0, &node_count, before_synapses.data(),
            static_cast<int32_t>(before_synapses.size()), &synapse_count),
            "pre-save evolved brain detail") ||
        !check(bgf_world_save_checkpoint(world, path), "save evolved brain"))
        return finish(false);
    // Give the selected child energy after saving so it can demonstrate a
    // second generation without depending on the fixture's pellet positions.
    boosted = read_test_file(path);
    if (!native_fixture_layout(boosted,body,rows)) return finish(false);
    uint32_t saved_version=0;
    std::memcpy(&saved_version,boosted.data()+8,4);
    size_t child_row=static_cast<size_t>(selected_child);
    if (saved_version>=12) {
        std::memcpy(&header_bytes,boosted.data()+12,4);
        child_row=rows;
        for (size_t row=0;row<rows;++row) {
            int32_t slot=-1;
            std::memcpy(&slot,boosted.data()+header_bytes+16+row*4,4);
            if (slot==selected_child) { child_row=row;break; }
        }
        if (!require(child_row<rows,"selected child is in the compact live-slot table"))
            return finish(false);
    }
    const size_t child_energy_offset = body+28u*rows +
        sizeof(float) * child_row;
    if (child_energy_offset + sizeof(float) > boosted.size()) return finish(false);
    std::memcpy(boosted.data() + child_energy_offset, &boosted_energy,
        sizeof(boosted_energy));
    if (!write_test_file(path, boosted)) return finish(false);
    BgfWorldHandle restored = nullptr;
    if (!check(bgf_world_load_checkpoint(path, config.device_index, &restored),
            "load evolved brain")) return finish(false);
    BgfWorldBibiteDetail after{};
    std::vector<BgfWorldBrainSynapseState> after_synapses(1024);
    int32_t after_count = 0;
    const bool loaded = check(bgf_world_get_bibite_detail(restored, selected_child, &after,
        nullptr, 0, &node_count, after_synapses.data(),
        static_cast<int32_t>(after_synapses.size()), &after_count),
        "post-load evolved brain detail");
    if (!loaded || !require(after.energy==boosted_energy,
            "compact fixture must boost the selected public slot, not its row number") ||
        !require(before.brain_synapses == after.brain_synapses &&
            synapse_count == after_count &&
            std::memcmp(before_synapses.data(), after_synapses.data(),
                static_cast<size_t>(after_count) * sizeof(BgfWorldBrainSynapseState)) == 0,
            "evolved topology and weights must survive a checkpoint")) {
        bgf_world_destroy(restored);
        return finish(false);
    }
    int32_t grandchild_slot = -1;
    BgfWorldBibiteDetail grandchild{};
    const bool second_generation =
        check(bgf_world_force_reproduction(restored, selected_child,
            &grandchild_slot), "second-generation birth") &&
        check(bgf_world_get_bibite_detail(restored, grandchild_slot, &grandchild,
            nullptr, 0, &node_count, nullptr, 0, &after_count),
            "second-generation brain detail") &&
        require(grandchild.generation == 2 && grandchild.brain_nodes == 73 &&
            std::abs(grandchild.brain_synapses - after.brain_synapses) <= 2,
            "evolved topology must remain inheritable across generations");
    bgf_world_destroy(restored);
    if (!second_generation) return finish(false);
    return finish(true);
}

bool dormant_hidden_link_regressions()
{
    BgfWorldConfig config{};
    bgf_world_default_config(&config);
    config.max_bibites = 64;
    config.initial_bibites = 1;
    config.pellet_count = 32;
    config.spatial_grid_width = 32;
    config.pheromone_grid_width = 32;
    config.pheromone_grid_height = 32;
    BgfWorldHandle world = nullptr;
    if (!check(bgf_world_create(&config, &world), "dormant-link fixture")) return false;
    const char* path = "bgf_dormant_brain_test.bgfgpu";
    const auto finish = [&](bool success) {
        if (world) bgf_world_destroy(world);
        std::remove(path);
        return success;
    };
    BgfWorldBibiteDetail detail{};
    std::vector<BgfWorldBrainSynapseState> synapses(1024);
    int32_t nodes = 0, count = 0;
    if (!check(bgf_world_get_bibite_detail(world, 0, &detail,
            nullptr, 0, &nodes, synapses.data(),
            static_cast<int32_t>(synapses.size()), &count),
            "read active hidden link") || count <= 96) return finish(false);
    const int32_t initial_count = count;
    int32_t edge = -1;
    float original = 0.0f;
    for (int32_t i = 96; i < count; ++i) {
        const auto& link = synapses[i];
        if (std::fabs(link.weight) < 0.0001f) continue;
        if (link.node_in < 16 && link.node_out >= 34 && link.node_out < 46)
            edge = (link.node_out - 34) * 16 + link.node_in;
        else if (link.node_in >= 34 && link.node_in < 46 &&
                link.node_out >= 46 && link.node_out < 58)
            edge = 192 + (link.node_out - 46) * 12 + link.node_in - 34;
        else if (link.node_in >= 46 && link.node_in < 58 &&
                link.node_out >= 58 && link.node_out < 64)
            edge = 336 + (link.node_out - 58) * 12 + link.node_in - 46;
        if (edge >= 0) {
            original = link.weight;
            break;
        }
    }
    if (!require(edge >= 0, "find a nonzero hidden link")) return finish(false);
    int32_t active = 0;
    float effective = 0.0f;
    if (!check(bgf_world_get_hidden_link(world, 0, edge, &active, &effective),
            "read effective hidden weight") ||
        !require(active == 1 && effective == original,
            "active link exposes its inherited FP16 weight") ||
        !check(bgf_world_set_hidden_link_enabled(world, 0, edge, 0),
            "disable hidden link") ||
        !check(bgf_world_get_hidden_link(world, 0, edge, &active, &effective),
            "read disabled hidden link") ||
        !require(active == 0 && effective == 0.0f,
            "disabled link must have exactly zero effective weight"))
        return finish(false);
    if (!check(bgf_world_get_bibite_detail(world, 0, &detail,
            nullptr, 0, &nodes, synapses.data(),
            static_cast<int32_t>(synapses.size()), &count),
            "read reduced connection count") ||
        !require(count == detail.brain_synapses &&
                count == initial_count - 1,
            "disabled link must leave the active graph")) return finish(false);
    int32_t extension_edge = -1;
    float extension_weight = 0.0f;
    for (int32_t i = 96; i < count; ++i) {
        const auto& link = synapses[i];
        if (link.node_in >= 16 && link.node_in < 34 &&
            link.node_out >= 34 && link.node_out < 46)
            extension_edge = 408 + (link.node_out - 34) * 18 + link.node_in - 16;
        else if (link.node_in >= 46 && link.node_in < 58 &&
            link.node_out >= 64 && link.node_out < 73)
            extension_edge = 408 + 216 + (link.node_out - 64) * 12 + link.node_in - 46;
        if (extension_edge >= 0 && std::fabs(link.weight) > 0.0001f) {
            extension_weight = link.weight;
            break;
        }
    }
    if (!require(extension_edge >= 408 && extension_edge < 732,
            "find an active stock I/O extension") ||
        !check(bgf_world_set_hidden_link_enabled(world, 0, extension_edge, 0),
            "disable stock I/O extension") ||
        !check(bgf_world_get_hidden_link(world, 0, extension_edge,
            &active, &effective), "read disabled stock I/O extension") ||
        !require(active == 0 && effective == 0.0f,
            "disabled extension has zero effective weight")) return finish(false);
    if (!check(bgf_world_save_checkpoint(world, path),
            "save dormant hidden weight")) return finish(false);
    std::vector<uint8_t> boosted = read_test_file(path);
    uint32_t header_bytes = 0;
    if (boosted.size() < 16u) return finish(false);
    std::memcpy(&header_bytes, boosted.data() + 12u, sizeof(header_bytes));
    size_t body=0,rows=config.max_bibites;
    if (!native_fixture_layout(boosted,body,rows)) return finish(false);
    const size_t energy_offset = body+28u*rows;
    if (energy_offset + sizeof(float) > boosted.size()) return finish(false);
    const float boosted_energy = 100000.0f;
    std::memcpy(boosted.data() + energy_offset, &boosted_energy,
        sizeof(boosted_energy));
    if (!write_test_file(path, boosted)) return finish(false);
    bgf_world_destroy(world);
    world = nullptr;
    if (!check(bgf_world_load_checkpoint(path, config.device_index, &world),
            "reload dormant hidden weight") ||
        !check(bgf_world_set_hidden_link_enabled(world, 0, edge, 1),
            "restore hidden link") ||
        !check(bgf_world_get_hidden_link(world, 0, edge, &active, &effective),
            "read restored hidden link") ||
        !require(active == 1 && effective == original,
            "reconnection must undo the zero using the saved weight") ||
        !check(bgf_world_set_hidden_link_enabled(world, 0, extension_edge, 1),
            "restore stock I/O extension") ||
        !check(bgf_world_get_hidden_link(world, 0, extension_edge,
            &active, &effective), "read restored stock I/O extension") ||
        !require(active == 1 && effective == extension_weight,
            "stock I/O extension restores its saved FP16 weight") ||
        !check(bgf_world_set_hidden_link_enabled(world, 0, edge, 0),
            "disable link before inheritance") ||
        !check(bgf_world_set_hidden_link_enabled(world, 0, extension_edge, 0),
            "disable extension before inheritance")) return finish(false);
    int32_t child = -1;
    if (!check(bgf_world_force_reproduction(world, 0, &child),
            "inherit dormant hidden link") ||
        !require(child > 0, "dormant-link offspring exists") ||
        !check(bgf_world_set_hidden_link_enabled(world, child, edge, 1),
            "reconnect inherited hidden link") ||
        !check(bgf_world_get_hidden_link(world, child, edge,
            &active, &effective), "read inherited dormant weight") ||
        !require(active == 1 && effective == original,
            "offspring must restore the parent's dormant weight") ||
        !check(bgf_world_set_hidden_link_enabled(world, child, extension_edge, 1),
            "restore inherited stock I/O extension") ||
        !check(bgf_world_get_hidden_link(world, child, extension_edge,
            &active, &effective), "read inherited stock I/O extension") ||
        !require(active == 1 && effective == extension_weight,
            "offspring restores inherited dormant stock I/O weight"))
        return finish(false);

    // A v4 checkpoint has zero-filled untouched edges. Its loader must mark
    // those as never initialised, so a newly grown link does not stay at zero.
    int32_t untouched = -1;
    for (int32_t candidate = 407; candidate >= 0; --candidate) {
        if (!check(bgf_world_get_hidden_link(world, 0, candidate,
                &active, &effective), "find untouched legacy link"))
            return finish(false);
        if (active == 0) {
            untouched = candidate;
            break;
        }
    }
    if (!require(untouched >= 0, "legacy fixture has an inactive link") ||
        !check(bgf_world_save_checkpoint_legacy_v11(world, path),
            "save v6 for synthetic v4 migration")) return finish(false);
    std::vector<uint8_t> legacy = read_test_file(path);
    const size_t bibites = static_cast<size_t>(config.max_bibites);
    const size_t version11_bytes = sizeof(BgfWorldDigestionSettings);
    if (legacy.size() <= version11_bytes + sizeof(uint64_t)) return finish(false);
    legacy.erase(legacy.end() - sizeof(uint64_t) - version11_bytes,
        legacy.end() - sizeof(uint64_t));
    const size_t version10_bytes = bibites *
        (sizeof(uint8_t) + 11u * sizeof(uint32_t) +
         324u * sizeof(uint16_t) + sizeof(uint16_t));
    if (legacy.size() <= version10_bytes + sizeof(uint64_t)) return finish(false);
    legacy.erase(legacy.end() - sizeof(uint64_t) - version10_bytes,
        legacy.end() - sizeof(uint64_t));
    const size_t weights_bytes = bibites * 408u * sizeof(uint16_t);
    const size_t extension_bytes =
        static_cast<size_t>(config.pellet_count) * sizeof(int32_t) + sizeof(float) +
        bibites * (4u * sizeof(float) + 2u * 4u * sizeof(float)) +
        4u * sizeof(float) +
        bibites * 40u +
        static_cast<size_t>(config.pellet_count) * sizeof(int32_t) +
        static_cast<size_t>(config.pheromone_grid_width) *
            config.pheromone_grid_height * 4u * 3u * sizeof(float);
    const size_t trailing_bytes = sizeof(uint64_t) + extension_bytes +
        bibites * (1u + 24u * 3u) * sizeof(uint16_t) + weights_bytes;
    if (legacy.size() <= trailing_bytes) return finish(false);
    const size_t weight_offset = legacy.size() - trailing_bytes +
        (static_cast<size_t>(untouched / 2) * bibites * 2u +
            static_cast<size_t>(untouched & 1)) * sizeof(uint16_t);
    if (weight_offset + sizeof(uint16_t) > legacy.size()) return finish(false);
    const uint16_t old_uninitialised_zero = 0;
    std::memcpy(legacy.data() + weight_offset,
        &old_uninitialised_zero, sizeof(old_uninitialised_zero));
    legacy.erase(legacy.end() - sizeof(uint64_t) - extension_bytes,
        legacy.end() - sizeof(uint64_t));
    const uint32_t version4 = 4u;
    std::memcpy(legacy.data() + 8u, &version4, sizeof(version4));
    if (!write_test_file(path, legacy)) return finish(false);
    bgf_world_destroy(world);
    world = nullptr;
    if (!check(bgf_world_load_checkpoint(path, config.device_index, &world),
            "load old v4 hidden-link storage") ||
        !check(bgf_world_set_hidden_link_enabled(world, 0, untouched, 1),
            "grow untouched v4 link") ||
        !check(bgf_world_get_hidden_link(world, 0, untouched,
            &active, &effective), "read migrated v4 link") ||
        !require(active == 1 && std::isfinite(effective) && effective != 0.0f,
            "v4 untouched zero must become a fresh initialized weight"))
        return finish(false);
    return finish(true);
}

bool food_zone_regressions()
{
    BgfWorldConfig config{};
    bgf_world_default_config(&config);
    config.max_bibites = 64;
    config.initial_bibites = 32;
    config.pellet_count = 256;
    config.world_half_extent = 100.0f;
    config.spatial_grid_width = 32;
    config.pheromone_grid_width = 32;
    config.pheromone_grid_height = 32;
    BgfWorldHandle world = nullptr;
    if (!check(bgf_world_create(&config, &world), "food-zone fixture")) return false;
    const char* save_path = "bgf_zone_size_test.bgfgpu";
    const auto finish = [&](bool success) {
        std::remove(save_path);
        bgf_world_destroy(world);
        return success;
    };
    const BgfWorldFoodZone zones[2] = {
        {5, -35.0f, -10.0f, 0.0f, 0.0f, 10.0f, 12.0f, 1.0f, 1.0f, 2.0f},
        {4, 35.0f, 10.0f, 18.0f, 0.5f, 0.0f, 0.0f, 1.0f, 1.0f, 0.5f}};
    if (!check(bgf_world_set_food_zones(world, zones, 2, 1),
            "apply plant zones") ||
        !require(bgf_world_set_food_zones(world, zones, 65, 1) != 0,
            "reject more than 64 food zones")) return finish(false);

    std::vector<BgfWorldBibite> starters(64);
    int32_t starters_written = 0;
    if (!check(bgf_world_download_bibites(world, starters.data(),
            static_cast<int32_t>(starters.size()), &starters_written),
            "download zoned starter population") ||
        !require(starters_written == config.initial_bibites,
            "retain all starter Bibites after zone projection"))
        return finish(false);
    for (int32_t i = 0; i < starters_written; ++i) {
        const float x = starters[i].position_x;
        const float y = starters[i].position_y;
        const bool rectangle = std::abs(x + 35.0f) <= 10.001f &&
            std::abs(y + 10.0f) <= 12.001f;
        const float ring_distance = std::hypot(x - 35.0f, y - 10.0f);
        const bool ring = ring_distance >= 8.999f &&
            ring_distance <= 18.001f;
        if (!require(rectangle || ring,
                "starter Bibite must begin inside a fertile zone"))
            return finish(false);
    }

    std::vector<BgfWorldPellet> pellets(256);
    int32_t written = 0;
    if (!check(bgf_world_download_pellets(world, pellets.data(),
            static_cast<int32_t>(pellets.size()), &written),
            "download zoned pellets") ||
        !require(written == 256, "zone setup retains the food target"))
        return finish(false);
    int32_t rectangle_count = 0;
    int32_t ring_count = 0;
    for (int32_t i = 0; i < written; ++i) {
        const float x = pellets[i].position_x;
        const float y = pellets[i].position_y;
        const bool rectangle = std::abs(x + 35.0f) <= 10.001f &&
            std::abs(y + 10.0f) <= 12.001f;
        const float ring_distance = std::hypot(x - 35.0f, y - 10.0f);
        const bool ring = ring_distance >= 8.999f &&
            ring_distance <= 18.001f;
        if (!require(rectangle || ring,
                "seeded plant pellet must belong to a fertile zone"))
            return finish(false);
        if (!require(std::abs(pellets[i].energy - config.pellet_energy *
                (rectangle ? 2.0f : 0.5f)) < 0.001f,
                "zone pellet size must change actual food energy"))
            return finish(false);
        rectangle_count += rectangle ? 1 : 0;
        ring_count += ring ? 1 : 0;
    }
    if (!require(rectangle_count > 0 && ring_count > 0,
            "both weighted plant zones must receive pellets"))
        return finish(false);

    const BgfWorldFoodZone moved_zone =
        {0, 0.0f, 0.0f, 7.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 3.0f};
    if (!check(bgf_world_set_food_zones(world, &moved_zone, 1, 1),
            "reseed after zone change") ||
        !check(bgf_world_download_pellets(world, pellets.data(),
            static_cast<int32_t>(pellets.size()), &written),
            "download reseeded pellets")) return finish(false);
    for (int32_t i = 0; i < written; ++i) {
        if (!require(std::hypot(pellets[i].position_x,
                pellets[i].position_y) <= 7.001f,
                "reseeded pellet must be in the new zone") ||
            !require(std::abs(pellets[i].energy - config.pellet_energy * 3.0f) <
                0.001f, "reseeded pellet must adopt the changed pellet size"))
            return finish(false);
    }
    float linear_drag = -1.0f;
    if (!require(bgf_world_set_linear_drag(world, -0.1f) != 0,
            "negative drag must be rejected") ||
        !check(bgf_world_set_linear_drag(world, 0.0f),
            "zero drag must be accepted") ||
        !check(bgf_world_get_linear_drag(world, &linear_drag),
            "read physical drag") ||
        !require(linear_drag == 0.0f, "zero drag must remain frictionless") ||
        !check(bgf_world_save_checkpoint(world, save_path),
            "save changed pellet sizes and drag")) return finish(false);
    BgfWorldHandle restored = nullptr;
    if (!check(bgf_world_load_checkpoint(save_path, config.device_index,
            &restored), "restore pellet sizes and drag")) return finish(false);
    bool restored_ok = check(bgf_world_get_linear_drag(restored,
            &linear_drag), "read restored physical drag") &&
        require(linear_drag == 0.0f,
            "physical drag must survive native checkpoint") &&
        check(bgf_world_download_pellets(restored, pellets.data(),
            static_cast<int32_t>(pellets.size()), &written),
            "read restored pellet sizes");
    for (int32_t i = 0; restored_ok && i < written; ++i)
        restored_ok = require(std::abs(pellets[i].energy -
            config.pellet_energy * 3.0f) < 0.001f,
            "zone pellet size must survive native checkpoint");
    bgf_world_destroy(restored);
    if (!restored_ok) return finish(false);
    return finish(true);
}

bool combat_and_meat_regressions()
{
    BgfWorldConfig config{};
    bgf_world_default_config(&config);
    config.max_bibites = 32;
    config.initial_bibites = 1;
    config.pellet_count = 64;
    config.world_half_extent = 50.0f;
    config.spatial_grid_width = 16;
    config.pheromone_grid_width = 16;
    config.pheromone_grid_height = 16;
    config.diagnostic_mask = 1u; // Isolate bites from contact repulsion.
    BgfWorldHandle world = nullptr;
    if (!check(bgf_world_create(&config, &world), "combat fixture")) return false;
    const char* checkpoint_path = "bgf_combat_meat_test.bgfgpu";
    const auto finish = [&](bool success) {
        std::remove(checkpoint_path);
        bgf_world_destroy(world);
        return success;
    };
    if (!check(bgf_world_kill_bibite(world, 0),
            "remove unrelated founder") ||
        !check(bgf_world_set_food_settings(world, 32, 1.0f,
            config.pellet_energy), "reserve pellet slots for meat") ||
        !check(bgf_world_set_combat_settings(world, 0.0f, 50.0f,
            0.0f, 80.0f), "disable bite damage")) return finish(false);

    BgfWorldTemplateSpawn spawn{};
    spawn.heading = 0.0f;
    spawn.energy = 100.0f;
    spawn.size = 1.0f;
    spawn.maximum_speed = 1.0f;
    spawn.turn_speed = 1.0f;
    spawn.metabolism = 0.001f;
    spawn.lifespan = 1000.0f;
    spawn.diet = 1.0f;
    const BgfWorldBrainNode inert[] = {{2, -1, 0, 0.0f}};
    const BgfWorldBrainNode attack[] = {{2, -1, 14, 1.0f}};
    int32_t victim = -1;
    int32_t predator = -1;
    spawn.position_y = 1.0f;
    if (!check(bgf_world_spawn_template_bibite(world, &spawn, inert, 1,
            nullptr, 0, &victim), "spawn passive victim")) return finish(false);
    spawn.position_y = 0.0f;
    if (!check(bgf_world_spawn_template_bibite(world, &spawn, attack, 1,
            nullptr, 0, &predator), "spawn predator")) return finish(false);
    BgfWorldStepMetrics metrics{};
    if (!check(bgf_world_step(world, 40, &metrics),
            "step with bite damage disabled")) return finish(false);
    BgfWorldBibiteDetail detail{};
    int32_t node_count = 0, synapse_count = 0;
    if (!check(bgf_world_get_bibite_detail(world, victim, &detail,
            nullptr, 0, &node_count, nullptr, 0, &synapse_count),
            "inspect unhurt victim") ||
        !require(detail.alive == 1 && std::fabs(detail.health - 100.0f) < 0.01f,
            "zero biting damage must leave health unchanged")) return finish(false);
    if (!check(bgf_world_set_combat_settings(world, 0.0f, 50.0f,
            5.0f, 80.0f), "enable bite damage") ||
        !check(bgf_world_step(world, 120, &metrics),
            "step predator and victim")) return finish(false);
    BgfWorldStats stats{};
    std::vector<BgfWorldPellet> pellets(64);
    int32_t written = 0;
    if (!check(bgf_world_get_stats(world, &stats), "inspect combat stats") ||
        !require(stats.deaths >= 1 && stats.active_meat >= 1 &&
                stats.meat_energy > 0.0f,
            "lethal bite must create meat with energy") ||
        !check(bgf_world_download_pellets(world, pellets.data(),
            static_cast<int32_t>(pellets.size()), &written),
            "download plant and meat pellets")) return finish(false);
    bool found_meat = false;
    for (int32_t i = 0; i < written; ++i)
        found_meat |= pellets[i].material == 1 && pellets[i].energy > 0.0f;
    const BgfWorldDigestionSettings digestion_settings{
        1.5f, 0.5f, -0.1f, 0.9f, 0.1f, 1.0f};
    if (!require(found_meat, "meat material must reach presentation") ||
        !check(bgf_world_set_digestion_settings(world, &digestion_settings),
            "apply live stock diet settings") ||
        !check(bgf_world_save_checkpoint(world, checkpoint_path),
            "save meat and combat checkpoint")) return finish(false);
    BgfWorldHandle restored = nullptr;
    if (!check(bgf_world_load_checkpoint(checkpoint_path,
            config.device_index, &restored),
            "reload meat and combat checkpoint")) return finish(false);
    bgf_world_destroy(world);
    world = restored;
    BgfWorldDigestionSettings restored_digestion{};
    if (!check(bgf_world_get_digestion_settings(world, &restored_digestion),
            "read restored diet settings") ||
        !require(std::fabs(restored_digestion.plant_affinity_power - 1.5f) < 0.0001f &&
                std::fabs(restored_digestion.meat_affinity_power - 0.5f) < 0.0001f &&
                std::fabs(restored_digestion.plant_min_efficiency + 0.1f) < 0.0001f &&
                std::fabs(restored_digestion.plant_max_efficiency - 0.9f) < 0.0001f &&
                std::fabs(restored_digestion.meat_min_efficiency - 0.1f) < 0.0001f &&
                std::fabs(restored_digestion.meat_max_efficiency - 1.0f) < 0.0001f,
            "diet settings must survive checkpoint")) return finish(false);
    if (!check(bgf_world_get_stats(world, &stats),
            "inspect restored meat") ||
        !require(stats.active_meat >= 1 && stats.meat_energy > 0.0f,
            "meat state must survive checkpoint") ||
        !check(bgf_world_get_bibite_detail(world, predator, &detail,
            nullptr, 0, &node_count, nullptr, 0, &synapse_count),
            "inspect restored predator") ||
        !require(detail.alive == 1 && detail.diet == 1.0f &&
                detail.attack_output > 0.0f,
            "diet and attack state must survive checkpoint") ||
        !check(bgf_world_kill_bibite(world, predator),
            "remove predator before feeding test") ||
        !check(bgf_world_set_food_settings(world, 0, 1.0f,
            config.pellet_energy), "isolate meat food")) return finish(false);

    const float available_meat = stats.meat_energy;
    const BgfWorldBrainNode feed[] = {
        {0, 13, -1, 0.0f}, {0, 16, -1, 0.0f},
        {2, -1, 8, 1.0f}, {2, -1, 9, 1.0f}};
    const BgfWorldBrainSynapse feed_links[] = {
        {0, 2, 0.0f}, {1, 2, 0.0f}};
    int32_t scavenger = -1;
    spawn.position_y = 1.0f;
    if (!check(bgf_world_spawn_template_bibite(world, &spawn, feed, 4,
            feed_links, 2, &scavenger), "spawn meat-eating scavenger") ||
        !check(bgf_world_step(world, 80, &metrics),
            "digest meat over time") ||
        !check(bgf_world_get_stats(world, &stats),
            "inspect meat consumption") ||
        !check(bgf_world_get_bibite_detail(world, scavenger, &detail,
            nullptr, 0, &node_count, nullptr, 0, &synapse_count),
            "inspect fed scavenger") ||
        !require(detail.alive == 1 && detail.energy > spawn.energy &&
                stats.meat_energy < available_meat,
            "meat diet must digest pellet energy into the scavenger"))
        return finish(false);
    BgfWorldBrainNodeState sensed_nodes[4]{};
    if (!check(bgf_world_get_bibite_detail(world, scavenger, &detail,
            sensed_nodes, 4, &node_count, nullptr, 0, &synapse_count),
            "inspect plant and meat sensors") ||
        !require(node_count == 4 && sensed_nodes[0].last_output == 0.0f &&
                sensed_nodes[1].last_output > 0.0f,
            "stock sensors must distinguish meat from absent plants"))
        return finish(false);
    return finish(true);
}

bool stock_brain_io_regressions()
{
    BgfWorldConfig config{};
    bgf_world_default_config(&config);
    config.max_bibites = 16;
    config.initial_bibites = 1;
    config.pellet_count = 32;
    config.world_half_extent = 50.0f;
    config.spatial_grid_width = 16;
    config.pheromone_grid_width = 16;
    config.pheromone_grid_height = 16;
    config.diagnostic_mask = 1u;
    BgfWorldHandle world = nullptr;
    if (!check(bgf_world_create(&config, &world), "stock brain I/O fixture"))
        return false;
    const char* path = "bgf_stock_brain_io_test.bgfgpu";
    const auto finish = [&](bool success) {
        std::remove(path);
        bgf_world_destroy(world);
        return success;
    };
    if (!check(bgf_world_kill_bibite(world, 0), "clear stock I/O founder") ||
        !check(bgf_world_set_food_settings(world, 0, 1.0f,
            config.pellet_energy), "remove competing food targets"))
        return finish(false);

    BgfWorldTemplateSpawn spawn{};
    spawn.energy = 500.0f;
    spawn.size = 0.9f;
    spawn.adult_size = 1.0f;
    spawn.maximum_speed = 1.0f;
    spawn.turn_speed = 1.0f;
    spawn.metabolism = 0.001f;
    spawn.lifespan = 1000.0f;
    spawn.lay_period = 2.0f;
    spawn.womb_capacity = 2.0f;
    spawn.clock_period = 1.0f;
    const BgfWorldBrainNode passive[] = {{2, -1, 0, 0.0f}};
    spawn.position_y = 1.0f;
    int32_t neighbour = -1;
    if (!check(bgf_world_spawn_template_bibite(world, &spawn, passive, 1,
            nullptr, 0, &neighbour), "spawn brain I/O neighbour"))
        return finish(false);

    // Every stock sensor must execute in the imported topology, and every
    // action must have a separately addressable node.
    std::vector<BgfWorldBrainNode> nodes;
    std::vector<BgfWorldBrainSynapse> synapses;
    for (int32_t sensor = 1; sensor <= 33; ++sensor)
        nodes.push_back({0, sensor, -1, 0.0f});
    nodes.push_back({0, 0, -1, 0.0f});
    const int32_t hidden = static_cast<int32_t>(nodes.size());
    nodes.push_back({2, -1, -1, 0.0f});
    for (int32_t sensor = 0; sensor < 34; ++sensor)
        synapses.push_back({sensor, hidden, 0.0f});
    const float drives[15] = {0.0f, 0.0f, 0.1f, 0.1f, 0.1f,
        0.0f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f};
    for (int32_t action = 0; action < 15; ++action)
        nodes.push_back({2, -1, action, drives[action]});
    spawn.position_y = 0.0f;
    int32_t actor = -1;
    if (!check(bgf_world_spawn_template_bibite(world, &spawn,
            nodes.data(), static_cast<int32_t>(nodes.size()),
            synapses.data(), static_cast<int32_t>(synapses.size()), &actor),
            "spawn full stock I/O brain")) return finish(false);
    BgfWorldStepMetrics metrics{};
    BgfWorldBibiteDetail detail{};
    std::vector<BgfWorldBrainNodeState> values(nodes.size());
    std::vector<BgfWorldBrainSynapseState> edges(synapses.size());
    int32_t node_count = 0, synapse_count = 0;
    if (!check(bgf_world_step(world, 4, &metrics), "evaluate all stock I/O nodes") ||
        !check(bgf_world_get_bibite_detail(world, actor, &detail,
            values.data(), static_cast<int32_t>(values.size()), &node_count,
            edges.data(), static_cast<int32_t>(edges.size()), &synapse_count),
            "inspect stock I/O outputs"))
        return finish(false);
    if (
        !require(node_count == static_cast<int32_t>(nodes.size()) &&
                synapse_count == 34 && detail.held_count == 1 &&
                values[6].last_output > 0.5f &&
                std::fabs(detail.grab_output - 1.0f) < 0.01f &&
                std::fabs(detail.herding_output - 0.5f) < 0.01f &&
                std::fabs(detail.egg_production_output - 1.0f) < 0.01f &&
                std::fabs(detail.growth_output - 1.0f) < 0.01f &&
                std::fabs(detail.clock_reset_output - 1.0f) < 0.01f,
            "all stock I/O nodes must execute and grab a nearby Bibite"))
        return finish(false);
    BgfWorldHandle restored = nullptr;
    if (!check(bgf_world_save_checkpoint(world, path), "save active grab") ||
        !check(bgf_world_load_checkpoint(path, config.device_index, &restored),
            "reload active grab")) return finish(false);
    bgf_world_destroy(world);
    world = restored;
    restored = nullptr;
    if (!check(bgf_world_get_bibite_detail(world, actor, &detail,
            nullptr, 0, &node_count, nullptr, 0, &synapse_count),
            "inspect reloaded grab") ||
        !require(detail.held_count == 1,
            "active grabbing state must survive checkpoint") ||
        !check(bgf_world_kill_bibite(world, neighbour), "remove held neighbour") ||
        !check(bgf_world_get_bibite_detail(world, actor, &detail,
            nullptr, 0, &node_count, nullptr, 0, &synapse_count),
            "inspect released grab") ||
        !require(detail.held_count == 0,
            "removing a held Bibite must release the holder immediately") ||
        !check(bgf_world_save_checkpoint(world, path), "save released grab"))
        return finish(false);
    if (!check(bgf_world_load_checkpoint(path, config.device_index, &restored),
            "reload stock brain I/O state")) return finish(false);
    bgf_world_destroy(world);
    world = restored;
    if (!check(bgf_world_step(world, 240, &metrics),
            "grow and produce eggs using stock outputs") ||
        !check(bgf_world_get_bibite_detail(world, actor, &detail,
            values.data(), static_cast<int32_t>(values.size()), &node_count,
            nullptr, 0, &synapse_count), "inspect developed Bibite") ||
        !require(detail.alive == 1 && detail.size > 0.98f &&
                detail.egg_progress > 0.0f && detail.clock_time < 0.03f &&
                values[1].last_output > 0.98f &&
                values[22].last_output < 0.001f &&
                values[23].last_output > 0.09f &&
                values[24].last_output > 0.0f,
            "growth, egg investment, resettable clock, age and pheromone sensors must update"))
        return finish(false);
    for (int32_t sensor = 0; sensor < 33; ++sensor) {
        if (!require(values[sensor].sensor == sensor + 1 &&
                std::isfinite(values[sensor].last_output),
                "every stock sensor must produce a finite value"))
            return finish(false);
    }
    if (!require(values[33].sensor == 0 && values[33].last_output == 1.0f,
            "stock Constant sensor must return one")) return finish(false);
    return finish(true);
}

} // namespace

namespace {

bool save_progress_regressions()
{
    BgfWorldConfig config{};
    bgf_world_default_config(&config);
    config.device_index = 0;
    config.max_bibites = 1024;
    config.initial_bibites = 512;
    config.pellet_count = 64;
    config.world_half_extent = 2000.0f;
    config.initial_energy = 400.0f;
    config.reproduction_energy = 1000000.0f;
    config.seed = 0x05a9ef1u;
    config.contact_grid_update_factor = 4;
    config.contact_solve_factor = 8;
    config.vision_lookup_factor = 1;
    config.brain_update_factor = 8;
    config.lock_food_target = 1;
    BgfWorldHandle world = nullptr;
    BgfWorldHandle restored = nullptr;
    const std::string path = "bgf_save_progress_" +
        std::to_string(GetCurrentProcessId()) + ".bgfgpu";
    if (!require(_access(path.c_str(), 0) != 0,
            "save regression must not overwrite an existing file")) return false;
    const auto finish = [&](bool success) {
        if (restored) bgf_world_destroy(restored);
        if (world) bgf_world_destroy(world);
        std::remove(path.c_str());
        return success;
    };
    if (!check(bgf_world_create(&config, &world), "create crowded save regression") ||
        !check(bgf_world_set_combat_settings(world, 0.0f, 200.0f, 0.0f, 32.0f),
            "disable combat damage in save regression")) return finish(false);

    // Dense overflow chains coexist with hundreds of sparse occupied cells.
    // Repeated sparse clears must finish before those chains are reused.
    for (int32_t i = 0; i < 80; ++i) {
        int32_t slot = -1;
        if (!check(bgf_world_spawn_bibite(world, 16.0f, 16.0f, 0.0f, &slot),
                "spawn dense overflow cluster") ||
            !require(slot >= 0, "dense cluster placement must succeed")) return finish(false);
    }
    // The old subtraction-based wrapping loop could never make progress for
    // this finite coordinate because one world width is below its FP32 ULP.
    int32_t extreme_slot = -1;
    if (!check(bgf_world_spawn_bibite(world,
            std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
            0.0f, &extreme_slot), "wrap an extreme finite coordinate")) return finish(false);
    BgfWorldBibiteDetail detail{};
    int32_t node_count = 0, synapse_count = 0;
    if (!check(bgf_world_get_bibite_detail(world, extreme_slot, &detail,
            nullptr, 0, &node_count, nullptr, 0, &synapse_count),
            "inspect bounded coordinate wrapping") ||
        !require(std::isfinite(detail.position_x) && std::isfinite(detail.position_y) &&
            detail.position_x >= -config.world_half_extent &&
            detail.position_x < config.world_half_extent &&
            detail.position_y >= -config.world_half_extent &&
            detail.position_y < config.world_half_extent,
            "finite extreme coordinates must wrap into the world")) return finish(false);

    const auto started = std::chrono::steady_clock::now();
    int saves = 0;
    for (int32_t batch = 0; batch < 24; ++batch) {
        BgfWorldStepMetrics metrics{};
        if (!check(bgf_world_step(world, 64, &metrics),
                "crowded grid rebuild must make progress")) return finish(false);
        if (batch % 3 != 2) continue;
        BgfWorldStats before{}, after{};
        if (!check(bgf_world_get_stats(world, &before), "read pre-save state") ||
            !check(bgf_world_save_checkpoint(world, path.c_str()),
                "crowded checkpoint must complete") ||
            !check(bgf_world_load_checkpoint(path.c_str(), config.device_index, &restored),
                "crowded checkpoint must reload") ||
            !check(bgf_world_get_stats(restored, &after), "read reloaded save state") ||
            !require(after.completed_steps == before.completed_steps &&
                after.living_bibites == before.living_bibites &&
                after.invalid_state_deaths == 0,
                "checkpoint must preserve progress and population") ||
            !check(bgf_world_step(restored, 16, &metrics),
                "reloaded crowded world must resume")) return finish(false);
        bgf_world_destroy(restored);
        restored = nullptr;
        ++saves;
    }
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    std::cout << "Crowded save progress: " << saves << " save/reload/resume cycles, "
              << 24 * 64 << " ticks, " << seconds << " wall seconds\n";
    return finish(saves == 8);
}

} // namespace

#include "bgf_optimization_tests.cuh"

int main(int argc, char** argv)
{
    if (!graphics_threading_regressions()) return 51;
    int32_t device_count = 0;
    if (!check(bgf_get_device_count(&device_count), "device discovery") ||
        !require(device_count > 0, "at least one CUDA device is required")) {
        return 1;
    }
    if (argc > 1 && std::strcmp(argv[1], "--food-zones-only") == 0)
        return food_zone_regressions() ? 0 : 53;
    if (argc > 1 && std::strcmp(argv[1], "--brain-links-only") == 0)
        return dormant_hidden_link_regressions() ? 0 : 54;
    if (argc > 1 && std::strcmp(argv[1], "--evolving-brain-only") == 0)
        return evolving_brain_regressions() ? 0 : 52;
    if (argc > 1 && std::strcmp(argv[1], "--checkpoint-only") == 0)
        return checkpoint_stability_regressions() ? 0 : 50;
    if (argc > 1 && std::strcmp(argv[1], "--biology-only") == 0)
        return combat_and_meat_regressions() ? 0 : 55;
    if (argc > 1 && std::strcmp(argv[1], "--brain-io-only") == 0)
        return stock_brain_io_regressions() ? 0 : 56;
    if (argc > 1 && std::strcmp(argv[1], "--save-progress-only") == 0)
        return save_progress_regressions() ? 0 : 57;
    if (argc > 1 && std::strcmp(argv[1], "--optimizations-only") == 0)
        return live_grid_regressions() && local_food_overflow_regressions() && crowded_contact_regressions() &&
            specialized_brain_regressions() && cadence_graph_cache_regressions() && compact_imported_pool_regressions() &&
            owned_checkpoint_regressions() && captured_graphics_regressions() ? 0 : 58;
    if (argc > 1 && std::strcmp(argv[1], "--kernel-parity-only") == 0)
        return specialized_brain_regressions() ? 0 : 58;
    if (!live_grid_regressions() || !local_food_overflow_regressions() || !crowded_contact_regressions() ||
        !specialized_brain_regressions() || !cadence_graph_cache_regressions() || !compact_imported_pool_regressions() ||
        !owned_checkpoint_regressions() || !captured_graphics_regressions()) return 58;
    if (!save_progress_regressions()) return 57;
    if (!dormant_hidden_link_regressions()) return 54;
    if (!food_zone_regressions()) return 53;
    if (!combat_and_meat_regressions()) return 55;
    if (!stock_brain_io_regressions()) return 56;
    if (!checkpoint_stability_regressions()) return 50;
    if (!evolving_brain_regressions()) return 52;

    BgfWorldConfig config{};
    bgf_world_default_config(&config);
    config.max_bibites = 512;
    config.initial_bibites = 256;
    config.pellet_count = 1024;
    config.spatial_grid_width = 64;
    config.pheromone_grid_width = 64;
    config.pheromone_grid_height = 64;
    config.seed = 123456789u;

    BgfWorldConfig invalid = config;
    invalid.initial_bibites = invalid.max_bibites + 1;
    BgfWorldHandle invalid_world = nullptr;
    if (!require(
            bgf_world_create(&invalid, &invalid_world) != 0 && invalid_world == nullptr,
            "invalid capacity must be rejected")) {
        return 2;
    }

    BgfWorldConfig above_hard_cap = config;
    above_hard_cap.max_bibites = 500001;
    above_hard_cap.initial_bibites = 1;
    BgfWorldHandle above_hard_cap_world = nullptr;
    if (!require(
            bgf_world_create(&above_hard_cap, &above_hard_cap_world) != 0 &&
                above_hard_cap_world == nullptr,
            "population above the 500,000 hard cap must be rejected")) {
        return 24;
    }

    BgfWorldConfig above_pellet_cap = config;
    above_pellet_cap.pellet_count = 32769;
    BgfWorldHandle above_pellet_cap_world = nullptr;
    if (!require(
            bgf_world_create(&above_pellet_cap, &above_pellet_cap_world) != 0 &&
                above_pellet_cap_world == nullptr,
            "pellet count above the graphical hard cap must be rejected")) {
        return 28;
    }

    BgfWorldConfig live_food_config = config;
    live_food_config.max_bibites = 64;
    live_food_config.initial_bibites = 1;
    live_food_config.pellet_count = 512;
    BgfWorldHandle live_food_world = nullptr;
    if (!check(bgf_world_create(&live_food_config, &live_food_world),
            "live-food world creation")) {
        return 44;
    }
    BgfWorldStats live_food_stats{};
    if (!require(bgf_world_set_food_settings(
            live_food_world, 513, 1.0f, 28.0f) != 0,
            "live food must reject a target above its reserved capacity") ||
        !check(bgf_world_set_food_settings(live_food_world, 128, 0.5f, 14.0f),
            "lower live-food settings") ||
        !check(bgf_world_get_stats(live_food_world, &live_food_stats),
            "lower live-food statistics") ||
        !require(live_food_stats.active_pellets == 128,
            "lowering density must remove live pellets immediately") ||
        !check(bgf_world_set_food_settings(live_food_world, 384, 2.0f, 56.0f),
            "raise live-food settings") ||
        !check(bgf_world_get_stats(live_food_world, &live_food_stats),
            "raised live-food statistics") ||
        !require(live_food_stats.active_pellets == 384,
            "raising density must add live pellets immediately")) {
        bgf_world_destroy(live_food_world);
        return 45;
    }
    std::vector<BgfWorldBibite> live_food_bibites(64);
    std::vector<BgfWorldPellet> live_food_pellets(512);
    int32_t live_bibites_written = 0;
    int32_t live_pellets_written = 0;
    if (!check(bgf_world_download_snapshot(
            live_food_world,
            live_food_bibites.data(), 64, &live_bibites_written,
            live_food_pellets.data(), 512, &live_pellets_written,
            &live_food_stats), "live-food display snapshot") ||
        !require(live_pellets_written == 384 &&
                live_food_stats.active_pellets == 384,
            "render snapshots must follow the changed live food target")) {
        bgf_world_destroy(live_food_world);
        return 48;
    }
    const char* live_food_path = "bgf_live_food_test.bgfgpu";
    std::remove(live_food_path);
    BgfWorldHandle restored_food_world = nullptr;
    int32_t restored_target = -1;
    float restored_growth = -1.0f;
    float restored_energy = -1.0f;
    if (!check(bgf_world_save_checkpoint(live_food_world, live_food_path),
            "live-food checkpoint save") ||
        !check(bgf_world_load_checkpoint(live_food_path, 0, &restored_food_world),
            "live-food checkpoint load") ||
        !check(bgf_world_get_food_settings(restored_food_world,
            &restored_target, &restored_growth, &restored_energy),
            "restored live-food settings") ||
        !require(restored_target == 384 &&
                std::fabs(restored_growth - 2.0f) < 0.0001f &&
                std::fabs(restored_energy - 56.0f) < 0.0001f,
            "food target, regrowth and energy must survive a save/load") ||
        !check(bgf_world_set_food_settings(restored_food_world, 0, 0.0f, 56.0f),
            "disable live food") ||
        !check(bgf_world_get_stats(restored_food_world, &live_food_stats),
            "disabled live-food statistics") ||
        !require(live_food_stats.active_pellets == 0,
            "zero biomass must remove all live food") ||
        !check(bgf_world_set_food_settings(restored_food_world, 512, 1.0f, 28.0f),
            "restore live food") ||
        !check(bgf_world_get_stats(restored_food_world, &live_food_stats),
            "restored live-food statistics") ||
        !require(live_food_stats.active_pellets == 512,
            "restoring biomass must reactivate all reserved food")) {
        bgf_world_destroy(restored_food_world);
        bgf_world_destroy(live_food_world);
        std::remove(live_food_path);
        return 46;
    }
    bgf_world_destroy(restored_food_world);
    bgf_world_destroy(live_food_world);
    std::remove(live_food_path);

    BgfWorldConfig non_finite_config = config;
    non_finite_config.world_half_extent =
        std::numeric_limits<float>::quiet_NaN();
    BgfWorldHandle non_finite_world = nullptr;
    if (!require(
            bgf_world_create(&non_finite_config, &non_finite_world) != 0 &&
                non_finite_world == nullptr,
            "non-finite world configuration must be rejected")) {
        return 29;
    }

    // Feeding must take many bites, not remove a pellet on first contact.
    // After enough time, fully eaten pellets enter the delayed-regrowth wheel.
    BgfWorldConfig food_delay_config = config;
    food_delay_config.max_bibites = 512;
    food_delay_config.initial_bibites = 512;
    food_delay_config.pellet_count = 256;
    food_delay_config.world_half_extent = 2.0f;
    food_delay_config.initial_energy = 10.0f;
    food_delay_config.pellet_energy = 28.0f;
    food_delay_config.reproduction_energy = 100.0f;
    food_delay_config.seed = 0x5eed1234u;
    BgfWorldHandle food_delay_world = nullptr;
    if (!check(bgf_world_create(&food_delay_config, &food_delay_world),
            "delayed-food world creation")) {
        return 30;
    }
    BgfWorldStepMetrics food_delay_metrics{};
    BgfWorldStats first_bite{};
    if (!check(bgf_world_step(food_delay_world, 1, &food_delay_metrics),
            "first partial feeding step") ||
        !check(bgf_world_get_stats(food_delay_world, &first_bite),
            "first partial feeding statistics") ||
        !require(first_bite.pellets_eaten == 0 &&
                first_bite.active_pellets == food_delay_config.pellet_count,
            "one update must not swallow whole pellets")) {
        bgf_world_destroy(food_delay_world);
        return 31;
    }
    std::vector<BgfWorldPellet> first_pellets(
        static_cast<size_t>(food_delay_config.pellet_count));
    int32_t first_pellet_count = 0;
    if (!check(bgf_world_download_pellets(food_delay_world,
            first_pellets.data(), food_delay_config.pellet_count,
            &first_pellet_count), "partial pellet snapshot") ||
        !require(first_pellet_count == food_delay_config.pellet_count &&
                std::any_of(first_pellets.begin(), first_pellets.end(),
                    [&](const BgfWorldPellet& pellet) {
                        return pellet.energy > 0.0f &&
                            pellet.energy < food_delay_config.pellet_energy;
                    }),
            "a first bite must reduce pellet food without removing the pellet")) {
        bgf_world_destroy(food_delay_world);
        return 31;
    }
    BgfWorldStats food_depleted{};
    if (!check(bgf_world_step(food_delay_world, 100, &food_delay_metrics),
            "bounded feeding duration") ||
        !check(bgf_world_get_stats(food_delay_world, &food_depleted),
            "post-feeding statistics") ||
        !require(food_depleted.pellets_eaten > 0 &&
                food_depleted.active_pellets < food_delay_config.pellet_count,
            "repeated bites must eventually consume pellets")) {
        bgf_world_destroy(food_delay_world);
        return 31;
    }
    for (int slot = 0; slot < food_delay_config.initial_bibites; ++slot) {
        if (!check(bgf_world_kill_bibite(food_delay_world, slot),
                "remove delayed-food test Bibite")) {
            bgf_world_destroy(food_delay_world);
            return 31;
        }
    }
    BgfWorldStats food_waiting{};
    if (!check(bgf_world_step(food_delay_world, 200, &food_delay_metrics),
            "delayed-food waiting steps") ||
        !check(bgf_world_get_stats(food_delay_world, &food_waiting),
            "delayed-food waiting statistics") ||
        !require(food_waiting.active_pellets == food_delay_config.pellet_count &&
                food_waiting.pellets_eaten == food_depleted.pellets_eaten,
            "all depleted food must regrow after the delay without new feeding")) {
        bgf_world_destroy(food_delay_world);
        return 32;
    }
    BgfWorldStats food_regrown{};
    bgf_world_destroy(food_delay_world);

    // The live fertility control must affect already queued regrowth, including
    // a zero-fertility interval, without turning eaten food into permanent loss.
    BgfWorldHandle sterile_food_world = nullptr;
    if (!check(bgf_world_create(&food_delay_config, &sterile_food_world),
            "sterile-food world creation") ||
        !check(bgf_world_step(sterile_food_world, 101, &food_delay_metrics),
            "sterile-food consumption steps")) {
        if (sterile_food_world) bgf_world_destroy(sterile_food_world);
        return 47;
    }
    for (int slot = 0; slot < food_delay_config.initial_bibites; ++slot) {
        if (!check(bgf_world_kill_bibite(sterile_food_world, slot),
                "remove sterile-food test Bibite")) {
            bgf_world_destroy(sterile_food_world);
            return 47;
        }
    }
    if (!check(bgf_world_set_food_settings(sterile_food_world, 256, 0.0f, 28.0f),
            "disable food regrowth") ||
        !check(bgf_world_step(sterile_food_world, 200, &food_delay_metrics),
            "sterile-food waiting steps") ||
        !check(bgf_world_get_stats(sterile_food_world, &food_waiting),
            "sterile-food waiting statistics") ||
        !require(food_waiting.active_pellets < 256,
            "zero fertility must prevent due pellets from regrowing") ||
        !check(bgf_world_set_food_settings(sterile_food_world, 256, 1.0f, 28.0f),
            "restore food regrowth") ||
        !check(bgf_world_step(sterile_food_world, 200, &food_delay_metrics),
            "restored-food waiting steps") ||
        !check(bgf_world_get_stats(sterile_food_world, &food_regrown),
            "restored-food statistics") ||
        !require(food_regrown.active_pellets == 256,
            "restored fertility must eventually regrow the queued food")) {
        bgf_world_destroy(sterile_food_world);
        return 47;
    }
    bgf_world_destroy(sterile_food_world);

    BgfWorldConfig contact_config = config;
    contact_config.max_bibites = 512;
    contact_config.initial_bibites = 512;
    contact_config.pellet_count = 256;
    contact_config.world_half_extent = 100.0f;
    contact_config.diagnostic_mask = 6u;
    BgfWorldConfig legacy_contact_config = contact_config;
    legacy_contact_config.diagnostic_mask = 14u;
    BgfWorldHandle tiled_contact_world = nullptr;
    BgfWorldHandle legacy_contact_world = nullptr;
    if (!check(bgf_world_create(&contact_config, &tiled_contact_world),
            "tiled-contact parity world creation") ||
        !check(bgf_world_create(&legacy_contact_config, &legacy_contact_world),
            "legacy-contact parity world creation")) {
        bgf_world_destroy(tiled_contact_world);
        bgf_world_destroy(legacy_contact_world);
        return 20;
    }
    BgfWorldStepMetrics tiled_contact_metrics{};
    BgfWorldStepMetrics legacy_contact_metrics{};
    if (!check(bgf_world_step(tiled_contact_world, 1, &tiled_contact_metrics),
            "tiled-contact parity step") ||
        !check(bgf_world_step(legacy_contact_world, 1, &legacy_contact_metrics),
            "legacy-contact parity step")) {
        bgf_world_destroy(tiled_contact_world);
        bgf_world_destroy(legacy_contact_world);
        return 21;
    }
    std::vector<BgfWorldBibite> tiled_contact_snapshot(512);
    std::vector<BgfWorldBibite> legacy_contact_snapshot(512);
    int32_t tiled_contact_written = 0;
    int32_t legacy_contact_written = 0;
    if (!check(bgf_world_download_bibites(
            tiled_contact_world,
            tiled_contact_snapshot.data(),
            static_cast<int32_t>(tiled_contact_snapshot.size()),
            &tiled_contact_written), "tiled-contact parity snapshot") ||
        !check(bgf_world_download_bibites(
            legacy_contact_world,
            legacy_contact_snapshot.data(),
            static_cast<int32_t>(legacy_contact_snapshot.size()),
            &legacy_contact_written), "legacy-contact parity snapshot") ||
        !require(tiled_contact_written == legacy_contact_written,
            "tiled and legacy contact paths must preserve population")) {
        bgf_world_destroy(tiled_contact_world);
        bgf_world_destroy(legacy_contact_world);
        return 22;
    }
    std::vector<BgfWorldBibite> tiled_by_slot(512);
    std::vector<BgfWorldBibite> legacy_by_slot(512);
    for (int32_t index = 0; index < tiled_contact_written; ++index) {
        tiled_by_slot[tiled_contact_snapshot[index].slot] = tiled_contact_snapshot[index];
        legacy_by_slot[legacy_contact_snapshot[index].slot] = legacy_contact_snapshot[index];
    }
    bool contact_parity = true;
    float maximum_contact_delta = 0.0f;
    int32_t maximum_contact_slot = -1;
    for (int32_t slot = 0; slot < 512; ++slot) {
        const BgfWorldBibite& tiled = tiled_by_slot[slot];
        const BgfWorldBibite& legacy = legacy_by_slot[slot];
        const float delta = std::fmax(
            std::fmax(
                std::fabs(tiled.position_x - legacy.position_x),
                std::fabs(tiled.position_y - legacy.position_y)),
            std::fmax(
                std::fabs(tiled.velocity_x - legacy.velocity_x),
                std::fabs(tiled.velocity_y - legacy.velocity_y)));
        if (delta > maximum_contact_delta) {
            maximum_contact_delta = delta;
            maximum_contact_slot = slot;
        }
        if (tiled.slot != legacy.slot || delta > 0.001f) {
            contact_parity = false;
        }
    }
    bgf_world_destroy(tiled_contact_world);
    bgf_world_destroy(legacy_contact_world);
    if (!require(contact_parity,
            "unique tiled contact forces must match the legacy per-Bibite solver")) {
        std::cerr << "Maximum contact-path delta " << maximum_contact_delta
                  << " at slot " << maximum_contact_slot << "\n";
        return 23;
    }

    BgfWorldHandle world = nullptr;
    if (!check(bgf_world_create(&config, &world), "world creation") ||
        !require(world != nullptr, "world handle must be populated")) {
        return 3;
    }

    char device_name[256]{};
    if (!check(bgf_world_get_device_name(world, device_name, sizeof(device_name)), "device name")) {
        bgf_world_destroy(world);
        return 4;
    }

    BgfWorldStats initial{};
    if (!check(bgf_world_get_stats(world, &initial), "initial stats") ||
        !require(initial.completed_steps == 0, "new world must start at step zero") ||
        !require(initial.living_bibites == config.initial_bibites, "initial population mismatch") ||
        !require(initial.active_pellets == config.pellet_count, "initial pellet count mismatch") ||
        !require(std::isfinite(initial.total_energy) && initial.total_energy > 0.0f,
            "initial energy must be finite and positive")) {
        bgf_world_destroy(world);
        return 5;
    }

    int32_t placed_slot = -1;
    if (!check(bgf_world_spawn_bibite(world, 10.0f, -20.0f, 1.5f, &placed_slot),
            "manual GPU Bibite placement") ||
        !require(placed_slot >= 0 && placed_slot < config.max_bibites,
            "manual placement must return a live GPU slot")) {
        bgf_world_destroy(world);
        return 6;
    }
    BgfWorldStats after_placement{};
    if (!check(bgf_world_get_stats(world, &after_placement), "post-placement stats") ||
        !require(after_placement.living_bibites == config.initial_bibites + 1,
            "manual placement must increase the GPU population") ||
        !require(after_placement.births == 1,
            "manual placement must be recorded in birth statistics")) {
        bgf_world_destroy(world);
        return 7;
    }

    BgfWorldTemplateSpawn template_spawn{};
    template_spawn.position_x = -31.5f;
    template_spawn.position_y = 47.25f;
    template_spawn.heading = -0.75f;
    template_spawn.energy = 277.0f;
    template_spawn.size = 1.4f;
    template_spawn.maximum_speed = 6.25f;
    template_spawn.turn_speed = 2.5f;
    template_spawn.metabolism = 0.18f;
    template_spawn.lifespan = 900.0f;
    template_spawn.color_r = 0.125f;
    template_spawn.color_g = 0.5f;
    template_spawn.color_b = 0.875f;
    template_spawn.gene_mutation_strength = 0.04f;
    template_spawn.brain_mutation_strength = 0.08f;
    template_spawn.generation = 12;
    template_spawn.lineage_id = 0x123456789abcdef0ull;
    template_spawn.tag_id = 0x0fedcba987654321ull;
    const BgfWorldBrainNode template_nodes[] = {
        {0, 1, -1, 0.0f},
        {3, -1, 0, 0.25f},
        {1, -1, 5, -5.0f}};
    const BgfWorldBrainSynapse template_synapses[] = {
        {0, 1, 1.5f},
        {0, 2, 0.5f}};
    int32_t template_slot = -1;
    if (!check(bgf_world_spawn_template_bibite(
            world,
            &template_spawn,
            template_nodes,
            static_cast<int32_t>(sizeof(template_nodes) / sizeof(template_nodes[0])),
            template_synapses,
            static_cast<int32_t>(sizeof(template_synapses) / sizeof(template_synapses[0])),
            &template_slot), "stock-template GPU Bibite placement") ||
        !require(template_slot >= 0 && template_slot < config.max_bibites,
            "template placement must return a live GPU slot")) {
        bgf_world_destroy(world);
        return 8;
    }
    BgfWorldStats after_template{};
    if (!check(bgf_world_get_stats(world, &after_template), "post-template stats") ||
        !require(after_template.living_bibites == config.initial_bibites + 2,
            "template placement must increase the GPU population") ||
        !require(after_template.births == 2,
            "template placement must be recorded in birth statistics")) {
        bgf_world_destroy(world);
        return 9;
    }
    std::vector<BgfWorldBibite> placement_snapshot(
        static_cast<size_t>(config.max_bibites));
    int32_t placement_written = 0;
    if (!check(bgf_world_download_bibites(
            world,
            placement_snapshot.data(),
            static_cast<int32_t>(placement_snapshot.size()),
            &placement_written), "template placement snapshot")) {
        bgf_world_destroy(world);
        return 10;
    }
    const BgfWorldBibite* placed_template = nullptr;
    for (int32_t index = 0; index < placement_written; ++index) {
        if (placement_snapshot[index].slot == template_slot) {
            placed_template = &placement_snapshot[index];
            break;
        }
    }
    if (!require(placed_template != nullptr, "template slot must be downloadable") ||
        !require(placed_template->generation == template_spawn.generation,
            "template generation must be preserved") ||
        !require(placed_template->lineage_id == template_spawn.lineage_id,
            "template lineage must be preserved") ||
        !require(placed_template->tag_id == template_spawn.tag_id,
            "template tag must be preserved") ||
        !require(placed_template->brain_nodes == 3 && placed_template->brain_synapses == 2,
            "template brain topology counts must be preserved") ||
        !require(std::fabs(placed_template->size - template_spawn.size) < 0.0001f,
            "template body size must be preserved") ||
        !require(std::fabs(placed_template->color_r - template_spawn.color_r) < 0.0001f &&
                std::fabs(placed_template->color_g - template_spawn.color_g) < 0.0001f &&
                std::fabs(placed_template->color_b - template_spawn.color_b) < 0.0001f,
            "template colour must be preserved")) {
        bgf_world_destroy(world);
        return 11;
    }

    std::vector<BgfWorldBrainNodeState> selected_nodes(64);
    std::vector<BgfWorldBrainSynapseState> selected_synapses(128);
    BgfWorldBibiteDetail selected_detail{};
    int32_t selected_nodes_written = 0;
    int32_t selected_synapses_written = 0;
    if (!check(bgf_world_get_bibite_detail(
            world,
            template_slot,
            &selected_detail,
            selected_nodes.data(),
            static_cast<int32_t>(selected_nodes.size()),
            &selected_nodes_written,
            selected_synapses.data(),
            static_cast<int32_t>(selected_synapses.size()),
            &selected_synapses_written), "selected Bibite detail download") ||
        !require(selected_detail.alive == 1 && selected_detail.slot == template_slot,
            "selected detail must identify the live requested slot") ||
        !require(selected_detail.template_brain == 1 &&
                selected_nodes_written == 3 && selected_synapses_written == 2,
            "selected detail must expose the complete template topology") ||
        !require(selected_detail.generation == template_spawn.generation &&
                selected_detail.lineage_id == template_spawn.lineage_id &&
                selected_detail.tag_id == template_spawn.tag_id,
            "selected detail must preserve identity fields") ||
        !require(selected_detail.maximum_speed >= template_spawn.maximum_speed &&
                std::isfinite(selected_detail.maximum_speed) &&
                std::fabs(selected_detail.turn_speed - template_spawn.turn_speed) < 0.0001f &&
                std::fabs(selected_detail.metabolism - template_spawn.metabolism) < 0.0001f &&
                std::fabs(selected_detail.lifespan - template_spawn.lifespan) < 0.0001f,
            "selected detail must expose effective GPU-native inherited traits") ||
        !require(selected_nodes[1].type == template_nodes[1].type &&
                selected_nodes[1].action == template_nodes[1].action &&
                std::fabs(selected_nodes[1].base_activation -
                    template_nodes[1].base_activation) < 0.0001f,
            "selected detail must expose template nodes") ||
        !require(selected_synapses[0].node_in == template_synapses[0].node_in &&
                selected_synapses[0].node_out == template_synapses[0].node_out &&
                std::fabs(selected_synapses[0].weight - template_synapses[0].weight) < 0.0001f,
            "selected detail must expose template synapses")) {
        bgf_world_destroy(world);
        return 40;
    }

    const uint64_t edited_tag = 0xa55a55aa12344321ull;
    if (!check(bgf_world_set_bibite_tag(world, template_slot, edited_tag),
            "selected Bibite tag update") ||
        !check(bgf_world_get_bibite_detail(
            world,
            template_slot,
            &selected_detail,
            nullptr,
            0,
            &selected_nodes_written,
            nullptr,
            0,
            &selected_synapses_written), "selected Bibite compact detail download") ||
        !require(selected_detail.tag_id == edited_tag &&
                selected_nodes_written == 0 && selected_synapses_written == 0,
            "tag update and header-only inspector download must round-trip")) {
        bgf_world_destroy(world);
        return 41;
    }

    int32_t child_slot = -1;
    BgfWorldBibiteDetail child_detail{};
    std::vector<BgfWorldBrainNodeState> child_nodes(64);
    std::vector<BgfWorldBrainSynapseState> child_synapses(128);
    if (!check(bgf_world_force_reproduction(world, template_slot, &child_slot),
            "selected Bibite lay-egg command") ||
        !require(child_slot >= 0 && child_slot < config.max_bibites,
            "lay-egg command must return a live child slot") ||
        !check(bgf_world_get_bibite_detail(
            world,
            child_slot,
            &child_detail,
            child_nodes.data(),
            static_cast<int32_t>(child_nodes.size()),
            &selected_nodes_written,
            child_synapses.data(),
            static_cast<int32_t>(child_synapses.size()),
            &selected_synapses_written), "selected child detail download") ||
        !require(child_detail.alive == 1 &&
                child_detail.generation == template_spawn.generation + 1 &&
                child_detail.lineage_id == template_spawn.lineage_id &&
                child_detail.tag_id == edited_tag && child_detail.template_brain == 1 &&
                selected_nodes_written == 3 && selected_synapses_written == 2,
            "lay-egg command must preserve lineage, tag, and brain topology") ||
        !require(
            std::fabs(child_synapses[0].weight - selected_synapses[0].weight) > 0.0001f ||
            std::fabs(child_synapses[1].weight - selected_synapses[1].weight) > 0.0001f ||
            std::fabs(child_detail.color_r - selected_detail.color_r) > 0.0001f ||
            std::fabs(child_detail.size - selected_detail.size) > 0.0001f,
            "offspring must inherit a mutated brain or body trait")) {
        bgf_world_destroy(world);
        return 42;
    }

    if (!check(bgf_world_kill_bibite(world, child_slot),
            "selected Bibite remove command") ||
        !check(bgf_world_get_bibite_detail(
            world,
            child_slot,
            &child_detail,
            nullptr,
            0,
            &selected_nodes_written,
            nullptr,
            0,
            &selected_synapses_written), "removed Bibite detail download") ||
        !require(child_detail.alive == 0,
            "removed selected Bibite must no longer be reported alive")) {
        bgf_world_destroy(world);
        return 43;
    }

    BgfWorldStepMetrics first{};
    if (!check(bgf_world_step(world, 64, &first), "first GPU-resident step batch") ||
        !require(first.requested_steps == 64, "reported step count mismatch") ||
        !require(first.gpu_milliseconds > 0.0f && first.wall_milliseconds > 0.0f,
            "timings must be positive") ||
        !require(first.gpu_offload_percent > 0.0f && first.gpu_offload_percent <= 100.0f,
            "offload percentage must be bounded") ||
        !require(first.prepare_milliseconds > 0.0f &&
                first.spatial_index_milliseconds > 0.0f &&
                first.contact_milliseconds > 0.0f &&
                first.decision_milliseconds > 0.0f &&
                first.motion_milliseconds > 0.0f &&
                first.lifecycle_milliseconds > 0.0f,
            "per-phase timings must be positive") ||
        !require(first.realtime_multiplier > 0.0, "simulation multiplier must be positive")) {
        bgf_world_destroy(world);
        return 12;
    }

    BgfWorldStepMetrics second{};
    if (!check(bgf_world_step(world, 64, &second), "second GPU-resident step batch")) {
        bgf_world_destroy(world);
        return 13;
    }

    BgfWorldStats stats{};
    if (!check(bgf_world_get_stats(world, &stats), "final stats") ||
        !require(stats.completed_steps == 128, "world step counter mismatch") ||
        !require(stats.deaths == stats.starvation_deaths + stats.age_deaths +
                stats.invalid_state_deaths,
            "death-cause counters must reconcile with total deaths") ||
        !require(stats.living_bibites > 0 && stats.living_bibites <= config.max_bibites,
            "living population must remain in bounds") ||
        !require(stats.active_pellets > 0 && stats.active_pellets <= config.pellet_count,
            "active food must remain bounded while eaten pellets regrow") ||
        !require(stats.pellets_eaten == 0 || stats.active_pellets < config.pellet_count,
            "eating must visibly reduce active food before delayed regrowth") ||
        !require(std::isfinite(stats.total_energy) && std::isfinite(stats.average_energy),
            "world energy must remain finite")) {
        bgf_world_destroy(world);
        return 14;
    }

    std::vector<BgfWorldBibite> snapshot(64);
    int32_t written = 0;
    if (!check(bgf_world_download_bibites(
            world,
            snapshot.data(),
            static_cast<int32_t>(snapshot.size()),
            &written), "snapshot download") ||
        !require(written > 0 && written <= static_cast<int32_t>(snapshot.size()),
            "snapshot result count must fit its capacity")) {
        bgf_world_destroy(world);
        return 15;
    }
    for (int32_t index = 0; index < written; ++index) {
        const BgfWorldBibite& bibite = snapshot[index];
        if (!require(
                std::isfinite(bibite.position_x) && std::isfinite(bibite.position_y) &&
                std::isfinite(bibite.energy) && bibite.energy > 0.0f &&
                bibite.position_x >= -config.world_half_extent &&
                bibite.position_x < config.world_half_extent &&
                bibite.position_y >= -config.world_half_extent &&
                bibite.position_y < config.world_half_extent,
                "snapshot contains an invalid living Bibite")) {
            bgf_world_destroy(world);
            return 16;
        }
    }

    std::vector<BgfWorldPellet> pellet_snapshot(
        static_cast<size_t>(config.pellet_count));
    int32_t pellets_written = 0;
    if (!check(bgf_world_download_pellets(
            world,
            pellet_snapshot.data(),
            static_cast<int32_t>(pellet_snapshot.size()),
            &pellets_written), "pellet snapshot download") ||
        !require(pellets_written == stats.active_pellets,
            "pellet snapshot count must match active pellet statistics")) {
        bgf_world_destroy(world);
        return 17;
    }
    for (int32_t index = 0; index < pellets_written; ++index) {
        const BgfWorldPellet& pellet = pellet_snapshot[index];
        if (!require(
                std::isfinite(pellet.position_x) && std::isfinite(pellet.position_y) &&
                pellet.energy > 0.0f &&
                pellet.position_x >= -config.world_half_extent &&
                pellet.position_x < config.world_half_extent &&
                pellet.position_y >= -config.world_half_extent &&
                pellet.position_y < config.world_half_extent,
                "pellet snapshot contains invalid state")) {
            bgf_world_destroy(world);
            return 18;
        }
    }

    std::vector<BgfWorldBibite> packed_bibites(
        static_cast<size_t>(config.max_bibites));
    std::vector<BgfWorldPellet> packed_pellets(
        static_cast<size_t>(config.pellet_count));
    int32_t packed_bibites_written = 0;
    int32_t packed_pellets_written = 0;
    BgfWorldStats packed_stats{};
    if (!check(bgf_world_download_snapshot(
            world,
            packed_bibites.data(),
            static_cast<int32_t>(packed_bibites.size()),
            &packed_bibites_written,
            packed_pellets.data(),
            static_cast<int32_t>(packed_pellets.size()),
            &packed_pellets_written,
            &packed_stats), "packed graphical snapshot download") ||
        !require(packed_bibites_written == packed_stats.living_bibites,
            "packed Bibite count must match packed statistics") ||
        !require(packed_pellets_written == packed_stats.active_pellets,
            "packed pellet count must match packed statistics") ||
        !require(packed_stats.completed_steps == stats.completed_steps,
            "packed snapshot step must match world statistics") ||
        !require(packed_stats.starvation_deaths == stats.starvation_deaths &&
                packed_stats.age_deaths == stats.age_deaths &&
                packed_stats.invalid_state_deaths == stats.invalid_state_deaths,
            "packed snapshot death causes must match world statistics") ||
        !require(std::isfinite(packed_stats.total_energy) &&
                std::isfinite(packed_stats.average_energy) &&
                packed_stats.total_energy > 0.0f,
            "packed snapshot energy must be finite and positive")) {
        bgf_world_destroy(world);
        return 19;
    }

    // The close-up sprite budget is filled from the camera rectangle, not
    // consumed by Bibites elsewhere in a large world. Charts still receive
    // their independent representative population snapshot.
    if (!require(packed_bibites_written > 0,
            "viewport test needs a living Bibite")) {
        bgf_world_destroy(world);
        return 49;
    }
    const float view_min_x = packed_bibites[0].position_x - 100.0f;
    const float view_max_x = packed_bibites[0].position_x + 100.0f;
    const float view_min_y = packed_bibites[0].position_y - 100.0f;
    const float view_max_y = packed_bibites[0].position_y + 100.0f;
    int32_t expected_visible = 0;
    for (int32_t index = 0; index < packed_bibites_written; ++index) {
        const BgfWorldBibite& bibite = packed_bibites[index];
        if (bibite.position_x >= view_min_x && bibite.position_x <= view_max_x &&
            bibite.position_y >= view_min_y && bibite.position_y <= view_max_y) {
            ++expected_visible;
        }
    }
    std::vector<BgfWorldBibite> visible_bibites(5);
    int32_t visible_written = 0;
    BgfWorldStats view_stats{};
    if (!check(bgf_world_download_snapshot_with_view(
            world,
            packed_bibites.data(), static_cast<int32_t>(packed_bibites.size()),
            &packed_bibites_written,
            packed_pellets.data(), static_cast<int32_t>(packed_pellets.size()),
            &packed_pellets_written,
            visible_bibites.data(), static_cast<int32_t>(visible_bibites.size()),
            &visible_written,
            view_min_x, view_min_y, view_max_x, view_max_y,
            &view_stats), "camera-filtered graphical snapshot") ||
        !require(visible_written == std::min(expected_visible, 5),
            "close-up budget must be spent only on Bibites in the camera rectangle") ||
        !require(packed_bibites_written == view_stats.living_bibites &&
                view_stats.living_bibites == packed_stats.living_bibites,
            "camera filtering must not change the world population snapshot")) {
        bgf_world_destroy(world);
        return 49;
    }
    for (int32_t index = 0; index < visible_written; ++index) {
        const BgfWorldBibite& bibite = visible_bibites[index];
        if (!require(bibite.position_x >= view_min_x && bibite.position_x <= view_max_x &&
                bibite.position_y >= view_min_y && bibite.position_y <= view_max_y,
                "close-up snapshot must not include off-screen Bibites")) {
            bgf_world_destroy(world);
            return 49;
        }
    }

    const char* checkpoint_path = "bgf_world_roundtrip_test.bgfgpu";
    std::remove(checkpoint_path);
    if (!check(bgf_world_save_checkpoint(world, checkpoint_path),
            "GPU checkpoint save")) {
        bgf_world_destroy(world);
        return 34;
    }
    BgfWorldConfig checkpoint_config{};
    if (!check(bgf_world_checkpoint_config(
            checkpoint_path,
            config.device_index,
            &checkpoint_config), "GPU checkpoint header") ||
        !require(checkpoint_config.max_bibites == config.max_bibites &&
                checkpoint_config.initial_bibites == config.initial_bibites &&
                checkpoint_config.pellet_count == config.pellet_count &&
                checkpoint_config.seed == config.seed,
            "checkpoint configuration must preserve world dimensions and seed")) {
        std::remove(checkpoint_path);
        bgf_world_destroy(world);
        return 35;
    }
    BgfWorldHandle restored_world = nullptr;
    if (!check(bgf_world_load_checkpoint(
            checkpoint_path,
            config.device_index,
            &restored_world), "GPU checkpoint load") ||
        !require(restored_world != nullptr,
            "checkpoint load must return a GPU world")) {
        std::remove(checkpoint_path);
        bgf_world_destroy(world);
        return 36;
    }
    BgfWorldStats restored_stats{};
    if (!check(bgf_world_get_stats(restored_world, &restored_stats),
            "restored checkpoint statistics") ||
        !require(restored_stats.completed_steps == packed_stats.completed_steps &&
                restored_stats.living_bibites == packed_stats.living_bibites &&
                restored_stats.active_pellets == packed_stats.active_pellets &&
                restored_stats.births == packed_stats.births &&
                restored_stats.deaths == packed_stats.deaths &&
                restored_stats.pellets_eaten == packed_stats.pellets_eaten,
            "checkpoint counters must round-trip exactly")) {
        bgf_world_destroy(restored_world);
        std::remove(checkpoint_path);
        bgf_world_destroy(world);
        return 37;
    }
    std::vector<BgfWorldBibite> restored_bibites(
        static_cast<size_t>(config.max_bibites));
    std::vector<BgfWorldPellet> restored_pellets(
        static_cast<size_t>(config.pellet_count));
    int32_t restored_bibites_written = 0;
    int32_t restored_pellets_written = 0;
    BgfWorldStats restored_snapshot_stats{};
    if (!check(bgf_world_download_snapshot(
            restored_world,
            restored_bibites.data(),
            static_cast<int32_t>(restored_bibites.size()),
            &restored_bibites_written,
            restored_pellets.data(),
            static_cast<int32_t>(restored_pellets.size()),
            &restored_pellets_written,
            &restored_snapshot_stats), "restored checkpoint snapshot") ||
        !require(restored_bibites_written == packed_bibites_written &&
                restored_pellets_written == packed_pellets_written,
            "checkpoint presentation counts must round-trip exactly")) {
        bgf_world_destroy(restored_world);
        std::remove(checkpoint_path);
        bgf_world_destroy(world);
        return 38;
    }
    bool checkpoint_state_matches = true;
    std::vector<BgfWorldBibite> restored_bibites_by_slot(
        static_cast<size_t>(config.max_bibites));
    for (int32_t index = 0; index < restored_bibites_written; ++index) {
        restored_bibites_by_slot[restored_bibites[index].slot] = restored_bibites[index];
    }
    for (int32_t index = 0; index < packed_bibites_written; ++index) {
        const BgfWorldBibite& before = packed_bibites[index];
        const BgfWorldBibite& after = restored_bibites_by_slot[before.slot];
        const bool bibite_matches =
            before.slot == after.slot &&
            before.position_x == after.position_x &&
            before.position_y == after.position_y &&
            before.velocity_x == after.velocity_x &&
            before.velocity_y == after.velocity_y &&
            before.heading == after.heading &&
            before.energy == after.energy &&
            before.age == after.age &&
            before.generation == after.generation &&
            before.lineage_id == after.lineage_id &&
            before.tag_id == after.tag_id &&
            before.brain_nodes == after.brain_nodes &&
            before.brain_synapses == after.brain_synapses;
        if (!bibite_matches && checkpoint_state_matches) {
            std::cerr << "Checkpoint Bibite mismatch at slot " << before.slot
                      << ": slots " << before.slot << "/" << after.slot
                      << ", positions " << before.position_x << "," << before.position_y
                      << "/" << after.position_x << "," << after.position_y
                      << ", energy " << before.energy << "/" << after.energy
                      << ", age " << before.age << "/" << after.age << "\n";
        }
        checkpoint_state_matches = checkpoint_state_matches && bibite_matches;
    }
    std::vector<BgfWorldPellet> restored_pellets_by_slot(
        static_cast<size_t>(config.pellet_count));
    for (int32_t index = 0; index < restored_pellets_written; ++index) {
        restored_pellets_by_slot[restored_pellets[index].slot] = restored_pellets[index];
    }
    for (int32_t index = 0; index < packed_pellets_written; ++index) {
        const BgfWorldPellet& before = packed_pellets[index];
        const BgfWorldPellet& after = restored_pellets_by_slot[before.slot];
        const bool pellet_matches =
            before.slot == after.slot &&
            before.position_x == after.position_x &&
            before.position_y == after.position_y &&
            before.energy == after.energy;
        if (!pellet_matches && checkpoint_state_matches) {
            std::cerr << "Checkpoint pellet mismatch at slot " << before.slot
                      << ": positions " << before.position_x << "," << before.position_y
                      << "/" << after.position_x << "," << after.position_y
                      << ", energy " << before.energy << "/" << after.energy << "\n";
        }
        checkpoint_state_matches = checkpoint_state_matches && pellet_matches;
    }
    BgfWorldStepMetrics restored_metrics{};
    if (!require(checkpoint_state_matches,
            "checkpoint entity state must round-trip exactly") ||
        !check(bgf_world_step(restored_world, 16, &restored_metrics),
            "restored checkpoint continuation") ||
        !check(bgf_world_get_stats(restored_world, &restored_stats),
            "continued checkpoint statistics") ||
        !require(restored_stats.completed_steps == packed_stats.completed_steps + 16 &&
                std::isfinite(restored_stats.total_energy),
            "restored checkpoint must continue simulating")) {
        bgf_world_destroy(restored_world);
        std::remove(checkpoint_path);
        bgf_world_destroy(world);
        return 39;
    }
    bgf_world_destroy(restored_world);
    // A v3 file predates hidden brains; v2 also lacks per-pellet food state.
    // Both must preserve their original direct 16x6 behaviour when migrated.
    const char* legacy_path = "bgf_world_legacy_food_test.bgfgpu";
    if (!check(bgf_world_save_checkpoint_legacy_v11(world,legacy_path),
            "export full-capacity legacy migration fixture")) return 39;
    std::vector<uint8_t> legacy_bytes = read_test_file(legacy_path);
    const size_t version11_bytes = sizeof(BgfWorldDigestionSettings);
    if (legacy_bytes.size() <= version11_bytes + sizeof(uint64_t)) return 39;
    legacy_bytes.erase(legacy_bytes.end() - sizeof(uint64_t) - version11_bytes,
        legacy_bytes.end() - sizeof(uint64_t));
    const size_t version10_bytes = static_cast<size_t>(config.max_bibites) *
        (sizeof(uint8_t) + 11u * sizeof(uint32_t) +
         324u * sizeof(uint16_t) + sizeof(uint16_t));
    if (legacy_bytes.size() <= version10_bytes + sizeof(uint64_t)) return 39;
    legacy_bytes.erase(legacy_bytes.end() - sizeof(uint64_t) - version10_bytes,
        legacy_bytes.end() - sizeof(uint64_t));
    const size_t extension_bytes =
        static_cast<size_t>(config.pellet_count) * sizeof(int32_t) + sizeof(float) +
        static_cast<size_t>(config.max_bibites) *
            (4u * sizeof(float) + 2u * 4u * sizeof(float)) +
        4u * sizeof(float) +
        static_cast<size_t>(config.max_bibites) * 40u +
        static_cast<size_t>(config.pellet_count) * sizeof(int32_t) +
        static_cast<size_t>(config.pheromone_grid_width) *
            config.pheromone_grid_height * 4u * 3u * sizeof(float);
    if (legacy_bytes.size() <= extension_bytes + sizeof(uint64_t)) return 39;
    legacy_bytes.erase(legacy_bytes.end() - sizeof(uint64_t) - extension_bytes,
        legacy_bytes.end() - sizeof(uint64_t));
    const size_t food_bytes = sizeof(int32_t) *
        static_cast<size_t>(config.pellet_count);
    const size_t hidden_brain_bytes = static_cast<size_t>(config.max_bibites) *
        (13u * sizeof(uint32_t) + 408u * sizeof(uint16_t) +
            24u * 3u * sizeof(uint16_t) + sizeof(uint16_t));
    bool legacy_ok = legacy_bytes.size() >
        food_bytes + hidden_brain_bytes + sizeof(uint64_t) + 12u;
    if (legacy_ok) {
        legacy_bytes.erase(
            legacy_bytes.end() - sizeof(uint64_t) - hidden_brain_bytes,
            legacy_bytes.end() - sizeof(uint64_t));
        const uint32_t version3 = 3u;
        std::memcpy(legacy_bytes.data() + 8u, &version3,
            sizeof(version3));
        legacy_ok = write_test_file(legacy_path, legacy_bytes);
    }
    BgfWorldHandle legacy_world = nullptr;
    legacy_ok = legacy_ok &&
        check(bgf_world_load_checkpoint(legacy_path, config.device_index,
            &legacy_world), "load v3 brain checkpoint") &&
        check(bgf_world_step(legacy_world, 1, &restored_metrics),
            "continue migrated v3 brain checkpoint");
    if (legacy_world) bgf_world_destroy(legacy_world);
    legacy_world = nullptr;
    std::remove(legacy_path);
    if (!require(legacy_ok, "v3 saves must remain loadable")) {
        std::remove(checkpoint_path);
        bgf_world_destroy(world);
        return 39;
    }
    legacy_bytes.erase(
        legacy_bytes.end() - sizeof(uint64_t) - food_bytes,
        legacy_bytes.end() - sizeof(uint64_t));
    const uint32_t version2 = 2u;
    std::memcpy(legacy_bytes.data() + 8u, &version2,
        sizeof(version2));
    legacy_ok = write_test_file(legacy_path, legacy_bytes) &&
        check(bgf_world_load_checkpoint(legacy_path, config.device_index,
            &legacy_world), "load v2 feeding checkpoint") &&
        check(bgf_world_step(legacy_world, 1, &restored_metrics),
            "continue migrated v2 feeding checkpoint");
    if (legacy_world) bgf_world_destroy(legacy_world);
    std::remove(legacy_path);
    if (!require(legacy_ok, "v2 saves must remain loadable")) {
        std::remove(checkpoint_path);
        bgf_world_destroy(world);
        return 39;
    }
    std::remove(checkpoint_path);

    bgf_world_destroy(world);

    // A camera close-up must still find Bibites omitted from the 8192-member
    // global UI sample in a larger population.
    BgfWorldConfig crowded_config{};
    bgf_world_default_config(&crowded_config);
    crowded_config.max_bibites = 8256;
    crowded_config.initial_bibites = crowded_config.max_bibites;
    crowded_config.pellet_count = 256;
    crowded_config.world_half_extent = 1000.0f;
    BgfWorldHandle crowded_world = nullptr;
    if (!check(bgf_world_create(&crowded_config, &crowded_world),
            "crowded viewport world creation")) return 52;
    std::vector<BgfWorldBibite> crowded_sample(8192);
    std::vector<BgfWorldBibite> crowded_all(
        static_cast<size_t>(crowded_config.max_bibites));
    std::vector<BgfWorldPellet> crowded_pellets(
        static_cast<size_t>(crowded_config.pellet_count));
    int32_t crowded_sample_written = 0;
    int32_t crowded_all_written = 0;
    int32_t crowded_pellets_written = 0;
    BgfWorldStats crowded_stats{};
    if (!check(bgf_world_download_snapshot(
            crowded_world, crowded_sample.data(), 8192,
            &crowded_sample_written,
            crowded_pellets.data(), crowded_config.pellet_count,
            &crowded_pellets_written, &crowded_stats),
            "crowded global UI sample") ||
        !check(bgf_world_download_bibites(
            crowded_world, crowded_all.data(), crowded_config.max_bibites,
            &crowded_all_written), "crowded full population") ||
        !require(crowded_sample_written == 8192 &&
                crowded_all_written == crowded_config.max_bibites,
            "crowded world must exceed the global UI sample")) {
        bgf_world_destroy(crowded_world);
        return 52;
    }
    std::vector<uint8_t> sampled_slots(
        static_cast<size_t>(crowded_config.max_bibites), 0);
    for (int32_t index = 0; index < crowded_sample_written; ++index)
        sampled_slots[crowded_sample[index].slot] = 1;
    int32_t omitted_slot = -1;
    float omitted_x = 0.0f;
    float omitted_y = 0.0f;
    for (int32_t index = 0; index < crowded_all_written; ++index) {
        if (sampled_slots[crowded_all[index].slot]) continue;
        omitted_slot = crowded_all[index].slot;
        omitted_x = crowded_all[index].position_x;
        omitted_y = crowded_all[index].position_y;
        break;
    }
    std::vector<BgfWorldBibite> crowded_visible(8);
    int32_t crowded_visible_written = 0;
    if (!require(omitted_slot >= 0,
            "crowded population must contain a Bibite outside the global UI sample") ||
        !check(bgf_world_download_snapshot_with_view(
            crowded_world, crowded_sample.data(), 8192,
            &crowded_sample_written,
            crowded_pellets.data(), crowded_config.pellet_count,
            &crowded_pellets_written,
            crowded_visible.data(), static_cast<int32_t>(crowded_visible.size()),
            &crowded_visible_written,
            omitted_x - 0.01f, omitted_y - 0.01f,
            omitted_x + 0.01f, omitted_y + 0.01f,
            &crowded_stats), "crowded close-up snapshot")) {
        bgf_world_destroy(crowded_world);
        return 52;
    }
    bool found_omitted = false;
    for (int32_t index = 0; index < crowded_visible_written; ++index)
        found_omitted |= crowded_visible[index].slot == omitted_slot;
    if (!require(found_omitted,
            "close-up textures must include Bibites omitted from the global UI sample")) {
        bgf_world_destroy(crowded_world);
        return 52;
    }
    bgf_world_destroy(crowded_world);

    // Regression for the large-map extinction bug. The original native world
    // kept a fixed 32-unit vision radius on a 30,000-unit-wide map, so almost
    // every starter starved in about five simulated minutes.
    BgfWorldConfig large_config{};
    bgf_world_default_config(&large_config);
    large_config.max_bibites = 4096;
    large_config.initial_bibites = 512;
    large_config.pellet_count = 2048;
    large_config.world_half_extent = 15000.0f;
    large_config.seed = 0x00c0ffeeu;
    large_config.contact_grid_update_factor = 4;
    large_config.contact_solve_factor = 8;
    large_config.vision_lookup_factor = 160;
    large_config.brain_update_factor = 8;
    large_config.lock_food_target = 1;
    BgfWorldHandle large_world = nullptr;
    if (!check(bgf_world_create(&large_config, &large_world),
            "large-map ecology world creation")) {
        return 25;
    }
    for (int32_t completed = 0; completed < 12000; completed += 256) {
        BgfWorldStepMetrics large_metrics{};
        if (!check(bgf_world_step(
                large_world,
                std::min(256, 12000 - completed),
                &large_metrics), "large-map ecology step")) {
            bgf_world_destroy(large_world);
            return 26;
        }
        if (completed % 3072 == 0) {
            BgfWorldStats progress{};
            if (check(bgf_world_get_stats(large_world, &progress),
                    "large-map ecology progress"))
                std::cout << "Ecology step " << progress.completed_steps << ": "
                          << progress.living_bibites << " living, "
                          << progress.pellets_eaten << " eaten, "
                          << progress.average_energy << " mean energy, "
                          << progress.active_pellets << " plants\n";
        }
    }
    BgfWorldStats large_stats{};
    if (!check(bgf_world_get_stats(large_world, &large_stats),
            "large-map ecology statistics")) {
        bgf_world_destroy(large_world);
        return 27;
    }
    std::cout << "Large-map feeding: " << large_stats.pellets_eaten
              << " pellets, " << large_stats.living_bibites << " living, "
              << large_stats.births << " births, " << large_stats.starvation_deaths
              << " starvation deaths, average energy " << large_stats.average_energy
              << ", " << large_stats.active_pellets << " plants"
              << "\n";
    if (!require(large_stats.living_bibites >= large_config.initial_bibites / 4,
            "large-map population must not undergo total five-minute extinction") ||
        !require(large_stats.pellets_eaten >=
                static_cast<uint64_t>(large_config.initial_bibites / 2),
            "large-map Bibites must finish substantial distributed food") ||
        !require(large_stats.age_deaths == 0,
            "starter lifespan must exceed the five-minute regression window") ||
        !require(large_stats.invalid_state_deaths == 0,
            "large-map integration must not create invalid state") ||
        !require(large_stats.deaths == large_stats.starvation_deaths +
                large_stats.age_deaths + large_stats.invalid_state_deaths,
            "large-map death causes must reconcile")) {
        bgf_world_destroy(large_world);
        return 27;
    }
    bgf_world_destroy(large_world);
    std::cout << "GPU world invariant tests passed on " << device_name
              << ": " << stats.living_bibites << " living Bibites, "
              << stats.pellets_eaten << " pellets eaten, "
              << first.gpu_offload_percent << "% first-batch GPU share; large-map "
              << large_stats.living_bibites << "/" << large_config.initial_bibites
              << " survived five simulated minutes.\n";
    return 0;
}
