#include "bgf_world.h"
#include <d3d11.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <io.h>
#include <iterator>
#include <iostream>
#include <limits>
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
    const size_t body = header_bytes + 120u;
    const auto corrupt_int = [&](size_t offset, int32_t value, const char* message) {
        std::vector<uint8_t> changed = original;
        if (offset + sizeof(value) > changed.size()) return false;
        std::memcpy(changed.data() + offset, &value, sizeof(value));
        return rejects(changed, message);
    };
    if (!corrupt_int(body, 2, "checkpoint must reject transient/invalid alive flags") ||
        !corrupt_int(body + 292u * config.max_bibites, 1,
            "checkpoint must reject out-of-range template topology IDs") ||
        !corrupt_int(body + 344u * config.max_bibites + sizeof(int32_t), config.max_bibites + 1,
            "checkpoint must reject oversized free-slot queue counts")) return finish(false);
    corrupt = original;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::memcpy(corrupt.data() + body + 8u * config.max_bibites, &nan, sizeof(nan));
    if (!rejects(corrupt, "checkpoint must reject non-finite living positions")) return finish(false);

    BgfWorldHandle restored = nullptr;
    if (!check(bgf_world_load_checkpoint(path, 0, &restored),
            "valid checkpoint must still load after corruption attempts")) return finish(false);
    const bool restored_identity = check(bgf_world_get_bibite_detail(restored, 0, &after,
        nullptr, 0, &nodes, nullptr, 0, &synapses), "restored inspector identity") &&
        require(after.alive == 1 && after.reserved != 0,
            "loaded live Bibites must receive a runtime inspector identity");
    bgf_world_destroy(restored);
    BgfWorldStepMetrics metrics{};
    if (!restored_identity || !check(bgf_world_step(world, 4, &metrics),
            "corrupt checkpoint rejection must not poison the CUDA context")) return finish(false);
    return finish(true);
}

} // namespace

int main()
{
    if (!graphics_threading_regressions()) return 51;
    int32_t device_count = 0;
    if (!check(bgf_get_device_count(&device_count), "device discovery") ||
        !require(device_count > 0, "at least one CUDA device is required")) {
        return 1;
    }
    if (!checkpoint_stability_regressions()) return 50;

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
    // Simulate a v2 checkpoint: it has the same prefix, but no per-pellet
    // remaining-food array. Loading must initialise the new feeding state.
    const char* legacy_path = "bgf_world_legacy_food_test.bgfgpu";
    std::vector<uint8_t> legacy_bytes = read_test_file(checkpoint_path);
    const size_t food_bytes = sizeof(int32_t) *
        static_cast<size_t>(config.pellet_count);
    bool legacy_ok = legacy_bytes.size() > food_bytes + sizeof(uint64_t) + 12u;
    if (legacy_ok) {
        const uint32_t legacy_version = 2u;
        std::memcpy(legacy_bytes.data() + 8u, &legacy_version,
            sizeof(legacy_version));
        legacy_bytes.erase(
            legacy_bytes.end() - sizeof(uint64_t) - food_bytes,
            legacy_bytes.end() - sizeof(uint64_t));
        legacy_ok = write_test_file(legacy_path, legacy_bytes);
    }
    BgfWorldHandle legacy_world = nullptr;
    legacy_ok = legacy_ok &&
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
    }
    BgfWorldStats large_stats{};
    if (!check(bgf_world_get_stats(large_world, &large_stats),
            "large-map ecology statistics")) {
        bgf_world_destroy(large_world);
        return 27;
    }
    std::cout << "Large-map feeding: " << large_stats.pellets_eaten
              << " pellets, " << large_stats.living_bibites << " living\n";
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
