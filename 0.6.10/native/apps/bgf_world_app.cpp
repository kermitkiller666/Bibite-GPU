#include "bgf_world.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

namespace {

struct Options {
    int32_t device = 0;
    int32_t bibites = 2048;
    int32_t capacity = 8192;
    int32_t pellets = 8192;
    int32_t extent = 500;
    int32_t steps = 4096;
    int32_t chunk = 256;
    uint32_t seed = 0x00c0ffeeu;
    uint32_t diagnostic_mask = 0u;
    bool list_gpus = false;
    bool extreme = false;
    bool pause = false;
    std::string report_path;
};

void print_usage()
{
    std::cout
        << "Bibites GPU Engine - persistent CUDA evolution simulation\n\n"
        << "Options:\n"
        << "  --device N       CUDA device index (default 0)\n"
        << "  --bibites N      Initial population (default 2048)\n"
        << "  --capacity N     Maximum population (default 8192)\n"
        << "  --pellets N      Persistent food pellets (default 8192)\n"
        << "  --extent N       World half-extent in simulation units (default 500)\n"
        << "  --steps N        Measured simulation steps (default 4096)\n"
        << "  --chunk N        Steps in each persistent GPU launch (default 256)\n"
        << "  --seed N         World seed\n"
        << "  --kernel MODE    fused (default), sparse, dense, or tensor (experimental)\n"
        << "  --body-layout M  flat (default), warp, half (fused-kernel comparisons)\n"
        << "  --fused-occupancy M  standard (default), high (experimental register cap)\n"
              << "  --diagnostic N   Profiling mask: 1 contact, 2 perception, 4 brains, 8 legacy contact,\n"
              << "                   16 force warp tiles, 32 scalar contact, 256/512 grid reuse,\n"
              << "                   1024/2048 solve contacts every 2/4 ticks\n"
              << "                   8388608 specialized sparse brain graph,\n"
              << "                   16777216 experimental FP16 Tensor Core brain,\n"
              << "                   33554432 specialized dense brain graph\n"
        << "  --report PATH    Save a JSON benchmark report\n"
        << "  --extreme        Lower refresh cadence; no speed multiplier is guaranteed\n"
        << "  --list-gpus      List available CUDA devices and exit\n"
        << "  --pause          Wait for Enter before closing\n"
        << "  --help           Show this help\n";
}

bool parse_i32(const char* text, int32_t& value)
{
    if (!text || !*text) return false;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (!end || *end != '\0' || parsed < 0 ||
        parsed > std::numeric_limits<int32_t>::max()) {
        return false;
    }
    value = static_cast<int32_t>(parsed);
    return true;
}

bool parse_u32(const char* text, uint32_t& value)
{
    if (!text || !*text) return false;
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (!end || *end != '\0' || parsed > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    value = static_cast<uint32_t>(parsed);
    return true;
}

bool parse_options(int argc, char** argv, Options& options)
{
    options.pause = argc == 1;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            print_usage();
            std::exit(0);
        }
        if (argument == "--list-gpus") {
            options.list_gpus = true;
            continue;
        }
        if (argument == "--pause") {
            options.pause = true;
            continue;
        }
        if (argument == "--extreme") {
            options.extreme = true;
            continue;
        }
        if (argument == "--report") {
            if (++index >= argc) return false;
            options.report_path = argv[index];
            continue;
        }
        if (++index >= argc) return false;
        if (argument == "--device") {
            if (!parse_i32(argv[index], options.device)) return false;
        } else if (argument == "--bibites") {
            if (!parse_i32(argv[index], options.bibites)) return false;
        } else if (argument == "--capacity") {
            if (!parse_i32(argv[index], options.capacity)) return false;
        } else if (argument == "--pellets") {
            if (!parse_i32(argv[index], options.pellets)) return false;
        } else if (argument == "--extent") {
            if (!parse_i32(argv[index], options.extent)) return false;
        } else if (argument == "--steps") {
            if (!parse_i32(argv[index], options.steps)) return false;
        } else if (argument == "--chunk") {
            if (!parse_i32(argv[index], options.chunk)) return false;
        } else if (argument == "--seed") {
            if (!parse_u32(argv[index], options.seed)) return false;
        } else if (argument == "--diagnostic") {
            if (!parse_u32(argv[index], options.diagnostic_mask)) return false;
        } else if (argument == "--kernel") {
            const std::string mode=argv[index];
            uint32_t flag=0;
            if (mode=="sparse") flag=8388608u;
            else if (mode=="dense") flag=33554432u;
            else if (mode=="tensor") flag=16777216u;
            else if (mode!="fused") return false;
            options.diagnostic_mask=(options.diagnostic_mask&~(8388608u|16777216u|33554432u))|flag;
        } else if (argument == "--body-layout") {
            const std::string mode=argv[index];
            uint32_t flag=0;
            if (mode=="warp") flag=67108864u;
            else if (mode=="half") flag=134217728u;
            else if (mode!="flat") return false;
            options.diagnostic_mask=(options.diagnostic_mask&~(67108864u|134217728u))|flag;
        } else if (argument == "--fused-occupancy") {
            const std::string mode=argv[index];
            if (mode=="high") options.diagnostic_mask|=268435456u;
            else if (mode=="standard") options.diagnostic_mask&=~268435456u;
            else return false;
        } else {
            return false;
        }
    }
    return options.bibites > 0 && options.capacity >= options.bibites &&
        options.pellets > 0 && options.extent >= 100 && options.extent <= 20000 &&
        options.steps > 0 && options.chunk > 0;
}

bool check(int status, const char* operation)
{
    if (status == 0) return true;
    std::cerr << operation << " failed with CUDA error " << status << ".\n";
    return false;
}

void wait_if_requested(bool pause)
{
    if (!pause) return;
    std::cout << "\nPress Enter to close..." << std::flush;
    std::cin.get();
}

} // namespace

int main(int argc, char** argv)
{
    Options options{};
    if (!parse_options(argc, argv, options)) {
        std::cerr << "Invalid arguments.\n\n";
        print_usage();
        return 2;
    }

    int32_t device_count = 0;
    if (!check(bgf_get_device_count(&device_count), "CUDA device discovery") || device_count <= 0) {
        std::cerr << "No supported NVIDIA CUDA GPU was found.\n";
        wait_if_requested(options.pause);
        return 3;
    }
    if (options.list_gpus) {
        for (int32_t device = 0; device < device_count; ++device) {
            char name[256]{};
            if (!check(bgf_get_device_name(device, name, sizeof(name)), "CUDA device name")) return 4;
            std::cout << device << ": " << name << "\n";
        }
        return 0;
    }
    if (options.device >= device_count) {
        std::cerr << "CUDA device " << options.device << " does not exist.\n";
        return 5;
    }

    BgfWorldConfig config{};
    bgf_world_default_config(&config);
    config.device_index = options.device;
    config.max_bibites = options.capacity;
    config.initial_bibites = options.bibites;
    config.pellet_count = options.pellets;
    config.world_half_extent = static_cast<float>(options.extent);
    config.seed = options.seed;
    config.diagnostic_mask = options.diagnostic_mask;
    if (options.extreme) {
        config.contact_grid_update_factor = 4;
        config.contact_solve_factor = 8;
        config.vision_lookup_factor = 160;
        config.brain_update_factor = 8;
        config.lock_food_target = 1;
    }

    BgfWorldHandle world = nullptr;
    if (!check(bgf_world_create(&config, &world), "GPU world creation")) {
        wait_if_requested(options.pause);
        return 6;
    }
    char device_name[256]{};
    if (!check(bgf_world_get_device_name(world, device_name, sizeof(device_name)), "GPU name")) {
        bgf_world_destroy(world);
        return 7;
    }

    std::cout << "Bibites GPU Engine\n"
              << "GPU: " << device_name << " (device " << options.device << ")\n"
              << "World: " << options.bibites << " initial / " << options.capacity
              << " maximum Bibites, " << options.pellets << " pellets\n"
              << "World half-extent: " << options.extent << "\n"
              << "Profile: " << (options.extreme ? "Extreme throughput" : "Exact cadence") << "\n"
              << "Precision: FP32 world state and neural accumulation; FP16 weights, hidden biases and cached hidden states\n"
              << "Pipeline: spatial grid + sensing + brains + physics + feeding + evolution + pheromones\n\n";

    const int32_t warmup_steps = std::min(options.chunk, 32);
    BgfWorldStepMetrics warmup{};
    if (!check(bgf_world_step(world, warmup_steps, &warmup), "GPU warmup")) {
        bgf_world_destroy(world);
        wait_if_requested(options.pause);
        return 8;
    }

    double total_gpu_ms = 0.0;
    double total_wall_ms = 0.0;
    double total_prepare_ms = 0.0;
    double total_spatial_index_ms = 0.0;
    double total_contact_ms = 0.0;
    double total_decision_ms = 0.0;
    double total_motion_ms = 0.0;
    double total_lifecycle_ms = 0.0;
    double total_food_ms=0.0,total_brain_ms=0.0,total_post_ms=0.0;
    double total_graph_build_ms=0.0;
    int total_graph_cache_hits=0;
    int pipeline=0;
    int32_t completed = 0;
    while (completed < options.steps) {
        const int32_t current = std::min(options.chunk, options.steps - completed);
        BgfWorldStepMetrics metrics{};
        if (!check(bgf_world_step(world, current, &metrics), "GPU simulation batch")) {
            bgf_world_destroy(world);
            wait_if_requested(options.pause);
            return 9;
        }
        total_gpu_ms += metrics.gpu_milliseconds;
        total_wall_ms += metrics.wall_milliseconds;
        total_prepare_ms += metrics.prepare_milliseconds;
        total_spatial_index_ms += metrics.spatial_index_milliseconds;
        total_contact_ms += metrics.contact_milliseconds;
        total_decision_ms += metrics.decision_milliseconds;
        total_motion_ms += metrics.motion_milliseconds;
        total_lifecycle_ms += metrics.lifecycle_milliseconds;
        total_food_ms+=metrics.food_milliseconds;
        total_brain_ms+=metrics.brain_milliseconds;
        total_post_ms+=metrics.post_decision_milliseconds;
        pipeline=metrics.pipeline;
        total_graph_build_ms+=metrics.graph_build_milliseconds;
        total_graph_cache_hits+=metrics.graph_cache_hits;
        completed += current;
    }

    BgfWorldStats stats{};
    if (!check(bgf_world_get_stats(world, &stats), "world statistics")) {
        bgf_world_destroy(world);
        return 10;
    }
    bgf_world_destroy(world);

    const double host_overhead_ms = std::max(total_wall_ms - total_gpu_ms, 0.0);
    const double measured_total_ms = total_gpu_ms + host_overhead_ms;
    const double offload_percent = measured_total_ms > 0.0
        ? total_gpu_ms / measured_total_ms * 100.0 : 0.0;
    const double simulated_seconds = static_cast<double>(options.steps) * config.fixed_delta_time;
    const double realtime_multiplier = total_wall_ms > 0.0
        ? simulated_seconds / (total_wall_ms / 1000.0) : 0.0;
    const double steps_per_second = total_wall_ms > 0.0
        ? static_cast<double>(options.steps) / (total_wall_ms / 1000.0) : 0.0;
    const bool target_passed = offload_percent >= 99.0;

    std::cout << std::fixed << std::setprecision(3)
              << "Measured GPU kernel time: " << total_gpu_ms << " ms\n"
              << "Measured host launch/sync overhead: " << host_overhead_ms << " ms\n"
              << "GPU simulation share: " << offload_percent << "%\n"
              << "GPU phases: prepare " << total_prepare_ms << " ms, spatial "
              << total_spatial_index_ms << " ms, contact " << total_contact_ms
              << " ms, decision " << total_decision_ms << " ms, motion " << total_motion_ms
              << " ms, lifecycle " << total_lifecycle_ms << " ms\n"
              << "Specialized decision phases (food/preparation, native brain, actions): "
              << total_food_ms << ", " << total_brain_ms << ", " << total_post_ms
              << " ms; pipeline " << pipeline << "\n"
              << "Throughput: " << steps_per_second << " steps/s, "
              << realtime_multiplier << "x real time\n"
              << "Population: " << stats.living_bibites << " living, "
              << stats.births << " births, " << stats.deaths << " deaths\n"
              << "Death causes: " << stats.starvation_deaths << " starvation, "
              << stats.age_deaths << " age, " << stats.invalid_state_deaths
              << " invalid state\n"
              << "Ecology: " << stats.pellets_eaten << " pellets eaten, average energy "
              << stats.average_energy << "\n"
              << "GPU_OFFLOAD_TARGET=" << (target_passed ? "PASS" : "FAIL")
              << " (required >= 99.000%)\n";

    if (!options.report_path.empty()) {
        std::ofstream report(options.report_path, std::ios::binary | std::ios::trunc);
        if (!report) {
            std::cerr << "Could not write report: " << options.report_path << "\n";
            wait_if_requested(options.pause);
            return 11;
        }
        report << std::fixed << std::setprecision(6)
               << "{\n"
               << "  \"engine\": \"Bibites GPU Engine\",\n"
               << "  \"device_index\": " << options.device << ",\n"
               << "  \"device_name\": \"" << device_name << "\",\n"
               << "  \"initial_bibites\": " << options.bibites << ",\n"
               << "  \"maximum_bibites\": " << options.capacity << ",\n"
               << "  \"pellets\": " << options.pellets << ",\n"
               << "  \"world_half_extent\": " << options.extent << ",\n"
               << "  \"profile\": \"" << (options.extreme ? "extreme" : "exact") << "\",\n"
               << "  \"measured_steps\": " << options.steps << ",\n"
               << "  \"steps_per_launch\": " << options.chunk << ",\n"
               << "  \"gpu_milliseconds\": " << total_gpu_ms << ",\n"
               << "  \"host_overhead_milliseconds\": " << host_overhead_ms << ",\n"
               << "  \"gpu_offload_percent\": " << offload_percent << ",\n"
               << "  \"phase_prepare_milliseconds\": " << total_prepare_ms << ",\n"
               << "  \"phase_spatial_index_milliseconds\": " << total_spatial_index_ms << ",\n"
               << "  \"phase_contact_milliseconds\": " << total_contact_ms << ",\n"
               << "  \"phase_decision_milliseconds\": " << total_decision_ms << ",\n"
               << "  \"phase_motion_milliseconds\": " << total_motion_ms << ",\n"
               << "  \"phase_lifecycle_milliseconds\": " << total_lifecycle_ms << ",\n"
               << "  \"phase_food_preparation_milliseconds\": " << total_food_ms << ",\n"
               << "  \"phase_native_brain_milliseconds\": " << total_brain_ms << ",\n"
               << "  \"phase_feeding_attack_milliseconds\": " << total_post_ms << ",\n"
               << "  \"pipeline\": " << pipeline << ",\n"
               << "  \"graph_build_milliseconds\": " << total_graph_build_ms << ",\n"
               << "  \"graph_cache_hits\": " << total_graph_cache_hits << ",\n"
               << "  \"full_call_wall_milliseconds\": " << total_wall_ms << ",\n"
               << "  \"steps_per_second\": " << steps_per_second << ",\n"
               << "  \"realtime_multiplier\": " << realtime_multiplier << ",\n"
               << "  \"living_bibites\": " << stats.living_bibites << ",\n"
               << "  \"births\": " << stats.births << ",\n"
               << "  \"deaths\": " << stats.deaths << ",\n"
               << "  \"starvation_deaths\": " << stats.starvation_deaths << ",\n"
               << "  \"age_deaths\": " << stats.age_deaths << ",\n"
               << "  \"invalid_state_deaths\": " << stats.invalid_state_deaths << ",\n"
               << "  \"pellets_eaten\": " << stats.pellets_eaten << ",\n"
               << "  \"target_percent\": 99.000000,\n"
               << "  \"target_passed\": " << (target_passed ? "true" : "false") << "\n"
               << "}\n";
    }

    wait_if_requested(options.pause);
    return target_passed ? 0 : 12;
}
