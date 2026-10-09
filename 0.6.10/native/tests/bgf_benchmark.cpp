#include "bgf_api.h"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

constexpr int brain_count = 8192;
constexpr int input_count = 32;
constexpr int target_count_per_brain = 32;
constexpr int node_count_per_brain = input_count + target_count_per_brain;
constexpr int edges_per_target = 8;
constexpr int measured_steps = 200;

bool check(int status, const char* operation)
{
    if (status == 0) return true;
    std::cerr << operation << " failed with CUDA error " << status << ".\n";
    return false;
}

} // namespace

int main()
{
    int32_t device_count = 0;
    if (!check(bgf_get_device_count(&device_count), "device discovery") || device_count == 0) {
        return 2;
    }

    std::vector<BgfBrain> brains(brain_count);
    std::vector<BgfNode> nodes(brain_count * node_count_per_brain);
    std::vector<BgfTarget> targets(brain_count * target_count_per_brain);
    std::vector<BgfEdge> edges(brain_count * target_count_per_brain * edges_per_target);

    for (int brain = 0; brain < brain_count; ++brain) {
        brains[brain] = {
            brain * node_count_per_brain,
            brain * target_count_per_brain,
            target_count_per_brain,
            0.025f};
        const int node_offset = brain * node_count_per_brain;
        for (int input = 0; input < input_count; ++input) {
            const float value = static_cast<float>((brain + input) % 17) / 16.0f;
            nodes[node_offset + input] = {value, value, 0.0f, 0.0f, 0};
        }
        for (int target = 0; target < target_count_per_brain; ++target) {
            const int target_index = brain * target_count_per_brain + target;
            const int edge_offset = target_index * edges_per_target;
            targets[target_index] = {input_count + target, edge_offset, edges_per_target};
            nodes[node_offset + input_count + target] = {0.0f, 0.0f, 0.0f, 0.05f, 1};
            for (int edge = 0; edge < edges_per_target; ++edge) {
                const float weight = static_cast<float>(((target * 3 + edge * 5) % 15) - 7) / 8.0f;
                edges[edge_offset + edge] = {
                    (target + edge * 3) % input_count,
                    bgf_float_to_half_bits(weight),
                    0};
            }
        }
    }

    std::cout << "Workload: " << brain_count << " brains, "
              << target_count_per_brain << " evaluated neurons/brain, "
              << edges_per_target << " synapses/neuron, "
              << measured_steps << " steps.\n";

    for (int device = 0; device < device_count; ++device) {
        char name[256]{};
        if (!check(bgf_get_device_name(device, name, sizeof(name)), "device name")) return 3;

        BgfContextHandle context = nullptr;
        if (!check(bgf_create_context(
                device,
                static_cast<int32_t>(brains.size()),
                static_cast<int32_t>(nodes.size()),
                static_cast<int32_t>(targets.size()),
                static_cast<int32_t>(edges.size()),
                &context), "context creation")) return 4;
        if (!check(bgf_upload_brains(context, brains.data(), static_cast<int32_t>(brains.size())), "brain upload") ||
            !check(bgf_upload_nodes(context, nodes.data(), static_cast<int32_t>(nodes.size())), "node upload") ||
            !check(bgf_upload_topology(
                context,
                targets.data(),
                static_cast<int32_t>(targets.size()),
                edges.data(),
                static_cast<int32_t>(edges.size())), "topology upload")) {
            bgf_destroy_context(context);
            return 5;
        }

        for (int step = 0; step < 10; ++step) {
            if (!check(bgf_step(context), "warmup")) return 6;
        }
        const auto started = std::chrono::steady_clock::now();
        for (int step = 0; step < measured_steps; ++step) {
            if (!check(bgf_step(context), "benchmark step")) return 7;
        }
        const auto stopped = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(stopped - started).count();
        const double brain_steps = static_cast<double>(brain_count) * measured_steps;
        const double synapse_ops = brain_steps * target_count_per_brain * edges_per_target;

        std::cout << name << ": " << std::fixed << std::setprecision(2)
                  << brain_steps / seconds / 1.0e6 << " million brain-steps/s, "
                  << synapse_ops / seconds / 1.0e9 << " billion synapse operations/s, "
                  << seconds * 1000.0 / measured_steps << " ms/step.\n";
        bgf_destroy_context(context);
    }

    return 0;
}
