#include "bgf_api.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

void require_close(float expected, float actual, float tolerance, const char* label)
{
    if (std::fabs(expected - actual) > tolerance) {
        std::cerr << label << ": expected " << expected << ", got " << actual << '\n';
        std::exit(1);
    }
}

} // namespace

int main()
{
    int32_t device_count = 0;
    int status = bgf_get_device_count(&device_count);
    if (status != 0 || device_count < 1) {
        std::cerr << "No CUDA device available; status=" << status << '\n';
        return 2;
    }

    for (int32_t device = 0; device < device_count; ++device) {
        if (bgf_set_device(device) != 0) {
            std::cerr << "Unable to select CUDA device " << device << ".\n";
            return 3;
        }
    }
    if (bgf_set_device(0) != 0) {
        std::cerr << "Unable to restore CUDA device 0.\n";
        return 3;
    }

    char device_name[256]{};
    if (bgf_get_device_name(0, device_name, sizeof(device_name)) != 0) {
        std::cerr << "Unable to read CUDA device name.\n";
        return 3;
    }

    // Two inputs feed a sigmoid output: sigmoid(0.5*2 + -0.25*4 + 0.1).
    BgfNode nodes[3] = {
        {2.0f, 2.0f, 0.0f, 0.0f, 0},
        {4.0f, 4.0f, 0.0f, 0.0f, 0},
        {0.0f, 0.0f, 0.0f, 0.1f, 1}
    };
    BgfEdge edges[2] = {
        {0, bgf_float_to_half_bits(0.5f), 0},
        {1, bgf_float_to_half_bits(-0.25f), 0}
    };
    BgfTarget targets[1] = {{2, 0, 2}};
    BgfBrain brains[1] = {{0, 0, 1, 0.025f}};

    status = bgf_evaluate_brains(brains, 1, nodes, 3, targets, 1, edges, 2);
    if (status != 0) {
        std::cerr << "GPU evaluation failed; status=" << status << '\n';
        return 4;
    }

    const float expected = 1.0f / (1.0f + std::exp(-0.1f));
    require_close(expected, nodes[2].value, 0.0002f, "sigmoid output");
    require_close(0.5f, bgf_half_bits_to_float(bgf_float_to_half_bits(0.5f)), 0.0001f, "half roundtrip");

    nodes[2] = {0.0f, 0.0f, 0.0f, 0.1f, 1};
    BgfContextHandle context = nullptr;
    status = bgf_create_context(0, 1, 3, 1, 2, &context);
    if (status != 0 || !context) {
        std::cerr << "Context creation failed; status=" << status << '\n';
        return 5;
    }
    status = bgf_upload_brains(context, brains, 1);
    if (status == 0) status = bgf_upload_nodes(context, nodes, 3);
    if (status == 0) status = bgf_upload_topology(context, targets, 1, edges, 2);
    if (status == 0) status = bgf_step(context);
    if (status == 0) status = bgf_download_nodes(context, nodes, 3);
    bgf_destroy_context(context);
    if (status != 0) {
        std::cerr << "Resident GPU evaluation failed; status=" << status << '\n';
        return 6;
    }
    require_close(expected, nodes[2].value, 0.0002f, "resident sigmoid output");

    BgfVisionQuery vision_query{
        0.0f, 0.0f,
        0.0f, 1.0f,
        10.0f, 180.0f,
        3.0f,
        1.0f, 1.0f, 0.0f,
        0, 4, 100, 7};
    BgfVisionEntity vision_entities[4] = {
        {0.0f, 5.0f, 0.0f, 1.0f, 0.0f, 2.0f, 0.0f, 1.0f,
         0.0f, 0.0f, 0.0f, BGF_VISION_PLANT, 0, 1, 0},
        {0.0f, 3.0f, 0.0f, 1.0f, 0.0f, 5.0f, 0.0f, 1.0f,
         0.0f, 0.0f, 0.0f, BGF_VISION_PLANT, BGF_VISION_HELD, 2, 100},
        {0.0f, 4.0f, 0.0f, 1.0f, 0.0f, 1.0f, 100.0f, 1.0f,
         0.2f, 0.3f, 0.4f, BGF_VISION_BIBITE, BGF_VISION_COLOR_READY, 3, 0},
        {0.0f, -3.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
         0.0f, 0.0f, 0.0f, BGF_VISION_MEAT, 0, 4, 0}
    };
    BgfVisionResult vision_result{};
    status = bgf_evaluate_vision(&vision_query, 1, vision_entities, 4, &vision_result);
    if (status != 0) {
        std::cerr << "GPU vision evaluation failed; status=" << status << '\n';
        return 7;
    }
    if (vision_result.plant_count != 1 || vision_result.meat_count != 0 ||
        vision_result.bibite_count != 1 || vision_result.has_herd != 1) {
        std::cerr << "GPU vision counts did not match the reference scene.\n";
        return 8;
    }
    require_close(0.0f, vision_result.plant_angle, 0.0002f, "plant angle");
    require_close(0.5f, vision_result.plant_weight, 0.0002f, "plant concentration");
    require_close(0.63525f, vision_result.max_plant_weight, 0.0002f, "plant max weight");
    require_close(0.0f, vision_result.bibite_angle, 0.0002f, "bibite angle");
    require_close(0.6f, vision_result.bibite_weight, 0.0002f, "bibite concentration");
    require_close(0.2f, vision_result.target_r, 0.0002f, "target red");
    require_close(0.3f, vision_result.target_g, 0.0002f, "target green");
    require_close(0.4f, vision_result.target_b, 0.0002f, "target blue");
    require_close(0.0f, vision_result.herd_direction, 0.0002f, "herd direction");
    require_close(1.0f, vision_result.herd_separation_projection, 0.0002f, "herd separation projection");

    BgfVisionContextHandle vision_context = nullptr;
    status = bgf_create_vision_context(0, 2, 8, &vision_context);
    if (status == 0) {
        BgfVisionResult resident_vision_result{};
        status = bgf_evaluate_vision_context(
            vision_context,
            &vision_query,
            1,
            vision_entities,
            4,
            &resident_vision_result);
        bgf_destroy_vision_context(vision_context);
        require_close(
            vision_result.max_plant_weight,
            resident_vision_result.max_plant_weight,
            0.0002f,
            "resident vision result");
    }
    if (status != 0) {
        std::cerr << "Resident GPU vision evaluation failed; status=" << status << '\n';
        return 9;
    }

    BgfPheromoneQuery pheromone_query{
        0.0f, 0.0f,
        0.0f, 1.0f,
        10.0f,
        100.0f,
        0, 0};
    BgfPheromoneEntity pheromone_entities[2] = {
        {0.0f, 5.0f, 1.0f, 0.0f, 10.0f, 0.0f, 0.0f, 0.0f},
        {5.0f, 0.0f, 0.0f, 1.0f, 0.0f, 5.0f, 0.0f, 0.0f}
    };
    BgfPheromoneContextHandle pheromone_context = nullptr;
    status = bgf_create_pheromone_context(0, 2, 4, &pheromone_context);
    BgfPheromoneResult pheromone_result{};
    if (status == 0) {
        status = bgf_evaluate_pheromones_context(
            pheromone_context,
            &pheromone_query,
            1,
            pheromone_entities,
            2,
            &pheromone_result);
        bgf_destroy_pheromone_context(pheromone_context);
    }
    if (status != 0) {
        std::cerr << "Resident GPU pheromone evaluation failed; status=" << status << '\n';
        return 10;
    }
    require_close(2.0f, pheromone_result.red_sum, 0.0002f, "pheromone red sum");
    require_close(1.0f, pheromone_result.green_sum, 0.0002f, "pheromone green sum");
    require_close(0.0f, pheromone_result.red_angle, 0.0002f, "pheromone red angle");
    require_close(-0.5f, pheromone_result.green_angle, 0.0002f, "pheromone green angle");
    require_close(-0.5f, pheromone_result.red_heading_angle, 0.0002f, "pheromone red heading");
    require_close(0.0f, pheromone_result.green_heading_angle, 0.0002f, "pheromone green heading");

    std::cout << "CUDA brain, vision and pheromone parity tests passed on " << device_name << ".\n";
    return 0;
}
