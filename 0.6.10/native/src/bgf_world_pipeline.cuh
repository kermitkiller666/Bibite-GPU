// Included inside bgf_world.cu's private namespace after its device stages.
__global__ void restart_profile_phase_kernel(DeviceWorld world)
{
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(world.counters->phase_started_cycle));
}

__global__ void reset_profile_kernel(DeviceWorld world)
{
    world.counters->phase_prepare_cycles = 0;
    world.counters->phase_spatial_index_cycles = 0;
    world.counters->phase_contact_cycles = 0;
    world.counters->phase_decision_cycles = 0;
    world.counters->phase_motion_cycles = 0;
    world.counters->phase_lifecycle_cycles = 0;
    world.timing->food_cycles = world.timing->brain_cycles = world.timing->post_cycles = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(world.counters->phase_started_cycle));
}

template <int Stage>
cudaError_t capture_stage(GpuWorldContext& context, cudaStream_t stream)
{
    int one = 1;
    void* args[] = {&context.world, &one};
    if constexpr (Stage==1 || Stage==7) {
        int blocks_per_sm=0;
        cudaError_t status=cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm,world_stage_kernel<Stage>,kThreads,0);
        if (status!=cudaSuccess) return status;
        const int blocks=std::max(1,blocks_per_sm*context.properties.multiProcessorCount);
        return cudaLaunchCooperativeKernel(reinterpret_cast<void*>(world_stage_kernel<Stage>),
            blocks,kThreads,args,0,stream);
    } else {
        // Kernel dependencies already fence contacts, inputs, actions and
        // motion. These stages have no whole-grid barrier or residency limit.
        const int work=context.latest_living*(Stage==2?8:1);
        const int blocks=std::max(1,std::min(256,(work+kThreads-1)/kThreads));
        return cudaLaunchKernel(reinterpret_cast<void*>(world_stage_kernel<Stage>),
            blocks,kThreads,args,0,stream);
    }
}

cudaError_t ensure_step_graph(GpuWorldContext& context, int steps)
{
    const auto& world=context.world;
    const int vision=(world.diagnostic_mask&262144u)?160:
        (world.diagnostic_mask&131072u)?80:(world.diagnostic_mask&65536u)?40:world.vision_lookup_factor;
    const int brain=(world.diagnostic_mask&1048576u)?8:
        (world.diagnostic_mask&524288u)?4:world.brain_update_factor;
    int period=std::lcm(kSenseUpdateFactor,brain);
    if (vision%kSenseUpdateFactor!=0) period=std::lcm(period,vision);
    const int phase=static_cast<int>(context.host_steps%period);
    if (!context.graph_world_valid ||
        std::memcmp(&context.graph_world,&context.world,sizeof(DeviceWorld))!=0) {
        // Serialized stepping has already retired every prior launch before
        // pointers/settings change. Invalidate all graphs holding old arguments.
        for (auto& entry:context.graph_cache) {
            if (entry.graph) cudaGraphExecDestroy(entry.graph);
            entry={};
        }
        context.step_graph=nullptr;
        context.graph_world=context.world;
        context.graph_world_valid=true;
    }
    for (auto& entry:context.graph_cache) {
        if (entry.graph && entry.steps==steps && entry.cadence_phase==phase) {
            entry.last_used=++context.graph_cache_clock;
            context.step_graph=entry.graph;
            ++context.graph_cache_hits;
            return cudaSuccess;
        }
    }
    const auto build_started=std::chrono::steady_clock::now();
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cudaError_t status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (status != cudaSuccess) return status;
    status = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    const void* brain_kernel=(context.world.diagnostic_mask&16777216u) ?
        reinterpret_cast<void*>(tensor_brain_kernel) :
        (context.world.diagnostic_mask&33554432u) ?
        reinterpret_cast<void*>(native_brain_kernel<true>) :
        reinterpret_cast<void*>(native_brain_kernel<false>);
    for (int step = 0; status == cudaSuccess && step < steps; ++step) {
        const uint64_t tick=context.host_steps+step;
        const bool run_brain=tick%brain==0 && !(world.diagnostic_mask&4u);
        const bool run_perception=tick%vision==0 || tick%kSenseUpdateFactor==0;
        status = capture_stage<1>(context, stream);
        if (status == cudaSuccess) status = capture_stage<2>(context, stream);
        if (status == cudaSuccess && (run_brain || run_perception)) status = capture_stage<3>(context, stream);
        if (status == cudaSuccess && run_brain) {
            void* args[] = {&context.world};
            const int bodies_per_block=(world.diagnostic_mask&16777216u)?4:128;
            const int blocks=std::max(1,std::min(256,
                (context.latest_living+bodies_per_block-1)/bodies_per_block));
            status = cudaLaunchKernel(brain_kernel,blocks,128,args,0,stream);
        }
        if (status == cudaSuccess) status = capture_stage<5>(context, stream);
        if (status == cudaSuccess) status = capture_stage<6>(context, stream);
        if (status == cudaSuccess) status = capture_stage<7>(context, stream);
    }
    const cudaError_t end_status = cudaStreamEndCapture(stream, &graph);
    if (status == cudaSuccess) status = end_status;
    if (status == cudaSuccess) status = cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0);
    if (graph) cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    context.graph_build_milliseconds+=static_cast<float>(
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-build_started).count());
    if (status != cudaSuccess) {
        if (executable) cudaGraphExecDestroy(executable);
        return status;
    }
    StepGraphCacheEntry* selected=&context.graph_cache[0];
    for (auto& entry:context.graph_cache)
        if (!entry.graph || entry.last_used<selected->last_used) selected=&entry;
    if (selected->graph) cudaGraphExecDestroy(selected->graph);
    selected->graph=executable;
    selected->steps=steps;
    selected->cadence_phase=phase;
    selected->last_used=++context.graph_cache_clock;
    context.step_graph = executable;
    return cudaSuccess;
}
