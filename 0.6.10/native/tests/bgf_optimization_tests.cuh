// Included by bgf_world_tests.cpp inside its private test namespace.
bool live_grid_regressions()
{
    BgfWorldConfig config{};
    bgf_world_default_config(&config);
    config.max_bibites=1024;config.initial_bibites=1;config.pellet_count=8192;
    config.pheromone_grid_width=config.pheromone_grid_height=16;
    config.reproduction_energy=1000000.0f;config.diagnostic_mask=6u;
    BgfWorldHandle world=nullptr;
    if (!check(bgf_world_create(&config,&world),"create density adaptation fixture")) return false;
    auto finish=[&](bool ok){bgf_world_destroy(world);return ok;};
    BgfWorldStepMetrics metrics{};BgfWorldStats stats{};BgfWorldRuntimeInfo info{};
    if (!check(bgf_world_set_food_settings(world,0,0,config.pellet_energy),"empty density fixture food") ||
        !check(bgf_world_get_stats(world,&stats),"update actual food occupancy") ||
        !check(bgf_world_step(world,1,&metrics),"shrink sparse spatial grids") ||
        !check(bgf_world_get_runtime_info(world,&info),"inspect sparse grids") ||
        !require(info.living_work_items==1 && info.contact_grid_width==32 &&
            info.sense_grid_width==32 && info.brain_workspace_bytes==0,
            "small native worlds must use small grids and no specialized brain workspace")) return finish(false);
    std::vector<int32_t> added;
    for (int i=0;i<300;++i) {
        int32_t slot=-1;
        if (!check(bgf_world_spawn_bibite(world,(i%20)*10.0f,(i/20)*10.0f,0,&slot),"grow live work list")) return finish(false);
        added.push_back(slot);
    }
    if (!check(bgf_world_step(world,1,&metrics),"grow contact grid from actual animals") ||
        !check(bgf_world_get_runtime_info(world,&info),"inspect grown contact grid") ||
        !require(info.living_work_items==301 && info.contact_grid_width==64,
            "contact grid must grow without renumbering public slots") ||
        !check(bgf_world_set_food_settings(world,8192,1,config.pellet_energy),"restore dense food") ||
        !check(bgf_world_get_stats(world,&stats),"update dense food occupancy") ||
        !check(bgf_world_step(world,1,&metrics),"grow food sensing grid") ||
        !check(bgf_world_get_runtime_info(world,&info),"inspect grown food grid") ||
        !require(info.sense_grid_width==64,"food grid must reflect actual active food")) return finish(false);
    std::vector<BgfWorldBibite> visible(1);
    int32_t count=0;
    BgfWorldBibiteDetail selected{};
    int32_t nodes=0,edges=0;
    if (!check(bgf_world_get_bibite_detail(world,added.back(),&selected,nullptr,0,&nodes,nullptr,0,&edges),"inspect stable selected ID") ||
        !check(bgf_world_download_visible_bibites(world,visible.data(),1,&count,
            selected.position_x-0.01f,selected.position_y-0.01f,
            selected.position_x+0.01f,selected.position_y+0.01f),"download viewport-only body") ||
        !require(count==1 && visible[0].slot==added.back(),"viewport-only updates must retain stable visible IDs")) return finish(false);
    for (int32_t slot:added)
        if (!check(bgf_world_kill_bibite(world,slot),"retire live work item")) return finish(false);
    if (!check(bgf_world_set_food_settings(world,0,0,config.pellet_energy),"retire food occupancy") ||
        !check(bgf_world_get_stats(world,&stats),"read reduced density") ||
        !check(bgf_world_step(world,1,&metrics),"shrink derived grids after deaths") ||
        !check(bgf_world_get_runtime_info(world,&info),"inspect reduced work") ||
        !require(info.living_work_items==1 && info.contact_grid_width==32 && info.sense_grid_width==32,
            "derived work/grids must shrink while original slot zero survives")) return finish(false);
    std::cout<<"Live work: grow/shrink grids, stable IDs, zero native workspace and visible-only download passed\n";
    return finish(true);
}

bool local_food_overflow_regressions()
{
    BgfWorldConfig config{};
    bgf_world_default_config(&config);
    config.max_bibites=8192;config.initial_bibites=1;config.pellet_count=1024;
    config.world_half_extent=256;config.pheromone_grid_width=config.pheromone_grid_height=16;
    BgfWorldHandle world=nullptr;
    if (!check(bgf_world_create(&config,&world),"create clustered food fixture")) return false;
    auto finish=[&](bool ok){bgf_world_destroy(world);return ok;};
    BgfWorldFoodZone zone{};
    zone.distribution=0;zone.radius=2;zone.seed_weight=zone.growth_weight=zone.pellet_size=1;
    if (!check(bgf_world_set_food_zones(world,&zone,1,1),"cluster food inside four overflowing cells")) return finish(false);
    BgfWorldStepMetrics overflow_metrics{};
    if (!check(bgf_world_step(world,1,&overflow_metrics),
            "build segmented overflow in the cooperative tick")) return finish(false);
    BgfWorldBibiteDetail body{};
    BgfWorldBrainNodeState nodes[73]{};
    int32_t n=0,e=0;
    if (!check(bgf_world_get_bibite_detail(world,0,&body,nodes,73,&n,nullptr,0,&e),
            "inspect food overflow senses") ||
        !require(nodes[15].last_output==256.0f,
            "all 1024 clustered pellets must be counted exactly once, including overflow")) return finish(false);
    std::vector<BgfWorldPellet> food(config.pellet_count);
    int32_t count=0;
    if (!check(bgf_world_download_pellets(world,food.data(),config.pellet_count,&count),"read clustered food reference")) return finish(false);
    const float sensing_radius=config.sense_radius*std::max(1.0f,std::min(32.0f,
        (config.world_half_extent/500.0f)*std::sqrt(8192.0f/config.pellet_count)));
    float nearest=sensing_radius;
    for (int i=0;i<count;++i)
        nearest=std::min(nearest,std::hypot(food[i].position_x-body.position_x,food[i].position_y-body.position_y));
    if (!require(std::fabs(nodes[13].last_output-(1.0f-nearest/sensing_radius))<0.000002f,
            "overflow nearest-food sensing must match a full CPU reference") ||
        !check(bgf_world_set_food_settings(world,0,1,config.pellet_energy),"remove clustered food") ||
        !check(bgf_world_step(world,2,&overflow_metrics),"invalidate dense food cache after clearing") ||
        !require(overflow_metrics.food_milliseconds==0.0f,
            "cleared food must not run or reuse the dense food prepass") ||
        !check(bgf_world_get_bibite_detail(world,0,&body,nodes,73,&n,nullptr,0,&e),"recheck cleared food") ||
        !require(nodes[13].last_output==0.0f && nodes[15].last_output==0.0f,
            "retired overflow pellets must not remain visible to brains")) return finish(false);
    std::cout<<"Food index: 1024-pellet overflow counts/nearest target match CPU; cleared food disappears\n";
    return finish(true);
}

bool crowded_contact_regressions()
{
    BgfWorldConfig config{};
    bgf_world_default_config(&config);
    config.max_bibites=512;config.initial_bibites=384;config.pellet_count=32;
    config.world_half_extent=256;config.pheromone_grid_width=config.pheromone_grid_height=16;
    config.diagnostic_mask=6u;
    BgfWorldHandle worlds[2]{};
    auto finish=[&](bool ok){for(auto world:worlds)if(world)bgf_world_destroy(world);return ok;};
    BgfWorldFoodZone zone{};
    zone.distribution=0;zone.radius=3;zone.seed_weight=zone.growth_weight=zone.pellet_size=1;
    BgfWorldTemplateSpawn giant{};
    giant.energy=500;giant.size=giant.adult_size=4;
    giant.maximum_speed=giant.turn_speed=1;giant.metabolism=0.001f;giant.lifespan=10000;
    giant.clock_period=giant.lay_period=giant.womb_capacity=1;
    BgfWorldBrainNode idle{2,-1,0,0};
    for (int mode=0;mode<2;++mode) {
        config.diagnostic_mask=mode ? 14u : 6u;
        if (!check(bgf_world_create(&config,&worlds[mode]),"create crowded contact fixture") ||
            !check(bgf_world_set_food_zones(worlds[mode],&zone,1,1),"cluster normal bodies")) return finish(false);
        int32_t slot=-1;
        if (!check(bgf_world_spawn_template_bibite(worlds[mode],&giant,&idle,1,nullptr,0,&slot),"insert oversized contact body")) return finish(false);
        BgfWorldStepMetrics metrics{};
        if (!check(bgf_world_step(worlds[mode],1,&metrics),"solve overflowing cells and oversized pairs")) return finish(false);
    }
    std::vector<BgfWorldBibite> bodies[2];
    for (int mode=0;mode<2;++mode) {
        bodies[mode].resize(config.max_bibites);
        int32_t count=0;
        if (!check(bgf_world_download_bibites(worlds[mode],bodies[mode].data(),config.max_bibites,&count),"read crowded contacts") ||
            !require(count==385,"crowded contact solvers must preserve every body")) return finish(false);
        bodies[mode].resize(count);
        std::sort(bodies[mode].begin(),bodies[mode].end(),[](const auto& a,const auto& b){return a.slot<b.slot;});
    }
    float maximum=0;
    for (size_t i=0;i<bodies[0].size();++i) {
        const auto& a=bodies[0][i];const auto& b=bodies[1][i];
        maximum=std::max(maximum,std::max(std::fabs(a.velocity_x-b.velocity_x),std::fabs(a.velocity_y-b.velocity_y)));
        if (!require(a.slot==b.slot && maximum<=0.003f,
                "local tiled/overflow/large-body ownership must match legacy contact forces")) return finish(false);
    }
    std::cout<<"Crowded contacts: 384 clustered bodies + giant, maximum solver delta "<<maximum<<"\n";
    return finish(true);
}

bool specialized_brain_regressions()
{
    for (int ticks : {1,37})
    for (bool dense : {false,true}) {
        std::vector<BgfWorldBrainNodeState> reference;
        for (uint32_t mode : {536870912u,0u,268435456u,67108864u,134217728u,8388608u,33554432u,16777216u}) {
            BgfWorldConfig config{};
            bgf_world_default_config(&config);
            config.max_bibites=64;config.initial_bibites=1;config.pellet_count=dense ? 1024 : 32;
            config.pheromone_grid_width=config.pheromone_grid_height=16;
            config.reproduction_energy=1000000.0f;config.diagnostic_mask=mode;
            BgfWorldHandle world=nullptr;
            if (!check(bgf_world_create(&config,&world),"create brain-kernel comparison")) return false;
            bool ok=true;
            if (dense) {
                BgfWorldFoodZone zone{};
                zone.radius=2;zone.seed_weight=zone.growth_weight=zone.pellet_size=1;
                ok=check(bgf_world_set_food_zones(world,&zone,1,1),
                    "compare kernels with clustered overflow food");
            }
            if (dense) for (int edge=0;edge<732 && ok;++edge)
                ok=check(bgf_world_set_hidden_link_enabled(world,0,edge,1),"enable dense brain fixture");
            BgfWorldStepMetrics metrics{};
            const int step_status=ok ? bgf_world_step(world,ticks,&metrics) : 0;
            if (ok && mode==16777216u && step_status==801) {
                std::cout<<"Tensor fixture skipped: this GPU has no supported tensor units\n";
                bgf_world_destroy(world);continue;
            }
            if (ok) ok=check(step_status,"evaluate specialized brain");
            BgfWorldBibiteDetail detail{};
            std::vector<BgfWorldBrainNodeState> nodes(73);
            int32_t n=0,e=0;
            if (ok) ok=check(bgf_world_get_bibite_detail(world,0,&detail,nodes.data(),73,&n,nullptr,0,&e),
                "inspect specialized activations");
            if (ok) ok=require(n==73 && detail.alive==1,"every kernel preserves the full native brain");
            if (ok && mode==536870912u) reference=nodes;
            if (ok && mode!=536870912u) {
                const float tolerance=mode==16777216u ? 0.006f : 0.000002f;
                for (int node=34;node<73 && ok;++node)
                    ok=require(std::isfinite(nodes[node].last_output) &&
                        std::fabs(nodes[node].last_output-reference[node].last_output)<=tolerance,
                        "sparse/dense kernels must match FP32; tensor rounding must be bounded");
                ok=ok && require(((mode==0u || mode==268435456u || mode==67108864u || mode==134217728u) || metrics.brain_milliseconds>0.0f) && metrics.food_milliseconds>=0.0f &&
                    metrics.wall_milliseconds>=metrics.gpu_milliseconds,
                    "specialized brain timings and full-call overhead must be accounted");
            }
            bgf_world_destroy(world);
            if (!ok) return false;
        }
    }
    std::cout<<"Brain kernels: sparse and dense FP32 parity; opt-in tensor rounding bounded\n";
    return true;
}

bool cadence_graph_cache_regressions()
{
    BgfWorldConfig config{};
    bgf_world_default_config(&config);
    config.max_bibites=64;config.initial_bibites=4;config.pellet_count=32;
    config.pheromone_grid_width=config.pheromone_grid_height=16;
    config.diagnostic_mask=8388608u;
    config.reproduction_energy=1000000;
    BgfWorldHandle world=nullptr;
    if (!check(bgf_world_create(&config,&world),"create graph cadence cache fixture")) return false;
    auto finish=[&](bool ok){bgf_world_destroy(world);return ok;};
    BgfWorldStepMetrics metrics{};
    int hits=0;
    for (int batch=0;batch<8;++batch) {
        if (!check(bgf_world_step(world,16,&metrics),"reuse graph across recurring cadence phases")) return finish(false);
        hits+=metrics.graph_cache_hits;
        if (!require(metrics.wall_milliseconds>=metrics.graph_build_milliseconds,
                "graph construction must remain inside full-call wall accounting")) return finish(false);
    }
    BgfWorldStats stats{};
    if (!check(bgf_world_get_stats(world,&stats),"inspect cached graph progress") ||
        !require(hits>=3 && stats.completed_steps==128,"recurring cadence phases must reuse cached graphs and advance all ticks") ||
        !check(bgf_world_set_food_settings(world,0,0,config.pellet_energy),"change captured scalar settings") ||
        !check(bgf_world_step(world,16,&metrics),"invalidate graph arguments after food changes") ||
        !require(metrics.graph_cache_hits==0,"graphs holding old scalar arguments must be invalidated")) return finish(false);
    std::cout<<"Graph cache: "<<hits<<" repeated cadence hits, 128 ticks, settings invalidation passed\n";
    return finish(true);
}

bool compact_imported_pool_regressions()
{
    BgfWorldConfig config{};
    bgf_world_default_config(&config);
    config.max_bibites=8192;config.initial_bibites=1;config.pellet_count=32;
    config.pheromone_grid_width=config.pheromone_grid_height=16;
    BgfWorldHandle world=nullptr,restored=nullptr;
    const char* path="bgf_imported_pool_test.bgfgpu";
    auto finish=[&](bool ok) {
        if (restored) bgf_world_destroy(restored);
        if (world) bgf_world_destroy(world);
        std::remove(path);return ok;
    };
    if (!check(bgf_world_create(&config,&world),"create imported-pool fixture")) return finish(false);
    BgfWorldTemplateSpawn spawn{};
    spawn.energy=1000000.0f;spawn.size=spawn.adult_size=1.0f;
    spawn.maximum_speed=spawn.turn_speed=1.0f;spawn.metabolism=0.001f;spawn.lifespan=10000;
    spawn.gene_mutation_strength=spawn.brain_mutation_strength=0.001f;
    spawn.clock_period=spawn.lay_period=1.0f;spawn.womb_capacity=1.0f;
    const BgfWorldBrainNode nodes[]={{0,0,-1,0.0f},{2,-1,-1,0.0f},{2,-1,0,0.0f}};
    const BgfWorldBrainSynapse edges[]={{0,1,0.3f},{1,2,0.9f}};
    int32_t parent=-1;
    if (!check(bgf_world_spawn_template_bibite(world,&spawn,nodes,3,edges,2,&parent),"import one stock brain")) return finish(false);
    BgfWorldRuntimeInfo info{};
    if (!check(bgf_world_get_runtime_info(world,&info),"read compact imported allocation") ||
        !require(info.imported_brains==1 && info.imported_pool_capacity==32 &&
            info.imported_pool_bytes==32u*4608u,
            "one imported brain must allocate only 32 pooled instances, not the population reservation")) return finish(false);
    std::vector<int32_t> children;
    for (int child=0;child<65;++child) {
        int32_t slot=-1;
        if (!check(bgf_world_force_reproduction(world,parent,&slot),"grow imported brain pool")) return finish(false);
        children.push_back(slot);
    }
    BgfWorldStepMetrics metrics{};
    if (!check(bgf_world_step(world,8,&metrics),"evaluate grown imported pool") ||
        !check(bgf_world_save_checkpoint(world,path),"save compact imported pool") ||
        !check(bgf_world_load_checkpoint(path,0,&restored),"load compact imported pool") ||
        !check(bgf_world_get_runtime_info(restored,&info),"read restored imported pool") ||
        !require(info.imported_brains==66 && info.imported_pool_capacity<=128,
            "save/load must reconstruct a compact pool for actual imported animals") ||
        !check(bgf_world_step(restored,8,&metrics),"resume compact imported pool")) return finish(false);
    for (int32_t child : children)
        if (!check(bgf_world_kill_bibite(restored,child),"release dead imported instances")) return finish(false);
    if (!check(bgf_world_get_runtime_info(restored,&info),"count released imported pool") ||
        !require(info.imported_brains==1 && info.living_work_items==2,
            "dead imported rows and work items must be released exactly once")) return finish(false);
    int32_t recycled=-1;
    if (!check(bgf_world_spawn_template_bibite(restored,&spawn,nodes,3,edges,2,&recycled),"reuse imported pool row") ||
        !check(bgf_world_step(restored,8,&metrics),"evaluate recycled imported row")) return finish(false);
    if (!check(bgf_world_kill_bibite(restored,parent),"release imported parent") ||
        !check(bgf_world_kill_bibite(restored,recycled),"release final imported animal") ||
        !check(bgf_world_step(restored,8,&metrics),"return to native fused kernel") ||
        !require(metrics.pipeline==0,"a historical imported pool must not force the slow imported pipeline")) return finish(false);
    std::cout<<"Imported brains: compact growth, save/load, death and row reuse passed\n";
    return finish(true);
}

bool owned_checkpoint_regressions()
{
    BgfWorldConfig config{};
    bgf_world_default_config(&config);
    config.max_bibites=8192;
    config.initial_bibites=2;
    config.pellet_count=32;
    config.pheromone_grid_width=config.pheromone_grid_height=16;
    config.reproduction_energy=1000000.0f;
    BgfWorldHandle world=nullptr,restored=nullptr;
    BgfCheckpointCaptureHandle capture=nullptr;
    const char* compact="bgf_owned_capture_test.bgfgpu";
    const char* legacy="bgf_owned_capture_legacy_test.bgfgpu";
    auto finish=[&](bool ok) {
        if (capture) bgf_world_release_checkpoint_capture(capture);
        if (restored) bgf_world_destroy(restored);
        if (world) bgf_world_destroy(world);
        std::remove(compact);std::remove(legacy);return ok;
    };
    if (!check(bgf_world_create(&config,&world),"create sparse checkpoint fixture")) return finish(false);
    BgfWorldStepMetrics metrics{};
    BgfWorldStats before{},after{},live{};
    if (!check(bgf_world_step(world,8,&metrics),"advance capture fixture") ||
        !check(bgf_world_get_stats(world,&before),"read captured progress") ||
        !check(bgf_world_capture_checkpoint(world,&capture),"capture owned checkpoint")) return finish(false);
    int writer_status=-1;
    std::thread writer([&] { writer_status=bgf_world_write_captured_checkpoint(capture,compact); });
    const int step_status=bgf_world_step(world,64,&metrics);
    writer.join();
    if (!check(step_status,"simulate while owned save writes") ||
        !check(writer_status,"write independently captured checkpoint") ||
        !check(bgf_world_load_checkpoint(compact,0,&restored),"reload owned capture") ||
        !check(bgf_world_get_stats(restored,&after),"read coherent captured world") ||
        !check(bgf_world_get_stats(world,&live),"read continuing world") ||
        !require(after.completed_steps==before.completed_steps &&
            live.completed_steps==before.completed_steps+64,
            "save captures its original tick while the world keeps advancing") ||
        !check(bgf_world_save_checkpoint_legacy_v11(restored,legacy),"export comparison checkpoint"))
        return finish(false);
    const auto compact_bytes=read_test_file(compact),legacy_bytes=read_test_file(legacy);
    if (!require(compact_bytes.size()*10<legacy_bytes.size(),
            "sparse checkpoints must not serialize empty genome reservations")) return finish(false);
    std::cout<<"Owned checkpoint: "<<compact_bytes.size()<<" versus "<<legacy_bytes.size()
        <<" legacy bytes; coherent asynchronous writer passed\n";
    // Reject duplicate compact public IDs before uploading body/genome arrays.
    auto corrupt=compact_bytes;
    uint32_t header=0;
    std::memcpy(&header,corrupt.data()+12,4);
    int32_t slot=0;
    std::memcpy(&slot,corrupt.data()+header+16,4);
    std::memcpy(corrupt.data()+header+20,&slot,4);
    if (!write_test_file(compact,corrupt)) return finish(false);
    BgfWorldHandle rejected=nullptr;
    const int result=bgf_world_load_checkpoint(compact,0,&rejected);
    if (rejected) bgf_world_destroy(rejected);
    return finish(require(result!=0 && !rejected,"duplicate compact slot IDs must be rejected"));
}

bool captured_graphics_regressions()
{
    // No window or game is opened. Only the specifically requested 4070 Ti
    // receives a D3D11 device; enumerating adapter names does not use the 5060 Ti.
    IDXGIFactory1* factory=nullptr;
    if (!require(SUCCEEDED(CreateDXGIFactory1(__uuidof(IDXGIFactory1),
            reinterpret_cast<void**>(&factory))),"create headless adapter inventory")) return false;
    IDXGIAdapter1* selected=nullptr;
    for (UINT index=0;;++index) {
        IDXGIAdapter1* adapter=nullptr;
        if (factory->EnumAdapters1(index,&adapter)==DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 desc{};
        adapter->GetDesc1(&desc);
        if (std::wstring(desc.Description).find(L"4070 Ti")!=std::wstring::npos) {
            selected=adapter;break;
        }
        adapter->Release();
    }
    factory->Release();
    if (!require(selected!=nullptr,"headless graphics test must select the 4070 Ti")) return false;
    ID3D11Device* device=nullptr;
    ID3D11DeviceContext* context=nullptr;
    const D3D_FEATURE_LEVEL feature=D3D_FEATURE_LEVEL_11_0;
    const HRESULT created=D3D11CreateDevice(selected,D3D_DRIVER_TYPE_UNKNOWN,nullptr,0,
        &feature,1,D3D11_SDK_VERSION,&device,nullptr,&context);
    selected->Release();
    if (!require(SUCCEEDED(created),"create windowless 4070 Ti render device")) return false;
    ID3D11Buffer *body=nullptr,*food=nullptr,*staging=nullptr;
    BgfWorldHandle world=nullptr;
    auto finish=[&](bool ok) {
        if (world) {
            const int release=bgf_world_unregister_d3d11_render_buffers(world);
            ok=check(release,"release captured graphics surfaces")&&ok;
            bgf_world_destroy(world);
        }
        if (staging) staging->Release();
        if (food) food->Release();
        if (body) body->Release();
        context->Release();device->Release();return ok;
    };
    BgfWorldConfig config{};
    bgf_world_default_config(&config);
    config.max_bibites=64;config.initial_bibites=1;config.pellet_count=32;
    config.pheromone_grid_width=config.pheromone_grid_height=16;
    config.reproduction_energy=1000000.0f;
    if (!check(bgf_world_create(&config,&world),"create captured-render world")) return finish(false);
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth=config.max_bibites*9u*9u*sizeof(float);
    desc.Usage=D3D11_USAGE_DEFAULT;desc.BindFlags=D3D11_BIND_VERTEX_BUFFER;
    if (!require(SUCCEEDED(device->CreateBuffer(&desc,nullptr,&body)),"create body surface")) return finish(false);
    desc.ByteWidth=config.pellet_count*4u*9u*sizeof(float);
    if (!require(SUCCEEDED(device->CreateBuffer(&desc,nullptr,&food)),"create food surface")) return finish(false);
    desc.ByteWidth=config.max_bibites*9u*9u*sizeof(float);
    desc.Usage=D3D11_USAGE_STAGING;desc.BindFlags=0;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    if (!require(SUCCEEDED(device->CreateBuffer(&desc,nullptr,&staging)),"create readback fixture")) return finish(false);
    uint32_t flags=0;int32_t protected_before=0;
    BgfWorldD3D11RenderConfig render{};
    render.bibite_vertex_buffer=body;render.pellet_vertex_buffer=food;
    render.bibite_capacity=config.max_bibites;render.pellet_capacity=config.pellet_count;
    render.pellet_half_width=render.pellet_half_height=1.0f;
    render.pellet_uv_max_x=render.pellet_uv_max_y=1.0f;render.pellet_color_a=1.0f;
    if (!check(bgf_world_prepare_d3d11_multithreading(body,&flags,&protected_before),"protect headless D3D context") ||
        !check(bgf_world_register_d3d11_render_buffer_set(world,0,&render),"register captured surface")) return finish(false);
    const auto read_vertices=[&](std::vector<float>& result) {
        context->CopyResource(staging,body);
        D3D11_MAPPED_SUBRESOURCE mapped{};
        if (!require(SUCCEEDED(context->Map(staging,0,D3D11_MAP_READ,0,&mapped)),"read rendered pose")) return false;
        result.resize(9u*9u);
        std::memcpy(result.data(),mapped.pData,result.size()*sizeof(float));
        context->Unmap(staging,0);return true;
    };
    int32_t bodies=0,pellets=0;
    std::vector<float> before,after;
    if (!check(bgf_world_capture_d3d11_render_buffer_set(world,0),"capture initial pose") ||
        !check(bgf_world_update_d3d11_render_buffer_set(world,0,&bodies,&pellets),"publish initial pose") ||
        !require(bodies==1 && pellets==32,"captured surfaces preserve entity counts") ||
        !read_vertices(before) ||
        !check(bgf_world_capture_d3d11_render_buffer_set(world,0),"capture immutable pose")) return finish(false);
    int32_t event=0;
    if (!check(bgf_world_create_unity_render_request(world,0,1,nullptr,&event),"queue owned update")) return finish(false);
    auto callback=reinterpret_cast<void(__stdcall*)(int32_t)>(bgf_world_unity_render_event_func());
    // Delayed callbacks must not read a newer mutable world instead of the
    // captured pose. Subsequent cycles also overlap callback and simulation.
    BgfWorldStepMetrics metrics{};
    if (!check(bgf_world_step(world,32,&metrics),"advance before delayed graphics callback")) return finish(false);
    callback(event);
    int state=0,status=0;
    if (!check(bgf_world_poll_unity_render_request(event,&state,&status,&bodies,&pellets),"poll captured callback") ||
        !require(state==2 && status==0,"captured callback completes") ||
        !check(bgf_world_release_unity_render_request(event),"retire captured callback") ||
        !read_vertices(after) ||
        !require(before==after,"delayed callback must draw its captured tick, not the current tick")) return finish(false);
    for (int cycle=0;cycle<20;++cycle) {
        if (!check(bgf_world_capture_d3d11_render_buffer_set(world,0),"capture concurrent render cycle")) return finish(false);
        std::atomic<bool> ready{false};int render_status=-1;
        std::thread renderer([&] {
            ready.store(true,std::memory_order_release);
            int32_t b=0,p=0;
            render_status=bgf_world_update_d3d11_render_buffer_set(world,0,&b,&p);
        });
        while (!ready.load(std::memory_order_acquire)) std::this_thread::yield();
        const int step_status=bgf_world_step(world,32,&metrics);
        renderer.join();
        if (!check(step_status,"simulate concurrently with owned graphics") ||
            !check(render_status,"publish concurrently with simulation")) return finish(false);
    }
    BgfWorldStats stats{};
    if (!check(bgf_world_get_stats(world,&stats),"read concurrent render progress") ||
        !require(stats.completed_steps==672 && stats.invalid_state_deaths==0,
            "concurrent graphics must not corrupt or stall simulation")) return finish(false);
    std::cout<<"Captured graphics: 20 concurrent updates, 672 ticks, RTX 4070 Ti, no window\n";
    return finish(true);
}
