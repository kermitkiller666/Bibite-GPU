// Runtime-only segmented overflow index. A dense fertile cell must not make
// every Bibite chase tens of thousands of dependent next pointers.
__global__ void clear_pellet_overflow_kernel(DeviceWorld world)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if (cell<world.sense_grid_width*world.sense_grid_width)
        world.pellet_overflow_counts[cell]=0;
}

__device__ __forceinline__ bool overflow_pellet(const DeviceWorld& world,int pellet)
{
    return world.pellet_hash_cell[pellet]==-2 &&
        (world.pellet_active[pellet]==1 || world.pellet_active[pellet]==2);
}

__global__ void count_pellet_overflow_kernel(DeviceWorld world)
{
    const int pellet=blockIdx.x*blockDim.x+threadIdx.x;
    if (pellet<world.pellet_count && overflow_pellet(world,pellet)) {
        const int cell=grid_cell(world,world.pellet_positions[pellet],world.sense_grid_width);
        atomicAdd(&world.pellet_overflow_counts[cell],1);
    }
}

// Selected-detail refresh is outside the cooperative tick. Its bounded prefix
// runs only on demand; the normal tick uses the parallel scan below.
__global__ void prefix_pellet_overflow_kernel(DeviceWorld world)
{
    if (blockIdx.x!=0 || threadIdx.x!=0) return;
    int offset=0;
    for (int cell=0;cell<world.sense_grid_width*world.sense_grid_width;++cell) {
        const int count=world.pellet_overflow_counts[cell];
        world.pellet_overflow_heads[cell]=offset;
        world.pellet_overflow_counts[cell]=0;
        offset+=count;
    }
}

__global__ void index_pellet_overflow_kernel(DeviceWorld world)
{
    const int pellet=blockIdx.x*blockDim.x+threadIdx.x;
    if (pellet<world.pellet_count && overflow_pellet(world,pellet)) {
        const int cell=grid_cell(world,world.pellet_positions[pellet],world.sense_grid_width);
        const int slot=atomicAdd(&world.pellet_overflow_counts[cell],1);
        world.pellet_overflow_next[world.pellet_overflow_heads[cell]+slot]=pellet;
    }
}

__device__ void build_pellet_overflow_segments(
    const DeviceWorld& world,cg::grid_group grid,int thread,int stride)
{
    static_assert(kThreads==256,"segmented food scan assumes eight full warps");
    __shared__ int warp_prefix[8];
    const int cells=world.sense_grid_width*world.sense_grid_width;
    const int tiles=(cells+kThreads-1)/kThreads;
    const int lane=threadIdx.x&31;
    const int warp=threadIdx.x>>5;
    for (int pellet=thread;pellet<world.pellet_count;pellet+=stride) {
        if (overflow_pellet(world,pellet)) {
            const int cell=grid_cell(world,world.pellet_positions[pellet],world.sense_grid_width);
            atomicAdd(&world.pellet_overflow_counts[cell],1);
        }
    }
    grid.sync();
    for (int tile=blockIdx.x;tile<tiles;tile+=gridDim.x) {
        const int cell=tile*kThreads+threadIdx.x;
        const int count=cell<cells ? world.pellet_overflow_counts[cell] : 0;
        int inclusive=count;
        for (int offset=1;offset<32;offset<<=1) {
            const int previous=__shfl_up_sync(0xffffffffu,inclusive,offset);
            if (lane>=offset) inclusive+=previous;
        }
        if (lane==31) warp_prefix[warp]=inclusive;
        __syncthreads();
        if (warp==0) {
            int warp_sum=lane<8 ? warp_prefix[lane] : 0;
            const int own=warp_sum;
            for (int offset=1;offset<32;offset<<=1) {
                const int previous=__shfl_up_sync(0xffffffffu,warp_sum,offset);
                if (lane>=offset) warp_sum+=previous;
            }
            if (lane<8) warp_prefix[lane]=warp_sum-own;
        }
        __syncthreads();
        if (cell<cells) world.pellet_overflow_heads[cell]=inclusive-count+warp_prefix[warp];
        if (threadIdx.x==kThreads-1)
            world.pellet_overflow_tile_prefix[tile]=inclusive+warp_prefix[warp];
        __syncthreads();
    }
    grid.sync();
    if (thread==0) {
        int offset=0;
        for (int tile=0;tile<tiles;++tile) {
            const int count=world.pellet_overflow_tile_prefix[tile];
            world.pellet_overflow_tile_prefix[tile]=offset;
            offset+=count;
        }
    }
    grid.sync();
    for (int cell=thread;cell<cells;cell+=stride) {
        world.pellet_overflow_heads[cell]+=world.pellet_overflow_tile_prefix[cell/kThreads];
        world.pellet_overflow_counts[cell]=0;
    }
    grid.sync();
    for (int pellet=thread;pellet<world.pellet_count;pellet+=stride) {
        if (overflow_pellet(world,pellet)) {
            const int cell=grid_cell(world,world.pellet_positions[pellet],world.sense_grid_width);
            const int slot=atomicAdd(&world.pellet_overflow_counts[cell],1);
            world.pellet_overflow_next[world.pellet_overflow_heads[cell]+slot]=pellet;
        }
    }
    grid.sync();
}
