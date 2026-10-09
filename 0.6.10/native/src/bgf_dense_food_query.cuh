// Dense overflow queries use a warp per Bibite, not one thread walking a very
// long cell. Ordinary worlds retain the inexpensive scalar query.
constexpr int kParallelFoodOverflowThreshold=256;

__device__ void cache_dense_food_senses(const DeviceWorld& world)
{
    const int lane=threadIdx.x&31;
    const int first=(blockIdx.x*blockDim.x+threadIdx.x)>>5;
    const int stride=(gridDim.x*blockDim.x)>>5;
    for (int work=first;work<*world.live_count;work+=stride) {
        const int index=world.live_items[work];
        if (world.alive[index]!=1 || (world.template_brain[index]==0 &&
                world.native_brain_version[index]==0)) continue;
        TemplateFoodSenses senses{};
        const float radius_squared=world.sense_radius*world.sense_radius;
        senses.plant_distance_squared=senses.meat_distance_squared=radius_squared;
        const float2 position=world.positions[index];
        const float heading=world.headings[index];
        const int width=world.sense_grid_width;
        const int origin=grid_cell(world,position,width);
        const int ox=origin%width,oy=origin/width,radius=world.sense_cell_radius;
        const bool all=radius*2+1>=width;
        const int span=all ? width : radius*2+1;
        for (int y=0;y<span;++y) {
            const int cy=all ? y : wrap_spatial_index(oy+y-radius,width);
            for (int x=0;x<span;++x) {
                const int cx=all ? x : wrap_spatial_index(ox+x-radius,width);
                const int cell=cy*width+cx;
                const uint32_t mask=world.pellet_cell_masks[cell];
                if ((mask&(1u<<lane))!=0u)
                    consider_template_food(world,index,position,heading,
                        world.pellet_cell_items[cell*kPelletBucketCapacity+lane],senses);
                const int start=world.pellet_overflow_heads[cell];
                const int count=start<0 || start>=world.pellet_count ? 0 :
                    max(0,min(world.pellet_overflow_counts[cell],world.pellet_count-start));
                for (int item=lane;item<count;item+=32)
                    consider_template_food(world,index,position,heading,
                        world.pellet_overflow_next[start+item],senses);
            }
        }
        for (int offset=16;offset>0;offset>>=1) {
            senses.plants_seen+=__shfl_down_sync(0xffffffffu,senses.plants_seen,offset);
            senses.meats_seen+=__shfl_down_sync(0xffffffffu,senses.meats_seen,offset);
            const int plant=__shfl_down_sync(0xffffffffu,senses.plant,offset);
            const int meat=__shfl_down_sync(0xffffffffu,senses.meat,offset);
            const float pd=__shfl_down_sync(0xffffffffu,senses.plant_distance_squared,offset);
            const float md=__shfl_down_sync(0xffffffffu,senses.meat_distance_squared,offset);
            if (plant>=0 && (senses.plant<0 || pd<senses.plant_distance_squared ||
                    (pd==senses.plant_distance_squared && plant<senses.plant))) {
                senses.plant=plant;senses.plant_distance_squared=pd;
            }
            if (meat>=0 && (senses.meat<0 || md<senses.meat_distance_squared ||
                    (md==senses.meat_distance_squared && meat<senses.meat))) {
                senses.meat=meat;senses.meat_distance_squared=md;
            }
        }
        if (lane==0) {
            float sn,cs;sincosf(heading,&sn,&cs);
            if (senses.plant>=0) {
                const float2 d=safe_normalize(wrapped_delta(position,
                    world.pellet_positions[senses.plant],world.world_half_extent));
                senses.plant_direction={d.x*cs-d.y*sn,d.x*sn+d.y*cs};
            }
            if (senses.meat>=0) {
                const float2 d=safe_normalize(wrapped_delta(position,
                    world.pellet_positions[senses.meat],world.world_half_extent));
                senses.meat_direction={d.x*cs-d.y*sn,d.x*sn+d.y*cs};
            }
            world.food_senses_cache[index]=senses;
        }
    }
}
