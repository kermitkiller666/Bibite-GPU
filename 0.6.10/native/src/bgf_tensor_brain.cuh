// Experimental, opt-in comparison kernel. Each animal owns different weights,
// so this is batched individual matvec, not shared-weight population GEMM.
// FP16 inputs/activations add rounding versus the default FP32 compute path.
__device__ __forceinline__ __half native_effective_weight(const DeviceWorld& world,
    int index,int layer,int output,int input,bool full)
{
    int edge=-1;
    const uint32_t* masks=world.hidden_masks;
    const __half* weights=world.hidden_weights;
    if (layer==0) {
        if (output>=kHiddenWidth || input>=(full?kStockSensorCount:kSensorCount))
            return __float2half_rn(0.0f);
        if (input<kSensorCount) edge=output*kSensorCount+input;
        else {
            edge=output*kExtraSensorCount+input-kSensorCount;
            masks=world.extension_masks;weights=world.extension_weights;
        }
    } else if (layer==1) {
        if (output>=kHiddenWidth || input>=kHiddenWidth) return __float2half_rn(0.0f);
        edge=kHiddenFirstEnd+output*kHiddenWidth+input;
    } else {
        if (input>=kHiddenWidth || output>=(full?kTemplateOutputCount:kOutputCount))
            return __float2half_rn(0.0f);
        if (output<kOutputCount) edge=kHiddenSecondEnd+output*kHiddenWidth+input;
        else {
            edge=kExtensionInputEnd+(output-kOutputCount)*kHiddenWidth+input;
            masks=world.extension_masks;weights=world.extension_weights;
        }
    }
    // Never load/multiply an inactive dormant weight: zero*NaN is not zero.
    return (masks[hidden_column_slot(world,index,edge>>5)]&(1u<<(edge&31)))
        ? weights[dense_weight_slot(world,index,edge)] : __float2half_rn(0.0f);
}

__global__ __launch_bounds__(128,4) void tensor_brain_kernel(DeviceWorld world)
{
    namespace wmma=nvcuda::wmma;
    __shared__ __align__(32) __half weight_tiles[4][256];
    __shared__ __align__(32) __half input_tiles[4][256];
    __shared__ __align__(32) float result_tiles[4][256];
    __shared__ float hidden[4][2][16];
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int global_warp=blockIdx.x*4+warp;
    if (blockIdx.x==0 && threadIdx.x==0) {
        const auto before=world.counters->phase_decision_cycles;
        finish_profile_phase(world.counters,&world.counters->phase_decision_cycles);
        world.timing->food_cycles+=world.counters->phase_decision_cycles-before;
    }
    const unsigned long long step=world.counters->completed_steps;
    const int factor=(world.diagnostic_mask&1048576u)?8:
        (world.diagnostic_mask&524288u)?4:world.brain_update_factor;
    if (step%factor==0 && !(world.diagnostic_mask&4u)) {
        for (int work=global_warp;work<*world.live_count;work+=gridDim.x*4) {
            const int index=world.live_items[work];
            if (world.alive[index]!=1 || world.template_brain[index]!=0) continue;
            const bool full=world.native_brain_version[index]!=0;
            for (int layer=0;layer<3;++layer) {
                wmma::fragment<wmma::accumulator,16,16,16,float> accum;
                wmma::fill_fragment(accum,0.0f);
                const int chunks=layer==0 ? (full?3:1) : 1;
                for (int chunk=0;chunk<chunks;++chunk) {
                    for (int element=lane;element<256;element+=32) {
                        const int row=element/16,col=element%16;
                        weight_tiles[warp][element]=native_effective_weight(world,index,
                            layer,row,chunk*16+col,full);
                        float input=0.0f;
                        if (element<16) {
                            const int sensor=chunk*16+element;
                            if (layer==0 && sensor<(full?kStockSensorCount:kSensorCount))
                                input=world.brain_inputs[static_cast<size_t>(sensor)*world.max_bibites+index];
                            else if (layer>0 && element<kHiddenWidth)
                                input=hidden[warp][layer-1][element];
                        }
                        // B is column-major; only column zero holds the vector.
                        input_tiles[warp][element]=__float2half_rn(input);
                    }
                    __syncwarp();
                    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> b;
                    wmma::load_matrix_sync(a,weight_tiles[warp],16);
                    wmma::load_matrix_sync(b,input_tiles[warp],16);
                    wmma::mma_sync(accum,a,b,accum);
                    __syncwarp();
                }
                wmma::store_matrix_sync(result_tiles[warp],accum,16,wmma::mem_row_major);
                __syncwarp();
                if (layer<2 && lane<16) {
                    const float value=result_tiles[warp][lane*16];
                    float activation=0.0f;
                    if (lane<kHiddenWidth) {
                        const size_t slot=hidden_column_slot(world,index,layer*kHiddenWidth+lane);
                        activation=tanhf(value+__half2float(world.hidden_biases[slot]));
                        world.hidden_last_input[slot]=__float2half_rn(value);
                        world.hidden_last_output[slot]=__float2half_rn(activation);
                    }
                    hidden[warp][layer][lane]=activation;
                } else if (layer==2 && lane<kTemplateOutputCount) {
                    float value=result_tiles[warp][lane*16];
                    if (lane<kOutputCount) {
                        for (int sensor=0;sensor<kSensorCount;++sensor)
                            value=fmaf(world.brain_inputs[static_cast<size_t>(sensor)*world.max_bibites+index],
                                __half2float(world.weights[dense_weight_slot(world,index,lane*kSensorCount+sensor)]),value);
                    }
                    float activation=tanhf(value);
                    if (full && (lane==7 || lane==8 || lane==9 || lane==12 || lane==13))
                        activation=(activation+1.0f)*0.5f;
                    world.brain_outputs[static_cast<size_t>(lane)*world.max_bibites+index]=
                        lane<kOutputCount || full ? activation : 0.0f;
                }
                __syncwarp();
            }
        }
    }
}
