// Private helpers: bounded transfers, live-slot checkpoint packing, imported pool.
constexpr size_t kCheckpointTransferBytes = 4u * 1024u * 1024u;

bool checkpoint_pooled_field(const char* field)
{
    return std::strcmp(field,"template_node_biases")==0 ||
        std::strcmp(field,"template_node_accum")==0 ||
        std::strcmp(field,"template_node_last_input")==0 ||
        std::strcmp(field,"template_node_last_output")==0 ||
        std::strcmp(field,"template_synapse_weights")==0;
}

bool checkpoint_live_field(const char* field)
{
    // Queues are slot references, not arrays of body state. Their exact free
    // order is retained so loading does not change which slot the next birth uses.
    return std::strcmp(field,"counters") != 0 &&
        std::strcmp(field,"reproduction_items") != 0 &&
        std::strcmp(field,"free_bibite_items") != 0 &&
        std::strcmp(field,"reproduction_count") != 0 &&
        std::strcmp(field,"free_bibite_count") != 0 &&
        std::strcmp(field,"instance_id") != 0 &&
        std::strncmp(field,"pellet_",7) != 0 &&
        std::strncmp(field,"eaten_pellet_",13) != 0 &&
        std::strncmp(field,"pheromone_",10) != 0;
}

cudaError_t ensure_checkpoint_staging(GpuWorldContext& context)
{
    if (!context.checkpoint_device_staging) {
        cudaError_t status = cudaMalloc(&context.checkpoint_device_staging,kCheckpointTransferBytes);
        if (status != cudaSuccess) return status;
    }
    if (!context.checkpoint_staging) {
        cudaError_t status=cudaMallocHost(reinterpret_cast<void**>(&context.checkpoint_staging),
            kCheckpointTransferBytes);
        if (status!=cudaSuccess) return status;
    }
    if (!context.checkpoint_slots) {
        cudaError_t status = allocate_device(&context.checkpoint_slots,context.world.max_bibites);
        if (status != cudaSuccess) return status;
    }
    if (!context.checkpoint_template_slots) {
        cudaError_t status=allocate_device(&context.checkpoint_template_slots,context.world.max_bibites);
        if (status!=cudaSuccess) return status;
    }
    return cudaSuccess;
}

cudaError_t prepare_checkpoint_rows(GpuWorldContext& context)
{
    cudaError_t status = rebuild_runtime(context);
    if (status != cudaSuccess) return status;
    status = ensure_checkpoint_staging(context);
    int count = 0;
    if (status == cudaSuccess) status = cudaMemcpy(&count,context.world.live_count,
        sizeof(int),cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return status;
    context.checkpoint_live.resize(count);
    status = cudaMemcpy(context.checkpoint_live.data(),context.world.live_items,
        sizeof(int)*count,cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return status;
    std::sort(context.checkpoint_live.begin(),context.checkpoint_live.end());
    std::vector<int> topology(context.world.max_bibites);
    status = cudaMemcpy(topology.data(),context.world.template_brain,
        sizeof(int)*topology.size(),cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return status;
    context.checkpoint_imported.clear();
    for (int slot : context.checkpoint_live)
        if (topology[slot] != 0) context.checkpoint_imported.push_back(slot);
    status = cudaMemcpy(context.checkpoint_slots,context.checkpoint_live.data(),
        sizeof(int)*count,cudaMemcpyHostToDevice);
    if (status == cudaSuccess) status = cudaMemcpy(context.checkpoint_template_slots,
        context.checkpoint_imported.data(),sizeof(int)*context.checkpoint_imported.size(),
        cudaMemcpyHostToDevice);
    return status;
}

__global__ void restore_template_pool_kernel(DeviceWorld world, const int* slots, int count)
{
    const int index = blockIdx.x*blockDim.x+threadIdx.x;
    if (index < count) world.template_instance[slots[index]] = index;
    if (index < world.template_instance_capacity-count)
        world.template_free_items[index] = count+index;
    if (index == 0) {
        *world.template_free_count=world.template_instance_capacity-count;
        *world.template_living_count=count;
    }
}

cudaError_t restore_checkpoint_pool(GpuWorldContext& context)
{
    const int count = static_cast<int>(context.checkpoint_imported.size());
    cudaError_t status = ensure_template_instance_storage(context,std::max(1,count));
    if (status != cudaSuccess) return status;
    status = cudaMemset(context.world.template_instance,0xff,
        sizeof(int)*context.world.max_bibites);
    if (status == cudaSuccess) status = cudaMemcpy(context.checkpoint_template_slots,
        context.checkpoint_imported.data(),sizeof(int)*count,cudaMemcpyHostToDevice);
    if (status != cudaSuccess) return status;
    restore_template_pool_kernel<<<(std::max(count,context.world.template_instance_capacity)+
        kThreads-1)/kThreads,kThreads>>>(context.world,context.checkpoint_template_slots,count);
    return cudaGetLastError();
}

template <typename T>
__global__ void pack_checkpoint_field(DeviceWorld world, const T* source, T* packed,
    const int* slots, int rows, size_t start, size_t count, bool paired, bool pooled,
    bool legacy_pool)
{
    const size_t item = static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (item >= count) return;
    const size_t logical = start+item;
    const int lanes = paired ? 2 : 1;
    const size_t column=logical/(static_cast<size_t>(rows)*lanes);
    const int row=(logical/lanes)%rows, lane=logical%lanes;
    const int public_slot=legacy_pool ? row : slots[row];
    const int slot=pooled ? world.template_instance[public_slot] : public_slot;
    const int stride=pooled ? world.template_instance_capacity : world.max_bibites;
    packed[item]=slot>=0 ? source[(column*stride+slot)*lanes+lane] : T{};
}

template <typename T>
__global__ void unpack_checkpoint_field(DeviceWorld world, const T* packed, T* destination,
    const int* slots, int rows, size_t start, size_t count, bool paired, bool pooled,
    bool legacy_pool)
{
    const size_t item=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (item>=count) return;
    const size_t logical=start+item;
    const int lanes=paired ? 2 : 1;
    const size_t column=logical/(static_cast<size_t>(rows)*lanes);
    const int row=(logical/lanes)%rows,lane=logical%lanes;
    const int public_slot=legacy_pool ? row : slots[row];
    const int slot=pooled ? world.template_instance[public_slot] : public_slot;
    if (slot<0) return;
    const int stride=pooled ? world.template_instance_capacity : world.max_bibites;
    destination[(column*stride+slot)*lanes+lane]=packed[item];
}

cudaError_t checkpoint_copy_bytes(GpuWorldContext& context, std::FILE* file,
    void* device, size_t bytes, bool upload)
{
    for (size_t offset=0;offset<bytes;offset+=kCheckpointTransferBytes) {
        const size_t count=std::min(kCheckpointTransferBytes,bytes-offset);
        if (upload && !read_host_bytes(file,context.checkpoint_staging,count))
            return cudaErrorInvalidValue;
        cudaError_t status=upload ? cudaMemcpy(static_cast<uint8_t*>(device)+offset,
            context.checkpoint_staging,count,cudaMemcpyHostToDevice) :
            cudaMemcpy(context.checkpoint_staging,static_cast<uint8_t*>(device)+offset,
                count,cudaMemcpyDeviceToHost);
        if (status!=cudaSuccess) return status;
        if (!upload && !write_host_bytes(file,context.checkpoint_staging,count))
            return cudaErrorUnknown;
    }
    return cudaSuccess;
}

template <typename T>
cudaError_t write_checkpoint_values(GpuWorldContext& context, std::FILE* file,
    const T* source, size_t count, const char* field)
{
    if (!checkpoint_live_field(field) ||
        (checkpoint_write_format<12 && !checkpoint_pooled_field(field)))
        return checkpoint_copy_bytes(context,file,
        const_cast<T*>(source),sizeof(T)*count,false);
    const bool pooled=checkpoint_pooled_field(field);
    const bool paired=std::strcmp(field,"weights")==0 ||
        std::strcmp(field,"hidden_weights")==0 || std::strcmp(field,"extension_weights")==0;
    const int rows=checkpoint_write_format<12 ? context.world.max_bibites :
        static_cast<int>((pooled ? context.checkpoint_imported : context.checkpoint_live).size());
    const size_t columns=count/context.world.max_bibites;
    const size_t total=columns*rows;
    const size_t per_chunk=kCheckpointTransferBytes/sizeof(T);
    for (size_t offset=0;offset<total;offset+=per_chunk) {
        const size_t n=std::min(per_chunk,total-offset);
        pack_checkpoint_field<<<(n+kThreads-1)/kThreads,kThreads>>>(context.world,source,
            static_cast<T*>(context.checkpoint_device_staging),
            pooled ? context.checkpoint_template_slots : context.checkpoint_slots,
            rows,offset,n,paired,pooled,checkpoint_write_format<12);
        cudaError_t status=cudaGetLastError();
        if (status==cudaSuccess) status=cudaMemcpy(context.checkpoint_staging,
            context.checkpoint_device_staging,n*sizeof(T),cudaMemcpyDeviceToHost);
        if (status!=cudaSuccess) return status;
        if (!write_host_bytes(file,context.checkpoint_staging,n*sizeof(T))) return cudaErrorUnknown;
    }
    return cudaSuccess;
}

template <typename T>
cudaError_t read_checkpoint_values(GpuWorldContext& context, std::FILE* file,
    T* destination, size_t count, const char* field, uint32_t version)
{
    const bool pooled=checkpoint_pooled_field(field);
    if (!checkpoint_live_field(field) || (version<12 && !pooled))
        return checkpoint_copy_bytes(context,file,destination,sizeof(T)*count,true);
    const bool paired=std::strcmp(field,"weights")==0 ||
        std::strcmp(field,"hidden_weights")==0 || std::strcmp(field,"extension_weights")==0;
    const int rows=version<12 ? context.world.max_bibites :
        static_cast<int>((pooled ? context.checkpoint_imported : context.checkpoint_live).size());
    const size_t columns=count/context.world.max_bibites;
    const size_t total=columns*rows,per_chunk=kCheckpointTransferBytes/sizeof(T);
    for (size_t offset=0;offset<total;offset+=per_chunk) {
        const size_t n=std::min(per_chunk,total-offset);
        if (!read_host_bytes(file,context.checkpoint_staging,n*sizeof(T))) return cudaErrorInvalidValue;
        cudaError_t status=cudaMemcpy(context.checkpoint_device_staging,
            context.checkpoint_staging,n*sizeof(T),cudaMemcpyHostToDevice);
        if (status!=cudaSuccess) return status;
        unpack_checkpoint_field<<<(n+kThreads-1)/kThreads,kThreads>>>(context.world,
            static_cast<T*>(context.checkpoint_device_staging),destination,
            pooled ? context.checkpoint_template_slots : context.checkpoint_slots,
            rows,offset,n,paired,pooled,version<12);
        status=cudaGetLastError();
        if (status!=cudaSuccess) return status;
    }
    return cudaSuccess;
}
