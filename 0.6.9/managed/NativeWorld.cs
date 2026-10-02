using System;
using System.Runtime.InteropServices;
using System.Text;

namespace BibitesGpuFork
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeWorldConfig
    {
        internal int DeviceIndex;
        internal int MaxBibites;
        internal int InitialBibites;
        internal int PelletCount;
        internal int SpatialGridWidth;
        internal int PheromoneGridWidth;
        internal int PheromoneGridHeight;
        internal uint Seed;
        internal float WorldHalfExtent;
        internal float FixedDeltaTime;
        internal float InitialEnergy;
        internal float PelletEnergy;
        internal float ReproductionEnergy;
        internal float SenseRadius;
        internal float PheromoneDiffusion;
        internal float PheromoneDecay;
        internal float MutationStrength;
        internal uint DiagnosticMask;
        internal int ContactGridUpdateFactor;
        internal int ContactSolveFactor;
        internal int VisionLookupFactor;
        internal int BrainUpdateFactor;
        internal int LockFoodTarget;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeWorldStepMetrics
    {
        internal int RequestedSteps;
        internal float GpuMilliseconds;
        internal float WallMilliseconds;
        internal float HostOverheadMilliseconds;
        internal float GpuOffloadPercent;
        internal double SimulatedSeconds;
        internal double RealtimeMultiplier;
        internal float PrepareMilliseconds;
        internal float SpatialIndexMilliseconds;
        internal float ContactMilliseconds;
        internal float DecisionMilliseconds;
        internal float MotionMilliseconds;
        internal float LifecycleMilliseconds;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeWorldStats
    {
        internal ulong CompletedSteps;
        internal ulong Births;
        internal ulong Deaths;
        internal ulong PelletsEaten;
        internal int LivingBibites;
        internal int ActivePellets;
        internal float TotalEnergy;
        internal float AverageEnergy;
        internal double SimulatedSeconds;
        internal ulong StarvationDeaths;
        internal ulong AgeDeaths;
        internal ulong InvalidStateDeaths;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeWorldBibite
    {
        internal int Slot;
        internal int Generation;
        internal ulong LineageId;
        internal ulong TagId;
        internal int BrainNodes;
        internal int BrainSynapses;
        internal float PositionX;
        internal float PositionY;
        internal float VelocityX;
        internal float VelocityY;
        internal float Heading;
        internal float Energy;
        internal float Age;
        internal float Size;
        internal float ColorR;
        internal float ColorG;
        internal float ColorB;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeWorldBibiteDetail
    {
        internal int Slot;
        internal int Alive;
        internal int Generation;
        internal int BrainNodes;
        internal int BrainSynapses;
        internal int TemplateBrain;
        internal int CachedFood;
        internal int CachedNeighbour;
        internal int NeighboursSeen;
        internal int Reserved;
        internal ulong LineageId;
        internal ulong TagId;
        internal float PositionX;
        internal float PositionY;
        internal float VelocityX;
        internal float VelocityY;
        internal float Heading;
        internal float Energy;
        internal float Age;
        internal float Size;
        internal float ReproductionCooldown;
        internal float MaximumSpeed;
        internal float TurnSpeed;
        internal float Metabolism;
        internal float Lifespan;
        internal float ColorR;
        internal float ColorG;
        internal float ColorB;
        internal float GeneMutationStrength;
        internal float BrainMutationStrength;
        internal float AccelerationOutput;
        internal float RotationOutput;
        internal float Pheromone1Output;
        internal float Pheromone2Output;
        internal float Pheromone3Output;
        internal float ReproductionOutput;
        internal float FoodDirectionX;
        internal float FoodDirectionY;
        internal float FoodDistanceSquared;
        internal float NeighbourDirectionX;
        internal float NeighbourDirectionY;
        internal float NeighbourDistanceSquared;
        internal float NeighbourColorR;
        internal float NeighbourColorG;
        internal float NeighbourColorB;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeWorldBrainNodeState
    {
        internal int Type;
        internal int Sensor;
        internal int Action;
        internal float BaseActivation;
        internal float LastInput;
        internal float LastOutput;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeWorldBrainSynapseState
    {
        internal int NodeIn;
        internal int NodeOut;
        internal float Weight;
    }

    internal sealed class NativeWorldSelectedBibite
    {
        internal long Sequence;
        internal NativeWorldBibiteDetail Detail;
        internal NativeWorldBrainNodeState[] Nodes;
        internal NativeWorldBrainSynapseState[] Synapses;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeWorldTemplateSpawn
    {
        internal float PositionX;
        internal float PositionY;
        internal float Heading;
        internal float Energy;
        internal float Size;
        internal float MaximumSpeed;
        internal float TurnSpeed;
        internal float Metabolism;
        internal float Lifespan;
        internal float ColorR;
        internal float ColorG;
        internal float ColorB;
        internal float GeneMutationStrength;
        internal float BrainMutationStrength;
        internal int Generation;
        internal ulong LineageId;
        internal ulong TagId;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeWorldBrainNode
    {
        internal int Type;
        internal int Sensor;
        internal int Action;
        internal float BaseActivation;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeWorldBrainSynapse
    {
        internal int NodeIn;
        internal int NodeOut;
        internal float Weight;
    }

    internal sealed class NativeWorldTemplate
    {
        internal NativeWorldTemplateSpawn Spawn;
        internal NativeWorldBrainNode[] Nodes;
        internal NativeWorldBrainSynapse[] Synapses;
        internal string TemplateName;
        internal string SpeciesName;
        internal string TagName;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeWorldPellet
    {
        internal int Slot;
        internal float PositionX;
        internal float PositionY;
        internal float Energy;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeD3D11RenderConfig
    {
        internal IntPtr BibiteVertexBuffer;
        internal IntPtr PelletVertexBuffer;
        internal int BibiteCapacity;
        internal int PelletCapacity;
        internal float PelletHalfWidth;
        internal float PelletHalfHeight;
        internal float PelletUvMinX;
        internal float PelletUvMinY;
        internal float PelletUvMaxX;
        internal float PelletUvMaxY;
        internal float PelletColorR;
        internal float PelletColorG;
        internal float PelletColorB;
        internal float PelletColorA;
    }

    internal sealed class NativeWorldContext : IDisposable
    {
        private const string LibraryName = "BibitesGpuNative";
        private IntPtr _handle;

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void bgf_world_default_config(ref NativeWorldConfig config);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_world_create(
            ref NativeWorldConfig config,
            out IntPtr world);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_world_destroy(IntPtr world);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_world_step(
            IntPtr world,
            int steps,
            out NativeWorldStepMetrics metrics);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_world_get_stats(
            IntPtr world,
            out NativeWorldStats stats);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_world_spawn_bibite(
            IntPtr world,
            float positionX,
            float positionY,
            float heading,
            out int spawnedSlot);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_world_spawn_template_bibite(
            IntPtr world,
            ref NativeWorldTemplateSpawn spawn,
            [In] NativeWorldBrainNode[] nodes,
            int nodeCount,
            [In] NativeWorldBrainSynapse[] synapses,
            int synapseCount,
            out int spawnedSlot);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_world_download_bibites(
            IntPtr world,
            [Out] NativeWorldBibite[] bibites,
            int capacity,
            out int written);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_world_download_pellets(
            IntPtr world,
            [Out] NativeWorldPellet[] pellets,
            int capacity,
            out int written);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_world_download_snapshot(
            IntPtr world,
            [Out] NativeWorldBibite[] bibites,
            int bibiteCapacity,
            out int bibitesWritten,
            [Out] NativeWorldPellet[] pellets,
            int pelletCapacity,
            out int pelletsWritten,
            out NativeWorldStats stats);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_world_get_bibite_detail(
            IntPtr world,
            int slot,
            out NativeWorldBibiteDetail detail,
            [Out] NativeWorldBrainNodeState[] nodes,
            int nodeCapacity,
            out int nodesWritten,
            [Out] NativeWorldBrainSynapseState[] synapses,
            int synapseCapacity,
            out int synapsesWritten);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_world_set_bibite_tag(
            IntPtr world,
            int slot,
            ulong tagId);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_world_kill_bibite(
            IntPtr world,
            int slot);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_world_force_reproduction(
            IntPtr world,
            int parentSlot,
            out int childSlot);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_world_set_food_settings(
            IntPtr world,
            int target,
            float growthFactor,
            float pelletEnergy);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_world_get_food_settings(
            IntPtr world,
            out int target,
            out float growthFactor,
            out float pelletEnergy);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_world_prepare_d3d11_multithreading(
            IntPtr vertexBuffer, out uint deviceFlags, out int protectionWasEnabled);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_world_register_d3d11_render_buffer_set(
            IntPtr world,
            int bufferIndex,
            ref NativeD3D11RenderConfig config);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_world_update_d3d11_render_buffer_set(
            IntPtr world, int bufferIndex, out int bibiteCount, out int pelletCount);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_world_unregister_d3d11_render_buffers(IntPtr world);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr bgf_world_unity_render_event_func();

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_world_create_unity_render_request(
            IntPtr world, int bufferIndex, int operation,
            ref NativeD3D11RenderConfig config, out int eventId);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_world_poll_unity_render_request(
            int eventId, out int state, out int result, out int bibites, out int pellets);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_world_cancel_unity_render_request(int eventId);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_world_release_unity_render_request(int eventId);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_world_abandon_d3d11_render_buffers(IntPtr world);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        private static extern int bgf_world_get_device_name(
            IntPtr world,
            StringBuilder buffer,
            int bufferLength);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        private static extern int bgf_world_save_checkpoint(
            IntPtr world,
            string path);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        private static extern int bgf_world_checkpoint_config(
            string path,
            int deviceIndex,
            out NativeWorldConfig config);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        private static extern int bgf_world_load_checkpoint(
            string path,
            int deviceIndex,
            out IntPtr world);

        internal NativeWorldContext(NativeWorldConfig config)
        {
            Check(bgf_world_create(ref config, out _handle), "create the GPU world");
            if (_handle == IntPtr.Zero)
            {
                throw new InvalidOperationException("The native GPU world returned an empty handle.");
            }
        }

        internal NativeWorldContext(string checkpointPath, int deviceIndex)
        {
            Check(bgf_world_load_checkpoint(
                checkpointPath,
                deviceIndex,
                out _handle), "load the GPU world checkpoint");
            if (_handle == IntPtr.Zero)
            {
                throw new InvalidOperationException("The native GPU checkpoint returned an empty handle.");
            }
        }

        internal static NativeWorldConfig CreateDefaultConfig()
        {
            NativeWorldConfig config = new NativeWorldConfig();
            bgf_world_default_config(ref config);
            return config;
        }

        internal static NativeWorldConfig ReadCheckpointConfig(
            string checkpointPath,
            int deviceIndex)
        {
            NativeWorldConfig config;
            Check(bgf_world_checkpoint_config(
                checkpointPath,
                deviceIndex,
                out config), "read the GPU checkpoint configuration");
            return config;
        }

        internal NativeWorldStepMetrics Step(int steps)
        {
            NativeWorldStepMetrics metrics;
            Check(bgf_world_step(_handle, steps, out metrics), "advance the GPU world");
            return metrics;
        }

        internal NativeWorldStats GetStats()
        {
            NativeWorldStats stats;
            Check(bgf_world_get_stats(_handle, out stats), "read GPU world statistics");
            return stats;
        }

        internal int SpawnBibite(float positionX, float positionY, float heading)
        {
            int spawnedSlot;
            Check(bgf_world_spawn_bibite(
                _handle,
                positionX,
                positionY,
                heading,
                out spawnedSlot), "place a Bibite in the GPU world");
            return spawnedSlot;
        }

        internal int SpawnTemplateBibite(NativeWorldTemplate template)
        {
            if (template == null || template.Nodes == null || template.Nodes.Length == 0)
            {
                throw new ArgumentException("A GPU template placement requires at least one brain node.", "template");
            }
            NativeWorldBrainSynapse[] synapses = template.Synapses ??
                new NativeWorldBrainSynapse[0];
            NativeWorldTemplateSpawn spawn = template.Spawn;
            int spawnedSlot;
            Check(bgf_world_spawn_template_bibite(
                _handle,
                ref spawn,
                template.Nodes,
                template.Nodes.Length,
                synapses,
                synapses.Length,
                out spawnedSlot), "place a stock-template Bibite in the GPU world");
            return spawnedSlot;
        }

        internal int DownloadBibites(NativeWorldBibite[] destination)
        {
            int written;
            Check(bgf_world_download_bibites(
                _handle,
                destination,
                destination.Length,
                out written), "download the Bibite display snapshot");
            return written;
        }

        internal int DownloadPellets(NativeWorldPellet[] destination)
        {
            int written;
            Check(bgf_world_download_pellets(
                _handle,
                destination,
                destination.Length,
                out written), "download the pellet display snapshot");
            return written;
        }

        internal NativeWorldStats DownloadSnapshot(
            NativeWorldBibite[] bibites,
            out int bibitesWritten,
            NativeWorldPellet[] pellets,
            out int pelletsWritten)
        {
            NativeWorldStats stats;
            Check(bgf_world_download_snapshot(
                _handle,
                bibites,
                bibites.Length,
                out bibitesWritten,
                pellets,
                pellets.Length,
                out pelletsWritten,
                out stats), "download the packed GPU display snapshot");
            return stats;
        }

        internal NativeWorldBibiteDetail DownloadBibiteDetail(
            int slot,
            NativeWorldBrainNodeState[] nodes,
            out int nodesWritten,
            NativeWorldBrainSynapseState[] synapses,
            out int synapsesWritten)
        {
            NativeWorldBibiteDetail detail;
            Check(bgf_world_get_bibite_detail(
                _handle,
                slot,
                out detail,
                nodes,
                nodes != null ? nodes.Length : 0,
                out nodesWritten,
                synapses,
                synapses != null ? synapses.Length : 0,
                out synapsesWritten), "read the selected GPU Bibite");
            return detail;
        }

        internal void SetBibiteTag(int slot, ulong tagId)
        {
            Check(bgf_world_set_bibite_tag(_handle, slot, tagId),
                "update the selected GPU Bibite tag");
        }

        internal void KillBibite(int slot)
        {
            Check(bgf_world_kill_bibite(_handle, slot),
                "remove the selected GPU Bibite");
        }

        internal int ForceReproduction(int parentSlot)
        {
            int childSlot;
            int status = bgf_world_force_reproduction(_handle, parentSlot, out childSlot);
            // These are normal biology/capacity refusals, not CUDA failures.
            // Only decode the explicit result when native reports InvalidValue;
            // a real driver error must retain its original diagnostic.
            if (status == 1)
            {
                string reason = childSlot == -2 ? "This Bibite is no longer alive."
                    : childSlot == -3 ? "Not enough energy to reproduce yet."
                    : childSlot == -4 ? "The world's population cap is full."
                    : childSlot == -5 ? "A free population slot could not be reserved; try again."
                    : null;
                if (reason != null) throw new InvalidOperationException(reason);
            }
            Check(status, "lay an egg from the selected GPU Bibite");
            return childSlot;
        }

        internal void SetFoodSettings(int target, float growthFactor, float pelletEnergy)
        {
            Check(bgf_world_set_food_settings(
                _handle, target, growthFactor, pelletEnergy),
                "update the live GPU food settings");
        }

        internal void GetFoodSettings(
            out int target,
            out float growthFactor,
            out float pelletEnergy)
        {
            Check(bgf_world_get_food_settings(
                _handle, out target, out growthFactor, out pelletEnergy),
                "read the live GPU food settings");
        }

        internal static void PrepareD3D11Multithreading(IntPtr vertexBuffer)
        {
            uint flags;
            int previouslyProtected;
            int status = bgf_world_prepare_d3d11_multithreading(
                vertexBuffer, out flags, out previouslyProtected);
            GpuInteropTrace.Write("main D3D device flags=" + flags +
                " prior-multithread-protection=" + previouslyProtected + " status=" + status);
            Check(status, "enable safe shared Direct3D 11 context access");
        }

        internal void RegisterD3D11RenderBuffers(int bufferIndex, NativeD3D11RenderConfig config)
        {
            Check(bgf_world_register_d3d11_render_buffer_set(
                _handle,
                bufferIndex,
                ref config), "register the Unity Direct3D 11 render buffers with CUDA");
        }

        internal void UnregisterD3D11RenderBuffers()
        {
            Check(bgf_world_unregister_d3d11_render_buffers(_handle),
                "unregister the Unity Direct3D 11 render buffers from CUDA");
        }

        internal void UpdateD3D11RenderBuffers(int bufferIndex, out int bibiteCount, out int pelletCount)
        {
            Check(bgf_world_update_d3d11_render_buffer_set(
                _handle, bufferIndex, out bibiteCount, out pelletCount),
                "update the shared Unity Direct3D 11 render buffers from CUDA");
        }

        internal bool HasHandle { get { return _handle != IntPtr.Zero; } }

        internal static IntPtr UnityRenderEventFunction
        {
            get { return bgf_world_unity_render_event_func(); }
        }

        internal int CreateUnityRenderRequest(int bufferIndex, int operation,
            NativeD3D11RenderConfig config)
        {
            int id;
            Check(bgf_world_create_unity_render_request(_handle, bufferIndex,
                operation, ref config, out id), "prepare a Unity render-thread handoff");
            return id;
        }

        internal static void PollUnityRenderRequest(int id, out int state, out int result,
            out int bibites, out int pellets)
        {
            Check(bgf_world_poll_unity_render_request(id, out state, out result,
                out bibites, out pellets), "poll a Unity render-thread handoff");
        }

        internal static void CancelUnityRenderRequest(int id)
        {
            Check(bgf_world_cancel_unity_render_request(id), "cancel a pending Unity render event");
        }

        internal static void ReleaseUnityRenderRequest(int id)
        {
            Check(bgf_world_release_unity_render_request(id), "retire a completed Unity render event");
        }

        internal static void CheckUnityRenderResult(int result)
        {
            Check(result, "execute CUDA/Direct3D interop on the Unity rendering thread");
        }

        internal void AbandonD3D11RenderBuffers()
        {
            Check(bgf_world_abandon_d3d11_render_buffers(_handle),
                "quarantine uncertain graphics registrations");
        }

        internal void RetainForProcessLifetime()
        {
            // Emergency only: a callback whose state cannot be read may still
            // reference this native world. Do not free it behind that callback.
            _handle = IntPtr.Zero;
        }

        internal string GetDeviceName()
        {
            StringBuilder buffer = new StringBuilder(256);
            Check(bgf_world_get_device_name(_handle, buffer, buffer.Capacity),
                "read the GPU world device name");
            return buffer.ToString();
        }

        internal void SaveCheckpoint(string path)
        {
            Check(bgf_world_save_checkpoint(_handle, path),
                "save the GPU world checkpoint");
        }

        public void Dispose()
        {
            if (_handle == IntPtr.Zero)
            {
                return;
            }
            int status = bgf_world_destroy(_handle);
            _handle = IntPtr.Zero;
            Check(status, "release the GPU world and its graphics registrations");
        }

        private static void Check(int status, string operation)
        {
            if (status != 0)
            {
                throw new InvalidOperationException(
                    "Failed to " + operation + " (CUDA error " + status + ").");
            }
        }
    }
}
