using System;
using System.Runtime.InteropServices;

namespace BibitesGpuFork
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct GpuPheromoneQuery
    {
        internal float PositionX;
        internal float PositionY;
        internal float UpX;
        internal float UpY;
        internal float SenseRadius;
        internal float RedDeathSafeRadius;
        internal int EnableRedDeath;
        internal int Reserved;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct GpuPheromoneEntity
    {
        internal float PositionX;
        internal float PositionY;
        internal float HeadingX;
        internal float HeadingY;
        internal float RedStrength;
        internal float GreenStrength;
        internal float BlueStrength;
        internal float Reserved;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct GpuPheromoneResult
    {
        internal float RedSum;
        internal float GreenSum;
        internal float BlueSum;
        internal float RedAngle;
        internal float GreenAngle;
        internal float BlueAngle;
        internal float RedHeadingAngle;
        internal float GreenHeadingAngle;
        internal float BlueHeadingAngle;
    }

    internal sealed class GpuPheromoneContext : IDisposable
    {
        private const string LibraryName = "BibitesGpuNative";
        private IntPtr _handle;

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_create_pheromone_context(
            int deviceIndex,
            int queryCapacity,
            int entityCapacity,
            out IntPtr context);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_destroy_pheromone_context(IntPtr context);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_evaluate_pheromones_context(
            IntPtr context,
            GpuPheromoneQuery[] queries,
            int queryCount,
            GpuPheromoneEntity[] entities,
            int entityCount,
            [Out] GpuPheromoneResult[] results);

        internal GpuPheromoneContext(int deviceIndex, int queryCapacity, int entityCapacity)
        {
            int status = bgf_create_pheromone_context(
                deviceIndex,
                queryCapacity,
                entityCapacity,
                out _handle);
            Check(status, "create persistent pheromone context");
        }

        internal void Evaluate(
            GpuPheromoneQuery[] queries,
            int queryCount,
            GpuPheromoneEntity[] entities,
            int entityCount,
            GpuPheromoneResult[] results)
        {
            ThrowIfDisposed();
            Check(
                bgf_evaluate_pheromones_context(
                    _handle,
                    queries,
                    queryCount,
                    entities,
                    entityCount,
                    results),
                "evaluate resident pheromone batch");
        }

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                bgf_destroy_pheromone_context(_handle);
                _handle = IntPtr.Zero;
            }
            GC.SuppressFinalize(this);
        }

        ~GpuPheromoneContext()
        {
            Dispose();
        }

        private void ThrowIfDisposed()
        {
            if (_handle == IntPtr.Zero)
            {
                throw new ObjectDisposedException(nameof(GpuPheromoneContext));
            }
        }

        private static void Check(int status, string operation)
        {
            if (status != 0)
            {
                throw new InvalidOperationException(
                    "CUDA failed to " + operation + " (error " + status + ").");
            }
        }
    }
}
