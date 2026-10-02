using System;
using System.Runtime.InteropServices;

namespace BibitesGpuFork
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct GpuVisionQuery
    {
        internal float PositionX;
        internal float PositionY;
        internal float UpX;
        internal float UpY;
        internal float ViewRadius;
        internal float ViewAngleDegrees;
        internal float HerdSeparationDistance;
        internal float SeparationWeight;
        internal float CohesionWeight;
        internal float AlignmentWeight;
        internal int EntityOffset;
        internal int EntityCount;
        internal int SelfEntityId;
        internal int TargetMask;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct GpuVisionEntity
    {
        internal float PositionX;
        internal float PositionY;
        internal float UpX;
        internal float UpY;
        internal float Radius;
        internal float SizeFactor;
        internal float MaxHealth;
        internal float HealthRatio;
        internal float ColorR;
        internal float ColorG;
        internal float ColorB;
        internal int Type;
        internal int Flags;
        internal int EntityId;
        internal int HeldByEntityId;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct GpuVisionResult
    {
        internal int PlantCount;
        internal int MeatCount;
        internal int BibiteCount;
        internal int HasHerd;
        internal float PlantAngle;
        internal float PlantWeight;
        internal float MeatAngle;
        internal float MeatWeight;
        internal float BibiteAngle;
        internal float BibiteWeight;
        internal float TargetR;
        internal float TargetG;
        internal float TargetB;
        internal float HerdDirection;
        internal float HerdSeparationProjection;
        internal float MaxPlantWeight;
        internal float MaxMeatWeight;
        internal float MaxBibiteWeight;
    }

    internal static class NativeVision
    {
        private const string LibraryName = "BibitesGpuNative";

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_evaluate_vision(
            GpuVisionQuery[] queries,
            int queryCount,
            GpuVisionEntity[] entities,
            int entityCount,
            [Out] GpuVisionResult[] results);

        internal static void Evaluate(
            GpuVisionQuery[] queries,
            GpuVisionEntity[] entities,
            GpuVisionResult[] results)
        {
            int status = bgf_evaluate_vision(
                queries,
                queries.Length,
                entities,
                entities.Length,
                results);
            if (status != 0)
            {
                throw new System.InvalidOperationException(
                    "CUDA failed to evaluate vision (error " + status + ").");
            }
        }
    }

    internal sealed class GpuVisionContext : IDisposable
    {
        private const string LibraryName = "BibitesGpuNative";
        private IntPtr _handle;

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_create_vision_context(
            int deviceIndex,
            int queryCapacity,
            int entityCapacity,
            out IntPtr context);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_destroy_vision_context(IntPtr context);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_evaluate_vision_context(
            IntPtr context,
            GpuVisionQuery[] queries,
            int queryCount,
            GpuVisionEntity[] entities,
            int entityCount,
            [Out] GpuVisionResult[] results);

        internal GpuVisionContext(int deviceIndex, int queryCapacity, int entityCapacity)
        {
            int status = bgf_create_vision_context(
                deviceIndex,
                queryCapacity,
                entityCapacity,
                out _handle);
            Check(status, "create persistent vision context");
        }

        internal void Evaluate(
            GpuVisionQuery[] queries,
            int queryCount,
            GpuVisionEntity[] entities,
            int entityCount,
            GpuVisionResult[] results)
        {
            ThrowIfDisposed();
            Check(
                bgf_evaluate_vision_context(
                    _handle,
                    queries,
                    queryCount,
                    entities,
                    entityCount,
                    results),
                "evaluate resident vision batch");
        }

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                bgf_destroy_vision_context(_handle);
                _handle = IntPtr.Zero;
            }
            GC.SuppressFinalize(this);
        }

        ~GpuVisionContext()
        {
            Dispose();
        }

        private void ThrowIfDisposed()
        {
            if (_handle == IntPtr.Zero)
            {
                throw new ObjectDisposedException(nameof(GpuVisionContext));
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
