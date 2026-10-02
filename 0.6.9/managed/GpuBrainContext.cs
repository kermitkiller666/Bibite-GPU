using System;
using System.Runtime.InteropServices;

namespace BibitesGpuFork
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct GpuBrain
    {
        internal int NodeOffset;
        internal int TargetOffset;
        internal int TargetCount;
        internal float Period;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct GpuNode
    {
        internal float Value;
        internal float LastOutput;
        internal float LastInput;
        internal float Bias;
        internal int Type;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct GpuTarget
    {
        internal int NodeIndex;
        internal int EdgeOffset;
        internal int EdgeCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct GpuEdge
    {
        internal int SourceNode;
        internal ushort WeightBits;
        internal ushort Reserved;
    }

    internal sealed class GpuBrainContext : IDisposable
    {
        private const string LibraryName = "BibitesGpuNative";
        private IntPtr _handle;

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_create_context(
            int deviceIndex,
            int brainCapacity,
            int nodeCapacity,
            int targetCapacity,
            int edgeCapacity,
            out IntPtr context);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_destroy_context(IntPtr context);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_upload_brains(IntPtr context, GpuBrain[] brains, int count);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_upload_nodes(IntPtr context, GpuNode[] nodes, int count);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_upload_topology(
            IntPtr context,
            GpuTarget[] targets,
            int targetCount,
            GpuEdge[] edges,
            int edgeCount);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_step(IntPtr context);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_download_nodes(IntPtr context, [Out] GpuNode[] nodes, int count);

        internal GpuBrainContext(
            int deviceIndex,
            int brainCapacity,
            int nodeCapacity,
            int targetCapacity,
            int edgeCapacity)
        {
            int status = bgf_create_context(
                deviceIndex,
                brainCapacity,
                nodeCapacity,
                targetCapacity,
                edgeCapacity,
                out _handle);
            Check(status, "create context");
        }

        internal void UploadBrains(GpuBrain[] brains)
        {
            ThrowIfDisposed();
            Check(bgf_upload_brains(_handle, brains, brains.Length), "upload brains");
        }

        internal void UploadNodes(GpuNode[] nodes)
        {
            ThrowIfDisposed();
            Check(bgf_upload_nodes(_handle, nodes, nodes.Length), "upload nodes");
        }

        internal void UploadTopology(GpuTarget[] targets, GpuEdge[] edges)
        {
            ThrowIfDisposed();
            Check(bgf_upload_topology(_handle, targets, targets.Length, edges, edges.Length), "upload topology");
        }

        internal void Step()
        {
            ThrowIfDisposed();
            Check(bgf_step(_handle), "step brains");
        }

        internal void DownloadNodes(GpuNode[] nodes)
        {
            ThrowIfDisposed();
            Check(bgf_download_nodes(_handle, nodes, nodes.Length), "download nodes");
        }

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                bgf_destroy_context(_handle);
                _handle = IntPtr.Zero;
            }
            GC.SuppressFinalize(this);
        }

        ~GpuBrainContext()
        {
            Dispose();
        }

        private void ThrowIfDisposed()
        {
            if (_handle == IntPtr.Zero)
            {
                throw new ObjectDisposedException(nameof(GpuBrainContext));
            }
        }

        private static void Check(int status, string operation)
        {
            if (status != 0)
            {
                throw new InvalidOperationException("CUDA failed to " + operation + " (error " + status + ").");
            }
        }
    }
}
