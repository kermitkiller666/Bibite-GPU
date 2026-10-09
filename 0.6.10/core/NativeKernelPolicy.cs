namespace BibitesGpuFork.Core
{
    // Only changes compute-pipeline flags. Cadence and phase-bypass diagnostics
    // retain their independent meaning; experimental Tensor is CLI-only.
    public static class NativeKernelPolicy
    {
        public const uint SparseGraph = 8388608u;
        public const uint TensorGraph = 16777216u;
        public const uint DenseGraph = 33554432u;
        private const uint PipelineMask = SparseGraph | TensorGraph | DenseGraph;

        public static int Normalize(int choice)
        {
            return choice >= 0 && choice <= 2 ? choice : 0;
        }

        public static uint Apply(uint diagnosticMask, int choice)
        {
            uint pipeline = Normalize(choice) == 1 ? SparseGraph :
                Normalize(choice) == 2 ? DenseGraph : 0u;
            return (diagnosticMask & ~PipelineMask) | pipeline;
        }
    }
}
