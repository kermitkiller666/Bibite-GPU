using System;

namespace BibitesGpuFork.Core
{
    public static class PresentationCadence
    {
        // Reserved simulation slots do not animate. Budget the actual CPU
        // presentation sample instead, so a small world in a large allocation
        // does not fall to one texture/chart update per second.
        public static int SnapshotFramesPerSecond(
            int livingBibites, int sampleCapacity, int requestedFramesPerSecond, bool paused)
        {
            if (paused) return 10;
            int requested = Math.Max(10, Math.Min(60, requestedFramesPerSecond));
            int sample = Math.Min(Math.Max(0, livingBibites), Math.Max(0, sampleCapacity));
            int budget = sample <= 1024 ? 60 : sample <= 4096 ? 30 : 15;
            return Math.Min(requested, budget);
        }

        public static double SnapshotIntervalSeconds(
            int livingBibites, int sampleCapacity, int requestedFramesPerSecond, bool paused)
        {
            return 1.0 / SnapshotFramesPerSecond(
                livingBibites, sampleCapacity, requestedFramesPerSecond, paused);
        }
    }
}
