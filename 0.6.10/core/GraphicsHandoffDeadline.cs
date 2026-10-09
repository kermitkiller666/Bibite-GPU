namespace BibitesGpuFork.Core
{
    // A late/running native callback must never be forcibly freed. Expiry only
    // tells the worker to quarantine ownership and report failure, not cancel
    // driver work or declare a graphics registration safe to destroy.
    public static class GraphicsHandoffDeadline
    {
        public const double MaximumWaitMilliseconds = 15000.0;

        public static bool IsExpired(double elapsedMilliseconds, int nativeState)
        {
            return nativeState != 2 && nativeState != 3 &&
                elapsedMilliseconds >= MaximumWaitMilliseconds;
        }
    }
}
