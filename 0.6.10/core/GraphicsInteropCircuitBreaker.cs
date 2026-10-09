using System;
using System.Threading;

namespace BibitesGpuFork.Core
{
    // Keep one instance for the process lifetime. There is intentionally no
    // reset operation: changing worlds must not retry uncertain driver-owned
    // resources and accumulate another quarantined allocation each time.
    public sealed class GraphicsInteropCircuitBreaker
    {
        private string _firstFailure;

        public bool IsOpen { get { return Volatile.Read(ref _firstFailure) != null; } }
        public string FirstFailure { get { return Volatile.Read(ref _firstFailure); } }

        public bool Trip(string reason)
        {
            string failure = string.IsNullOrWhiteSpace(reason)
                ? "graphics resource ownership could not be verified" : reason;
            return Interlocked.CompareExchange(ref _firstFailure, failure, null) == null;
        }
    }
}
