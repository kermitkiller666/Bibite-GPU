using System;
using System.Threading;
using BibitesGpuFork.Core;

internal static class PresentationSafetyTests
{
    internal static void Run()
    {
        Equal(60, Rate(512, 131072, 60), "large reserved capacity cannot throttle a small world");
        Equal(Rate(512, 512, 60), Rate(512, 500000, 60), "capacity does not change a living sample");
        Equal(30, Rate(512, 8192, 30), "small worlds honor the selected graphics rate");
        Equal(60, Rate(1024, 8192, 60), "small-sample boundary");
        Equal(30, Rate(1025, 8192, 60), "medium samples have a bounded download rate");
        Equal(30, Rate(4096, 8192, 60), "medium-sample boundary");
        Equal(15, Rate(4097, 8192, 60), "large samples have a bounded download rate");
        Equal(15, Rate(500000, 8192, 60), "large worlds are based on their capped sample");
        Equal(60, Rate(500000, 512, 60), "small presentation samples do not pay for unrendered population");
        Equal(10, Rate(500000, 8192, 10), "requested rate remains an upper bound");
        Equal(10, Rate(1, 8192, int.MinValue), "invalid low rate is clamped");
        Equal(60, Rate(1, 8192, int.MaxValue), "invalid high rate is clamped");
        Equal(60, Rate(-1, -1, 60), "empty or unavailable samples remain responsive");
        foreach (int living in new[] { 0, 1, 1024, 1025, 4096, 4097, 500000, int.MaxValue })
        {
            Equal(10, PresentationCadence.SnapshotFramesPerSecond(living, 8192, 60, true),
                "paused food and action updates stay responsive");
            double interval = PresentationCadence.SnapshotIntervalSeconds(living, 8192, 60, false);
            Check(interval >= 1.0 / 60.0 && interval <= 1.0 / 10.0 && !double.IsNaN(interval),
                "snapshot interval must remain finite and bounded");
        }

        GraphicsInteropCircuitBreaker breaker = new GraphicsInteropCircuitBreaker();
        Check(!breaker.IsOpen && breaker.FirstFailure == null, "interop starts permitted");
        Check(breaker.Trip("first driver failure"), "first interop fault trips the process guard");
        Check(breaker.IsOpen && breaker.FirstFailure == "first driver failure", "first fault is retained");
        for (int world = 0; world < 100; ++world)
            Check(!breaker.Trip("later world fault") && breaker.FirstFailure == "first driver failure",
                "new worlds cannot clear or replace the process fault");
        GraphicsInteropCircuitBreaker empty = new GraphicsInteropCircuitBreaker();
        Check(empty.Trip(null) && empty.IsOpen && !string.IsNullOrEmpty(empty.FirstFailure),
            "missing diagnostic cannot accidentally leave interop enabled");

        GraphicsInteropCircuitBreaker raced = new GraphicsInteropCircuitBreaker();
        int firstFaults = 0;
        Thread[] threads = new Thread[8];
        for (int index = 0; index < threads.Length; ++index)
        {
            int fault = index;
            threads[index] = new Thread(() =>
            {
                if (raced.Trip("thread " + fault)) Interlocked.Increment(ref firstFaults);
            }) { IsBackground = true };
            threads[index].Start();
        }
        foreach (Thread thread in threads) thread.Join();
        Check(firstFaults == 1 && raced.IsOpen && raced.FirstFailure.StartsWith("thread "),
            "worker and Unity faults atomically retain exactly one process failure");

        DeferredApplicationQuit immediate = new DeferredApplicationQuit();
        Check(immediate.Request(false) && immediate.Allowed && !immediate.Requested,
            "a main-menu or stock-only quit needs no GPU cleanup delay");
        Check(!immediate.TryComplete(true), "an immediate quit must not be reissued");
        DeferredApplicationQuit deferred = new DeferredApplicationQuit();
        Check(!deferred.TryComplete(true), "cleanup alone cannot initiate a user quit");
        Check(!deferred.Request(true) && deferred.Requested && !deferred.Allowed,
            "live or retiring resources defer quit while the render loop is available");
        for (int frame = 0; frame < 1000; ++frame)
        {
            Check(!deferred.TryComplete(false), "pending callbacks or saves must keep ownership");
            Check(!deferred.Request(true), "repeated close requests cannot bypass cleanup");
        }
        Check(deferred.TryComplete(true) && deferred.Allowed,
            "confirmed cleanup completion reissues quit exactly once");
        Check(!deferred.TryComplete(true) && deferred.Request(false),
            "the reissued Unity quit is accepted without another drain");
    }

    private static int Rate(int living, int capacity, int requested)
    {
        return PresentationCadence.SnapshotFramesPerSecond(living, capacity, requested, false);
    }

    private static void Equal(int expected, int actual, string message)
    {
        Check(expected == actual, message + ": expected " + expected + ", got " + actual);
    }

    private static void Check(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException(message);
    }
}
