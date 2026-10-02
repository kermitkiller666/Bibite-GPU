using System;
using BibitesGpuFork.Core;

internal static class Program
{
    private static int Main(string[] args)
    {
        try
        {
            if (args.Length == 1 && args[0] == "--verify-failure-exit")
                throw new InvalidOperationException("Intentional noninteractive failure-exit verification.");
            RunTests();
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine("Core regression tests failed: " + ex);
            return 1;
        }
    }

    private static void RunTests()
    {
        AssertEqual(1f, TimeWarpSpeeds.Snap(float.NaN), "NaN");
        AssertEqual(1f, TimeWarpSpeeds.Snap(-4f), "negative");
        AssertEqual(2f, TimeWarpSpeeds.Snap(2.2f), "nearest lower");
        AssertEqual(3f, TimeWarpSpeeds.Snap(2.7f), "nearest upper");
        AssertEqual(5000f, TimeWarpSpeeds.Snap(4000f), "high-warp snap");
        AssertEqual(10000f, TimeWarpSpeeds.Snap(20000f), "upper bound");
        AssertEqual(10000f, TimeWarpSpeeds.Snap(float.PositiveInfinity), "infinite upper bound");
        AssertEqual(10000f, TimeWarpSpeeds.Snap(float.MaxValue), "large finite upper bound");
        AssertEqual(1f, TimeWarpSpeeds.Snap(float.NegativeInfinity), "infinite lower bound");
        AssertEqual(1f, TimeWarpSpeeds.Snap(float.MinValue), "large finite lower bound");
        AssertEqual(25f, TimeWarpSpeeds.Next(10f), "next");
        AssertEqual(10f, TimeWarpSpeeds.Previous(25f), "previous");
        AssertEqual(2500f, TimeWarpSpeeds.Next(1000f), "next high warp");
        AssertEqual(1000f, TimeWarpSpeeds.Previous(2500f), "previous high warp");
        AssertEqual(10000f, TimeWarpSpeeds.Next(10000f), "next saturates");
        AssertEqual(1f, TimeWarpSpeeds.Previous(1f), "previous saturates");
        for (int i = 0; i < TimeWarpSpeeds.Values.Length; i++)
        {
            float speed = TimeWarpSpeeds.Values[i];
            AssertEqual(speed, TimeWarpSpeeds.Snap(speed), "exact selectable warp " + speed);
            AssertEqual(TimeWarpSpeeds.Values[Math.Min(i + 1, TimeWarpSpeeds.Values.Length - 1)],
                TimeWarpSpeeds.Next(speed), "next selectable warp " + speed);
            AssertEqual(TimeWarpSpeeds.Values[Math.Max(i - 1, 0)],
                TimeWarpSpeeds.Previous(speed), "previous selectable warp " + speed);
        }

        AssertHalf(0x0000, 0f, "positive zero");
        AssertHalf(0x8000, -0f, "negative zero");
        AssertHalf(0x3c00, 1f, "one");
        AssertHalf(0xc000, -2f, "negative two");
        AssertHalf(0x7bff, 65504f, "largest finite");
        AssertHalf(0x0001, 5.9604645e-8f, "smallest subnormal");
        AssertHalf(0x7c00, float.PositiveInfinity, "positive infinity");
        AssertHalf(0xfc00, float.NegativeInfinity, "negative infinity");
        ushort nan = HalfConverter.ToHalfBits(float.NaN);
        if ((nan & 0x7c00) != 0x7c00 || (nan & 0x03ff) == 0)
        {
            throw new InvalidOperationException("NaN did not remain NaN in FP16.");
        }

        SaveTransactionTests.Run();
        MenuSafetyTests.Run();
        PresentationSafetyTests.Run();
        Console.WriteLine("Time-warp, FP16 conversion, save-transaction, menu-session, presentation-metadata, Bibite-action notice, snapshot-cadence, interop-circuit-breaker and deferred-quit tests passed.");
    }

    private static void AssertEqual(float expected, float actual, string name)
    {
        if (expected != actual)
        {
            throw new InvalidOperationException(
                name + ": expected " + expected + ", got " + actual);
        }
    }

    private static void AssertHalf(ushort expected, float value, string name)
    {
        ushort actual = HalfConverter.ToHalfBits(value);
        if (actual != expected)
        {
            throw new InvalidOperationException(
                name + ": expected 0x" + expected.ToString("x4") + ", got 0x" + actual.ToString("x4"));
        }
    }
}
