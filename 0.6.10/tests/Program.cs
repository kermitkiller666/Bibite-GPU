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
        FoodPelletCapTests();
        DetailedSpriteVisibilityTests();
        BrainDiagramTests();
        if (NativeKernelPolicy.Normalize(-1) != 0 || NativeKernelPolicy.Normalize(3) != 0 ||
            NativeKernelPolicy.Apply(65536u | NativeKernelPolicy.TensorGraph, 0) != 65536u ||
            NativeKernelPolicy.Apply(65536u | NativeKernelPolicy.DenseGraph, 1) !=
                (65536u | NativeKernelPolicy.SparseGraph) ||
            NativeKernelPolicy.Apply(0u, 2) != NativeKernelPolicy.DenseGraph)
            throw new InvalidOperationException("Native kernel selection changed unrelated diagnostics or retained conflicting pipelines.");
        Console.WriteLine("Time-warp, FP16 conversion, brain-diagram, food-pellet cap, detailed-sprite visibility, save-transaction, menu-session, presentation-metadata, Bibite-action notice, snapshot-cadence, interop-circuit-breaker, graphics-handoff deadline and deferred-quit tests passed.");
    }

    private static void FoodPelletCapTests()
    {
        AssertEqual(25.0, FoodPelletCap.EstimateZonePellets(500.0, 2.0, 10.0),
            "stock biomass-to-pellet estimate");
        AssertEqual(0.0, FoodPelletCap.EstimateZonePellets(0.0, 1.0, 200.0),
            "zero biomass");
        AssertEqual(25, FoodPelletCap.Resolve(25.0, 8192, 32768),
            "low stock setting must not be scaled to 8192");
        AssertEqual(0, FoodPelletCap.Resolve(0.0, 8192, 32768),
            "zero stock food cap");
        AssertEqual(12, FoodPelletCap.Resolve(1000.0, 12, 32768),
            "explicit GPU pellet cap");
        AssertEqual(32768, FoodPelletCap.Resolve(50000.0, 50000, 32768),
            "native allocation ceiling");
        AssertEqual(0, FoodPelletCap.Resolve(double.NaN, 8192, 32768),
            "invalid biomass estimate");
        AssertEqual(0, FoodPelletCap.ResolveLoaded(0.0, 0.0, 0, 8192, 32768),
            "reloading a zero-biomass growth zone must not spawn the GPU cap");
        AssertEqual(0, FoodPelletCap.ResolveLoaded(100.0, 100.0, 0, 8192, 32768),
            "unchanged stock settings must preserve an explicitly saved zero target");
        AssertEqual(25, FoodPelletCap.ResolveLoaded(25.0, 0.0, 0, 8192, 32768),
            "raising biomass from zero must use the stock count, not the GPU cap");
        AssertEqual(120, FoodPelletCap.ResolveLoaded(200.0, 100.0, 60, 8192, 32768),
            "nonzero saved food levels retain proportional live settings");
        AssertEqual(0, FoodPelletCap.ResolveLoaded(0.0, 100.0, 60, 8192, 32768),
            "setting loaded food biomass to zero clears the target");
        AssertEqual(12, FoodPelletCap.ResolveLoaded(200.0, 100.0, 60, 12, 32768),
            "saved food levels still obey the GPU cap");
    }

    private static void DetailedSpriteVisibilityTests()
    {
        if (!DetailedSpriteVisibility.ShouldRender(550f, 1080))
            throw new InvalidOperationException(
                "Readable close-up sprites must remain enabled at the old zoom cutoff.");
        if (!DetailedSpriteVisibility.ShouldRender(1500f, 1080))
            throw new InvalidOperationException(
                "Visible multi-pixel sprites must not be disabled by a fixed world-size cutoff.");
        if (DetailedSpriteVisibility.ShouldRender(2500f, 1080))
            throw new InvalidOperationException(
                "Sub-pixel distant detail should fall back to the batched renderer.");
        if (DetailedSpriteVisibility.ShouldRender(float.NaN, 1080) ||
            DetailedSpriteVisibility.ShouldRender(100f, 0))
            throw new InvalidOperationException(
                "Invalid camera dimensions must disable detailed sprites safely.");
        if (!DetailedSpriteVisibility.CanReplaceLowDetailMesh(120, 2048, 512))
            throw new InvalidOperationException(
                "A viewport smaller than the texture pool must fully replace silhouettes.");
        if (!DetailedSpriteVisibility.CanReplaceLowDetailMesh(512, 512, 512))
            throw new InvalidOperationException(
                "A fully visible population that fits exactly must replace silhouettes.");
        if (DetailedSpriteVisibility.CanReplaceLowDetailMesh(512, 2048, 512))
            throw new InvalidOperationException(
                "A truncated full texture pool must retain low-detail population coverage.");
        if (DetailedSpriteVisibility.CanReplaceLowDetailMesh(0, 512, 512))
            throw new InvalidOperationException(
                "An empty detail list must not hide the low-detail mesh.");
        DetailedSpriteViewport sampled = new DetailedSpriteViewport(-112f, -112f, 112f, 112f);
        if (!sampled.Contains(new DetailedSpriteViewport(-100f, -100f, 100f, 100f)))
            throw new InvalidOperationException("Padded snapshots must cover the captured camera.");
        if (!sampled.Contains(new DetailedSpriteViewport(-110f, -100f, 90f, 100f)))
            throw new InvalidOperationException("Small camera moves inside the padding retain coverage.");
        if (sampled.Contains(new DetailedSpriteViewport(0f, -100f, 200f, 100f)) ||
            sampled.Contains(new DetailedSpriteViewport(-200f, -200f, 200f, 200f)))
            throw new InvalidOperationException("Panning or zooming beyond a snapshot must restore the mesh.");
        if (sampled.Contains(new DetailedSpriteViewport(float.NaN, 0f, 1f, 1f)) ||
            new DetailedSpriteViewport(0f, 0f, float.PositiveInfinity, 1f).Contains(sampled))
            throw new InvalidOperationException("Invalid viewport bounds must not claim population coverage.");
    }

    private static void BrainDiagramTests()
    {
        for (int i = 0; i < BrainDiagramMath.NativeNodeCount; i++)
        {
            int expected = i < 16 ? 0 : i < 28 ? 1 : i < 40 ? 2 : 3;
            if (BrainDiagramMath.NativeColumn(i) != expected)
                throw new InvalidOperationException("Native brain layer " + i);
        }
        if (BrainDiagramMath.NativeColumn(-1) != -1 ||
            BrainDiagramMath.NativeColumn(46) != -1)
            throw new InvalidOperationException("Out-of-range native brain layer");
        for (int i = 0; i < BrainDiagramMath.FullNativeNodeCount; i++)
        {
            int expected = i < 34 ? 0 : i < 46 ? 1 : i < 58 ? 2 : 3;
            if (BrainDiagramMath.FullNativeColumn(i) != expected)
                throw new InvalidOperationException("Full stock-I/O brain layer " + i);
        }
        if (BrainDiagramMath.FullNativeColumn(-1) != -1 ||
            BrainDiagramMath.FullNativeColumn(73) != -1)
            throw new InvalidOperationException("Out-of-range stock-I/O brain layer");
        float reference = BrainDiagramMath.ReferenceMagnitude(
            new[] { 0.1f, 0.2f, 0.4f, 1f, 100f });
        AssertEqual(1f, reference, "outlier-resistant brain brightness reference");
        float zero = BrainDiagramMath.StrengthBrightness(0f, reference);
        float weak = BrainDiagramMath.StrengthBrightness(0.1f, reference);
        float medium = BrainDiagramMath.StrengthBrightness(0.4f, reference);
        float strong = BrainDiagramMath.StrengthBrightness(1f, reference);
        if (!(zero < weak && weak < medium && medium < strong && strong <= 1f) ||
            BrainDiagramMath.StrengthBrightness(-0.4f, reference) != medium ||
            BrainDiagramMath.StrengthBrightness(float.NaN, reference) != 0f)
            throw new InvalidOperationException("Connection brightness must follow absolute strength");
    }

    private static void AssertEqual(float expected, float actual, string name)
    {
        if (expected != actual)
        {
            throw new InvalidOperationException(
                name + ": expected " + expected + ", got " + actual);
        }
    }

    private static void AssertEqual(double expected, double actual, string name)
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
