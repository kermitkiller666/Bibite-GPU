using System;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Threading;
using HarmonyLib;
using SimulationScripts;
using SimulationScripts.BibiteScripts;

namespace BibitesGpuFork
{
    internal enum PerformanceCategory
    {
        BibiteBody,
        VisionLookup,
        VisionSenses,
        PheromoneSense,
        Mouth,
        Stomach,
        Propulsion,
        PheromoneOrgan,
        FatOrgan,
        Armor,
        EggLaying,
        Growth,
        InternalClock,
        Egg,
        MatterDecay,
        Zone,
        PheromoneSpot,
        Count
    }

    internal static class PerformanceDiagnostics
    {
        private static readonly long[] ElapsedTicks = new long[(int)PerformanceCategory.Count];
        private static readonly long[] Calls = new long[(int)PerformanceCategory.Count];

        internal static bool Enabled { get; private set; }

        internal static void Start()
        {
            for (int i = 0; i < ElapsedTicks.Length; i++)
            {
                ElapsedTicks[i] = 0;
                Calls[i] = 0;
            }
            Enabled = true;
        }

        internal static void Install(Harmony harmony)
        {
            Patch(harmony, typeof(BibiteBody), "FixedUpdate", typeof(ProfileBibiteBodyPatch));
            Patch(harmony, typeof(FieldOfView), "FindSeenEntities", typeof(ProfileVisionLookupPatch));
            Patch(harmony, typeof(FieldOfView), "ComputeSenses", typeof(ProfileVisionSensesPatch));
            Patch(harmony, typeof(Pherosense), "PherosenseAround", typeof(ProfilePheromoneSensePatch));
            Patch(harmony, typeof(BibiteMouth), "UpdateOrgan", typeof(ProfileMouthPatch));
            Patch(harmony, typeof(BibiteStomach), "UpdateOrgan", typeof(ProfileStomachPatch));
            Patch(harmony, typeof(BibitePropulsion), "UpdateOrgan", typeof(ProfilePropulsionPatch));
            Patch(harmony, typeof(BibitePheromoneOrgan), "UpdateOrgan", typeof(ProfilePheromoneOrganPatch));
            Patch(harmony, typeof(BibiteFatOrgan), "UpdateOrgan", typeof(ProfileFatOrganPatch));
            Patch(harmony, typeof(BibiteArmor), "UpdateOrgan", typeof(ProfileArmorPatch));
            Patch(harmony, typeof(BibiteEggLayingOrgan), "UpdateOrgan", typeof(ProfileEggLayingPatch));
            Patch(harmony, typeof(BibiteGrowth), "FixedUpdate", typeof(ProfileGrowthPatch));
            Patch(harmony, typeof(InternalClock), "FixedUpdate", typeof(ProfileInternalClockPatch));
            Patch(harmony, typeof(EggHatching), "FixedUpdate", typeof(ProfileEggPatch));
            Patch(harmony, typeof(MatterDecayProcessor), "FixedUpdate", typeof(ProfileMatterDecayPatch));
            Patch(harmony, typeof(Zone), "FixedUpdate", typeof(ProfileZonePatch));
            Patch(harmony, typeof(PheromoneSpot), "FixedUpdate", typeof(ProfilePheromoneSpotPatch));
        }

        private static void Patch(Harmony harmony, Type targetType, string methodName, Type patchType)
        {
            harmony.Patch(
                AccessTools.DeclaredMethod(targetType, methodName),
                new HarmonyMethod(AccessTools.DeclaredMethod(patchType, "Prefix")),
                new HarmonyMethod(AccessTools.DeclaredMethod(patchType, "Postfix")));
        }

        internal static long Begin()
        {
            return Enabled ? Stopwatch.GetTimestamp() : 0L;
        }

        internal static void End(PerformanceCategory category, long started)
        {
            if (started == 0L)
            {
                return;
            }
            Interlocked.Add(ref ElapsedTicks[(int)category], Stopwatch.GetTimestamp() - started);
            Interlocked.Increment(ref Calls[(int)category]);
        }

        internal static string StopAndReport(long fixedSteps, float wallSeconds)
        {
            Enabled = false;
            StringBuilder report = new StringBuilder("BGF_PROFILE");
            double wallMilliseconds = Math.Max(0.001, wallSeconds * 1000.0);
            for (int i = 0; i < (int)PerformanceCategory.Count; i++)
            {
                long calls = Interlocked.Read(ref Calls[i]);
                if (calls == 0)
                {
                    continue;
                }
                double milliseconds = Interlocked.Read(ref ElapsedTicks[i]) * 1000.0 / Stopwatch.Frequency;
                double perFixed = fixedSteps > 0 ? milliseconds / fixedSteps : 0.0;
                double microsecondsPerCall = milliseconds * 1000.0 / calls;
                double wallPercent = 100.0 * milliseconds / wallMilliseconds;
                report.Append(" | ").Append(((PerformanceCategory)i).ToString())
                    .Append(' ').Append(perFixed.ToString("0.000", CultureInfo.InvariantCulture)).Append(" ms/fixed")
                    .Append(", ").Append(microsecondsPerCall.ToString("0.0", CultureInfo.InvariantCulture)).Append(" us/call")
                    .Append(", ").Append(wallPercent.ToString("0.0", CultureInfo.InvariantCulture)).Append("% wall");
            }
            return report.ToString();
        }
    }

    internal static class ProfileBibiteBodyPatch
    {
        internal static void Prefix(out long __state) { __state = PerformanceDiagnostics.Begin(); }
        internal static void Postfix(long __state) { PerformanceDiagnostics.End(PerformanceCategory.BibiteBody, __state); }
    }

    internal static class ProfileVisionLookupPatch
    {
        internal static void Prefix(out long __state) { __state = PerformanceDiagnostics.Begin(); }
        internal static void Postfix(long __state) { PerformanceDiagnostics.End(PerformanceCategory.VisionLookup, __state); }
    }

    internal static class ProfileVisionSensesPatch
    {
        internal static void Prefix(out long __state) { __state = PerformanceDiagnostics.Begin(); }
        internal static void Postfix(long __state) { PerformanceDiagnostics.End(PerformanceCategory.VisionSenses, __state); }
    }

    internal static class ProfilePheromoneSensePatch
    {
        internal static void Prefix(out long __state) { __state = PerformanceDiagnostics.Begin(); }
        internal static void Postfix(long __state) { PerformanceDiagnostics.End(PerformanceCategory.PheromoneSense, __state); }
    }

    internal static class ProfileMouthPatch
    {
        internal static void Prefix(out long __state) { __state = PerformanceDiagnostics.Begin(); }
        internal static void Postfix(long __state) { PerformanceDiagnostics.End(PerformanceCategory.Mouth, __state); }
    }

    internal static class ProfileStomachPatch
    {
        internal static void Prefix(out long __state) { __state = PerformanceDiagnostics.Begin(); }
        internal static void Postfix(long __state) { PerformanceDiagnostics.End(PerformanceCategory.Stomach, __state); }
    }

    internal static class ProfilePropulsionPatch
    {
        internal static void Prefix(out long __state) { __state = PerformanceDiagnostics.Begin(); }
        internal static void Postfix(long __state) { PerformanceDiagnostics.End(PerformanceCategory.Propulsion, __state); }
    }

    internal static class ProfilePheromoneOrganPatch
    {
        internal static void Prefix(out long __state) { __state = PerformanceDiagnostics.Begin(); }
        internal static void Postfix(long __state) { PerformanceDiagnostics.End(PerformanceCategory.PheromoneOrgan, __state); }
    }

    internal static class ProfileFatOrganPatch
    {
        internal static void Prefix(out long __state) { __state = PerformanceDiagnostics.Begin(); }
        internal static void Postfix(long __state) { PerformanceDiagnostics.End(PerformanceCategory.FatOrgan, __state); }
    }

    internal static class ProfileArmorPatch
    {
        internal static void Prefix(out long __state) { __state = PerformanceDiagnostics.Begin(); }
        internal static void Postfix(long __state) { PerformanceDiagnostics.End(PerformanceCategory.Armor, __state); }
    }

    internal static class ProfileEggLayingPatch
    {
        internal static void Prefix(out long __state) { __state = PerformanceDiagnostics.Begin(); }
        internal static void Postfix(long __state) { PerformanceDiagnostics.End(PerformanceCategory.EggLaying, __state); }
    }

    internal static class ProfileGrowthPatch
    {
        internal static void Prefix(out long __state) { __state = PerformanceDiagnostics.Begin(); }
        internal static void Postfix(long __state) { PerformanceDiagnostics.End(PerformanceCategory.Growth, __state); }
    }

    internal static class ProfileInternalClockPatch
    {
        internal static void Prefix(out long __state) { __state = PerformanceDiagnostics.Begin(); }
        internal static void Postfix(long __state) { PerformanceDiagnostics.End(PerformanceCategory.InternalClock, __state); }
    }

    internal static class ProfileEggPatch
    {
        internal static void Prefix(out long __state) { __state = PerformanceDiagnostics.Begin(); }
        internal static void Postfix(long __state) { PerformanceDiagnostics.End(PerformanceCategory.Egg, __state); }
    }

    internal static class ProfileMatterDecayPatch
    {
        internal static void Prefix(out long __state) { __state = PerformanceDiagnostics.Begin(); }
        internal static void Postfix(long __state) { PerformanceDiagnostics.End(PerformanceCategory.MatterDecay, __state); }
    }

    internal static class ProfileZonePatch
    {
        internal static void Prefix(out long __state) { __state = PerformanceDiagnostics.Begin(); }
        internal static void Postfix(long __state) { PerformanceDiagnostics.End(PerformanceCategory.Zone, __state); }
    }

    internal static class ProfilePheromoneSpotPatch
    {
        internal static void Prefix(out long __state) { __state = PerformanceDiagnostics.Begin(); }
        internal static void Postfix(long __state) { PerformanceDiagnostics.End(PerformanceCategory.PheromoneSpot, __state); }
    }
}
