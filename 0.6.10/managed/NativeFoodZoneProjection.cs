using System;
using System.Collections.Generic;
using SettingScripts;
using SimulationScripts;
using UnityEngine;

namespace BibitesGpuFork
{
    internal static class NativeFoodZoneProjection
    {
        internal const int MaximumZones = 64;

        internal static NativeWorldFoodZone[] Capture(
            ScenarioSettings scenario, float worldHalfExtent, out int omitted)
        {
            omitted = 0;
            if (scenario == null || scenario.allZones == null)
                return new NativeWorldFoodZone[0];

            List<NativeWorldFoodZone> projected = new List<NativeWorldFoodZone>();
            foreach (ZoneSettings settings in scenario.allZones)
            {
                if (settings == null || settings.spawnMaterial == null ||
                    settings.spawnMaterial.val != MatterMaterialManager.Plant)
                    continue;

                float radius = settings.absoluteRadius;
                float halfWidth = settings.absoluteWidth * 0.5f;
                float halfHeight = settings.absoluteHeight * 0.5f;
                bool rectangle = settings.distribution.val == SpawnDistribution.Rect;
                if (rectangle ? !PositiveFinite(halfWidth) || !PositiveFinite(halfHeight)
                    : !PositiveFinite(radius))
                    continue;

                float seedWeight = Mathf.Max(0f, settings.maxBiomass);
                float growthWeight = Mathf.Max(0f, settings.totalGrowth);
                if (!Finite(seedWeight) || !Finite(growthWeight) ||
                    (seedWeight == 0f && growthWeight == 0f))
                    continue;

                Vector2 center = settings.posX != null && settings.posY != null
                    ? new Vector2(settings.posX.val * worldHalfExtent,
                        settings.posY.val * worldHalfExtent)
                    : Vector2.zero;
                ZoneManager manager = ZoneManager.instance;
                if (manager != null)
                {
                    foreach (Zone zone in manager.zones)
                    {
                        if (zone != null && zone.settings == settings)
                        {
                            center = zone.pos;
                            break;
                        }
                    }
                }
                if (!Finite(center.x) || !Finite(center.y)) continue;
                if (projected.Count >= MaximumZones)
                {
                    omitted++;
                    continue;
                }
                projected.Add(new NativeWorldFoodZone
                {
                    Distribution = (int)settings.distribution.val,
                    CenterX = Mathf.Clamp(center.x, -worldHalfExtent,
                        worldHalfExtent),
                    CenterY = Mathf.Clamp(center.y, -worldHalfExtent,
                        worldHalfExtent),
                    Radius = rectangle ? 0f : radius,
                    InnerRadius = Mathf.Clamp(settings.insideRadius.val,
                        0f, 0.999f),
                    HalfWidth = rectangle ? halfWidth : 0f,
                    HalfHeight = rectangle ? halfHeight : 0f,
                    SeedWeight = seedWeight,
                    GrowthWeight = growthWeight,
                    PelletSize = Mathf.Clamp(settings.pelletSize.val, 0.01f, 50f)
                });
            }
            return projected.ToArray();
        }

        internal static bool Same(
            NativeWorldFoodZone[] first, NativeWorldFoodZone[] second)
        {
            if (ReferenceEquals(first, second)) return true;
            if (first == null || second == null || first.Length != second.Length)
                return false;
            for (int i = 0; i < first.Length; i++)
            {
                NativeWorldFoodZone a = first[i];
                NativeWorldFoodZone b = second[i];
                if (a.Distribution != b.Distribution ||
                    a.CenterX != b.CenterX || a.CenterY != b.CenterY ||
                    a.Radius != b.Radius || a.InnerRadius != b.InnerRadius ||
                    a.HalfWidth != b.HalfWidth || a.HalfHeight != b.HalfHeight ||
                    a.SeedWeight != b.SeedWeight ||
                    a.GrowthWeight != b.GrowthWeight ||
                    a.PelletSize != b.PelletSize)
                    return false;
            }
            return true;
        }

        internal static float TotalGrowth(NativeWorldFoodZone[] zones)
        {
            double total = 0.0;
            foreach (NativeWorldFoodZone zone in zones)
                total += zone.GrowthWeight;
            return (float)Math.Min(float.MaxValue, total);
        }

        private static bool PositiveFinite(float value)
        {
            return value > 0f && Finite(value);
        }

        private static bool Finite(float value)
        {
            return !float.IsNaN(value) && !float.IsInfinity(value);
        }
    }
}
