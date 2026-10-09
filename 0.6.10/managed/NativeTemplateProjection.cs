using System;
using System.Collections.Generic;
using SettingScripts;
using SimulationScripts.BibiteScripts;
using UnityEngine;
using Utility;

namespace BibitesGpuFork
{
    internal static class NativeTemplateProjection
    {
        internal const int MaximumNodes = 256;
        internal const int MaximumSynapses = 512;

        internal static bool TryCreate(
            BibiteTemplate template,
            RandomizeGenes randomizeGenes,
            Tagging tagging,
            GrowthAtSpawn growth,
            Vector3 position,
            float heading,
            string customTag,
            out NativeWorldTemplate result,
            out string error)
        {
            result = null;
            error = null;
            if (template == null)
            {
                error = "the selected Bibite template is missing";
                return false;
            }
            if (template.nodes == null || template.nodes.Length == 0)
            {
                error = "the selected template has no brain nodes";
                return false;
            }
            if (template.nodes.Length > MaximumNodes)
            {
                error = "the selected brain has " + template.nodes.Length +
                    " nodes; the GPU template limit is " + MaximumNodes;
                return false;
            }

            float[] genes = template.genes != null
                ? (float[])template.genes.Clone()
                : new float[0];
            if (genes.Length < 11)
            {
                error = "the selected template does not contain a complete 0.6.x genome";
                return false;
            }
            ApplyPlacementRandomization(genes, randomizeGenes);

            Dictionary<int, int> nodeIndices = new Dictionary<int, int>();
            NativeWorldBrainNode[] nodes = new NativeWorldBrainNode[template.nodes.Length];
            for (int index = 0; index < template.nodes.Length; index++)
            {
                NEATBrain.Node source = template.nodes[index];
                nodeIndices[source.Index] = index;
                nodes[index] = new NativeWorldBrainNode
                {
                    Type = Mathf.Clamp((int)source.Type, 0, 13),
                    Sensor = SensorOf(source.Desc),
                    Action = ActionOf(source.Desc),
                    BaseActivation = FiniteOr(source.baseActivation, 0f)
                };
            }

            List<NativeWorldBrainSynapse> synapses = new List<NativeWorldBrainSynapse>();
            if (template.synapses != null)
            {
                for (int index = 0; index < template.synapses.Length; index++)
                {
                    NEATBrain.Synaps source = template.synapses[index];
                    if (!source.En || !IsFinite(source.Weight))
                    {
                        continue;
                    }
                    int nodeIn;
                    int nodeOut;
                    if (!ResolveNode(source.NodeIn, nodeIndices, template.nodes.Length, out nodeIn) ||
                        !ResolveNode(source.NodeOut, nodeIndices, template.nodes.Length, out nodeOut))
                    {
                        continue;
                    }
                    float weight = source.Weight;
                    if (randomizeGenes == RandomizeGenes.NormalMutations ||
                        randomizeGenes == RandomizeGenes.AllGenes)
                    {
                        weight += UnityEngine.Random.Range(-1f, 1f) *
                            Mathf.Clamp(GetGene(genes, 10, 0.1f), 0f, 4f);
                    }
                    synapses.Add(new NativeWorldBrainSynapse
                    {
                        NodeIn = nodeIn,
                        NodeOut = nodeOut,
                        Weight = Mathf.Clamp(weight, -16f, 16f)
                    });
                }
            }
            if (synapses.Count > MaximumSynapses)
            {
                error = "the selected brain has " + synapses.Count +
                    " enabled synapses; the GPU template limit is " + MaximumSynapses;
                return false;
            }

            string templateName = CleanName(template.name, "Placed Bibite");
            string speciesName = CleanName(template.speciesName, templateName);
            string tagName = ResolveTagName(tagging, customTag, speciesName);
            ulong lineageId = StableId("species:" + speciesName + "|template:" + templateName);
            ulong tagId = string.Equals(tagName, "Untagged", StringComparison.Ordinal)
                ? 0ul
                : StableId("tag:" + tagName);

            float sizeRatio = Mathf.Clamp(GetGene(genes, 3, 1f), 0.35f, 2.5f);
            float speedRatio = Mathf.Clamp(GetGene(genes, 4, 1f), 0.15f, 3f);
            float growthScale = GrowthScale(genes, growth);
            float size = Mathf.Clamp(sizeRatio * growthScale, 0.2f, 3.5f);
            float maximumSpeed = Mathf.Clamp(4.5f * speedRatio / Mathf.Sqrt(size), 1f, 10f);
            float turnSpeed = Mathf.Clamp(2.7f * speedRatio / Mathf.Max(size, 0.35f), 0.5f, 8f);
            float metabolism = Mathf.Clamp(
                0.10f + 0.08f * speedRatio * speedRatio + 0.06f * size,
                0.05f,
                0.8f);
            float lifecycle = GetGene(genes, 0, 15f) + GetGene(genes, 1, 20f) +
                GetGene(genes, 2, 10f);
            // The first native projection used benchmark-scale 4-30 minute
            // lifespans. That made healthy placed Bibites disappear during an
            // ordinary observation session. Preserve lifecycle selection while
            // keeping the useful lifetime in the same range as the procedural
            // GPU population (roughly 30-90 minutes for typical genomes).
            float lifespan = Mathf.Clamp(1800f + lifecycle * 60f, 900f, 21600f);

            result = new NativeWorldTemplate
            {
                Spawn = new NativeWorldTemplateSpawn
                {
                    PositionX = position.x,
                    PositionY = position.y,
                    Heading = heading,
                    Energy = 90f,
                    Size = size,
                    MaximumSpeed = maximumSpeed,
                    TurnSpeed = turnSpeed,
                    Metabolism = metabolism,
                    Lifespan = lifespan,
                    ColorR = Mathf.Clamp01(GetGene(genes, 5, 0.5f)),
                    ColorG = Mathf.Clamp01(GetGene(genes, 6, 0.5f)),
                    ColorB = Mathf.Clamp01(GetGene(genes, 7, 0.5f)),
                    GeneMutationStrength = Mathf.Clamp(GetGene(genes, 8, 0.08f), 0f, 4f),
                    BrainMutationStrength = Mathf.Clamp(GetGene(genes, 10, 0.08f), 0f, 4f),
                    Diet = Mathf.Clamp01(GetGene(genes, 16, 0f)),
                    AdultSize = sizeRatio,
                    ClockPeriod = Mathf.Clamp(GetGene(genes, 14, 1f), 0.05f, 3600f),
                    LayPeriod = Mathf.Clamp(GetGene(genes, 0, 15f), 0.25f, 3600f),
                    WombCapacity = Mathf.Clamp(1f + GetGene(genes, 27, 0.25f) * 4f, 1f, 8f),
                    Generation = Math.Max(0, template.generation),
                    LineageId = lineageId,
                    TagId = tagId
                },
                Nodes = nodes,
                Synapses = synapses.ToArray(),
                TemplateName = templateName,
                SpeciesName = speciesName,
                TagName = tagName
            };
            return true;
        }

        private static bool ResolveNode(
            int requested,
            Dictionary<int, int> indices,
            int nodeCount,
            out int resolved)
        {
            if (indices.TryGetValue(requested, out resolved))
            {
                return true;
            }
            resolved = requested;
            return requested >= 0 && requested < nodeCount;
        }

        private static float GrowthScale(float[] genes, GrowthAtSpawn growth)
        {
            if (growth == GrowthAtSpawn.Adult)
            {
                return 1f;
            }
            if (growth == GrowthAtSpawn.Elder)
            {
                return 1.35f;
            }
            try
            {
                float birth = Mathf.Max(BibiteGenes.GrowthAtBirth(genes), 0.0001f);
                float mature = Mathf.Max(BibiteGenes.GrowthAtMature(genes), birth);
                return Mathf.Clamp(Mathf.Sqrt(birth / mature), 0.2f, 1f);
            }
            catch
            {
                return 0.45f;
            }
        }

        private static void ApplyPlacementRandomization(float[] genes, RandomizeGenes mode)
        {
            if (mode == RandomizeGenes.No)
            {
                return;
            }
            if (mode == RandomizeGenes.Color || mode == RandomizeGenes.AllGenes)
            {
                SetGene(genes, 5, UnityEngine.Random.value);
                SetGene(genes, 6, UnityEngine.Random.value);
                SetGene(genes, 7, UnityEngine.Random.value);
            }
            if (mode == RandomizeGenes.AllGenes)
            {
                SetGene(genes, 0, UnityEngine.Random.Range(3f, 30f));
                SetGene(genes, 1, UnityEngine.Random.Range(5f, 45f));
                SetGene(genes, 2, UnityEngine.Random.Range(3f, 25f));
                SetGene(genes, 3, UnityEngine.Random.Range(0.4f, 2f));
                SetGene(genes, 4, UnityEngine.Random.Range(0.25f, 2f));
                SetGene(genes, 8, UnityEngine.Random.Range(0.01f, 0.25f));
                SetGene(genes, 10, UnityEngine.Random.Range(0.02f, 0.3f));
            }
            else if (mode == RandomizeGenes.NormalMutations)
            {
                float sigma = Mathf.Clamp(GetGene(genes, 8, 0.08f), 0f, 1f);
                int[] mappedGenes = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 10 };
                for (int index = 0; index < mappedGenes.Length; index++)
                {
                    int gene = mappedGenes[index];
                    SetGene(genes, gene, GetGene(genes, gene, 0f) +
                        UnityEngine.Random.Range(-sigma, sigma));
                }
            }
        }

        private static int SensorOf(string description)
        {
            switch (description ?? string.Empty)
            {
                case "Constant": return 0;
                case "EnergyRatio": return 1;
                case "Maturity": return 2;
                case "LifeRatio": return 3;
                case "Fullness": return 4;
                case "Speed": return 5;
                case "RotationSpeed": return 6;
                case "IsGrabbing": return 7;
                case "AttackedDamage": return 8;
                case "EggStored": return 9;
                case "BibiteCloseness": return 10;
                case "BibiteAngle": return 11;
                case "NBibites": return 12;
                case "PlantCloseness": return 13;
                case "PlantAngle": return 14;
                case "NPlants": return 15;
                case "MeatCloseness": return 16;
                case "MeatAngle": return 17;
                case "NMeats": return 18;
                case "RedBibite": return 19;
                case "GreenBibite": return 20;
                case "BlueBibite": return 21;
                case "Tic": return 22;
                case "Minute": return 23;
                case "TimeAlive": return 24;
                case "PheroSense1": return 25;
                case "PheroSense2": return 26;
                case "PheroSense3": return 27;
                case "Phero1Angle": return 28;
                case "Phero2Angle": return 29;
                case "Phero3Angle": return 30;
                case "Phero1Heading": return 31;
                case "Phero2Heading": return 32;
                case "Phero3Heading": return 33;
                default: return -1;
            }
        }

        private static int ActionOf(string description)
        {
            switch (description ?? string.Empty)
            {
                case "Accelerate": return 0;
                case "Rotate": return 1;
                case "PhereOut1": return 2;
                case "PhereOut2": return 3;
                case "PhereOut3": return 4;
                case "EggProduction":
                    return 7;
                case "Want2Lay": return 5;
                case "Herding": return 6;
                case "Want2Eat": return 8;
                case "Digestion": return 9;
                case "Grab": return 10;
                case "ClkReset": return 11;
                case "Want2Grow": return 12;
                case "Want2Heal": return 13;
                case "Want2Attack": return 14;
                default: return -1;
            }
        }

        private static string ResolveTagName(Tagging tagging, string customTag, string speciesName)
        {
            if (tagging == Tagging.SpeciesTagging)
            {
                return speciesName;
            }
            if (tagging == Tagging.CustomTagging && !string.IsNullOrWhiteSpace(customTag))
            {
                return customTag.Trim();
            }
            if (tagging == Tagging.RandomTagging)
            {
                return "GPU tag " + UnityEngine.Random.Range(1, 1000).ToString("000");
            }
            return "Untagged";
        }

        private static string CleanName(string value, string fallback)
        {
            return string.IsNullOrWhiteSpace(value) ? fallback : value.Trim();
        }

        private static float GetGene(float[] genes, int index, float fallback)
        {
            return genes != null && index >= 0 && index < genes.Length && IsFinite(genes[index])
                ? genes[index]
                : fallback;
        }

        private static void SetGene(float[] genes, int index, float value)
        {
            if (genes != null && index >= 0 && index < genes.Length)
            {
                genes[index] = value;
            }
        }

        private static float FiniteOr(float value, float fallback)
        {
            return IsFinite(value) ? value : fallback;
        }

        private static bool IsFinite(float value)
        {
            return !float.IsNaN(value) && !float.IsInfinity(value);
        }

        internal static ulong StableId(string value)
        {
            const ulong offset = 14695981039346656037ul;
            const ulong prime = 1099511628211ul;
            ulong hash = offset;
            string text = value ?? string.Empty;
            for (int index = 0; index < text.Length; index++)
            {
                char character = text[index];
                hash ^= (byte)(character & 0xff);
                hash *= prime;
                hash ^= (byte)(character >> 8);
                hash *= prime;
            }
            return hash == 0ul ? 1ul : hash;
        }
    }
}
