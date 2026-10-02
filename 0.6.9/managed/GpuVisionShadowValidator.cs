using System;
using System.Collections.Generic;
using System.Reflection;
using BepInEx.Logging;
using SimulationScripts;
using SimulationScripts.BibiteScripts;
using UnityEngine;

namespace BibitesGpuFork
{
    internal sealed class VisionShadowSample
    {
        internal GpuVisionQuery Query;
        internal GpuVisionEntity[] Entities;
        internal GpuVisionResult Expected;
    }

    internal sealed class GpuVisionShadowValidator
    {
        private const float Tolerance = 0.004f;
        private static readonly FieldInfo PlantsInRange = PrivateField("nPlantsInRange");
        private static readonly FieldInfo MeatsInRange = PrivateField("nMeatsInRange");
        private static readonly FieldInfo BibitesInRange = PrivateField("nBibitesInRange");

        private readonly ManualLogSource _log;
        private readonly List<VisionShadowSample> _completed = new List<VisionShadowSample>();
        private long _valuesCompared;
        private long _mismatches;
        private float _largestError;
        private string _lastFailure;

        internal GpuVisionShadowValidator(ManualLogSource log)
        {
            _log = log;
        }

        internal string StatusText
        {
            get
            {
                if (!string.IsNullOrEmpty(_lastFailure))
                {
                    return "GPU vision error: " + _lastFailure;
                }
                if (_valuesCompared == 0)
                {
                    return "GPU vision validation on | waiting for active vision";
                }
                return "GPU vision validation | " + _valuesCompared + " values | " +
                    _mismatches + " outside tolerance | max error " + _largestError.ToString("0.000000");
            }
        }

        internal VisionShadowSample Begin(FieldOfView vision, int sampleLimit)
        {
            if (vision == null || !vision.needToSee || _completed.Count >= sampleLimit)
            {
                return null;
            }

            int plantCount = ReadCount(PlantsInRange, vision);
            int meatCount = ReadCount(MeatsInRange, vision);
            int bibiteCount = ReadCount(BibitesInRange, vision);
            List<GpuVisionEntity> entities = new List<GpuVisionEntity>(plantCount + meatCount + bibiteCount);
            for (int i = 0; i < plantCount; i++)
            {
                AddPellet(entities, vision.seenPlantPellets[i], 0, vision);
            }
            for (int i = 0; i < meatCount; i++)
            {
                AddPellet(entities, vision.seenMeatPellets[i], 1, vision);
            }
            for (int i = 0; i < bibiteCount; i++)
            {
                AddBibite(entities, vision.seenBibites[i]);
            }

            Vector3 position = vision.transform.position;
            Vector3 up = vision.transform.up;
            BibiteBody owner = vision.GetComponent<BibiteBody>();
            return new VisionShadowSample
            {
                Query = new GpuVisionQuery
                {
                    PositionX = position.x,
                    PositionY = position.y,
                    UpX = up.x,
                    UpY = up.y,
                    ViewRadius = vision.viewRadius,
                    ViewAngleDegrees = vision.viewAngle,
                    HerdSeparationDistance = vision.herdSeparationDistance,
                    SeparationWeight = vision.separationWeight,
                    CohesionWeight = vision.cohesionWeight,
                    AlignmentWeight = vision.alignmentWeight,
                    EntityCount = entities.Count,
                    SelfEntityId = owner != null ? owner.GetInstanceID() : 0,
                    TargetMask = vision.targetMask
                },
                Entities = entities.ToArray()
            };
        }

        internal void Complete(FieldOfView vision, VisionShadowSample sample)
        {
            if (sample == null)
            {
                return;
            }
            sample.Expected = Capture(vision);
            _completed.Add(sample);
        }

        internal void CompareAuthoritative(GpuVisionResult expected, GpuVisionResult actual)
        {
            Compare(expected, actual);
            _lastFailure = null;
        }

        internal static GpuVisionResult Capture(FieldOfView vision)
        {
            return new GpuVisionResult
            {
                PlantCount = vision.nPlants,
                MeatCount = vision.nMeat,
                BibiteCount = vision.nBibite,
                HasHerd = vision.hasHerd ? 1 : 0,
                PlantAngle = vision.pelletConcentrationAngle,
                PlantWeight = vision.pelletConcentrationWeight,
                MeatAngle = vision.meatConcentrationAngle,
                MeatWeight = vision.meatConcentrationWeight,
                BibiteAngle = vision.bibiteConcentrationAngle,
                BibiteWeight = vision.bibiteConcentrationWeight,
                TargetR = vision.targetR,
                TargetG = vision.targetG,
                TargetB = vision.targetB,
                HerdDirection = vision.herdSumDirection,
                HerdSeparationProjection = vision.herdSeparationProjection,
                MaxPlantWeight = vision.maxPlantWeight,
                MaxMeatWeight = vision.maxMeatWeight,
                MaxBibiteWeight = vision.maxBibiteWeight
            };
        }

        internal void Flush()
        {
            if (_completed.Count == 0)
            {
                return;
            }
            try
            {
                GpuVisionQuery[] queries = new GpuVisionQuery[_completed.Count];
                int entityCount = 0;
                for (int i = 0; i < _completed.Count; i++)
                {
                    entityCount += _completed[i].Entities.Length;
                }
                GpuVisionEntity[] entities = new GpuVisionEntity[entityCount];
                int entityOffset = 0;
                for (int i = 0; i < _completed.Count; i++)
                {
                    VisionShadowSample sample = _completed[i];
                    GpuVisionQuery query = sample.Query;
                    query.EntityOffset = entityOffset;
                    query.EntityCount = sample.Entities.Length;
                    queries[i] = query;
                    Array.Copy(sample.Entities, 0, entities, entityOffset, sample.Entities.Length);
                    entityOffset += sample.Entities.Length;
                }

                GpuVisionResult[] actual = new GpuVisionResult[queries.Length];
                NativeVision.Evaluate(queries, entities, actual);
                for (int i = 0; i < actual.Length; i++)
                {
                    Compare(_completed[i].Expected, actual[i]);
                }
                _lastFailure = null;
            }
            catch (Exception ex)
            {
                _lastFailure = ex.GetType().Name + ": " + ex.Message;
                _log.LogError("GPU vision shadow validation failed: " + ex);
            }
            finally
            {
                _completed.Clear();
            }
        }

        internal void ClearPending()
        {
            _completed.Clear();
        }

        private void Compare(GpuVisionResult expected, GpuVisionResult actual)
        {
            CompareInt(expected.PlantCount, actual.PlantCount);
            CompareInt(expected.MeatCount, actual.MeatCount);
            CompareInt(expected.BibiteCount, actual.BibiteCount);
            CompareInt(expected.HasHerd, actual.HasHerd);
            CompareFloat(expected.PlantAngle, actual.PlantAngle);
            CompareFloat(expected.PlantWeight, actual.PlantWeight);
            CompareFloat(expected.MeatAngle, actual.MeatAngle);
            CompareFloat(expected.MeatWeight, actual.MeatWeight);
            CompareFloat(expected.BibiteAngle, actual.BibiteAngle);
            CompareFloat(expected.BibiteWeight, actual.BibiteWeight);
            CompareFloat(expected.TargetR, actual.TargetR);
            CompareFloat(expected.TargetG, actual.TargetG);
            CompareFloat(expected.TargetB, actual.TargetB);
            CompareFloat(expected.HerdDirection, actual.HerdDirection);
            CompareFloat(expected.HerdSeparationProjection, actual.HerdSeparationProjection);
            CompareFloat(expected.MaxPlantWeight, actual.MaxPlantWeight);
            CompareFloat(expected.MaxMeatWeight, actual.MaxMeatWeight);
            CompareFloat(expected.MaxBibiteWeight, actual.MaxBibiteWeight);
        }

        private void CompareInt(int expected, int actual)
        {
            _valuesCompared++;
            if (expected != actual) _mismatches++;
        }

        private void CompareFloat(float expected, float actual)
        {
            _valuesCompared++;
            float error = Math.Abs(expected - actual);
            if (float.IsNaN(expected) != float.IsNaN(actual) || error > Tolerance)
            {
                _mismatches++;
            }
            if (!float.IsNaN(error) && error > _largestError)
            {
                _largestError = error;
            }
        }

        private static void AddPellet(
            ICollection<GpuVisionEntity> entities,
            MatterPellet pellet,
            int type,
            FieldOfView vision)
        {
            if (pellet == null || pellet.amount < 0.01f)
            {
                return;
            }
            Vector3 position = pellet.transform.position;
            Vector3 up = pellet.transform.up;
            entities.Add(new GpuVisionEntity
            {
                PositionX = position.x,
                PositionY = position.y,
                UpX = up.x,
                UpY = up.y,
                Radius = pellet.radius,
                SizeFactor = pellet.sizeFactor,
                Type = type,
                Flags = IsHeld(vision, pellet.transform) ? 1 : 0,
                EntityId = pellet.GetInstanceID()
            });
        }

        private static void AddBibite(ICollection<GpuVisionEntity> entities, BibiteBody body)
        {
            if (body == null)
            {
                return;
            }
            Vector3 position = body.transform.position;
            Vector3 up = body.transform.up;
            float[] genes = body.gene != null ? body.gene.genes : null;
            entities.Add(new GpuVisionEntity
            {
                PositionX = position.x,
                PositionY = position.y,
                UpX = up.x,
                UpY = up.y,
                Radius = body.sideRadius,
                MaxHealth = body.maxHealth,
                HealthRatio = body.maxHealth > 0f ? body.health / body.maxHealth : 0f,
                ColorR = genes != null && genes.Length > 7 ? genes[5] : 0f,
                ColorG = genes != null && genes.Length > 7 ? genes[6] : 0f,
                ColorB = genes != null && genes.Length > 7 ? genes[7] : 0f,
                Type = body.dead ? 3 : 2,
                Flags = genes != null && genes.Length > 7 ? 2 : 0,
                EntityId = body.GetInstanceID()
            });
        }

        private static bool IsHeld(FieldOfView vision, Transform target)
        {
            for (int i = 0; i < vision.nHeld; i++)
            {
                if (vision.held[i] == target) return true;
            }
            return false;
        }

        private static int ReadCount(FieldInfo field, FieldOfView vision)
        {
            return (int)field.GetValue(vision);
        }

        private static FieldInfo PrivateField(string name)
        {
            FieldInfo field = typeof(FieldOfView).GetField(
                name,
                BindingFlags.Instance | BindingFlags.NonPublic);
            if (field == null)
            {
                throw new MissingFieldException(typeof(FieldOfView).FullName, name);
            }
            return field;
        }
    }
}
