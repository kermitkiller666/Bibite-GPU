using System;
using System.Collections.Generic;
using System.Diagnostics;
using BepInEx.Logging;
using ManagementScripts;
using SettingScripts;
using SimulationScripts;
using SimulationScripts.BibiteScripts;
using UnityEngine;

namespace BibitesGpuFork
{
    internal sealed class GpuAuthoritativeVisionScheduler : IDisposable
    {
        private sealed class VisionEntry
        {
            internal FieldOfView Vision;
            internal BibiteBody Body;
        }

        private readonly ManualLogSource _log;
        private readonly HashSet<FieldOfView> _registered = new HashSet<FieldOfView>();
        private readonly HashSet<int> _clearedDisplays = new HashSet<int>();
        private readonly List<VisionEntry> _worldBodies = new List<VisionEntry>();
        private readonly List<VisionEntry> _queries = new List<VisionEntry>();
        private readonly Dictionary<int, int> _heldByBody = new Dictionary<int, int>();
        private GpuVisionContext _context;
        private GpuVisionQuery[] _gpuQueries = new GpuVisionQuery[0];
        private GpuVisionEntity[] _gpuEntities = new GpuVisionEntity[0];
        private GpuVisionResult[] _gpuResults = new GpuVisionResult[0];
        private int _queryCapacity;
        private int _entityCapacity;
        private int _tickCounter = int.MaxValue;
        private int _deviceIndex;
        private string _failure;
        private int _lastQueryCount;
        private int _lastEntityCount;
        private double _lastMilliseconds;
        private long _batchCount;
        private long _fixedCallCount;

        internal GpuAuthoritativeVisionScheduler(ManualLogSource log, int deviceIndex)
        {
            _log = log;
            _deviceIndex = deviceIndex;
        }

        internal bool Active { get; private set; }
        internal bool Failed { get { return !string.IsNullOrEmpty(_failure); } }

        internal string StatusText
        {
            get
            {
                if (Failed) return "GPU vision fast mode failed: " + _failure;
                if (!Active) return "GPU vision fast mode preparing";
                return "GPU vision fast mode | " + _lastQueryCount + " viewers | " +
                    _lastEntityCount + " world entities | " +
                    _lastMilliseconds.ToString("0.000") + " ms/batch | " +
                    _batchCount + " batches | " + _fixedCallCount + " fixed calls";
            }
        }

        internal void Register(FieldOfView vision)
        {
            if (vision != null)
            {
                _registered.Add(vision);
            }
        }

        internal bool FixedStep()
        {
            try
            {
                _fixedCallCount++;
                int factor = Math.Max(1, ScenarioIndependentSettings.Instance.brainTPS.val);
                if (_tickCounter != int.MaxValue)
                {
                    _tickCounter++;
                }
                if (_tickCounter < factor)
                {
                    Active = true;
                    return true;
                }
                _tickCounter = 0;

                GatherBodiesAndQueries();
                if (_queries.Count == 0)
                {
                    _lastQueryCount = 0;
                    _lastEntityCount = 0;
                    Active = true;
                    return true;
                }

                int maximumEntities = _worldBodies.Count;
                WorldObjectsSpawner spawner = WorldObjectsSpawner.Instance;
                if (spawner != null && spawner.allPellets != null)
                {
                    maximumEntities += spawner.allPellets.Count;
                }
                EnsureCapacity(_queries.Count, maximumEntities);

                Stopwatch timer = Stopwatch.StartNew();
                BuildHeldMap();
                int entityCount = BuildEntities(spawner);
                BuildQueries(entityCount);
                _context.Evaluate(
                    _gpuQueries,
                    _queries.Count,
                    _gpuEntities,
                    entityCount,
                    _gpuResults);
                if (Plugin.Instance != null)
                {
                    for (int queryIndex = 0; queryIndex < _queries.Count; queryIndex++)
                    {
                        if (!Plugin.Instance.ValidateAuthoritativeVision(
                            _queries[queryIndex].Vision,
                            _gpuResults[queryIndex]))
                        {
                            break;
                        }
                    }
                }
                ApplyResults();
                timer.Stop();

                _lastQueryCount = _queries.Count;
                _lastEntityCount = entityCount;
                _lastMilliseconds = timer.Elapsed.TotalMilliseconds;
                _batchCount++;
                _failure = null;
                Active = true;
                return true;
            }
            catch (Exception ex)
            {
                _failure = ex.GetType().Name + ": " + ex.Message;
                Active = false;
                _log.LogError("Authoritative GPU vision failed; returning to stock CPU vision: " + ex);
                return false;
            }
        }

        internal void Reset()
        {
            Active = false;
            _failure = null;
            _tickCounter = int.MaxValue;
            _registered.Clear();
            _worldBodies.Clear();
            _queries.Clear();
            _heldByBody.Clear();
            _clearedDisplays.Clear();
            if (_context != null)
            {
                _context.Dispose();
                _context = null;
            }
            _queryCapacity = 0;
            _entityCapacity = 0;
            _gpuQueries = new GpuVisionQuery[0];
            _gpuEntities = new GpuVisionEntity[0];
            _gpuResults = new GpuVisionResult[0];
        }

        internal void SetDevice(int deviceIndex)
        {
            if (_deviceIndex == deviceIndex)
            {
                return;
            }
            Reset();
            _deviceIndex = deviceIndex;
        }

        private void GatherBodiesAndQueries()
        {
            _worldBodies.Clear();
            _queries.Clear();
            _registered.RemoveWhere(delegate(FieldOfView candidate) { return candidate == null; });
            foreach (FieldOfView vision in _registered)
            {
                if (vision == null)
                {
                    continue;
                }
                BibiteBody body = vision.GetComponent<BibiteBody>();
                if (body == null || body.destroyed || !body.born)
                {
                    continue;
                }
                VisionEntry entry = new VisionEntry { Vision = vision, Body = body };
                _worldBodies.Add(entry);
                if (!body.dead && vision.needToSee)
                {
                    _queries.Add(entry);
                }
            }
        }

        private void BuildHeldMap()
        {
            _heldByBody.Clear();
            for (int bodyIndex = 0; bodyIndex < _worldBodies.Count; bodyIndex++)
            {
                BibiteBody body = _worldBodies[bodyIndex].Body;
                BibiteMouth mouth = body.mouth;
                if (mouth == null)
                {
                    continue;
                }
                int heldCount = Math.Min(mouth.nHeld, mouth.links.Length);
                for (int heldIndex = 0; heldIndex < heldCount; heldIndex++)
                {
                    FixedJoint2D joint = mouth.links[heldIndex];
                    if (joint == null || joint.attachedRigidbody == null)
                    {
                        continue;
                    }
                    MatterPellet pellet = joint.attachedRigidbody.GetComponent<MatterPellet>();
                    if (pellet != null)
                    {
                        _heldByBody[pellet.GetInstanceID()] = body.GetInstanceID();
                    }
                }
            }
        }

        private int BuildEntities(WorldObjectsSpawner spawner)
        {
            int entityCount = 0;
            if (spawner != null && spawner.allPellets != null)
            {
                for (int pelletIndex = 0; pelletIndex < spawner.allPellets.Count; pelletIndex++)
                {
                    MatterPellet pellet = spawner.allPellets[pelletIndex];
                    if (pellet == null || pellet.amount < 0.01f)
                    {
                        continue;
                    }
                    int type;
                    if (pellet.material == MatterMaterialManager.Plant)
                    {
                        type = 0;
                    }
                    else if (pellet.material == MatterMaterialManager.Meat)
                    {
                        type = 1;
                    }
                    else
                    {
                        continue;
                    }
                    Vector3 position = pellet.transform.position;
                    Vector3 up = pellet.transform.up;
                    int pelletId = pellet.GetInstanceID();
                    int heldBy;
                    _heldByBody.TryGetValue(pelletId, out heldBy);
                    _gpuEntities[entityCount++] = new GpuVisionEntity
                    {
                        PositionX = position.x,
                        PositionY = position.y,
                        UpX = up.x,
                        UpY = up.y,
                        Radius = pellet.radius,
                        SizeFactor = pellet.sizeFactor,
                        Type = type,
                        EntityId = pelletId,
                        HeldByEntityId = heldBy
                    };
                }
            }

            for (int bodyIndex = 0; bodyIndex < _worldBodies.Count; bodyIndex++)
            {
                BibiteBody body = _worldBodies[bodyIndex].Body;
                Vector3 position = body.transform.position;
                Vector3 up = body.transform.up;
                float[] genes = body.gene != null ? body.gene.genes : null;
                _gpuEntities[entityCount++] = new GpuVisionEntity
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
                };
            }
            return entityCount;
        }

        private void BuildQueries(int entityCount)
        {
            for (int queryIndex = 0; queryIndex < _queries.Count; queryIndex++)
            {
                VisionEntry entry = _queries[queryIndex];
                FieldOfView vision = entry.Vision;
                Vector3 position = vision.transform.position;
                Vector3 up = vision.transform.up;
                _gpuQueries[queryIndex] = new GpuVisionQuery
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
                    EntityOffset = 0,
                    EntityCount = entityCount,
                    SelfEntityId = entry.Body.GetInstanceID(),
                    TargetMask = vision.targetMask
                };
            }
        }

        private void ApplyResults()
        {
            for (int queryIndex = 0; queryIndex < _queries.Count; queryIndex++)
            {
                FieldOfView vision = _queries[queryIndex].Vision;
                GpuVisionResult result = _gpuResults[queryIndex];
                int visionId = vision.GetInstanceID();
                if (_clearedDisplays.Add(visionId))
                {
                    Array.Clear(vision.plantWeights, 0, vision.plantWeights.Length);
                    Array.Clear(vision.meatWeights, 0, vision.meatWeights.Length);
                    Array.Clear(vision.bibiteWeights, 0, vision.bibiteWeights.Length);
                }
                vision.seenThisFrame = true;
                vision.nPlants = Math.Min(vision.plantWeights.Length, result.PlantCount);
                vision.nMeat = Math.Min(vision.meatWeights.Length, result.MeatCount);
                vision.nBibite = Math.Min(vision.bibiteWeights.Length, result.BibiteCount);
                vision.hasHerd = result.HasHerd != 0;
                vision.pelletConcentrationAngle = result.PlantAngle;
                vision.pelletConcentrationWeight = result.PlantWeight;
                vision.meatConcentrationAngle = result.MeatAngle;
                vision.meatConcentrationWeight = result.MeatWeight;
                vision.bibiteConcentrationAngle = result.BibiteAngle;
                vision.bibiteConcentrationWeight = result.BibiteWeight;
                vision.targetR = result.TargetR;
                vision.targetG = result.TargetG;
                vision.targetB = result.TargetB;
                vision.herdSumDirection = result.HerdDirection;
                vision.herdSeparationProjection = result.HerdSeparationProjection;
                vision.maxPlantWeight = result.MaxPlantWeight;
                vision.maxMeatWeight = result.MaxMeatWeight;
                vision.maxBibiteWeight = result.MaxBibiteWeight;
                vision.plantWeightSum = 0f;
                vision.meatWeightSum = 0f;
                vision.bibiteWeightSum = 0f;
                vision.dist2Bibite = float.PositiveInfinity;
            }
        }

        private void EnsureCapacity(int queryCount, int entityCount)
        {
            if (_context != null && queryCount <= _queryCapacity && entityCount <= _entityCapacity)
            {
                return;
            }
            if (_context != null)
            {
                _context.Dispose();
            }
            _queryCapacity = Grow(_queryCapacity, queryCount);
            _entityCapacity = Grow(_entityCapacity, entityCount);
            _gpuQueries = new GpuVisionQuery[_queryCapacity];
            _gpuEntities = new GpuVisionEntity[_entityCapacity];
            _gpuResults = new GpuVisionResult[_queryCapacity];
            _context = new GpuVisionContext(_deviceIndex, _queryCapacity, _entityCapacity);
            _log.LogInfo("Authoritative GPU vision buffers resized for " +
                _queryCapacity + " viewers and " + _entityCapacity + " entities.");
        }

        private static int Grow(int current, int required)
        {
            int value = Math.Max(16, current);
            while (value < required)
            {
                value *= 2;
            }
            return value;
        }

        public void Dispose()
        {
            Reset();
        }
    }
}
