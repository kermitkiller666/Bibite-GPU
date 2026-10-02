using System;
using System.Collections.Generic;
using System.Diagnostics;
using BepInEx.Logging;
using SettingScripts;
using SimulationScripts;
using SimulationScripts.BibiteScripts;
using SimulationScripts.Events;
using UnityEngine;

namespace BibitesGpuFork
{
    internal sealed class GpuAuthoritativePheromoneScheduler : IDisposable
    {
        private sealed class SensorEntry
        {
            internal Pherosense Sensor;
            internal BibiteBody Body;
            internal BibiteGenes Genes;
        }

        private readonly ManualLogSource _log;
        private readonly HashSet<Pherosense> _registeredSensors = new HashSet<Pherosense>();
        private readonly HashSet<PheromoneSpot> _registeredSpots = new HashSet<PheromoneSpot>();
        private readonly List<SensorEntry> _queries = new List<SensorEntry>();
        private readonly List<PheromoneSpot> _spots = new List<PheromoneSpot>();
        private GpuPheromoneContext _context;
        private GpuPheromoneQuery[] _gpuQueries = new GpuPheromoneQuery[0];
        private GpuPheromoneEntity[] _gpuEntities = new GpuPheromoneEntity[0];
        private GpuPheromoneResult[] _gpuResults = new GpuPheromoneResult[0];
        private int _queryCapacity;
        private int _entityCapacity;
        private int _tickCounter = int.MaxValue;
        private int _deviceIndex;
        private string _failure;
        private int _lastQueryCount;
        private int _lastEntityCount;
        private double _lastMilliseconds;
        private long _batchCount;
        private long _valuesCompared;
        private long _mismatches;
        private float _largestError;

        internal GpuAuthoritativePheromoneScheduler(ManualLogSource log, int deviceIndex)
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
                if (Failed) return "GPU pheromone mode failed: " + _failure;
                if (!Active) return "GPU pheromone mode preparing";
                string validation = _valuesCompared > 0
                    ? " | validation " + _valuesCompared + " values, " + _mismatches +
                        " outside tolerance, max error " + _largestError.ToString("0.000000")
                    : string.Empty;
                return "GPU pheromone mode | " + _lastQueryCount + " sensors | " +
                    _lastEntityCount + " spots | " + _lastMilliseconds.ToString("0.000") +
                    " ms/batch | " + _batchCount + " batches" + validation;
            }
        }

        internal void RegisterSensor(Pherosense sensor)
        {
            if (sensor != null)
            {
                _registeredSensors.Add(sensor);
            }
        }

        internal void RegisterSpot(PheromoneSpot spot)
        {
            if (spot != null)
            {
                _registeredSpots.Add(spot);
            }
        }

        internal bool FixedStep()
        {
            try
            {
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

                Gather();
                if (_queries.Count == 0)
                {
                    _lastQueryCount = 0;
                    _lastEntityCount = _spots.Count;
                    Active = true;
                    return true;
                }
                EnsureCapacity(_queries.Count, _spots.Count);

                Stopwatch timer = Stopwatch.StartNew();
                BuildQueries();
                int entityCount = BuildEntities();
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
                        if (!Plugin.Instance.ValidateAuthoritativePheromone(
                            _queries[queryIndex].Sensor,
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
                _log.LogError("Authoritative GPU pheromone sensing failed; returning to stock CPU sensing: " + ex);
                return false;
            }
        }

        internal void Reset()
        {
            Active = false;
            _failure = null;
            _tickCounter = int.MaxValue;
            _registeredSensors.Clear();
            _registeredSpots.Clear();
            _queries.Clear();
            _spots.Clear();
            _valuesCompared = 0;
            _mismatches = 0;
            _largestError = 0f;
            if (_context != null)
            {
                _context.Dispose();
                _context = null;
            }
            _queryCapacity = 0;
            _entityCapacity = 0;
            _gpuQueries = new GpuPheromoneQuery[0];
            _gpuEntities = new GpuPheromoneEntity[0];
            _gpuResults = new GpuPheromoneResult[0];
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

        internal void CompareStock(Pherosense sensor, GpuPheromoneResult actual)
        {
            Compare(sensor.pheroSum1, actual.RedSum);
            Compare(sensor.pheroSum2, actual.GreenSum);
            Compare(sensor.pheroSum3, actual.BlueSum);
            Compare(sensor.angleToPhero1, actual.RedAngle);
            Compare(sensor.angleToPhero2, actual.GreenAngle);
            Compare(sensor.angleToPhero3, actual.BlueAngle);
            Compare(sensor.angleToPhero1Heading, actual.RedHeadingAngle);
            Compare(sensor.angleToPhero2Heading, actual.GreenHeadingAngle);
            Compare(sensor.angleToPhero3Heading, actual.BlueHeadingAngle);
        }

        private void Compare(float expected, float actual)
        {
            const float tolerance = 0.004f;
            _valuesCompared++;
            float error = Math.Abs(expected - actual);
            if (float.IsNaN(expected) != float.IsNaN(actual) || error > tolerance)
            {
                _mismatches++;
            }
            if (!float.IsNaN(error) && error > _largestError)
            {
                _largestError = error;
            }
        }

        private void Gather()
        {
            _queries.Clear();
            _spots.Clear();
            _registeredSensors.RemoveWhere(delegate(Pherosense sensor) { return sensor == null; });
            _registeredSpots.RemoveWhere(delegate(PheromoneSpot spot) { return spot == null; });
            foreach (Pherosense sensor in _registeredSensors)
            {
                if (sensor == null || !sensor.usePheroSense)
                {
                    continue;
                }
                BibiteBody body = sensor.GetComponent<BibiteBody>();
                BibiteGenes genes = sensor.GetComponent<BibiteGenes>();
                if (body == null || genes == null || body.destroyed || body.dead || !body.born ||
                    genes.genes == null || genes.genes.Length <= 15)
                {
                    continue;
                }
                _queries.Add(new SensorEntry { Sensor = sensor, Body = body, Genes = genes });
            }
            foreach (PheromoneSpot spot in _registeredSpots)
            {
                if (spot != null &&
                    (spot.Rstrength > 0f || spot.Gstrength > 0f || spot.Bstrength > 0f))
                {
                    _spots.Add(spot);
                }
            }
        }

        private void BuildQueries()
        {
            bool redDeath = ScenarioSettings.Instance.enableRedDeath.val;
            float safeRadius = RedDeathBloomManager.safeRadius;
            for (int queryIndex = 0; queryIndex < _queries.Count; queryIndex++)
            {
                SensorEntry entry = _queries[queryIndex];
                Vector3 position = entry.Sensor.transform.position;
                Vector3 up = entry.Sensor.transform.up;
                _gpuQueries[queryIndex] = new GpuPheromoneQuery
                {
                    PositionX = position.x,
                    PositionY = position.y,
                    UpX = up.x,
                    UpY = up.y,
                    SenseRadius = entry.Genes.genes[15],
                    RedDeathSafeRadius = safeRadius,
                    EnableRedDeath = redDeath ? 1 : 0
                };
            }
        }

        private int BuildEntities()
        {
            for (int entityIndex = 0; entityIndex < _spots.Count; entityIndex++)
            {
                PheromoneSpot spot = _spots[entityIndex];
                Vector3 position = spot.transform.position;
                Vector2 heading = spot.heading;
                _gpuEntities[entityIndex] = new GpuPheromoneEntity
                {
                    PositionX = position.x,
                    PositionY = position.y,
                    HeadingX = heading.x,
                    HeadingY = heading.y,
                    RedStrength = spot.Rstrength,
                    GreenStrength = spot.Gstrength,
                    BlueStrength = spot.Bstrength
                };
            }
            return _spots.Count;
        }

        private void ApplyResults()
        {
            for (int queryIndex = 0; queryIndex < _queries.Count; queryIndex++)
            {
                Pherosense sensor = _queries[queryIndex].Sensor;
                GpuPheromoneResult result = _gpuResults[queryIndex];
                sensor.pheroSum1 = result.RedSum;
                sensor.pheroSum2 = result.GreenSum;
                sensor.pheroSum3 = result.BlueSum;
                sensor.angleToPhero1 = result.RedAngle;
                sensor.angleToPhero2 = result.GreenAngle;
                sensor.angleToPhero3 = result.BlueAngle;
                sensor.angleToPhero1Heading = result.RedHeadingAngle;
                sensor.angleToPhero2Heading = result.GreenHeadingAngle;
                sensor.angleToPhero3Heading = result.BlueHeadingAngle;
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
            _gpuQueries = new GpuPheromoneQuery[_queryCapacity];
            _gpuEntities = new GpuPheromoneEntity[_entityCapacity];
            _gpuResults = new GpuPheromoneResult[_queryCapacity];
            _context = new GpuPheromoneContext(_deviceIndex, _queryCapacity, _entityCapacity);
            _log.LogInfo("Authoritative GPU pheromone buffers resized for " +
                _queryCapacity + " sensors and " + _entityCapacity + " spots.");
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
