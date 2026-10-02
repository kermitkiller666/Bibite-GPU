using System;
using System.Collections.Generic;
using BepInEx.Logging;
using BibitesGpuFork.Core;
using SimulationScripts.BibiteScripts;

namespace BibitesGpuFork
{
    internal sealed class GpuShadowValidator : IDisposable
    {
        private const float AbsoluteTolerance = 0.003f;
        private const float RelativeTolerance = 0.003f;

        private sealed class Sample
        {
            internal NEATBrain Brain;
            internal NEATBrain.Node[] Before;
            internal NEATBrain.Node[] Expected;
            internal NEATBrain.Synaps[] Synapses;
            internal int[] Targets;
            internal int[][] Links;
        }

        private readonly ManualLogSource _log;
        private readonly Dictionary<NEATBrain, Sample> _active = new Dictionary<NEATBrain, Sample>();
        private readonly List<Sample> _completed = new List<Sample>();
        private GpuBrainContext _context;
        private int _brainCapacity;
        private int _nodeCapacity;
        private int _targetCapacity;
        private int _edgeCapacity;
        private long _totalCompared;
        private long _totalMismatches;
        private float _largestError;
        private string _lastFailure;
        private int _deviceIndex;

        internal GpuShadowValidator(ManualLogSource log, int deviceIndex)
        {
            _log = log;
            _deviceIndex = deviceIndex;
        }

        internal string StatusText
        {
            get
            {
                if (!string.IsNullOrEmpty(_lastFailure))
                {
                    return "GPU validation error: " + _lastFailure;
                }
                if (_totalCompared == 0)
                {
                    return "GPU brain validation on | waiting for live brains";
                }
                return "GPU brain validation | " + _totalCompared + " values | " +
                    _totalMismatches + " outside tolerance | max error " + _largestError.ToString("0.000000");
            }
        }

        internal bool Begin(NEATBrain brain, int sampleLimit)
        {
            if (brain == null || brain.Nodes == null || brain.Synapses == null ||
                brain.brainTargets == null || brain.brainLinks == null ||
                _active.ContainsKey(brain) || _completed.Count + _active.Count >= sampleLimit)
            {
                return false;
            }

            _active.Add(brain, new Sample { Brain = brain });
            return true;
        }

        internal void CaptureAfterSenses(NEATBrain brain)
        {
            Sample sample;
            if (!_active.TryGetValue(brain, out sample) || sample.Before != null)
            {
                return;
            }

            sample.Before = (NEATBrain.Node[])brain.Nodes.Clone();
            sample.Synapses = (NEATBrain.Synaps[])brain.Synapses.Clone();
            sample.Targets = brain.brainTargets.ToArray();
            sample.Links = new int[brain.brainLinks.Count][];
            for (int i = 0; i < sample.Links.Length; i++)
            {
                sample.Links[i] = brain.brainLinks[i].ToArray();
            }
        }

        internal void CompleteAfterCpu(NEATBrain brain)
        {
            Sample sample;
            if (!_active.TryGetValue(brain, out sample))
            {
                return;
            }

            _active.Remove(brain);
            if (sample.Before == null)
            {
                return;
            }
            sample.Expected = (NEATBrain.Node[])brain.Nodes.Clone();
            _completed.Add(sample);
        }

        internal void Flush()
        {
            if (_completed.Count == 0)
            {
                return;
            }

            try
            {
                RunBatch();
                _lastFailure = null;
            }
            catch (Exception ex)
            {
                _lastFailure = ex.GetType().Name + ": " + ex.Message;
                _log.LogError("GPU shadow validation failed: " + ex);
            }
            finally
            {
                _completed.Clear();
            }
        }

        internal void ClearPending()
        {
            _active.Clear();
            _completed.Clear();
        }

        private void RunBatch()
        {
            int nodeCount = 0;
            int targetCount = 0;
            int edgeCount = 0;
            for (int i = 0; i < _completed.Count; i++)
            {
                Sample sample = _completed[i];
                nodeCount += sample.Before.Length;
                targetCount += sample.Targets.Length;
                for (int target = 0; target < sample.Links.Length; target++)
                {
                    edgeCount += sample.Links[target].Length;
                }
            }

            GpuBrain[] brains = new GpuBrain[_completed.Count];
            GpuNode[] nodes = new GpuNode[nodeCount];
            GpuTarget[] targets = new GpuTarget[targetCount];
            GpuEdge[] edges = new GpuEdge[edgeCount];

            int nodeOffset = 0;
            int targetOffset = 0;
            int edgeOffset = 0;
            for (int brainIndex = 0; brainIndex < _completed.Count; brainIndex++)
            {
                Sample sample = _completed[brainIndex];
                brains[brainIndex] = new GpuBrain
                {
                    NodeOffset = nodeOffset,
                    TargetOffset = targetOffset,
                    TargetCount = sample.Targets.Length,
                    Period = NEATBrain.brainPeriod
                };

                for (int nodeIndex = 0; nodeIndex < sample.Before.Length; nodeIndex++)
                {
                    NEATBrain.Node source = sample.Before[nodeIndex];
                    nodes[nodeOffset + nodeIndex] = new GpuNode
                    {
                        Value = source.Value,
                        LastOutput = source.LastOutput,
                        LastInput = source.LastInput,
                        Bias = source.baseActivation,
                        Type = (int)source.Type
                    };
                }

                for (int targetIndex = 0; targetIndex < sample.Targets.Length; targetIndex++)
                {
                    int[] links = sample.Links[targetIndex];
                    targets[targetOffset + targetIndex] = new GpuTarget
                    {
                        NodeIndex = sample.Targets[targetIndex],
                        EdgeOffset = edgeOffset,
                        EdgeCount = links.Length
                    };
                    for (int linkIndex = 0; linkIndex < links.Length; linkIndex++)
                    {
                        NEATBrain.Synaps synapse = sample.Synapses[links[linkIndex]];
                        edges[edgeOffset++] = new GpuEdge
                        {
                            SourceNode = synapse.NodeIn,
                            WeightBits = HalfConverter.ToHalfBits(synapse.Weight)
                        };
                    }
                }

                nodeOffset += sample.Before.Length;
                targetOffset += sample.Targets.Length;
            }

            EnsureCapacity(brains.Length, nodes.Length, targets.Length, edges.Length);
            _context.UploadBrains(brains);
            _context.UploadNodes(nodes);
            _context.UploadTopology(targets, edges);
            _context.Step();
            _context.DownloadNodes(nodes);

            nodeOffset = 0;
            for (int brainIndex = 0; brainIndex < _completed.Count; brainIndex++)
            {
                Sample sample = _completed[brainIndex];
                for (int targetIndex = 0; targetIndex < sample.Targets.Length; targetIndex++)
                {
                    int localNodeIndex = sample.Targets[targetIndex];
                    float expected = sample.Expected[localNodeIndex].Value;
                    float actual = nodes[nodeOffset + localNodeIndex].Value;
                    float error = Math.Abs(expected - actual);
                    float tolerance = AbsoluteTolerance + RelativeTolerance * Math.Abs(expected);
                    _totalCompared++;
                    if (error > tolerance || float.IsNaN(actual) != float.IsNaN(expected))
                    {
                        _totalMismatches++;
                    }
                    if (!float.IsNaN(error) && error > _largestError)
                    {
                        _largestError = error;
                    }
                }
                nodeOffset += sample.Before.Length;
            }
        }

        private void EnsureCapacity(int brains, int nodes, int targets, int edges)
        {
            if (_context != null && brains <= _brainCapacity && nodes <= _nodeCapacity &&
                targets <= _targetCapacity && edges <= _edgeCapacity)
            {
                return;
            }

            if (_context != null)
            {
                _context.Dispose();
            }
            _brainCapacity = GrowCapacity(_brainCapacity, brains);
            _nodeCapacity = GrowCapacity(_nodeCapacity, nodes);
            _targetCapacity = GrowCapacity(_targetCapacity, targets);
            _edgeCapacity = GrowCapacity(_edgeCapacity, edges);
            _context = new GpuBrainContext(_deviceIndex, _brainCapacity, _nodeCapacity, _targetCapacity, _edgeCapacity);
            _log.LogInfo("GPU shadow buffers: " + _brainCapacity + " brains, " + _nodeCapacity +
                " nodes, " + _targetCapacity + " targets, " + _edgeCapacity + " edges.");
        }

        private static int GrowCapacity(int current, int required)
        {
            int capacity = Math.Max(16, current);
            while (capacity < required)
            {
                capacity *= 2;
            }
            return capacity;
        }

        public void Dispose()
        {
            ClearPending();
            if (_context != null)
            {
                _context.Dispose();
                _context = null;
            }
        }

        internal void SetDevice(int deviceIndex)
        {
            if (_deviceIndex == deviceIndex)
            {
                return;
            }
            ClearPending();
            if (_context != null)
            {
                _context.Dispose();
                _context = null;
            }
            _deviceIndex = deviceIndex;
            _brainCapacity = 0;
            _nodeCapacity = 0;
            _targetCapacity = 0;
            _edgeCapacity = 0;
            _totalCompared = 0;
            _totalMismatches = 0;
            _largestError = 0f;
            _lastFailure = null;
        }
    }

}
