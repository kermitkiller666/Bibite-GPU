using System;
using System.Collections.Generic;
using System.Diagnostics;
using BepInEx.Logging;
using BibitesGpuFork.Core;
using ManagementScripts;
using SettingScripts;
using SimulationScripts.BibiteScripts;
using UnityEngine;

namespace BibitesGpuFork
{
    internal sealed class GpuAuthoritativeBrainScheduler : IDisposable
    {
        private sealed class BrainLayout
        {
            internal NEATBrain Brain;
            internal NEATBrain.Node[] NodesReference;
            internal NEATBrain.Synaps[] SynapsesReference;
            internal List<int> TargetsReference;
            internal List<List<int>> LinksReference;
            internal int NodeOffset;
        }

        private readonly ManualLogSource _log;
        private readonly List<NEATBrain> _currentBrains = new List<NEATBrain>();
        private readonly HashSet<NEATBrain> _registeredBrains = new HashSet<NEATBrain>();
        private readonly List<BrainLayout> _layouts = new List<BrainLayout>();
        private GpuBrainContext _context;
        private GpuBrain[] _gpuBrains;
        private GpuNode[] _gpuNodes;
        private GpuTarget[] _gpuTargets;
        private GpuEdge[] _gpuEdges;
        private int _brainCapacity;
        private int _nodeCapacity;
        private int _targetCapacity;
        private int _edgeCapacity;
        private int _tickCounter = int.MaxValue;
        private string _failure;
        private int _lastBrainCount;
        private double _lastMilliseconds;
        private long _batchCount;
        private long _fixedCallCount;
        private int _lastDiscoveredCount;
        private long _registrationAttempts;
        private int _deviceIndex;
        private bool _autoCpu;
        private int _lastSynapseCount;

        internal GpuAuthoritativeBrainScheduler(ManualLogSource log, int deviceIndex)
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
                if (Failed) return "GPU brain fast mode failed: " + _failure;
                if (_autoCpu) return "GPU brain auto mode | CPU selected for " +
                    _lastSynapseCount + " total synapses (batch too small for CUDA)";
                if (!Active) return "GPU brain fast mode preparing";
                return "GPU brain fast mode | " + _lastBrainCount + "/" + _lastDiscoveredCount +
                    " brains | " + _lastSynapseCount + " synapses | " +
                    _lastMilliseconds.ToString("0.000") + " ms/batch | " +
                    _batchCount + " batches | " + _fixedCallCount + " fixed calls | " +
                    _registrationAttempts + " registrations";
            }
        }

        internal bool FixedStep(int minimumBatchSynapses)
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

                GatherBrains();
                if (_currentBrains.Count == 0)
                {
                    _lastBrainCount = 0;
                    _lastSynapseCount = 0;
                    _autoCpu = false;
                    Active = true;
                    return true;
                }

                int synapseCount = 0;
                for (int brainIndex = 0; brainIndex < _currentBrains.Count; brainIndex++)
                {
                    synapseCount += _currentBrains[brainIndex].Synapses.Length;
                }
                _lastBrainCount = _currentBrains.Count;
                _lastSynapseCount = synapseCount;
                if (minimumBatchSynapses > 0 && synapseCount < minimumBatchSynapses)
                {
                    _autoCpu = true;
                    Active = false;
                    _failure = null;
                    return true;
                }
                _autoCpu = false;

                bool rebuild = LayoutChanged();
                if (rebuild)
                {
                    RebuildLayout();
                }

                Stopwatch timer = Stopwatch.StartNew();
                UploadDynamicState();
                _context.Step();
                _context.DownloadNodes(_gpuNodes);
                ApplyResults();
                timer.Stop();

                _lastBrainCount = _currentBrains.Count;
                _lastMilliseconds = timer.Elapsed.TotalMilliseconds;
                _batchCount++;
                Active = true;
                _failure = null;
                return true;
            }
            catch (Exception ex)
            {
                _failure = ex.GetType().Name + ": " + ex.Message;
                Active = false;
                _log.LogError("Authoritative GPU brain scheduler failed; returning to CPU brains: " + ex);
                return false;
            }
        }

        internal void Register(NEATBrain brain)
        {
            _registrationAttempts++;
            if (brain != null)
            {
                _registeredBrains.Add(brain);
            }
        }

        internal void Reset()
        {
            Active = false;
            _failure = null;
            _tickCounter = int.MaxValue;
            _autoCpu = false;
            _lastSynapseCount = 0;
            _layouts.Clear();
            _currentBrains.Clear();
            _registeredBrains.Clear();
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
            Reset();
            _deviceIndex = deviceIndex;
        }

        private void GatherBrains()
        {
            _currentBrains.Clear();
            _registeredBrains.RemoveWhere(delegate(NEATBrain candidate) { return candidate == null; });
            _lastDiscoveredCount = _registeredBrains.Count;
            foreach (NEATBrain brain in _registeredBrains)
            {
                BibiteBody body = brain != null ? brain.GetComponentInParent<BibiteBody>(true) : null;
                if (body == null || body.dead ||
                    brain.Nodes == null || brain.Synapses == null ||
                    brain.brainTargets == null || brain.brainLinks == null)
                {
                    continue;
                }
                _currentBrains.Add(brain);
            }
        }

        private bool LayoutChanged()
        {
            if (_layouts.Count != _currentBrains.Count || _context == null) return true;
            for (int i = 0; i < _layouts.Count; i++)
            {
                BrainLayout layout = _layouts[i];
                NEATBrain brain = _currentBrains[i];
                if (!ReferenceEquals(layout.Brain, brain) ||
                    !ReferenceEquals(layout.NodesReference, brain.Nodes) ||
                    !ReferenceEquals(layout.SynapsesReference, brain.Synapses) ||
                    !ReferenceEquals(layout.TargetsReference, brain.brainTargets) ||
                    !ReferenceEquals(layout.LinksReference, brain.brainLinks))
                {
                    return true;
                }
            }
            return false;
        }

        private void RebuildLayout()
        {
            _layouts.Clear();
            int nodeCount = 0;
            int targetCount = 0;
            int edgeCount = 0;
            for (int i = 0; i < _currentBrains.Count; i++)
            {
                NEATBrain brain = _currentBrains[i];
                _layouts.Add(new BrainLayout
                {
                    Brain = brain,
                    NodesReference = brain.Nodes,
                    SynapsesReference = brain.Synapses,
                    TargetsReference = brain.brainTargets,
                    LinksReference = brain.brainLinks,
                    NodeOffset = nodeCount
                });
                nodeCount += brain.Nodes.Length;
                targetCount += brain.brainTargets.Count;
                for (int target = 0; target < brain.brainLinks.Count; target++)
                {
                    edgeCount += brain.brainLinks[target].Count;
                }
            }

            EnsureContext(_currentBrains.Count, nodeCount, targetCount, edgeCount);
            _gpuBrains = new GpuBrain[_currentBrains.Count];
            _gpuNodes = new GpuNode[nodeCount];
            _gpuTargets = new GpuTarget[targetCount];
            _gpuEdges = new GpuEdge[edgeCount];

            int targetOffset = 0;
            int edgeOffset = 0;
            for (int brainIndex = 0; brainIndex < _layouts.Count; brainIndex++)
            {
                BrainLayout layout = _layouts[brainIndex];
                NEATBrain brain = layout.Brain;
                _gpuBrains[brainIndex] = new GpuBrain
                {
                    NodeOffset = layout.NodeOffset,
                    TargetOffset = targetOffset,
                    TargetCount = brain.brainTargets.Count,
                    Period = NEATBrain.brainPeriod
                };
                for (int targetIndex = 0; targetIndex < brain.brainTargets.Count; targetIndex++)
                {
                    List<int> links = brain.brainLinks[targetIndex];
                    _gpuTargets[targetOffset++] = new GpuTarget
                    {
                        NodeIndex = brain.brainTargets[targetIndex],
                        EdgeOffset = edgeOffset,
                        EdgeCount = links.Count
                    };
                    for (int linkIndex = 0; linkIndex < links.Count; linkIndex++)
                    {
                        NEATBrain.Synaps synapse = brain.Synapses[links[linkIndex]];
                        _gpuEdges[edgeOffset++] = new GpuEdge
                        {
                            SourceNode = synapse.NodeIn,
                            WeightBits = HalfConverter.ToHalfBits(synapse.Weight)
                        };
                    }
                }
            }
            _context.UploadBrains(_gpuBrains);
            _context.UploadTopology(_gpuTargets, _gpuEdges);
            _log.LogInfo("Authoritative GPU brain layout rebuilt for " + _layouts.Count + " brains, " +
                nodeCount + " nodes and " + edgeCount + " synapses.");
        }

        private void UploadDynamicState()
        {
            for (int brainIndex = 0; brainIndex < _layouts.Count; brainIndex++)
            {
                BrainLayout layout = _layouts[brainIndex];
                NEATBrain brain = layout.Brain;
                brain.UpdateSenses();
                _gpuBrains[brainIndex].Period = NEATBrain.brainPeriod;
                for (int nodeIndex = 0; nodeIndex < brain.Nodes.Length; nodeIndex++)
                {
                    NEATBrain.Node source = brain.Nodes[nodeIndex];
                    _gpuNodes[layout.NodeOffset + nodeIndex] = new GpuNode
                    {
                        Value = source.Value,
                        LastOutput = source.LastOutput,
                        LastInput = source.LastInput,
                        Bias = source.baseActivation,
                        Type = (int)source.Type
                    };
                }
            }
            _context.UploadBrains(_gpuBrains);
            _context.UploadNodes(_gpuNodes);
        }

        private void ApplyResults()
        {
            for (int brainIndex = 0; brainIndex < _layouts.Count; brainIndex++)
            {
                BrainLayout layout = _layouts[brainIndex];
                NEATBrain brain = layout.Brain;
                for (int targetIndex = 0; targetIndex < brain.brainTargets.Count; targetIndex++)
                {
                    int nodeIndex = brain.brainTargets[targetIndex];
                    GpuNode source = _gpuNodes[layout.NodeOffset + nodeIndex];
                    NEATBrain.Node destination = brain.Nodes[nodeIndex];
                    destination.Value = source.Value;
                    destination.LastOutput = source.LastOutput;
                    destination.LastInput = source.LastInput;
                    brain.Nodes[nodeIndex] = destination;
                }
            }
        }

        private void EnsureContext(int brains, int nodes, int targets, int edges)
        {
            if (_context != null && brains <= _brainCapacity && nodes <= _nodeCapacity &&
                targets <= _targetCapacity && edges <= _edgeCapacity)
            {
                return;
            }
            if (_context != null) _context.Dispose();
            _brainCapacity = Grow(_brainCapacity, brains);
            _nodeCapacity = Grow(_nodeCapacity, nodes);
            _targetCapacity = Grow(_targetCapacity, targets);
            _edgeCapacity = Grow(_edgeCapacity, edges);
            _context = new GpuBrainContext(_deviceIndex, _brainCapacity, _nodeCapacity, _targetCapacity, _edgeCapacity);
        }

        private static int Grow(int current, int required)
        {
            int value = Math.Max(16, current);
            while (value < required) value *= 2;
            return value;
        }

        public void Dispose()
        {
            Reset();
        }
    }
}
