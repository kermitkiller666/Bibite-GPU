using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Globalization;
using System.Reflection;
using System.Threading;
using BepInEx.Logging;
using Newtonsoft.Json.Linq;
using BibitesGpuFork.Core;
using ManagementScripts;
using OneUseScripts;
using SettingScripts;
using SimulationScripts;
using SimulationScripts.BibiteScripts;
using UIScripts;
using UIScripts.InfoHandles;
using UIScripts.UIReferences;
using UnityEngine;
using UnityEngine.Rendering;
using Utility;

namespace BibitesGpuFork
{
    internal static class GpuInteropProcessSafety
    {
        internal static readonly GraphicsInteropCircuitBreaker CircuitBreaker =
            new GraphicsInteropCircuitBreaker();
        internal static volatile bool RenderingShuttingDown;
    }

    internal static class GpuInteropTrace
    {
        internal static ManualLogSource Logger;
        internal static readonly bool Enabled = string.Equals(
            Environment.GetEnvironmentVariable("BIBITES_GPU_TRACE_D3D_INTEROP"),
            "1", StringComparison.Ordinal);

        internal static void Write(string phase)
        {
            if (!Enabled) return;
            ManualLogSource logger = Logger;
            if (logger != null) logger.LogInfo("D3D_TRACE thread=" +
                Thread.CurrentThread.ManagedThreadId + " " + phase);
        }
    }

    internal sealed class GpuWorldSnapshot
    {
        internal long Sequence;
        internal long SummarySequence;
        internal NativeWorldBibite[] Bibites;
        internal int BibiteCount;
        internal NativeWorldBibite[] VisibleBibites;
        internal int VisibleBibiteCount;
        internal DetailedSpriteViewport? VisibleViewport;
        internal NativeWorldPellet[] Pellets;
        internal int PelletCount;
        internal NativeWorldStats Stats;
        internal NativeWorldStepMetrics Metrics;
        internal double AchievedMultiplier;
        internal float PelletEnergy;
        internal double SnapshotMilliseconds;
    }

    internal sealed class GpuWorldRunner : IDisposable
    {
        // Full GPU populations stay resident. CPU presentation only needs a
        // representative set for charts, detail sprites, and interaction.
        // Keeping this array independent of the 500k simulation cap also
        // prevents Mono's P/Invoke marshaller from walking hundreds of
        // thousands of structs during every statistics refresh.
        private const int MaximumPresentationBibites = 8192;

        private sealed class VisibleViewport
        {
            internal float MinX;
            internal float MinY;
            internal float MaxX;
            internal float MaxY;
        }

        private struct SpawnRequest
        {
            internal NativeWorldTemplate Template;
        }

        private sealed class FoodSettingsRequest
        {
            internal int Target;
            internal float GrowthFactor;
            internal float PelletEnergy;
        }

        private sealed class CombatSettingsRequest
        {
            internal float CollisionDamageConstant;
            internal float CollisionDamageThreshold;
            internal float BitingDamageFactor;
            internal float BitingPressure;
        }

        private enum BibiteCommandKind
        {
            SetTag,
            Kill,
            ForceReproduction
        }

        private struct BibiteCommandRequest
        {
            internal BibiteCommandKind Kind;
            internal int Slot;
            internal ulong TagId;
            internal string TagName;
            internal int ExpectedInstanceId;
        }

        internal sealed class RenderBufferRequest
        {
            internal NativeD3D11RenderConfig Config;
            internal int BufferIndex;
            internal bool UpdateOnly;
            internal bool ReleaseOnly;
            internal int NativeEventId;
            internal int EventIssued;
            internal volatile string Error;
            internal int BibiteCount;
            internal int PelletCount;
            // 0 queued, 1 registering/mapping/writing, 2 complete. The main
            // thread only polls; it never waits for CUDA or displays this
            // back buffer before the completion publication barrier.
            internal int State;
            internal bool IsCompleted { get { return Volatile.Read(ref State) == 2; } }
        }

        internal sealed class CheckpointRequest
        {
            internal string Path;
            internal ManualResetEvent Completed;
            internal string Error;
        }

        private readonly NativeWorldConfig _config;
        private readonly NativeWorldFoodZone[] _foodZones;
        private readonly int _presentationBibiteCapacity;
        private readonly int _visibleBibiteCapacity;
        private readonly object _snapshotLock = new object();
        private readonly object _commandLock = new object();
        private readonly Queue<SpawnRequest> _spawnRequests = new Queue<SpawnRequest>();
        private readonly Queue<RenderBufferRequest> _renderBufferRequests =
            new Queue<RenderBufferRequest>();
        private readonly Queue<CheckpointRequest> _checkpointRequests =
            new Queue<CheckpointRequest>();
        private readonly Queue<BibiteCommandRequest> _bibiteCommandRequests =
            new Queue<BibiteCommandRequest>();
        private readonly object _selectedDetailLock = new object();
        private readonly Thread _thread;
        private readonly string _checkpointPath;
        private readonly int _initialFoodTarget;
        private volatile bool _stopRequested;
        private volatile bool _stopped;
        private volatile bool _paused;
        private volatile int _targetMultiplier = 1;
        private volatile bool _ready;
        private volatile string _error;
        private volatile string _deviceName;
        private volatile bool _directRenderRegistered;
        private volatile bool _hasRegisteredRenderResources;
        private volatile bool _directRenderFaulted;
        private volatile bool _retainRenderResources;
        private volatile RenderBufferRequest _pendingRenderEvent;
        private volatile int _renderFps = 30;
        private volatile VisibleViewport _visibleViewport;
        private FoodSettingsRequest _pendingFoodSettings;
        private NativeWorldFoodZone[] _pendingFoodZones;
        private object _pendingLinearDrag;
        private CombatSettingsRequest _pendingCombatSettings;
        private object _pendingDigestionSettings;
        private volatile int _currentFoodTarget;
        private volatile float _currentFoodGrowthFactor = 1f;
        private volatile float _currentPelletEnergy;
        private int _adaptiveBatchSteps = 64;
        private GpuWorldSnapshot _published;
        private NativeWorldBibite[] _writeBibites;
        private NativeWorldBibite[] _writeVisibleBibites;
        private NativeWorldPellet[] _writePellets;
        private NativeWorldBrainNodeState[] _selectedWriteNodes =
            new NativeWorldBrainNodeState[NativeTemplateProjection.MaximumNodes];
        private NativeWorldBrainSynapseState[] _selectedWriteSynapses =
            new NativeWorldBrainSynapseState[1024];
        private NativeWorldSelectedBibite _selectedPublished;
        private volatile int _selectedSlot = -1;
        private volatile bool _selectedBrainRequested;
        private long _selectedSequence;
        private volatile string _lastBibiteCommandStatus;
        private long _sequence;
        private int _successfulPlacements;
        private int _failedPlacements;
        private volatile string _lastPlacementError;
        private volatile int _latestLivingBibites;
        private ulong _completedSimulationSteps;
        private long _summarySequence;
        private long _lastSummaryTimestamp;
        private NativeWorldStats _lastSummaryStats;
        private double _maximumStepMilliseconds;
        private long _lastRenderCompletionTimestamp;
        private double _maximumRenderGapMilliseconds;
        private int _completedRenderUpdates;
        private int _checkpointRequestsInFlight;
        private volatile string _workerPhase = "Starting GPU world";
        private long _workerPhaseStarted = Stopwatch.GetTimestamp();

        internal string WorkerState
        {
            get
            {
                double seconds = (Stopwatch.GetTimestamp() -
                    Interlocked.Read(ref _workerPhaseStarted)) / (double)Stopwatch.Frequency;
                return _workerPhase + " (" + seconds.ToString("0.0", CultureInfo.InvariantCulture) + " s)";
            }
        }

        private void SetWorkerPhase(string phase)
        {
            Interlocked.Exchange(ref _workerPhaseStarted, Stopwatch.GetTimestamp());
            _workerPhase = phase;
        }

        internal GpuWorldRunner(
            NativeWorldConfig config,
            int initialFoodTarget,
            NativeWorldFoodZone[] foodZones,
            int detailedBibiteLimit,
            string checkpointPath = null)
        {
            _config = config;
            _foodZones = foodZones ?? new NativeWorldFoodZone[0];
            _checkpointPath = checkpointPath;
            _initialFoodTarget = initialFoodTarget;
            _presentationBibiteCapacity = Math.Min(
                config.MaxBibites,
                MaximumPresentationBibites);
            _visibleBibiteCapacity = Math.Min(
                config.MaxBibites,
                Math.Max(0, detailedBibiteLimit));
            _writeBibites = new NativeWorldBibite[_presentationBibiteCapacity];
            _writeVisibleBibites = new NativeWorldBibite[_visibleBibiteCapacity];
            _writePellets = new NativeWorldPellet[config.PelletCount];
            _thread = new Thread(Run)
            {
                IsBackground = true,
                Name = "Bibites GPU world"
            };
            _thread.Start();
        }

        internal bool IsReady
        {
            get { return _ready; }
        }

        internal bool IsStopped
        {
            get { return _stopped; }
        }

        internal string ShutdownState
        {
            get
            {
                RenderBufferRequest pending = _pendingRenderEvent;
                return "stop=" + _stopRequested + ", stopped=" + _stopped +
                    ", renderEvent=" + (pending == null ? "none" :
                        pending.NativeEventId + "/issued=" + pending.EventIssued +
                        "/completed=" + pending.IsCompleted +
                        "/release=" + pending.ReleaseOnly);
            }
        }

        internal bool RetainRenderResources
        {
            get { return _retainRenderResources; }
        }

        internal string Error
        {
            get { return _error; }
        }

        internal string DeviceName
        {
            get { return _deviceName ?? "initializing CUDA"; }
        }

        internal int SuccessfulPlacements
        {
            get { return _successfulPlacements; }
        }

        internal int FailedPlacements
        {
            get { return _failedPlacements; }
        }

        internal long LatestSequence
        {
            get { return Interlocked.Read(ref _sequence); }
        }

        internal string LastPlacementError
        {
            get { return _lastPlacementError; }
        }

        internal double MaximumRenderGapMilliseconds
        {
            get { return _maximumRenderGapMilliseconds; }
        }

        internal double MaximumStepMilliseconds
        {
            get { return _maximumStepMilliseconds; }
        }

        internal int CompletedRenderUpdates
        {
            get { return _completedRenderUpdates; }
        }

        internal int CurrentFoodTarget
        {
            get { return _currentFoodTarget; }
        }

        internal float CurrentPelletEnergy
        {
            get { return _currentPelletEnergy; }
        }

        internal float CurrentFoodGrowthFactor
        {
            get { return _currentFoodGrowthFactor; }
        }

        internal void QueueFoodSettings(int target, float growthFactor, float pelletEnergy)
        {
            Interlocked.Exchange(ref _pendingFoodSettings, new FoodSettingsRequest
            {
                Target = target,
                GrowthFactor = growthFactor,
                PelletEnergy = pelletEnergy
            });
        }

        internal void QueueFoodZones(NativeWorldFoodZone[] zones)
        {
            Interlocked.Exchange(ref _pendingFoodZones,
                zones ?? new NativeWorldFoodZone[0]);
        }

        internal void QueueLinearDrag(float drag)
        {
            Interlocked.Exchange(ref _pendingLinearDrag, (object)drag);
        }

        internal void QueueCombatSettings(float collisionDamageConstant,
            float collisionDamageThreshold, float bitingDamageFactor,
            float bitingPressure)
        {
            Interlocked.Exchange(ref _pendingCombatSettings,
                new CombatSettingsRequest
                {
                    CollisionDamageConstant = collisionDamageConstant,
                    CollisionDamageThreshold = collisionDamageThreshold,
                    BitingDamageFactor = bitingDamageFactor,
                    BitingPressure = bitingPressure
                });
        }

        internal void QueueDigestionSettings(NativeWorldDigestionSettings settings)
        {
            Interlocked.Exchange(ref _pendingDigestionSettings, (object)settings);
        }

        internal void SetControl(int targetMultiplier, bool paused, int renderFps)
        {
            _targetMultiplier = Math.Max(
                1,
                Math.Min(TimeWarpSpeeds.Maximum, targetMultiplier));
            _paused = paused;
            _renderFps = Math.Max(10, Math.Min(60, renderFps));
        }

        // Called on Unity's main thread. Publishing one immutable object keeps
        // all four camera bounds from the same frame on the CUDA worker.
        internal void SetVisibleViewport(Camera camera)
        {
            if (_visibleBibiteCapacity == 0 || camera == null ||
                !camera.orthographic || !DetailedSpriteVisibility.ShouldRender(
                    camera.orthographicSize, Screen.height))
            {
                _visibleViewport = null;
                return;
            }
            float halfHeight = camera.orthographicSize;
            float halfWidth = halfHeight * camera.aspect;
            Vector3 position = camera.transform.position;
            _visibleViewport = new VisibleViewport
            {
                MinX = position.x - halfWidth - DetailedSpriteVisibility.EdgePadding,
                MinY = position.y - halfHeight - DetailedSpriteVisibility.EdgePadding,
                MaxX = position.x + halfWidth + DetailedSpriteVisibility.EdgePadding,
                MaxY = position.y + halfHeight + DetailedSpriteVisibility.EdgePadding
            };
        }

        internal bool QueueSpawn(NativeWorldTemplate template)
        {
            if (template == null || !_ready || _stopRequested || !string.IsNullOrEmpty(_error))
            {
                return false;
            }
            lock (_commandLock)
            {
                if (!_ready || _stopRequested) return false;
                if (_spawnRequests.Count >= 128)
                {
                    _lastPlacementError = "placement queue is full";
                    return false;
                }
                _spawnRequests.Enqueue(new SpawnRequest
                {
                    Template = template
                });
                return true;
            }
        }

        internal string LastBibiteCommandStatus
        {
            get { return _lastBibiteCommandStatus; }
        }

        internal void SetSelectedSlot(int slot)
        {
            int normalized = slot >= 0 && slot < _config.MaxBibites ? slot : -1;
            if (Interlocked.Exchange(ref _selectedSlot, normalized) == normalized)
            {
                return;
            }
            lock (_selectedDetailLock)
            {
                _selectedPublished = null;
            }
        }

        internal void SetSelectedInspectorNeedsBrain(bool requested)
        {
            _selectedBrainRequested = requested;
        }

        internal bool WithSelectedDetail(
            int slot,
            long afterSequence,
            Action<NativeWorldSelectedBibite> consumer)
        {
            if (consumer == null)
            {
                return false;
            }
            lock (_selectedDetailLock)
            {
                if (_selectedPublished == null ||
                    _selectedPublished.Sequence <= afterSequence ||
                    _selectedPublished.Detail.Slot != slot)
                {
                    return false;
                }
                consumer(_selectedPublished);
                return true;
            }
        }

        internal bool QueueSetTag(int slot, ulong tagId, string tagName,
            NativeWorldBibiteDetail expectedIdentity)
        {
            return QueueBibiteCommand(new BibiteCommandRequest
            {
                Kind = BibiteCommandKind.SetTag,
                Slot = slot,
                TagId = tagId,
                TagName = tagName,
                ExpectedInstanceId = expectedIdentity.Reserved
            });
        }

        internal bool QueueKill(int slot, NativeWorldBibiteDetail expectedIdentity)
        {
            return QueueBibiteCommand(new BibiteCommandRequest
            {
                Kind = BibiteCommandKind.Kill,
                Slot = slot,
                ExpectedInstanceId = expectedIdentity.Reserved
            });
        }

        internal bool QueueForceReproduction(int slot, NativeWorldBibiteDetail expectedIdentity)
        {
            return QueueBibiteCommand(new BibiteCommandRequest
            {
                Kind = BibiteCommandKind.ForceReproduction,
                Slot = slot,
                ExpectedInstanceId = expectedIdentity.Reserved
            });
        }

        private bool QueueBibiteCommand(BibiteCommandRequest request)
        {
            if (!_ready || _stopRequested || !string.IsNullOrEmpty(_error) ||
                request.Slot < 0 || request.ExpectedInstanceId == 0)
            {
                return false;
            }
            lock (_commandLock)
            {
                if (!_ready || _stopRequested) return false;
                if (_bibiteCommandRequests.Count >= 64)
                {
                    _lastBibiteCommandStatus = "Bibite command queue is full";
                    return false;
                }
                _bibiteCommandRequests.Enqueue(request);
                _lastBibiteCommandStatus = "Queued " + CommandName(request.Kind) + " for GPU Bibite #" +
                    request.Slot + ".";
                return true;
            }
        }

        internal RenderBufferRequest BeginRegisterRenderBuffers(
            int bufferIndex,
            NativeD3D11RenderConfig config,
            out string error)
        {
            RenderBufferRequest request = new RenderBufferRequest
            {
                BufferIndex = bufferIndex,
                Config = config,
                UpdateOnly = false
            };
            return QueueRenderRequest(request, out error);
        }

        internal RenderBufferRequest BeginUpdateRenderBuffers(int bufferIndex, out string error)
        {
            error = null;
            if (Volatile.Read(ref _checkpointRequestsInFlight) > 0)
            {
                return null;
            }
            RenderBufferRequest request = new RenderBufferRequest
            {
                BufferIndex = bufferIndex,
                UpdateOnly = true
            };
            return QueueRenderRequest(request, out error);
        }

        internal void DisableDirectRendering(string reason)
        {
            _directRenderRegistered = false;
            _directRenderFaulted = true;
            GpuInteropProcessSafety.CircuitBreaker.Trip(reason);
        }

        // Main thread only, including for retired runners. Queueing the native
        // callback never waits for CUDA; all actual D3D operations run on
        // Unity's rendering thread in its command order.
        internal void PumpRenderThreadEvent()
        {
            // Application.quitting is too late to schedule graphics work. A
            // normal quit drains it earlier through Application.wantsToQuit.
            if (GpuInteropProcessSafety.RenderingShuttingDown) return;
            RenderBufferRequest request = _pendingRenderEvent;
            if (request == null || request.IsCompleted ||
                Interlocked.CompareExchange(ref request.EventIssued, 1, 0) != 0) return;
            try
            {
                if (_completedRenderUpdates < 8 || request.ReleaseOnly)
                    GpuInteropTrace.Write("main issue native render event " + request.NativeEventId);
                GL.IssuePluginEvent(NativeWorldContext.UnityRenderEventFunction, request.NativeEventId);
            }
            catch (Exception ex)
            {
                request.Error = "could not schedule the Unity render callback: " + ex.Message;
                GpuInteropProcessSafety.CircuitBreaker.Trip(request.Error);
                try
                {
                    NativeWorldContext.CancelUnityRenderRequest(request.NativeEventId);
                }
                catch (Exception cancellationError)
                {
                    // Stop may have already cancelled and retired this ID on
                    // the worker. Never let that benign race escape Update.
                    if (!request.IsCompleted && !_stopRequested)
                    {
                        _retainRenderResources = true;
                        _error = cancellationError.ToString();
                        _stopRequested = true;
                    }
                }
            }
        }

        private RenderBufferRequest QueueRenderRequest(RenderBufferRequest request, out string error)
        {
            error = null;
            lock (_commandLock)
            {
                if (!_ready || _stopRequested || _directRenderFaulted ||
                    !string.IsNullOrEmpty(_error))
                {
                    error = _error ?? (_directRenderFaulted
                        ? "CUDA graphics interop is disabled after a driver error"
                        : "the GPU world is not ready");
                    return null;
                }
                if (_renderBufferRequests.Count >= 2) return null;
                _renderBufferRequests.Enqueue(request);
            }
            return request;
        }

        internal CheckpointRequest BeginSaveCheckpoint(string path, out string error)
        {
            error = null;
            if (string.IsNullOrEmpty(path) || !_ready || _stopRequested ||
                !string.IsNullOrEmpty(_error))
            {
                error = _error ?? "the GPU world is not ready";
                return null;
            }
            CheckpointRequest request = new CheckpointRequest
            {
                Path = path,
                Completed = new ManualResetEvent(false)
            };
            Interlocked.Increment(ref _checkpointRequestsInFlight);
            lock (_commandLock)
            {
                if (!_ready || _stopRequested)
                {
                    Interlocked.Decrement(ref _checkpointRequestsInFlight);
                    request.Completed.Close();
                    error = _error ?? "the GPU world is stopping";
                    return null;
                }
                _checkpointRequests.Enqueue(request);
            }
            return request;
        }

        internal bool WithLatest(long afterSequence, Action<GpuWorldSnapshot> consumer)
        {
            lock (_snapshotLock)
            {
                if (_published == null || _published.Sequence <= afterSequence)
                {
                    return false;
                }
                consumer(_published);
                return true;
            }
        }

        private void Run()
        {
            try
            {
                NativeGpu.SelectDevice(_config.DeviceIndex);
                using (NativeWorldContext world = string.IsNullOrEmpty(_checkpointPath)
                    ? new NativeWorldContext(_config)
                    : new NativeWorldContext(_checkpointPath, _config.DeviceIndex))
                {
                    try
                    {
                    world.SetFoodZones(_foodZones,
                        string.IsNullOrEmpty(_checkpointPath));
                    if (string.IsNullOrEmpty(_checkpointPath))
                    {
                        world.SetFoodSettings(
                            _initialFoodTarget, 1f, _config.PelletEnergy);
                    }
                    float initialGrowth;
                    int initialTarget;
                    float initialEnergy;
                    world.GetFoodSettings(
                        out initialTarget, out initialGrowth, out initialEnergy);
                    _currentFoodTarget = initialTarget;
                    _currentFoodGrowthFactor = initialGrowth;
                    _currentPelletEnergy = initialEnergy;
                    _deviceName = world.GetDeviceName();
                    NativeWorldStats initialStats = PublishSnapshot(
                        world,
                        new NativeWorldStepMetrics(),
                        0.0,
                        0.0);
                    _ready = true;

                    Stopwatch clock = Stopwatch.StartNew();
                    double nextDeadline = clock.Elapsed.TotalSeconds;
                    double previousSnapshotWall = clock.Elapsed.TotalSeconds;
                    double previousSnapshotSim = initialStats.SimulatedSeconds;
                    double nextSnapshot = previousSnapshotWall + 1.0 / 30.0;
                    double nextSelectedDetail = previousSnapshotWall;
                    int previousTarget = _targetMultiplier;

                    while (!_stopRequested)
                    {
                        SetWorkerPhase("Applying queued GPU commands");
                        DrainRenderBufferRequests(world);
                        DrainSpawnRequests(world);
                        DrainBibiteCommandRequests(world);
                        ApplyFoodZones(world);
                        ApplyFoodSettings(world);
                        ApplyLinearDrag(world);
                        ApplyCombatSettings(world);
                        ApplyDigestionSettings(world);
                        DrainCheckpointRequests(world);
                        if (_paused)
                        {
                            double pausedNow = clock.Elapsed.TotalSeconds;
                            // Placement, remove/tag actions and food controls
                            // still operate while paused. Publish their results
                            // instead of leaving charts and fallback rendering
                            // frozen at the last running snapshot.
                            if (pausedNow >= nextSnapshot)
                            {
                                NativeWorldStats pausedStats = PublishSnapshot(
                                    world, new NativeWorldStepMetrics(),
                                    previousSnapshotSim,
                                    Math.Max(0.000001, pausedNow - previousSnapshotWall));
                                previousSnapshotSim = pausedStats.SimulatedSeconds;
                                previousSnapshotWall = pausedNow;
                                nextSnapshot = pausedNow + PresentationCadence.SnapshotIntervalSeconds(
                                    _latestLivingBibites, _presentationBibiteCapacity, _renderFps, true);
                            }
                            if (_selectedSlot >= 0 && pausedNow >= nextSelectedDetail)
                            {
                                PublishSelectedDetail(world);
                                nextSelectedDetail = pausedNow + 0.1;
                            }
                            SetWorkerPhase("Paused");
                            Thread.Sleep(8);
                            nextDeadline = pausedNow;
                            continue;
                        }

                        int target = _targetMultiplier;
                        if (target != previousTarget)
                        {
                            previousTarget = target;
                            nextDeadline = clock.Elapsed.TotalSeconds;
                        }

                        int targetStepsPerSecond = Math.Max(
                            1,
                            (int)Math.Round(target / _config.FixedDeltaTime));
                        // High-warp runs use longer launches to amortise driver
                        // synchronisation and keep more than 99% of the measured
                        // simulation interval inside CUDA. Lower speeds retain the
                        // smaller batch for responsive pause and speed controls.
                        int profileMaximumBatchSteps = target >= 250 ? 512 : 256;
                        if (_directRenderRegistered)
                        {
                            // A population can multiply rapidly inside one
                            // persistent launch. Keep large visual worlds in
                            // shorter, pre-emptible batches so graphics requests
                            // cannot sit behind a once-small 512-step launch.
                            int living = _latestLivingBibites;
                            if (living >= 65536)
                            {
                                profileMaximumBatchSteps = Math.Min(
                                    profileMaximumBatchSteps,
                                    32);
                            }
                            else if (living >= 16384)
                            {
                                profileMaximumBatchSteps = Math.Min(
                                    profileMaximumBatchSteps,
                                    64);
                            }
                            else if (living >= 4096)
                            {
                                profileMaximumBatchSteps = Math.Min(
                                    profileMaximumBatchSteps,
                                    128);
                            }
                        }
                        int maximumBatchSteps = Math.Max(
                            8,
                            Math.Min(profileMaximumBatchSteps, _adaptiveBatchSteps));
                        int steps = Math.Max(
                            1,
                            Math.Min(maximumBatchSteps, targetStepsPerSecond / 30));
                        SetWorkerPhase("Running GPU simulation batch");
                        NativeWorldStepMetrics metrics = world.Step(steps);
                        _completedSimulationSteps += (ulong)steps;
                        SetWorkerPhase("Publishing GPU batch results");
                        _maximumStepMilliseconds = Math.Max(
                            _maximumStepMilliseconds,
                            metrics.WallMilliseconds);
                        AdaptBatchSize(steps, profileMaximumBatchSteps, metrics.WallMilliseconds);
                        double now = clock.Elapsed.TotalSeconds;
                        nextDeadline += steps * _config.FixedDeltaTime / target;

                        if (_selectedSlot >= 0 && now >= nextSelectedDetail)
                        {
                            PublishSelectedDetail(world);
                            nextSelectedDetail = now + 0.1;
                        }

                        double snapshotInterval = PresentationCadence.SnapshotIntervalSeconds(
                            _latestLivingBibites, _presentationBibiteCapacity, _renderFps, false);
                        if (now >= nextSnapshot)
                        {
                            double wallDelta = Math.Max(0.000001, now - previousSnapshotWall);
                            NativeWorldStats stats = PublishSnapshot(
                                world,
                                metrics,
                                previousSnapshotSim,
                                wallDelta);
                            previousSnapshotWall = now;
                            previousSnapshotSim = stats.SimulatedSeconds;
                            // Direct rendering refreshes the D3D buffer separately
                            // at the chosen graphics cadence. CPU snapshots are then
                            // needed only for charts, selection, and detail sprites.
                            // Budget the actual sample, not the reserved world
                            // capacity; small worlds keep smooth detailed sprites.
                            nextSnapshot += snapshotInterval;
                            if (nextSnapshot < now - snapshotInterval)
                            {
                                nextSnapshot = now + snapshotInterval;
                            }
                        }

                        now = clock.Elapsed.TotalSeconds;
                        double remaining = nextDeadline - now;
                        if (remaining > 0.002)
                        {
                            Thread.Sleep(Math.Max(0, (int)(remaining * 1000.0) - 1));
                        }
                        while (!_stopRequested && !_paused &&
                            clock.Elapsed.TotalSeconds < nextDeadline)
                        {
                            Thread.SpinWait(64);
                        }
                        if (clock.Elapsed.TotalSeconds - nextDeadline > 0.25)
                        {
                            nextDeadline = clock.Elapsed.TotalSeconds;
                        }
                    }
                    }
                    finally
                    {
                        ReleaseGraphicsOnRenderThread(world);
                    }
                }
            }
            catch (Exception ex)
            {
                _error = ex.ToString();
                // A failed native teardown must not make Unity free resources
                // that the driver might still have registered or mapped.
                if (_hasRegisteredRenderResources)
                {
                    _retainRenderResources = true;
                    GpuInteropProcessSafety.CircuitBreaker.Trip(_error);
                }
            }
            finally
            {
                lock (_commandLock)
                {
                    _ready = false;
                    _stopRequested = true;
                }
                try
                {
                    CompletePendingRenderRequests(_error ?? "the GPU world stopped");
                    CompletePendingCheckpointRequests(_error ?? "the GPU world stopped");
                }
                finally
                {
                    // Native disposal has returned. Failed interop resources
                    // remain quarantined even after this worker has stopped.
                    _stopped = true;
                }
            }
        }

        private void CompletePendingCheckpointRequests(string error)
        {
            while (true)
            {
                CheckpointRequest request;
                lock (_commandLock)
                {
                    if (_checkpointRequests.Count == 0)
                    {
                        return;
                    }
                    request = _checkpointRequests.Dequeue();
                }
                request.Error = error;
                Interlocked.Decrement(ref _checkpointRequestsInFlight);
                request.Completed.Set();
            }
        }

        private void ApplyFoodZones(NativeWorldContext world)
        {
            NativeWorldFoodZone[] zones = Interlocked.Exchange(
                ref _pendingFoodZones, null);
            if (zones != null)
                world.SetFoodZones(zones, false);
        }

        private void ApplyFoodSettings(NativeWorldContext world)
        {
            FoodSettingsRequest request = Interlocked.Exchange(
                ref _pendingFoodSettings, null);
            if (request == null)
            {
                return;
            }
            world.SetFoodSettings(
                request.Target, request.GrowthFactor, request.PelletEnergy);
            _currentFoodTarget = request.Target;
            _currentFoodGrowthFactor = request.GrowthFactor;
            _currentPelletEnergy = request.PelletEnergy;
        }

        private void ApplyLinearDrag(NativeWorldContext world)
        {
            object pending = Interlocked.Exchange(ref _pendingLinearDrag, null);
            if (pending != null) world.SetLinearDrag((float)pending);
        }

        private void ApplyCombatSettings(NativeWorldContext world)
        {
            CombatSettingsRequest pending = Interlocked.Exchange(
                ref _pendingCombatSettings, null);
            if (pending != null)
                world.SetCombatSettings(pending.CollisionDamageConstant,
                    pending.CollisionDamageThreshold,
                    pending.BitingDamageFactor, pending.BitingPressure);
        }

        private void ApplyDigestionSettings(NativeWorldContext world)
        {
            object pending = Interlocked.Exchange(ref _pendingDigestionSettings, null);
            if (pending != null)
                world.SetDigestionSettings((NativeWorldDigestionSettings)pending);
        }

        private void CompletePendingRenderRequests(string error)
        {
            while (true)
            {
                RenderBufferRequest request;
                lock (_commandLock)
                {
                    if (_renderBufferRequests.Count == 0)
                    {
                        return;
                    }
                    request = _renderBufferRequests.Dequeue();
                }
                request.Error = error;
                Interlocked.Exchange(ref request.State, 2);
            }
        }

        private void DrainSpawnRequests(NativeWorldContext world)
        {
            while (!_stopRequested)
            {
                SpawnRequest request;
                lock (_commandLock)
                {
                    if (_spawnRequests.Count == 0)
                    {
                        return;
                    }
                    request = _spawnRequests.Dequeue();
                }
                try
                {
                    int slot = world.SpawnTemplateBibite(request.Template);
                    if (slot >= 0)
                    {
                        Interlocked.Increment(ref _successfulPlacements);
                        _lastPlacementError = null;
                    }
                    else
                    {
                        Interlocked.Increment(ref _failedPlacements);
                        _lastPlacementError = "GPU population cap reached";
                    }
                }
                catch (Exception ex)
                {
                    Interlocked.Increment(ref _failedPlacements);
                    _lastPlacementError = ex.Message;
                }
            }
        }

        private void DrainBibiteCommandRequests(NativeWorldContext world)
        {
            while (!_stopRequested)
            {
                BibiteCommandRequest request;
                lock (_commandLock)
                {
                    if (_bibiteCommandRequests.Count == 0)
                    {
                        return;
                    }
                    request = _bibiteCommandRequests.Dequeue();
                }
                try
                {
                    int ignoredNodes;
                    int ignoredSynapses;
                    NativeWorldBibiteDetail current = world.DownloadBibiteDetail(
                        request.Slot, null, out ignoredNodes, null, out ignoredSynapses);
                    // Commands and stepping run on this worker, so this identity
                    // check and the command cannot be interleaved by slot reuse.
                    if (current.Alive != 1 || current.Reserved != request.ExpectedInstanceId)
                    {
                        _lastBibiteCommandStatus = "GPU Bibite #" + request.Slot +
                            " has died or been replaced; " +
                            CommandName(request.Kind) + " was cancelled.";
                        continue;
                    }
                    switch (request.Kind)
                    {
                        case BibiteCommandKind.SetTag:
                            world.SetBibiteTag(request.Slot, request.TagId);
                            _lastBibiteCommandStatus = "Updated GPU Bibite #" + request.Slot +
                                " tag to " + (string.IsNullOrEmpty(request.TagName)
                                    ? "Untagged"
                                    : request.TagName) + ".";
                            break;
                        case BibiteCommandKind.Kill:
                            world.KillBibite(request.Slot);
                            _lastBibiteCommandStatus = "Removed GPU Bibite #" + request.Slot + ".";
                            break;
                        case BibiteCommandKind.ForceReproduction:
                            int child = world.ForceReproduction(request.Slot);
                            _lastBibiteCommandStatus = "GPU Bibite #" + request.Slot +
                                " laid offspring #" + child + ".";
                            break;
                    }
                    if (_selectedSlot == request.Slot)
                    {
                        PublishSelectedDetail(world);
                    }
                }
                catch (Exception ex)
                {
                    _lastBibiteCommandStatus = CommandName(request.Kind) + " failed for GPU Bibite #" +
                        request.Slot + ": " + ex.Message;
                }
            }
        }

        private void PublishSelectedDetail(NativeWorldContext world)
        {
            int requestedSlot = _selectedSlot;
            if (requestedSlot < 0)
            {
                return;
            }
            bool includeBrain = _selectedBrainRequested;
            int nodeCount;
            int synapseCount;
            NativeWorldBibiteDetail detail = world.DownloadBibiteDetail(
                requestedSlot,
                includeBrain ? _selectedWriteNodes : null,
                out nodeCount,
                includeBrain ? _selectedWriteSynapses : null,
                out synapseCount);
            if (requestedSlot != _selectedSlot)
            {
                return;
            }
            NativeWorldBrainNodeState[] nodes = includeBrain
                ? new NativeWorldBrainNodeState[nodeCount]
                : new NativeWorldBrainNodeState[0];
            NativeWorldBrainSynapseState[] synapses = includeBrain
                ? new NativeWorldBrainSynapseState[synapseCount]
                : new NativeWorldBrainSynapseState[0];
            if (nodeCount > 0)
            {
                Array.Copy(_selectedWriteNodes, nodes, nodeCount);
            }
            if (synapseCount > 0)
            {
                Array.Copy(_selectedWriteSynapses, synapses, synapseCount);
            }
            lock (_selectedDetailLock)
            {
                _selectedPublished = new NativeWorldSelectedBibite
                {
                    Sequence = ++_selectedSequence,
                    Detail = detail,
                    Nodes = nodes,
                    Synapses = synapses
                };
            }
        }

        private static string CommandName(BibiteCommandKind kind)
        {
            switch (kind)
            {
                case BibiteCommandKind.SetTag: return "tag update";
                case BibiteCommandKind.Kill: return "remove";
                case BibiteCommandKind.ForceReproduction: return "lay egg";
                default: return "command";
            }
        }

        private void DrainCheckpointRequests(NativeWorldContext world)
        {
            while (!_stopRequested)
            {
                CheckpointRequest request;
                lock (_commandLock)
                {
                    if (_checkpointRequests.Count == 0)
                    {
                        return;
                    }
                    request = _checkpointRequests.Dequeue();
                }
                IntPtr capture = IntPtr.Zero;
                bool queued = false;
                try
                {
                    SetWorkerPhase("Capturing GPU checkpoint");
                    capture = world.CaptureCheckpoint();
                    IntPtr ownedCapture = capture;
                    queued = ThreadPool.QueueUserWorkItem(delegate
                    {
                        try
                        {
                            NativeWorldContext.WriteCapturedCheckpoint(ownedCapture, request.Path);
                        }
                        catch (Exception ex) { request.Error = ex.Message; }
                        finally
                        {
                            NativeWorldContext.ReleaseCheckpointCapture(ownedCapture);
                            Interlocked.Decrement(ref _checkpointRequestsInFlight);
                            request.Completed.Set();
                        }
                    });
                    if (!queued) throw new InvalidOperationException("could not start the save writer");
                    capture = IntPtr.Zero; // Writer owns an independent, coherent snapshot.
                }
                catch (Exception ex) { request.Error = ex.Message; }
                finally
                {
                    SetWorkerPhase("GPU checkpoint capture finished");
                    if (!queued)
                    {
                        if (capture != IntPtr.Zero) NativeWorldContext.ReleaseCheckpointCapture(capture);
                        Interlocked.Decrement(ref _checkpointRequestsInFlight);
                        request.Completed.Set();
                    }
                }
            }
        }

        private void DrainRenderBufferRequests(NativeWorldContext world)
        {
            // A captured update owns its input independently of the world.
            // Poll without waiting, so simulation advances even if Unity has
            // not reached the callback yet. Do not reuse that surface early.
            if (!CompleteCapturedRenderRequest(world, false)) return;
            while (!_stopRequested)
            {
                RenderBufferRequest request;
                lock (_commandLock)
                {
                    if (_renderBufferRequests.Count == 0)
                    {
                        return;
                    }
                    request = _renderBufferRequests.Dequeue();
                }
                if (Interlocked.CompareExchange(ref request.State, 1, 0) != 0)
                {
                    continue;
                }
                try
                {
                    if (request.UpdateOnly)
                    {
                        world.CaptureD3D11RenderBuffers(request.BufferIndex);
                        request.NativeEventId = world.CreateUnityRenderRequest(
                            request.BufferIndex, 1, request.Config);
                        _pendingRenderEvent = request;
                        return;
                    }
                    if (!request.UpdateOnly) _hasRegisteredRenderResources = true;
                    bool completed = ExecuteRenderThreadRequest(world, request, true);
                    if (!completed)
                    {
                        request.Error = "the GPU world stopped before the render event executed";
                    }
                    else if (request.UpdateOnly) RecordRenderCompletion();
                    else
                    {
                        _directRenderRegistered = true;
                    }
                }
                catch (Exception ex)
                {
                    GpuInteropTrace.Write("worker render error " + ex.Message);
                    request.Error = ex.Message;
                    GpuInteropProcessSafety.CircuitBreaker.Trip(request.Error);
                    _directRenderFaulted = true;
                    _directRenderRegistered = false;
                    _retainRenderResources = true;
                }
                finally
                {
                    if (_pendingRenderEvent != request)
                        Interlocked.Exchange(ref request.State, 2);
                }
            }
        }

        private bool CompleteCapturedRenderRequest(NativeWorldContext world, bool wait)
        {
            RenderBufferRequest request = _pendingRenderEvent;
            if (request == null) return true;
            bool terminal = false;
            Stopwatch deadline = wait ? Stopwatch.StartNew() : null;
            try
            {
                if (wait) NativeWorldContext.CancelUnityRenderRequest(request.NativeEventId);
                while (true)
                {
                    int state, result;
                    NativeWorldContext.PollUnityRenderRequest(request.NativeEventId,
                        out state, out result, out request.BibiteCount, out request.PelletCount);
                    if (state == 2 || state == 3)
                    {
                        terminal = true;
                        if (!string.IsNullOrEmpty(request.Error))
                            throw new InvalidOperationException(request.Error);
                        if (state == 2)
                        {
                            NativeWorldContext.CheckUnityRenderResult(result);
                            RecordRenderCompletion();
                        }
                        else request.Error = "the captured graphics update was cancelled";
                        return true;
                    }
                    if (!wait) return false;
                    if (GraphicsHandoffDeadline.IsExpired(deadline.Elapsed.TotalMilliseconds, state))
                        throw new TimeoutException("The captured graphics callback did not finish within 15 seconds; its resources are quarantined until game exit.");
                    Thread.Sleep(1); // Only teardown waits for a running callback.
                }
            }
            catch (Exception ex)
            {
                request.Error = ex.Message;
                _directRenderFaulted = true;
                _directRenderRegistered = false;
                _retainRenderResources = true;
                GpuInteropProcessSafety.CircuitBreaker.Trip(request.Error);
                if (!terminal)
                {
                    world.RetainForProcessLifetime();
                    _pendingRenderEvent = null;
                    Interlocked.Exchange(ref request.State, 2);
                    throw;
                }
                return true;
            }
            finally
            {
                if (terminal)
                {
                    NativeWorldContext.ReleaseUnityRenderRequest(request.NativeEventId);
                    _pendingRenderEvent = null;
                    Interlocked.Exchange(ref request.State, 2);
                }
            }
        }

        private bool ExecuteRenderThreadRequest(NativeWorldContext world,
            RenderBufferRequest request, bool allowCancellation)
        {
            request.NativeEventId = world.CreateUnityRenderRequest(request.BufferIndex,
                request.ReleaseOnly ? 2 : request.UpdateOnly ? 1 : 0, request.Config);
            bool terminal = false;
            _pendingRenderEvent = request;
            SetWorkerPhase("Waiting for graphics callback");
            Stopwatch deadline = Stopwatch.StartNew();
            try
            {
                if (_completedRenderUpdates < 8 || request.ReleaseOnly)
                    GpuInteropTrace.Write("worker yields world to render event " + request.NativeEventId);
                // Only the worker waits. It performs no world/CUDA calls until
                // the render callback has returned exclusive ownership. There
                // is no timed-out map that Unity could accidentally draw.
                while (true)
                {
                    int state, result;
                    NativeWorldContext.PollUnityRenderRequest(request.NativeEventId,
                        out state, out result, out request.BibiteCount, out request.PelletCount);
                    if (state == 2 || state == 3)
                    {
                        terminal = true;
                        if (!string.IsNullOrEmpty(request.Error))
                            throw new InvalidOperationException(request.Error);
                        if (state == 3) return false;
                        NativeWorldContext.CheckUnityRenderResult(result);
                        if (_completedRenderUpdates < 8 || request.ReleaseOnly)
                            GpuInteropTrace.Write("worker reclaimed world after render event " + request.NativeEventId);
                        return true;
                    }
                    if (allowCancellation && _stopRequested)
                        NativeWorldContext.CancelUnityRenderRequest(request.NativeEventId);
                    if (GraphicsHandoffDeadline.IsExpired(deadline.Elapsed.TotalMilliseconds, state))
                        throw new TimeoutException("The graphics callback did not finish within 15 seconds; its resources are quarantined until game exit.");
                    Thread.Sleep(1);
                }
            }
            finally
            {
                _pendingRenderEvent = null;
                SetWorkerPhase("Graphics callback finished");
                if (terminal) NativeWorldContext.ReleaseUnityRenderRequest(request.NativeEventId);
                else
                {
                    _retainRenderResources = true;
                    GpuInteropProcessSafety.CircuitBreaker.Trip(
                        "native render callback completion could not be verified");
                    world.RetainForProcessLifetime();
                }
            }
        }

        private void ReleaseGraphicsOnRenderThread(NativeWorldContext world)
        {
            // A failed handoff can have quarantined the world already. Never
            // poll it again or enter CUDA teardown through an abandoned handle.
            if (!world.HasHandle) return;
            // Never destroy an owned capture while its callback is running.
            CompleteCapturedRenderRequest(world, true);
            if (!_hasRegisteredRenderResources || !world.HasHandle) return;
            RenderBufferRequest cleanup = new RenderBufferRequest { ReleaseOnly = true, State = 1 };
            try
            {
                // Retirement polling in Update continues issuing cleanup even
                // after the old scene and its renderer have been hidden.
                if (!ExecuteRenderThreadRequest(world, cleanup, false))
                    throw new InvalidOperationException("graphics cleanup was cancelled");
            }
            catch (Exception ex)
            {
                _retainRenderResources = true;
                GpuInteropProcessSafety.CircuitBreaker.Trip(ex.Message);
                if (world.HasHandle) world.AbandonD3D11RenderBuffers();
                if (string.IsNullOrEmpty(_error)) _error = ex.ToString();
            }
            finally
            {
                Interlocked.Exchange(ref cleanup.State, 2);
            }
        }

        private void RecordRenderCompletion()
        {
            long timestamp = Stopwatch.GetTimestamp();
            long previous = Interlocked.Exchange(
                ref _lastRenderCompletionTimestamp,
                timestamp);
            if (previous > 0)
            {
                double gap = (timestamp - previous) * 1000.0 / Stopwatch.Frequency;
                _maximumRenderGapMilliseconds = Math.Max(
                    _maximumRenderGapMilliseconds,
                    gap);
            }
            Interlocked.Increment(ref _completedRenderUpdates);
        }

        private void AdaptBatchSize(int completedSteps, int maximumSteps, float wallMilliseconds)
        {
            if (completedSteps <= 0 || wallMilliseconds <= 0.01f)
            {
                return;
            }
            double targetBatchMilliseconds = Math.Max(
                4.0,
                Math.Min(15.0, 180.0 / Math.Max(10, _renderFps)));
            int ideal = Math.Max(
                8,
                Math.Min(
                    maximumSteps,
                    (int)Math.Round(completedSteps * targetBatchMilliseconds / wallMilliseconds)));
            int current = Math.Max(8, Math.Min(maximumSteps, _adaptiveBatchSteps));
            _adaptiveBatchSteps = ideal < current
                ? ideal
                : Math.Max(8, Math.Min(maximumSteps, (current * 3 + ideal) / 4));
        }

        private NativeWorldStats PublishSnapshot(
            NativeWorldContext world,
            NativeWorldStepMetrics metrics,
            double previousSimulatedSeconds,
            double wallDelta)
        {
            SetWorkerPhase("Downloading presentation snapshot");
            Stopwatch snapshotTimer = Stopwatch.StartNew();
            int bibiteCount;
            int pelletCount;
            int visibleBibiteCount;
            VisibleViewport viewport = _visibleViewport;
            long timestamp = Stopwatch.GetTimestamp();
            bool summaryDue = !_directRenderRegistered || _lastSummaryTimestamp == 0 ||
                (timestamp - _lastSummaryTimestamp) / (double)Stopwatch.Frequency >= 0.2;
            NativeWorldStats stats;
            if (summaryDue)
            {
            stats = world.DownloadSnapshot(
                _writeBibites,
                out bibiteCount,
                _directRenderRegistered ? null : _writePellets,
                out pelletCount,
                _writeVisibleBibites,
                viewport != null ? _writeVisibleBibites.Length : 0,
                out visibleBibiteCount,
                viewport != null ? viewport.MinX : 0f,
                viewport != null ? viewport.MinY : 0f,
                viewport != null ? viewport.MaxX : 0f,
                viewport != null ? viewport.MaxY : 0f);
                _lastSummaryStats = stats;
                _completedSimulationSteps = stats.CompletedSteps;
                _lastSummaryTimestamp = Stopwatch.GetTimestamp();
                ++_summarySequence;
            }
            else
            {
                stats = _lastSummaryStats;
                stats.CompletedSteps = _completedSimulationSteps;
                stats.SimulatedSeconds = _completedSimulationSteps * (double)_config.FixedDeltaTime;
                bibiteCount = _published != null ? _published.BibiteCount : 0;
                pelletCount = _published != null ? _published.PelletCount : 0;
                visibleBibiteCount = world.DownloadVisibleBibites(_writeVisibleBibites,
                    viewport != null ? _writeVisibleBibites.Length : 0,
                    viewport != null ? viewport.MinX : 0f,
                    viewport != null ? viewport.MinY : 0f,
                    viewport != null ? viewport.MaxX : 0f,
                    viewport != null ? viewport.MaxY : 0f);
            }
            snapshotTimer.Stop();
            double achieved = wallDelta > 0.0
                ? (stats.SimulatedSeconds - previousSimulatedSeconds) / wallDelta
                : 0.0;
            _latestLivingBibites = stats.LivingBibites;
            lock (_snapshotLock)
            {
                NativeWorldBibite[] oldBibites = _published != null
                    ? _published.Bibites
                    : new NativeWorldBibite[_presentationBibiteCapacity];
                NativeWorldBibite[] oldVisibleBibites = _published != null
                    ? _published.VisibleBibites
                    : new NativeWorldBibite[_visibleBibiteCapacity];
                NativeWorldPellet[] oldPellets = _published != null
                    ? _published.Pellets
                    : new NativeWorldPellet[_config.PelletCount];
                _published = new GpuWorldSnapshot
                {
                    Sequence = ++_sequence,
                    SummarySequence = _summarySequence,
                    Bibites = summaryDue ? _writeBibites : oldBibites,
                    BibiteCount = bibiteCount,
                    VisibleBibites = _writeVisibleBibites,
                    VisibleBibiteCount = visibleBibiteCount,
                    VisibleViewport = viewport != null
                        ? (DetailedSpriteViewport?)new DetailedSpriteViewport(
                            viewport.MinX, viewport.MinY, viewport.MaxX, viewport.MaxY)
                        : null,
                    Pellets = summaryDue ? _writePellets : oldPellets,
                    PelletCount = pelletCount,
                    Stats = stats,
                    Metrics = metrics,
                    AchievedMultiplier = achieved,
                    PelletEnergy = _currentPelletEnergy,
                    SnapshotMilliseconds = snapshotTimer.Elapsed.TotalMilliseconds
                };
                if (summaryDue) _writeBibites = oldBibites;
                _writeVisibleBibites = oldVisibleBibites;
                if (summaryDue) _writePellets = oldPellets;
            }
            return stats;
        }

        public void Dispose()
        {
            _stopRequested = true;
            // The main thread must retain render resources until IsStopped,
            // not freeze Unity or destroy still-registered buffers on timeout.
        }
    }

    internal sealed class BibiteSpriteTemplate : IDisposable
    {
        internal sealed class Part
        {
            internal string Name;
            internal Sprite Sprite;
            internal Material Material;
            internal Vector3 LocalPosition;
            internal Quaternion LocalRotation;
            internal Vector3 LocalScale;
            internal Color Color;
            internal bool FlipX;
            internal bool FlipY;
            internal int SortingLayerId;
            internal int SortingOrder;
        }

        internal readonly List<Part> Parts = new List<Part>();
        private readonly List<Material> _materials = new List<Material>();

        internal static BibiteSpriteTemplate Capture(BibiteProceduralSpriter source)
        {
            if (source == null)
            {
                return null;
            }
            SpriteRenderer[] renderers = source.GetComponentsInChildren<SpriteRenderer>(true);
            if (renderers.Length == 0)
            {
                return null;
            }
            BibiteSpriteTemplate result = new BibiteSpriteTemplate();
            Dictionary<Material, Material> materialCopies = new Dictionary<Material, Material>();
            Transform root = source.transform;
            for (int index = 0; index < renderers.Length; index++)
            {
                SpriteRenderer renderer = renderers[index];
                if (renderer == null || renderer.sprite == null)
                {
                    continue;
                }
                Material material = null;
                if (renderer.sharedMaterial != null)
                {
                    if (!materialCopies.TryGetValue(renderer.sharedMaterial, out material))
                    {
                        material = new Material(renderer.sharedMaterial)
                        {
                            name = "GPU detail " + renderer.sharedMaterial.name
                        };
                        materialCopies.Add(renderer.sharedMaterial, material);
                        result._materials.Add(material);
                    }
                }
                result.Parts.Add(new Part
                {
                    Name = renderer.gameObject.name,
                    Sprite = renderer.sprite,
                    Material = material,
                    LocalPosition = root.InverseTransformPoint(renderer.transform.position),
                    LocalRotation = Quaternion.Inverse(root.rotation) * renderer.transform.rotation,
                    LocalScale = renderer.transform.localScale,
                    Color = renderer.color,
                    FlipX = renderer.flipX,
                    FlipY = renderer.flipY,
                    SortingLayerId = renderer.sortingLayerID,
                    SortingOrder = renderer.sortingOrder
                });
            }
            if (result.Parts.Count == 0)
            {
                result.Dispose();
                return null;
            }
            return result;
        }

        public void Dispose()
        {
            for (int index = 0; index < _materials.Count; index++)
            {
                if (_materials[index] != null)
                {
                    UnityEngine.Object.Destroy(_materials[index]);
                }
            }
            _materials.Clear();
            Parts.Clear();
        }
    }

    internal sealed class BibiteDetailedSpritePool : IDisposable
    {
        private const int DetailSortingOrderOffset = 8;

        private sealed class Proxy
        {
            internal GameObject Root;
            internal SpriteRenderer[] Renderers;
            internal readonly MaterialPropertyBlock Block = new MaterialPropertyBlock();
            internal NativeWorldBibite Previous;
            internal bool HasPrevious;
        }

        internal const int MaximumDetailedBibites = 2048;
        private static readonly int RootPosition = Shader.PropertyToID("_RootPos");
        private static readonly int Direction = Shader.PropertyToID("_Dir");
        private static readonly int BodyColor = Shader.PropertyToID("_Color");
        private readonly BibiteSpriteTemplate _template;
        private readonly Proxy[] _proxies;

        internal BibiteDetailedSpritePool(
            BibiteSpriteTemplate template,
            Transform parent,
            int maximumDetailedBibites)
        {
            _template = template;
            int detailedBibiteLimit = Mathf.Clamp(
                maximumDetailedBibites,
                0,
                MaximumDetailedBibites);
            if (template == null || template.Parts.Count == 0 || detailedBibiteLimit == 0)
            {
                _proxies = new Proxy[0];
                return;
            }
            _proxies = new Proxy[detailedBibiteLimit];
            for (int index = 0; index < _proxies.Length; index++)
            {
                Proxy proxy = new Proxy
                {
                    Root = new GameObject("Detailed GPU Bibite " + index),
                    Renderers = new SpriteRenderer[template.Parts.Count]
                };
                proxy.Root.transform.SetParent(parent, false);
                for (int partIndex = 0; partIndex < template.Parts.Count; partIndex++)
                {
                    BibiteSpriteTemplate.Part part = template.Parts[partIndex];
                    GameObject child = new GameObject(part.Name);
                    child.transform.SetParent(proxy.Root.transform, false);
                    child.transform.localPosition = part.LocalPosition;
                    child.transform.localRotation = part.LocalRotation;
                    child.transform.localScale = part.LocalScale;
                    SpriteRenderer renderer = child.AddComponent<SpriteRenderer>();
                    renderer.sprite = part.Sprite;
                    renderer.sharedMaterial = part.Material;
                    renderer.color = part.Color;
                    renderer.flipX = part.FlipX;
                    renderer.flipY = part.FlipY;
                    renderer.sortingLayerID = part.SortingLayerId;
                    // Preserve the stock body/arms/mouth/eyes draw order while
                    // keeping the original sprite above the batched silhouette.
                    renderer.sortingOrder = part.SortingOrder +
                        DetailSortingOrderOffset;
                    proxy.Renderers[partIndex] = renderer;
                }
                proxy.Root.SetActive(false);
                _proxies[index] = proxy;
            }
        }

        internal bool Render(
            NativeWorldBibite[] bibites,
            int count,
            int totalCount,
            int selectedSlot)
        {
            if (_proxies.Length == 0)
            {
                return false;
            }
            Camera camera = Camera.main;
            if (camera == null || !camera.orthographic)
            {
                DisableFrom(0);
                return false;
            }
            float vertical = camera.orthographicSize;
            if (!DetailedSpriteVisibility.ShouldRender(vertical, Screen.height))
            {
                DisableFrom(0);
                return false;
            }
            float horizontal = vertical * camera.aspect;
            Vector3 cameraPosition = camera.transform.position;
            int used = 0;

            int selectedIndex = -1;
            if (selectedSlot >= 0)
            {
                for (int index = 0; index < count; index++)
                {
                    if (bibites[index].Slot == selectedSlot)
                    {
                        selectedIndex = index;
                        break;
                    }
                }
                if (selectedIndex >= 0 && InView(
                    bibites[selectedIndex], cameraPosition, horizontal, vertical))
                {
                    Apply(_proxies[used], bibites[selectedIndex]);
                    used++;
                }
            }
            for (int index = 0; index < count && used < _proxies.Length; index++)
            {
                if (index == selectedIndex) continue;
                NativeWorldBibite bibite = bibites[index];
                if (!InView(bibite, cameraPosition, horizontal, vertical))
                {
                    continue;
                }
                Apply(_proxies[used], bibite);
                used++;
            }
            DisableFrom(used);
            return used == count && DetailedSpriteVisibility.CanReplaceLowDetailMesh(
                count,
                totalCount,
                _proxies.Length);
        }

        private static bool InView(
            NativeWorldBibite bibite,
            Vector3 cameraPosition,
            float horizontal,
            float vertical)
        {
            const float edgePadding = DetailedSpriteVisibility.EdgePadding;
            return Mathf.Abs(bibite.PositionX - cameraPosition.x) <=
                    horizontal + edgePadding &&
                Mathf.Abs(bibite.PositionY - cameraPosition.y) <=
                    vertical + edgePadding;
        }

        internal void Clear()
        {
            DisableFrom(0);
        }

        private static void Apply(Proxy proxy, NativeWorldBibite bibite)
        {
            if (!proxy.Root.activeSelf)
            {
                proxy.Root.SetActive(true);
            }
            NativeWorldBibite previous = proxy.Previous;
            if (proxy.HasPrevious && previous.Slot == bibite.Slot &&
                previous.PositionX == bibite.PositionX && previous.PositionY == bibite.PositionY &&
                previous.Heading == bibite.Heading && previous.Size == bibite.Size &&
                previous.ColorR == bibite.ColorR && previous.ColorG == bibite.ColorG &&
                previous.ColorB == bibite.ColorB) return;
            proxy.Previous = bibite;
            proxy.HasPrevious = true;
            Vector3 position = new Vector3(bibite.PositionX, bibite.PositionY, -0.15f);
            Vector2 direction = new Vector2(
                Mathf.Sin(bibite.Heading),
                Mathf.Cos(bibite.Heading));
            proxy.Root.transform.position = position;
            proxy.Root.transform.rotation = Quaternion.Euler(
                0f,
                0f,
                -bibite.Heading * Mathf.Rad2Deg);
            float scale = Mathf.Clamp(bibite.Size, 0.55f, 1.8f);
            proxy.Root.transform.localScale = new Vector3(scale, scale, 1f);
            Color color = new Color(
                Mathf.Clamp01(bibite.ColorR),
                Mathf.Clamp01(bibite.ColorG),
                Mathf.Clamp01(bibite.ColorB),
                1f);
            proxy.Block.SetVector(RootPosition, position);
            proxy.Block.SetVector(Direction, direction);
            proxy.Block.SetColor(BodyColor, color);
            for (int index = 0; index < proxy.Renderers.Length; index++)
                proxy.Renderers[index].SetPropertyBlock(proxy.Block);
        }

        private void DisableFrom(int first)
        {
            for (int index = first; index < _proxies.Length; index++)
            {
                if (_proxies[index].Root.activeSelf)
                {
                    _proxies[index].Root.SetActive(false);
                }
            }
        }

        public void Dispose()
        {
            for (int index = 0; index < _proxies.Length; index++)
            {
                if (_proxies[index] != null && _proxies[index].Root != null)
                {
                    UnityEngine.Object.Destroy(_proxies[index].Root);
                }
            }
            if (_template != null)
            {
                _template.Dispose();
            }
        }
    }

    internal sealed class PelletSpriteTemplate : IDisposable
    {
        internal Sprite Sprite;
        internal Material Material;
        internal Vector2 DisplaySize;
        internal Vector2 UvMin;
        internal Vector2 UvMax;
        internal Color VertexColor;

        internal static PelletSpriteTemplate Capture(SpriteRenderer source)
        {
            if (source == null || source.sprite == null)
            {
                return null;
            }
            Sprite sprite = source.sprite;
            Vector2 uvMin = new Vector2(float.PositiveInfinity, float.PositiveInfinity);
            Vector2 uvMax = new Vector2(float.NegativeInfinity, float.NegativeInfinity);
            Vector2[] spriteUvs = sprite.uv;
            for (int index = 0; index < spriteUvs.Length; index++)
            {
                uvMin = Vector2.Min(uvMin, spriteUvs[index]);
                uvMax = Vector2.Max(uvMax, spriteUvs[index]);
            }
            if (spriteUvs.Length == 0 || float.IsNaN(uvMin.x) || float.IsInfinity(uvMin.x) ||
                float.IsNaN(uvMax.x) || float.IsInfinity(uvMax.x))
            {
                uvMin = Vector2.zero;
                uvMax = Vector2.one;
            }

            Material material;
            if (source.sharedMaterial != null)
            {
                material = new Material(source.sharedMaterial)
                {
                    name = "GPU plant pellet display material"
                };
            }
            else
            {
                Shader shader = Shader.Find("Sprites/Default");
                if (shader == null)
                {
                    return null;
                }
                material = new Material(shader)
                {
                    name = "GPU plant pellet display material"
                };
            }
            material.mainTexture = sprite.texture;
            if (material.HasProperty("_MainTex"))
            {
                material.SetTexture("_MainTex", sprite.texture);
            }
            if (material.HasProperty("_Color"))
            {
                material.SetColor("_Color", Color.white);
            }

            Vector3 worldSize = source.bounds.size;
            Vector2 displaySize = new Vector2(
                Mathf.Clamp(worldSize.x, 1.5f, 12f),
                Mathf.Clamp(worldSize.y, 1.5f, 12f));
            return new PelletSpriteTemplate
            {
                Sprite = sprite,
                Material = material,
                DisplaySize = displaySize,
                UvMin = uvMin,
                UvMax = uvMax,
                VertexColor = source.color
            };
        }

        public void Dispose()
        {
            if (Material != null)
            {
                UnityEngine.Object.Destroy(Material);
                Material = null;
            }
        }
    }

    // End-of-frame fences include every camera and GUI draw. A fence created
    // during Update would precede that frame's draws and would not establish
    // exclusive CUDA ownership of a previously displayed surface.
    internal sealed class GpuRenderFrameFence : MonoBehaviour
    {
        internal Action AfterFrame;

        private IEnumerator Start()
        {
            WaitForEndOfFrame endOfFrame = new WaitForEndOfFrame();
            while (true)
            {
                yield return endOfFrame;
                Action callback = AfterFrame;
                if (callback != null) callback();
            }
        }
    }

    internal sealed class GpuWorldMeshRenderer : IDisposable
    {
        private const int BibiteVertices = 9;
        private const int BibiteIndices = 24;
        private const int PelletVertices = 4;
        private const int PelletIndices = 6;

        private readonly GameObject _root;
        private readonly Mesh _bibiteMesh;
        private readonly Mesh _pelletMesh;
        private readonly MeshFilter _bibiteFilter;
        private readonly MeshFilter _pelletFilter;
        private readonly MeshRenderer _bibiteRenderer;
        private readonly Mesh[] _directBibiteMeshes = new Mesh[2];
        private readonly Mesh[] _directPelletMeshes = new Mesh[2];
        private readonly NativeD3D11RenderConfig[] _directConfigs = new NativeD3D11RenderConfig[2];
        private readonly GraphicsFence[] _drawFences = new GraphicsFence[2];
        private readonly bool[] _drawFenceValid = new bool[2];
        private readonly bool[] _surfaceWasDisplayed = new bool[2];
        private GpuWorldRunner.RenderBufferRequest _renderRequest;
        private int _registeredSurfaces;
        private int _frontSurface = -1;
        private bool _directInteropRequested;
        private string _directInteropError;
        private int _traceFrames;
        private readonly Material _bibiteMaterial;
        private readonly Material _pelletMaterial;
        private Vector3[] _bibitePositions;
        private Color[] _bibiteColors;
        private readonly int[] _bibiteTriangles;
        private Vector3[] _pelletPositions;
        private Color[] _pelletColors;
        private readonly Vector2[] _pelletUvs;
        private readonly int[] _pelletTriangles;
        private readonly float _worldHalfExtent;
        private readonly BibiteDetailedSpritePool _detailedSprites;
        private readonly PelletSpriteTemplate _pelletSpriteTemplate;
        private readonly int _maxBibites;
        private readonly int _maxPellets;
        private bool _directInteropEnabled;
        private int _selectedSlot = -1;
        private DetailedSpriteViewport? _detailedViewport;

        internal GpuWorldMeshRenderer(
            int maxBibites,
            int maxPellets,
            float worldHalfExtent,
            BibiteSpriteTemplate spriteTemplate,
            PelletSpriteTemplate pelletSpriteTemplate,
            int detailedBibiteLimit)
        {
            _worldHalfExtent = worldHalfExtent;
            _pelletSpriteTemplate = pelletSpriteTemplate;
            _maxBibites = maxBibites;
            _maxPellets = maxPellets;
            _bibiteTriangles = new int[maxBibites * BibiteIndices];
            _pelletUvs = new Vector2[maxPellets * PelletVertices];
            _pelletTriangles = new int[maxPellets * PelletIndices];
            BuildStaticIndicesAndUvs(maxBibites, maxPellets);

            Shader shader = Shader.Find("Sprites/Default");
            if (shader == null)
            {
                shader = Shader.Find("Unlit/Transparent");
            }
            if (shader == null)
            {
                throw new InvalidOperationException("No compatible Unity sprite shader was found.");
            }

            _root = new GameObject("Bibites GPU native visuals");
            // Scene changes must not free registered D3D resources before the
            // asynchronous worker shutdown has released CUDA ownership.
            UnityEngine.Object.DontDestroyOnLoad(_root);
            GameObject pelletObject = new GameObject("GPU pellets");
            GameObject bibiteObject = new GameObject("GPU Bibites");
            pelletObject.transform.SetParent(_root.transform, false);
            bibiteObject.transform.SetParent(_root.transform, false);

            _pelletMesh = new Mesh { name = "GPU pellet display mesh" };
            _pelletMesh.MarkDynamic();
            _bibiteMesh = new Mesh { name = "GPU Bibite display mesh" };
            _bibiteMesh.MarkDynamic();
            _pelletMesh.indexFormat = IndexFormat.UInt32;
            _bibiteMesh.indexFormat = IndexFormat.UInt32;
            _pelletFilter = pelletObject.AddComponent<MeshFilter>();
            MeshRenderer pelletRenderer = pelletObject.AddComponent<MeshRenderer>();
            _bibiteFilter = bibiteObject.AddComponent<MeshFilter>();
            _bibiteRenderer = bibiteObject.AddComponent<MeshRenderer>();
            _pelletFilter.sharedMesh = _pelletMesh;
            _bibiteFilter.sharedMesh = _bibiteMesh;
            _pelletMaterial = pelletSpriteTemplate != null && pelletSpriteTemplate.Material != null
                ? pelletSpriteTemplate.Material
                : new Material(shader) { name = "GPU pellet display material" };
            _bibiteMaterial = new Material(shader) { name = "GPU Bibite display material" };
            pelletRenderer.sharedMaterial = _pelletMaterial;
            _bibiteRenderer.sharedMaterial = _bibiteMaterial;
            pelletRenderer.sortingOrder = 0;
            _bibiteRenderer.sortingOrder = 2;
            _detailedSprites = new BibiteDetailedSpritePool(
                spriteTemplate,
                _root.transform,
                detailedBibiteLimit);
        }

        internal int SelectedSlot
        {
            get { return _selectedSlot; }
        }

        internal void SetSelectedSlot(int slot)
        {
            _selectedSlot = slot;
        }

        internal bool DirectInteropEnabled
        {
            get { return _directInteropEnabled; }
        }

        internal void Render(GpuWorldSnapshot snapshot)
        {
            if (_traceFrames < 8) GpuInteropTrace.Write("main CPU presentation begin");
            if (!_directInteropEnabled || _frontSurface < 0)
            {
                RenderBibites(snapshot.Bibites, snapshot.BibiteCount);
                RenderPellets(snapshot.Pellets, snapshot.PelletCount,
                    snapshot.PelletEnergy);
            }
            bool detailsReplaceLowDetail = _detailedSprites.Render(
                snapshot.VisibleBibites,
                snapshot.VisibleBibiteCount,
                snapshot.Stats.LivingBibites,
                _selectedSlot) && CoversCamera(snapshot.VisibleViewport);
            _detailedViewport = snapshot.VisibleViewport;
            _bibiteRenderer.enabled = !detailsReplaceLowDetail;
            if (_traceFrames < 8) GpuInteropTrace.Write("main CPU presentation end");
        }

        internal void RefreshViewport()
        {
            // Camera movement can happen between worker snapshots, including
            // while paused. Never hide the whole mesh with an old visible list.
            if (!_detailedViewport.HasValue || CoversCamera(_detailedViewport)) return;
            _detailedSprites.Clear();
            _bibiteRenderer.enabled = true;
            _detailedViewport = null;
        }

        private static bool CoversCamera(DetailedSpriteViewport? viewport)
        {
            Camera camera = Camera.main;
            if (!viewport.HasValue || camera == null || !camera.orthographic ||
                !DetailedSpriteVisibility.ShouldRender(camera.orthographicSize, Screen.height))
                return false;
            Vector3 position = camera.transform.position;
            float halfHeight = camera.orthographicSize;
            float halfWidth = halfHeight * camera.aspect;
            return viewport.Value.Contains(new DetailedSpriteViewport(
                position.x - halfWidth, position.y - halfHeight,
                position.x + halfWidth, position.y + halfHeight));
        }

        private bool TryCreateDirectRenderConfig(int bufferIndex, out NativeD3D11RenderConfig config)
        {
            config = new NativeD3D11RenderConfig();
            if (SystemInfo.graphicsDeviceType != GraphicsDeviceType.Direct3D11 ||
                SystemInfo.graphicsDeviceVendorID != 0x10de ||
                !SystemInfo.supportsGraphicsFence)
            {
                return false;
            }
            if (_directBibiteMeshes[bufferIndex] == null)
            {
                GpuInteropTrace.Write("main create mesh set " + bufferIndex + " begin");
                Mesh bibites = new Mesh { name = "GPU Bibite back buffer " + bufferIndex };
                Mesh pellets = new Mesh { name = "GPU pellet back buffer " + bufferIndex };
                _directBibiteMeshes[bufferIndex] = bibites;
                _directPelletMeshes[bufferIndex] = pellets;
                // These meshes keep a fixed layout for their entire registered
                // lifetime. CPU fallback owns separate, never-registered meshes.
                InitializeDirectMesh(bibites, _maxBibites * BibiteVertices,
                    _bibiteTriangles, _maxBibites * BibiteIndices);
                InitializeDirectMesh(pellets, _maxPellets * PelletVertices,
                    _pelletTriangles, _maxPellets * PelletIndices);
                GpuInteropTrace.Write("main create mesh set " + bufferIndex + " end");
            }
            GpuInteropTrace.Write("main get native pointers " + bufferIndex + " begin");
            IntPtr bibiteBuffer = _directBibiteMeshes[bufferIndex].GetNativeVertexBufferPtr(0);
            IntPtr pelletBuffer = _directPelletMeshes[bufferIndex].GetNativeVertexBufferPtr(0);
            GpuInteropTrace.Write("main get native pointers " + bufferIndex + " end");
            if (bibiteBuffer == IntPtr.Zero || pelletBuffer == IntPtr.Zero) return false;
            Vector2 pelletSize = _pelletSpriteTemplate != null
                ? _pelletSpriteTemplate.DisplaySize
                : new Vector2(2.7f, 2.7f);
            Vector2 uvMin = _pelletSpriteTemplate != null
                ? _pelletSpriteTemplate.UvMin
                : Vector2.zero;
            Vector2 uvMax = _pelletSpriteTemplate != null
                ? _pelletSpriteTemplate.UvMax
                : Vector2.one;
            Color pelletColor = _pelletSpriteTemplate != null
                ? _pelletSpriteTemplate.VertexColor
                : new Color(0.25f, 0.8f, 0.15f, 0.95f);
            config.BibiteVertexBuffer = bibiteBuffer;
            config.PelletVertexBuffer = pelletBuffer;
            config.BibiteCapacity = _maxBibites;
            config.PelletCapacity = _maxPellets;
            config.PelletHalfWidth = pelletSize.x * 0.5f;
            config.PelletHalfHeight = pelletSize.y * 0.5f;
            config.PelletUvMinX = uvMin.x;
            config.PelletUvMinY = uvMin.y;
            config.PelletUvMaxX = uvMax.x;
            config.PelletUvMaxY = uvMax.y;
            config.PelletColorR = pelletColor.r;
            config.PelletColorG = pelletColor.g;
            config.PelletColorB = pelletColor.b;
            config.PelletColorA = pelletColor.a;
            return true;
        }

        internal bool BeginDirectInterop(GpuWorldRunner runner, out string error)
        {
            error = null;
            try
            {
                if (!TryCreateDirectRenderConfig(0, out _directConfigs[0]) ||
                    !TryCreateDirectRenderConfig(1, out _directConfigs[1]))
                {
                    error = "NVIDIA Direct3D 11 graphics buffers and CPU-query fences are required";
                    return false;
                }
                NativeWorldContext.PrepareD3D11Multithreading(_directConfigs[0].BibiteVertexBuffer);
                // Install the end-of-frame observer before giving any resource
                // to CUDA; initialization failures can then fall back safely.
                GpuRenderFrameFence fence = _root.AddComponent<GpuRenderFrameFence>();
                fence.AfterFrame = RecordEndOfFrameFence;
                GpuInteropTrace.Write("main queue register 0 begin");
                _renderRequest = runner.BeginRegisterRenderBuffers(0, _directConfigs[0], out error);
                GpuInteropTrace.Write("main queue register 0 end");
                if (_renderRequest == null) return false;
                _directInteropRequested = true;
                return true;
            }
            catch (Exception ex)
            {
                error = ex.Message;
                return false;
            }
        }

        private void RecordEndOfFrameFence()
        {
            if (!_directInteropRequested || !_directInteropEnabled || _frontSurface < 0) return;
            try
            {
                if (_traceFrames < 8) GpuInteropTrace.Write("main end-of-frame fence " + _frontSurface + " begin");
                // A CPU-query fence works on D3D11 without async-compute queue
                // support. We only poll it; never insert a GPU-side wait.
                _drawFences[_frontSurface] = Graphics.CreateGraphicsFence(
                    GraphicsFenceType.CPUSynchronisation,
                    SynchronisationStageFlags.PixelProcessing);
                _drawFenceValid[_frontSurface] = true;
                if (_traceFrames < 8) GpuInteropTrace.Write("main end-of-frame fence " + _frontSurface + " end");
            }
            catch (Exception ex)
            {
                _directInteropError = "could not fence Unity's previous render: " + ex.Message;
            }
        }

        internal bool PumpDirectInterop(GpuWorldRunner runner, bool requestFrame, out string error)
        {
            error = null;
            runner.PumpRenderThreadEvent();
            if (!_directInteropRequested) return true;
            ++_traceFrames;
            if (_traceFrames < 8) GpuInteropTrace.Write("main pump begin");
            try
            {
                if (!string.IsNullOrEmpty(_directInteropError))
                    throw new InvalidOperationException(_directInteropError);
                if (_renderRequest != null)
                {
                    if (!_renderRequest.IsCompleted) return true;
                    GpuWorldRunner.RenderBufferRequest completed = _renderRequest;
                    _renderRequest = null;
                    if (!string.IsNullOrEmpty(completed.Error))
                        throw new InvalidOperationException(completed.Error);
                    if (completed.UpdateOnly)
                    {
                        // Both writing and unmapping have completed. Counts
                        // belong to this exact frame, not an older CPU snapshot.
                        int surface = completed.BufferIndex;
                        if (_traceFrames < 8) GpuInteropTrace.Write("main swap " + surface + " begin");
                        UpdateDirectDrawCounts(surface, completed.BibiteCount, completed.PelletCount);
                        _bibiteFilter.sharedMesh = _directBibiteMeshes[surface];
                        _pelletFilter.sharedMesh = _directPelletMeshes[surface];
                        _frontSurface = surface;
                        _surfaceWasDisplayed[surface] = true;
                        _drawFenceValid[surface] = false;
                        if (_traceFrames < 8) GpuInteropTrace.Write("main swap " + surface + " end");
                    }
                    else
                    {
                        ++_registeredSurfaces;
                    }
                }
                if (_registeredSurfaces < 2)
                {
                    GpuInteropTrace.Write("main queue register " + _registeredSurfaces);
                    _renderRequest = runner.BeginRegisterRenderBuffers(_registeredSurfaces,
                        _directConfigs[_registeredSurfaces], out error);
                    if (!string.IsNullOrEmpty(error)) throw new InvalidOperationException(error);
                    return true;
                }
                _directInteropEnabled = true;
                if (!requestFrame) return true;
                int backSurface = _frontSurface < 0 ? 0 : 1 - _frontSurface;
                if (_traceFrames < 8) GpuInteropTrace.Write("main check draw fence " + backSurface);
                if (_surfaceWasDisplayed[backSurface] &&
                    (!_drawFenceValid[backSurface] || !_drawFences[backSurface].passed))
                    return true;
                _renderRequest = runner.BeginUpdateRenderBuffers(backSurface, out error);
                if (_traceFrames < 8) GpuInteropTrace.Write("main queued update " + backSurface);
                if (!string.IsNullOrEmpty(error)) throw new InvalidOperationException(error);
                return true;
            }
            catch (Exception ex)
            {
                error = ex.Message;
                _directInteropRequested = false;
                _directInteropEnabled = false;
                runner.DisableDirectRendering(error);
                // Never recycle, rewrite, display, or destroy a failed back
                // buffer. Separate CPU meshes keep the GUI/world responsive.
                _bibiteFilter.sharedMesh = _bibiteMesh;
                _pelletFilter.sharedMesh = _pelletMesh;
                return false;
            }
        }

        private void InitializeDirectMesh(
            Mesh mesh,
            int vertexCount,
            int[] indices,
            int indexCount)
        {
            VertexAttributeDescriptor[] attributes =
            {
                new VertexAttributeDescriptor(
                    VertexAttribute.Position,
                    VertexAttributeFormat.Float32,
                    3,
                    0),
                new VertexAttributeDescriptor(
                    VertexAttribute.Color,
                    VertexAttributeFormat.Float32,
                    4,
                    0),
                new VertexAttributeDescriptor(
                    VertexAttribute.TexCoord0,
                    VertexAttributeFormat.Float32,
                    2,
                    0)
            };
            MeshUpdateFlags flags = MeshUpdateFlags.DontRecalculateBounds |
                MeshUpdateFlags.DontValidateIndices;
            mesh.SetVertexBufferParams(vertexCount, attributes);
            mesh.SetIndexBufferParams(indexCount, IndexFormat.UInt32);
            mesh.SetIndexBufferData(indices, 0, 0, indexCount, flags);
            mesh.subMeshCount = 1;
            mesh.SetSubMesh(
                0,
                new SubMeshDescriptor(0, 0, MeshTopology.Triangles)
                {
                    firstVertex = 0,
                    vertexCount = vertexCount,
                    bounds = new Bounds(Vector3.zero,
                        new Vector3(_worldHalfExtent * 2.2f, _worldHalfExtent * 2.2f, 10f))
                },
                flags);
            mesh.bounds = new Bounds(
                Vector3.zero,
                new Vector3(_worldHalfExtent * 2.2f, _worldHalfExtent * 2.2f, 10f));
        }

        internal bool TrySelectAtScreenPoint(
            Vector3 screenPoint,
            NativeWorldBibite[] bibites,
            int count,
            out NativeWorldBibite selected)
        {
            selected = new NativeWorldBibite();
            Camera camera = Camera.main;
            if (camera == null || count <= 0)
            {
                return false;
            }
            Vector3 world = camera.ScreenToWorldPoint(screenPoint);
            float worldPerPixel = camera.orthographic
                ? camera.orthographicSize * 2f / Math.Max(1, Screen.height)
                : 1f;
            // Match the visible body, not only its centre point. This remains easy
            // to click at close zoom and grows to a 28-pixel target when zoomed out.
            float threshold = Mathf.Max(12f, worldPerPixel * 28f);
            float best = threshold * threshold;
            int bestIndex = -1;
            for (int index = 0; index < count; index++)
            {
                float dx = bibites[index].PositionX - world.x;
                float dy = bibites[index].PositionY - world.y;
                float distance = dx * dx + dy * dy;
                if (distance < best)
                {
                    best = distance;
                    bestIndex = index;
                }
            }
            if (bestIndex < 0)
            {
                _selectedSlot = -1;
                return false;
            }
            selected = bibites[bestIndex];
            _selectedSlot = selected.Slot;
            return true;
        }

        private void RenderBibites(NativeWorldBibite[] bibites, int count)
        {
            EnsureCpuFallbackArrays();
            int vertexCount = count * BibiteVertices;
            int indexCount = count * BibiteIndices;
            for (int index = 0; index < count; index++)
            {
                NativeWorldBibite bibite = bibites[index];
                int start = index * BibiteVertices;
                Vector2 center = new Vector2(bibite.PositionX, bibite.PositionY);
                Vector2 forward = new Vector2(
                    Mathf.Sin(bibite.Heading),
                    Mathf.Cos(bibite.Heading));
                Vector2 right = new Vector2(forward.y, -forward.x);
                float length = 3.8f + 3.2f * Mathf.Clamp(bibite.Size, 0.45f, 2.2f);
                float width = length * 0.48f;
                Color body = new Color(
                    Mathf.Clamp01(bibite.ColorR),
                    Mathf.Clamp01(bibite.ColorG),
                    Mathf.Clamp01(bibite.ColorB),
                    1f);
                if (bibite.Slot == _selectedSlot)
                {
                    body = Color.Lerp(body, Color.yellow, 0.55f);
                }
                Color noseColor = Color.Lerp(body, Color.white, 0.2f);
                Color tailColor = Color.Lerp(body, Color.black, 0.25f);

                // One contiguous low-detail silhouette replaces the old detached
                // fin and eye triangles. The stock sprites still overlay this shape
                // when zoomed in, while distant populations remain a single cheap mesh.
                SetVertex(start + 0, center, -0.05f, body);
                SetVertex(start + 1, center + forward * length, -0.05f, noseColor);
                SetVertex(start + 2, center + forward * length * 0.55f - right * width * 0.75f, -0.05f, body);
                SetVertex(start + 3, center - forward * length * 0.20f - right * width, -0.05f, body);
                SetVertex(start + 4, center - forward * length * 0.80f - right * width * 0.55f, -0.05f, tailColor);
                SetVertex(start + 5, center - forward * length * 0.95f, -0.05f, tailColor);
                SetVertex(start + 6, center - forward * length * 0.80f + right * width * 0.55f, -0.05f, tailColor);
                SetVertex(start + 7, center - forward * length * 0.20f + right * width, -0.05f, body);
                SetVertex(start + 8, center + forward * length * 0.55f + right * width * 0.75f, -0.05f, body);
            }

            _bibiteMesh.Clear(false);
            _bibiteMesh.SetVertices(_bibitePositions, 0, vertexCount);
            _bibiteMesh.SetColors(_bibiteColors, 0, vertexCount);
            _bibiteMesh.SetTriangles(_bibiteTriangles, 0, indexCount, 0, false);
            _bibiteMesh.bounds = new Bounds(
                Vector3.zero,
                new Vector3(_worldHalfExtent * 2.2f, _worldHalfExtent * 2.2f, 10f));
        }

        private void RenderPellets(NativeWorldPellet[] pellets, int count,
            float referenceEnergy)
        {
            EnsureCpuFallbackArrays();
            int vertexCount = count * PelletVertices;
            int indexCount = count * PelletIndices;
            Color pelletColor = _pelletSpriteTemplate != null
                ? _pelletSpriteTemplate.VertexColor
                : new Color(0.25f, 0.8f, 0.15f, 0.95f);
            Vector2 displaySize = _pelletSpriteTemplate != null
                ? _pelletSpriteTemplate.DisplaySize
                : new Vector2(2.7f, 2.7f);
            for (int index = 0; index < count; index++)
            {
                NativeWorldPellet pellet = pellets[index];
                int start = index * PelletVertices;
                float scale = Mathf.Sqrt(Mathf.Max(0.01f,
                    pellet.Energy / Mathf.Max(0.001f, referenceEnergy)));
                float radiusX = displaySize.x * 0.5f * scale;
                float radiusY = displaySize.y * 0.5f * scale;
                Color drawColor = pellet.Material == 1
                    ? new Color(0.8f, 0.18f, 0.24f, pelletColor.a)
                    : pelletColor;
                _pelletPositions[start + 0] = new Vector3(
                    pellet.PositionX - radiusX, pellet.PositionY - radiusY, 0.05f);
                _pelletPositions[start + 1] = new Vector3(
                    pellet.PositionX - radiusX, pellet.PositionY + radiusY, 0.05f);
                _pelletPositions[start + 2] = new Vector3(
                    pellet.PositionX + radiusX, pellet.PositionY + radiusY, 0.05f);
                _pelletPositions[start + 3] = new Vector3(
                    pellet.PositionX + radiusX, pellet.PositionY - radiusY, 0.05f);
                _pelletColors[start + 0] = drawColor;
                _pelletColors[start + 1] = drawColor;
                _pelletColors[start + 2] = drawColor;
                _pelletColors[start + 3] = drawColor;
            }

            _pelletMesh.Clear(false);
            _pelletMesh.SetVertices(_pelletPositions, 0, vertexCount);
            _pelletMesh.SetColors(_pelletColors, 0, vertexCount);
            _pelletMesh.SetUVs(0, _pelletUvs, 0, vertexCount);
            _pelletMesh.SetTriangles(_pelletTriangles, 0, indexCount, 0, false);
            _pelletMesh.bounds = new Bounds(
                Vector3.zero,
                new Vector3(_worldHalfExtent * 2.2f, _worldHalfExtent * 2.2f, 10f));
        }

        private void SetVertex(int index, Vector2 point, float z, Color color)
        {
            _bibitePositions[index] = new Vector3(point.x, point.y, z);
            _bibiteColors[index] = color;
        }

        private void EnsureCpuFallbackArrays()
        {
            if (_bibitePositions == null)
            {
                _bibitePositions = new Vector3[Math.Min(_maxBibites, 8192) * BibiteVertices];
                _bibiteColors = new Color[_bibitePositions.Length];
            }
            if (_pelletPositions == null)
            {
                _pelletPositions = new Vector3[_maxPellets * PelletVertices];
                _pelletColors = new Color[_pelletPositions.Length];
            }
        }

        private void BuildStaticIndicesAndUvs(int maxBibites, int maxPellets)
        {
            for (int index = 0; index < maxBibites; index++)
            {
                int vertex = index * BibiteVertices;
                int triangle = index * BibiteIndices;
                for (int segment = 0; segment < 8; segment++)
                {
                    int offset = triangle + segment * 3;
                    _bibiteTriangles[offset + 0] = vertex;
                    _bibiteTriangles[offset + 1] = vertex + 1 + segment;
                    _bibiteTriangles[offset + 2] = vertex + 1 + ((segment + 1) % 8);
                }
            }
            for (int index = 0; index < maxPellets; index++)
            {
                int vertex = index * PelletVertices;
                int triangle = index * PelletIndices;
                _pelletTriangles[triangle + 0] = vertex + 0;
                _pelletTriangles[triangle + 1] = vertex + 1;
                _pelletTriangles[triangle + 2] = vertex + 2;
                _pelletTriangles[triangle + 3] = vertex + 0;
                _pelletTriangles[triangle + 4] = vertex + 2;
                _pelletTriangles[triangle + 5] = vertex + 3;
                Vector2 uvMin = _pelletSpriteTemplate != null
                    ? _pelletSpriteTemplate.UvMin
                    : Vector2.zero;
                Vector2 uvMax = _pelletSpriteTemplate != null
                    ? _pelletSpriteTemplate.UvMax
                    : Vector2.one;
                _pelletUvs[vertex + 0] = new Vector2(uvMin.x, uvMin.y);
                _pelletUvs[vertex + 1] = new Vector2(uvMin.x, uvMax.y);
                _pelletUvs[vertex + 2] = new Vector2(uvMax.x, uvMax.y);
                _pelletUvs[vertex + 3] = new Vector2(uvMax.x, uvMin.y);
            }
        }

        private void UpdateDirectDrawCounts(int bufferIndex, int bibiteCount, int pelletCount)
        {
            MeshUpdateFlags flags = MeshUpdateFlags.DontRecalculateBounds |
                MeshUpdateFlags.DontValidateIndices;
            bibiteCount = Mathf.Clamp(bibiteCount, 0, _maxBibites);
            pelletCount = Mathf.Clamp(pelletCount, 0, _maxPellets);
            Bounds bounds = new Bounds(Vector3.zero,
                new Vector3(_worldHalfExtent * 2.2f, _worldHalfExtent * 2.2f, 10f));
            _directBibiteMeshes[bufferIndex].SetSubMesh(
                0,
                new SubMeshDescriptor(
                    0,
                    bibiteCount * BibiteIndices,
                    MeshTopology.Triangles)
                {
                    firstVertex = 0,
                    vertexCount = bibiteCount * BibiteVertices,
                    bounds = bounds
                },
                flags);
            _directPelletMeshes[bufferIndex].SetSubMesh(
                0,
                new SubMeshDescriptor(
                    0,
                    pelletCount * PelletIndices,
                    MeshTopology.Triangles)
                {
                    firstVertex = 0,
                    vertexCount = pelletCount * PelletVertices,
                    bounds = bounds
                },
                flags);
        }

        internal void HideForRetirement()
        {
            if (_root != null) _root.SetActive(false);
        }

        public void Dispose()
        {
            if (_detailedSprites != null) _detailedSprites.Dispose();
            if (_root != null) UnityEngine.Object.Destroy(_root);
            if (_bibiteMesh != null) UnityEngine.Object.Destroy(_bibiteMesh);
            if (_pelletMesh != null) UnityEngine.Object.Destroy(_pelletMesh);
            for (int index = 0; index < 2; ++index)
            {
                if (_directBibiteMeshes[index] != null) UnityEngine.Object.Destroy(_directBibiteMeshes[index]);
                if (_directPelletMeshes[index] != null) UnityEngine.Object.Destroy(_directPelletMeshes[index]);
            }
            if (_bibiteMaterial != null) UnityEngine.Object.Destroy(_bibiteMaterial);
            if (_pelletSpriteTemplate != null)
            {
                _pelletSpriteTemplate.Dispose();
            }
            else if (_pelletMaterial != null)
            {
                UnityEngine.Object.Destroy(_pelletMaterial);
            }
        }
    }

    internal sealed class GpuNativeWorldBridge : IDisposable
    {
        // A driver error can leave resource ownership indeterminate. Preserve
        // hidden DDOL roots until process exit rather than risking a use-after-
        // free, even if native disposal has returned an error.
        private static readonly List<GpuWorldMeshRenderer> QuarantinedRenderers =
            new List<GpuWorldMeshRenderer>();
        private const int MaximumPresentationSamples = 8192;

        private sealed class RetiredWorldResources
        {
            internal GpuWorldRunner Runner;
            internal GpuWorldMeshRenderer Renderer;
        }

        private readonly List<RetiredWorldResources> _retiredWorlds =
            new List<RetiredWorldResources>();
        private bool _applicationQuitRequested;

        internal bool HasResourcesForApplicationQuit
        {
            get { return _runner != null || _renderer != null || _retiredWorlds.Count != 0; }
        }

        internal string ApplicationQuitState
        {
            get
            {
                string retired = _retiredWorlds.Count == 0
                    ? "none"
                    : _retiredWorlds[0].Runner.ShutdownState;
                return "activeRunner=" + (_runner != null) +
                    ", worker=" + (_runner != null ? _runner.WorkerState : "none") +
                    ", renderer=" + (_renderer != null) +
                    ", retired=" + _retiredWorlds.Count +
                    ", firstRetired=" + retired;
            }
        }

        internal bool AdvanceApplicationQuit(bool mayStopWorld)
        {
            if (mayStopWorld && !_applicationQuitRequested)
            {
                _applicationQuitRequested = true;
                _pendingCheckpointPath = null;
                _sceneAttempted = true;
                StopWorld();
            }
            // A save may still need the current runner. Complete any existing
            // render handoff so it cannot hold up that save, but queue no new
            // frames. Retired workers require the same pump for unregister.
            ReleaseStoppedWorldResources();
            if (_runner != null) _runner.PumpRenderThreadEvent();
            return !HasResourcesForApplicationQuit;
        }

        internal enum InspectorTab
        {
            Stats,
            Genes,
            Biology,
            Brain,
            ExpandedBrain
        }

        internal const int MaximumPopulation = 500000;

        internal sealed class BreakdownEntry
        {
            internal ulong Id;
            internal string Name;
            internal int Count;
            internal float Energy;
        }

        private static readonly FieldInfo TimeKeeperTps = typeof(TimeKeeper).GetField(
            "tps",
            BindingFlags.Instance | BindingFlags.NonPublic);
        private static readonly FieldInfo TimeKeeperStps = typeof(TimeKeeper).GetField(
            "stps",
            BindingFlags.Instance | BindingFlags.NonPublic);
        private static readonly FieldInfo InformationSpeciesItems = typeof(InformationPanel).GetField(
            "speciesItems",
            BindingFlags.Instance | BindingFlags.NonPublic);
        private static readonly FieldInfo InformationSpeciesCount = typeof(InformationPanel).GetField(
            "NSpecies",
            BindingFlags.Instance | BindingFlags.NonPublic);
        private static readonly FieldInfo InformationTopSpeciesCount = typeof(InformationPanel).GetField(
            "NTopSpecies",
            BindingFlags.Instance | BindingFlags.NonPublic);
        private static readonly FieldInfo InformationSpeciesPrefab = typeof(InformationPanel).GetField(
            "speciesItemPrefab",
            BindingFlags.Instance | BindingFlags.NonPublic);
        private static readonly FieldInfo InformationSpeciesHolder = typeof(InformationPanel).GetField(
            "speciesInfoHolder",
            BindingFlags.Instance | BindingFlags.NonPublic);
        private static readonly FieldInfo InformationSpeciesSection = typeof(InformationPanel).GetField(
            "speciesSection",
            BindingFlags.Instance | BindingFlags.NonPublic);
        private static readonly FieldInfo InformationTagItems = typeof(InformationPanel).GetField(
            "tagItems",
            BindingFlags.Instance | BindingFlags.NonPublic);
        private static readonly FieldInfo InformationTagPrefab = typeof(InformationPanel).GetField(
            "tagItemPrefab",
            BindingFlags.Instance | BindingFlags.NonPublic);
        private static readonly FieldInfo InformationTagHolder = typeof(InformationPanel).GetField(
            "tagsInfoHolder",
            BindingFlags.Instance | BindingFlags.NonPublic);
        private static readonly FieldInfo InformationTagSection = typeof(InformationPanel).GetField(
            "tagSection",
            BindingFlags.Instance | BindingFlags.NonPublic);
        private readonly ManualLogSource _logger;
        private readonly Dictionary<ulong, string> _lineageNames = new Dictionary<ulong, string>();
        private readonly Dictionary<ulong, string> _tagNames = new Dictionary<ulong, string>();
        private readonly Dictionary<ulong, TagElementHandle> _nativeTagItems =
            new Dictionary<ulong, TagElementHandle>();
        private readonly Dictionary<int, ulong> _speciesHandleLineages =
            new Dictionary<int, ulong>();
        private readonly Dictionary<int, ulong> _tagHandleTags =
            new Dictionary<int, ulong>();
        private GpuWorldRunner _runner;
        private GpuWorldMeshRenderer _renderer;
        private bool _sceneAttempted;
        private bool _presentationActive;
        private bool _loadedCheckpointActive;
        private bool _stockWorldCleared;
        private bool _autoSaveSuppressed;
        private float _simulationReadyAt;
        private long _lastRenderedSequence;
        private GpuWorldSnapshot _lastSnapshot;
        private NativeWorldBibite _selectedBibite;
        private NativeWorldBibiteDetail _selectedDetail;
        private NativeWorldBrainNodeState[] _selectedBrainNodes =
            new NativeWorldBrainNodeState[0];
        private NativeWorldBrainSynapseState[] _selectedBrainSynapses =
            new NativeWorldBrainSynapseState[0];
        private long _selectedDetailSequence = -1;
        private bool _hasSelection;
        private bool _followSelection;
        private InspectorTab _selectedInspectorTab;
        private Vector2 _selectedInspectorScroll;
        private int _expandedBrainPage;
        private bool _expandedBrainShowsSynapses;
        private string _selectedTagEditor = string.Empty;
        private string _selectedInspectorNotice;
        private GpuNativeInspector _stockInspector;
        private int _selectedInstanceToken;
        private int _renderFps = 30;
        private bool _extremeMode;
        private int _populationCap;
        private int _activePelletCount;
        private int _configuredFoodCap;
        private int _foodBaselineTarget;
        private double _foodBaselineEstimate;
        private float _foodBaselineGrowth;
        private float _foodBaselineGlobalGrowth;
        private float _foodBaselineStockEnergy;
        private float _foodBaselineNativeEnergy;
        private float _foodBaselineNativeGrowth = 1f;
        private int _lastQueuedFoodTarget = -1;
        private float _lastQueuedFoodGrowth = -1f;
        private float _lastQueuedPelletEnergy = -1f;
        private float _lastQueuedLinearDrag = -1f;
        private float _lastQueuedCollisionDamage = -1f;
        private float _lastQueuedCollisionThreshold = -1f;
        private float _lastQueuedBitingDamage = -1f;
        private float _lastQueuedBitingPressure = -1f;
        private bool _hasQueuedDigestionSettings;
        private NativeWorldDigestionSettings _lastQueuedDigestionSettings;
        private NativeWorldFoodZone[] _lastQueuedFoodZones;
        private float _nextFoodSettingsPollAt;
        private float _activeWorldHalfExtent;
        private string _pendingCheckpointPath;
        private GpuWorldRunner.CheckpointRequest _pendingSaveRequest;
        private string _pendingSaveWorldPath;
        private string _pendingSaveCheckpointPath;
        private string _pendingSaveTemporaryPath;
        private double _maximumSnapshotMilliseconds;
        private TimeKeeper _timeKeeper;
        private DataLogger _boundDataLogger;
        private Func<bool, DataPoint4> _stockBiomassLogAction;
        private Func<bool, DataPoint3> _stockCountLogAction;
        private Func<bool, DataPoint6> _stockBrainSizeLogAction;
        private Func<bool, DataPoint4> _stockAgeLogAction;
        private Func<bool, DataPoint4> _stockDeathAgeLogAction;
        private Func<bool, DataPoint6> _stockEggsLaidLogAction;
        private Func<bool, DataPoint2> _stockBirthDeathLogAction;
        private double _nextHistorySampleSeconds;
        private ulong _historyBirthBaseline;
        private ulong _historyDeathBaseline;
        private float _nextInformationUpdateAt;
        private bool _informationPanelFaulted;
        private bool _directInteropFaultReported;
        private float _nextDirectRenderAt;
        private int _detailedBibiteLimit = 96;
        private int _lastReportedPlacementFailures;
        private bool _nativeSpeciesPanelOpen;
        private Rect _nativeSpeciesWindow = new Rect(70f, 190f, 560f, 520f);
        private Rect _selectedBibiteWindow = new Rect(0f, 150f, 620f, 560f);
        private Vector2 _nativeSpeciesScroll;
        private List<BreakdownEntry> _lineageBreakdown = new List<BreakdownEntry>();
        private readonly float[] _presentationAgeSamples =
            new float[MaximumPresentationSamples];
        private long _presentationSampleSequence = -1;
        private long _lastChartSummarySequence = -1;
        private DataPoint6 _cachedBrainSizeData;
        private DataPoint4 _cachedAgeData;

        internal GpuNativeWorldBridge(ManualLogSource logger)
        {
            _logger = logger;
            GpuInteropTrace.Logger = logger;
        }

        internal bool OwnsSimulation
        {
            get { return _presentationActive; }
        }

        internal bool HasNativeSelection { get { return _hasSelection; } }
        internal NativeWorldBibite SelectedBibite { get { return _selectedBibite; } }
        internal NativeWorldBibiteDetail SelectedDetail { get { return _selectedDetail; } }
        internal NativeWorldBrainNodeState[] SelectedBrainNodes { get { return _selectedBrainNodes; } }
        internal NativeWorldBrainSynapseState[] SelectedBrainSynapses { get { return _selectedBrainSynapses; } }
        internal InspectorTab SelectedInspectorTab { get { return _selectedInspectorTab; } }
        internal bool FollowingSelection { get { return _followSelection; } }
        internal string SelectedNotice
        {
            get
            {
                if (!string.IsNullOrEmpty(_selectedInspectorNotice)) return _selectedInspectorNotice;
                string status = _runner != null ? _runner.LastBibiteCommandStatus : null;
                return GpuBibiteActionNotice.MatchesSlot(status, _selectedBibite.Slot) ? status : null;
            }
        }
        internal string SelectedLineageName { get { return LineageName(_selectedBibite.LineageId); } }
        internal string SelectedTagName { get { return TagName(_selectedBibite.TagId); } }

        internal void ToggleSelectionFollow() { _followSelection = !_followSelection; }
        internal void CloseNativeSelection() { ClearSelection(); }
        internal void ShowNativeInspectorTab(InspectorTab tab) { SelectInspectorTab(tab); }
        internal void CopyNativeTag() { GUIUtility.systemCopyBuffer = _selectedBibite.TagId == 0 ? string.Empty : SelectedTagName; }
        internal void SetNativeTag(string tag) { _selectedTagEditor = tag ?? string.Empty; ApplySelectedTag(); }
        internal void KillNativeSelection() { if (_hasSelection) QueueSelectedKill(); }
        internal void ReproduceNativeSelection() { if (_hasSelection) QueueSelectedReproduction(); }
        internal void SetNativeInspectorBrainVisible(bool visible)
        {
            if (_runner != null) _runner.SetSelectedInspectorNeedsBrain(visible);
        }

        internal JObject ExportPresentationMetadata()
        {
            return GpuPresentationMetadata.Export(_lineageNames, _tagNames);
        }

        internal void ImportPresentationMetadata(JObject metadata)
        {
            GpuPresentationMetadata.Import(metadata, _lineageNames, _tagNames);
        }

        internal bool IsStarting
        {
            get { return _runner != null && !_presentationActive && string.IsNullOrEmpty(_runner.Error); }
        }

        internal int RenderFps
        {
            get { return _renderFps; }
        }

        internal int SuccessfulPlacements
        {
            get { return _runner != null ? _runner.SuccessfulPlacements : 0; }
        }

        internal int LivingBibites
        {
            get { return _lastSnapshot != null ? _lastSnapshot.Stats.LivingBibites : 0; }
        }

        internal int ActivePellets
        {
            get { return _lastSnapshot != null ? _lastSnapshot.Stats.ActivePellets : 0; }
        }

        internal ulong PelletsEaten
        {
            get { return _lastSnapshot != null ? _lastSnapshot.Stats.PelletsEaten : 0ul; }
        }

        internal double AchievedMultiplier
        {
            get { return _lastSnapshot != null ? _lastSnapshot.AchievedMultiplier : 0.0; }
        }

        internal double LatestSnapshotMilliseconds
        {
            get { return _lastSnapshot != null ? _lastSnapshot.SnapshotMilliseconds : 0.0; }
        }

        internal double MaximumSnapshotMilliseconds
        {
            get { return _maximumSnapshotMilliseconds; }
        }

        internal int PresentationBibites
        {
            get { return _lastSnapshot != null ? _lastSnapshot.BibiteCount : 0; }
        }

        internal double MaximumRenderGapMilliseconds
        {
            get { return _runner != null ? _runner.MaximumRenderGapMilliseconds : 0.0; }
        }

        internal double MaximumStepMilliseconds
        {
            get { return _runner != null ? _runner.MaximumStepMilliseconds : 0.0; }
        }

        internal int CompletedRenderUpdates
        {
            get { return _runner != null ? _runner.CompletedRenderUpdates : 0; }
        }

        internal static string CheckpointPathForWorld(string worldPath)
        {
            return string.IsNullOrEmpty(worldPath) ? null : worldPath + ".bgfgpu";
        }

        internal bool BeginSaveWorldCheckpoint(string worldPath, out string error)
        {
            error = null;
            if (!_presentationActive || _runner == null || !_runner.IsReady)
            {
                error = "the GPU world is not ready";
                return false;
            }
            if (_pendingSaveRequest != null)
            {
                error = "a GPU world save is already in progress";
                return false;
            }

            string checkpointPath = CheckpointPathForWorld(worldPath);
            string directory = Path.GetDirectoryName(checkpointPath);
            string temporaryPath = Path.Combine(
                string.IsNullOrEmpty(directory) ? "." : directory,
                ".bgfgpu-" + Guid.NewGuid().ToString("N") + ".tmp");
            try
            {
                if (!string.IsNullOrEmpty(directory))
                {
                    Directory.CreateDirectory(directory);
                }
                GpuWorldRunner.CheckpointRequest request =
                    _runner.BeginSaveCheckpoint(temporaryPath, out error);
                if (request == null)
                {
                    return false;
                }
                _pendingSaveRequest = request;
                _pendingSaveWorldPath = worldPath;
                _pendingSaveCheckpointPath = checkpointPath;
                _pendingSaveTemporaryPath = temporaryPath;
                _logger.LogInfo("Queued an asynchronous GPU-native checkpoint for " +
                    worldPath + ".");
                return true;
            }
            catch (Exception ex)
            {
                error = ex.Message;
                _logger.LogError("Could not start the GPU-native checkpoint: " + ex);
                return false;
            }
        }

        internal string PendingSaveState
        {
            get { return _runner != null ? _runner.WorkerState : "GPU worker unavailable"; }
        }

        internal bool PollSaveWorldCheckpoint(
            out bool completed,
            out string worldPath,
            out string error)
        {
            completed = false;
            worldPath = _pendingSaveWorldPath;
            error = null;
            if (_pendingSaveRequest == null)
            {
                return false;
            }
            if (!_pendingSaveRequest.Completed.WaitOne(0))
            {
                return true;
            }

            completed = true;
            error = _pendingSaveRequest.Error;
            _pendingSaveRequest.Completed.Close();
            _pendingSaveRequest = null;
            string checkpointPath = _pendingSaveCheckpointPath;
            string temporaryPath = _pendingSaveTemporaryPath;
            _pendingSaveWorldPath = null;
            _pendingSaveCheckpointPath = null;
            _pendingSaveTemporaryPath = null;
            try
            {
                if (string.IsNullOrEmpty(error))
                {
                    if (File.Exists(checkpointPath))
                    {
                        File.Replace(temporaryPath, checkpointPath, null);
                    }
                    else
                    {
                        File.Move(temporaryPath, checkpointPath);
                    }
                    _logger.LogInfo(
                        "Saved GPU-native checkpoint beside the stock wrapper: " +
                        checkpointPath);
                }
            }
            catch (Exception ex)
            {
                error = ex.Message;
                _logger.LogError("Could not finish the GPU-native checkpoint: " + ex);
            }
            finally
            {
                try
                {
                    if (!string.IsNullOrEmpty(temporaryPath) && File.Exists(temporaryPath))
                    {
                        File.Delete(temporaryPath);
                    }
                }
                catch (Exception cleanupError)
                {
                    _logger.LogWarning("Could not remove temporary GPU checkpoint: " +
                        cleanupError.Message);
                }
            }
            return true;
        }

        internal bool PrepareCheckpointLoad(string worldPath, out string error)
        {
            error = null;
            string checkpointPath = CheckpointPathForWorld(worldPath);
            if (string.IsNullOrEmpty(checkpointPath) || !File.Exists(checkpointPath))
            {
                error = "the GPU checkpoint sidecar is missing";
                return false;
            }
            try
            {
                // Validate the header before the stock wrapper starts changing
                // the current scene. The actual device allocation happens on
                // the worker after SaveSystem has restored scenario settings.
                NativeWorldContext.ReadCheckpointConfig(checkpointPath, 0);
                StopWorld();
                _pendingCheckpointPath = checkpointPath;
                _sceneAttempted = false;
                _simulationReadyAt = 0f;
                return true;
            }
            catch (Exception ex)
            {
                error = ex.Message;
                _logger.LogError("Could not prepare GPU checkpoint load: " + ex);
                return false;
            }
        }

        internal void PrepareStockWorldLoad()
        {
            StopWorld();
            _pendingCheckpointPath = null;
            _sceneAttempted = true;
            _simulationReadyAt = 0f;
        }

        internal void ResetForSceneChange()
        {
            StopWorld();
            _pendingCheckpointPath = null;
            _sceneAttempted = false;
            _simulationReadyAt = 0f;
        }

        internal bool QueuePlacedBibite(
            BibiteTemplate template,
            RandomizeGenes randomizeGenes,
            Tagging tagging,
            GrowthAtSpawn growth,
            Vector3 position,
            float? angleDegrees,
            string customTag)
        {
            if (!_presentationActive || _runner == null || !_runner.IsReady)
            {
                return false;
            }
            float heading = angleDegrees.HasValue
                ? angleDegrees.Value * Mathf.Deg2Rad
                : UnityEngine.Random.Range(0f, Mathf.PI * 2f);
            NativeWorldTemplate projected;
            string projectionError;
            if (!NativeTemplateProjection.TryCreate(
                    template,
                    randomizeGenes,
                    tagging,
                    growth,
                    position,
                    heading,
                    customTag,
                    out projected,
                    out projectionError))
            {
                _logger.LogWarning("Could not project the placed Bibite into CUDA: " + projectionError);
                PopupManager.DisplayError(
                    "GPU Bibite placement",
                    "This Bibite was not placed because " + projectionError + ".");
                return true;
            }
            bool queued = _runner.QueueSpawn(projected);
            if (queued)
            {
                _lineageNames[projected.Spawn.LineageId] = projected.SpeciesName;
                _tagNames[projected.Spawn.TagId] = projected.TagName;
                _logger.LogInfo("Routed placed template '" + projected.TemplateName +
                    "' into the GPU-native world at " + position.x.ToString("0.0") + ", " +
                    position.y.ToString("0.0") + " with " + projected.Nodes.Length +
                    " nodes and " + projected.Synapses.Length + " enabled synapses.");
            }
            else
            {
                _logger.LogWarning("Could not queue the placed Bibite for the GPU world: " +
                    (_runner.LastPlacementError ?? "GPU world is not ready"));
                PopupManager.DisplayError(
                    "GPU Bibite placement",
                    "The selected Bibite could not be queued: " +
                    (_runner.LastPlacementError ?? "GPU world is not ready") + ".");
            }
            return true;
        }

        internal bool TrySelectSpeciesHandle(SpeciesInfoHandle handle)
        {
            ulong lineageId;
            if (!_presentationActive || handle == null ||
                !_speciesHandleLineages.TryGetValue(handle.GetInstanceID(), out lineageId))
            {
                return false;
            }
            SelectFirstMatching(delegate(NativeWorldBibite bibite)
            {
                return bibite.LineageId == lineageId;
            });
            return true;
        }

        internal bool TrySelectTagHandle(TagElementHandle handle)
        {
            ulong tagId;
            if (!_presentationActive || handle == null ||
                !_tagHandleTags.TryGetValue(handle.GetInstanceID(), out tagId))
            {
                return false;
            }
            SelectFirstMatching(delegate(NativeWorldBibite bibite)
            {
                return bibite.TagId == tagId;
            });
            return true;
        }

        internal bool TrySelectAtScreenPoint(Vector3 screenPoint)
        {
            if (!_presentationActive || _renderer == null || _lastSnapshot == null)
            {
                return false;
            }
            if (UnityEngine.EventSystems.EventSystem.current != null &&
                UnityEngine.EventSystems.EventSystem.current.IsPointerOverGameObject())
            {
                return true;
            }
            Vector2 guiPoint = new Vector2(screenPoint.x, Screen.height - screenPoint.y);
            if ((_hasSelection && (_stockInspector == null || !_stockInspector.Ready) && _selectedBibiteWindow.Contains(guiPoint)) ||
                (_nativeSpeciesPanelOpen && _nativeSpeciesWindow.Contains(guiPoint)))
            {
                return true;
            }
            NativeWorldBibite selected;
            bool found = _renderer.TrySelectAtScreenPoint(
                screenPoint,
                _lastSnapshot.VisibleBibites,
                _lastSnapshot.VisibleBibiteCount,
                out selected);
            if (!found)
            {
                found = _renderer.TrySelectAtScreenPoint(
                    screenPoint,
                    _lastSnapshot.Bibites,
                    _lastSnapshot.BibiteCount,
                    out selected);
            }
            if (found)
            {
                SelectBibite(selected);
            }
            else
            {
                ClearSelection();
            }
            return true;
        }

        internal bool OpenNativeSpeciesPanel()
        {
            if (!_presentationActive)
            {
                return false;
            }
            if (_lastSnapshot != null)
            {
                _lineageBreakdown = BuildBreakdown(
                    _lastSnapshot,
                    delegate(NativeWorldBibite bibite) { return bibite.LineageId; },
                    LineageName);
            }
            _nativeSpeciesPanelOpen = true;
            if (_stockInspector == null || !_stockInspector.Ready)
            {
                if (_stockInspector != null) _stockInspector.Dispose();
                _stockInspector = new GpuNativeInspector(this);
            }
            _stockInspector.ShowSpecies(_lineageBreakdown,
                _lastSnapshot != null && _lastSnapshot.Stats.LivingBibites > Math.Min(_lastSnapshot.BibiteCount, MaximumPresentationSamples));
            _nativeSpeciesWindow.x = Mathf.Clamp(
                _nativeSpeciesWindow.x,
                8f,
                Mathf.Max(8f, Screen.width - _nativeSpeciesWindow.width - 8f));
            _nativeSpeciesWindow.y = Mathf.Clamp(
                _nativeSpeciesWindow.y,
                8f,
                Mathf.Max(8f, Screen.height - _nativeSpeciesWindow.height - 8f));
            return true;
        }

        internal void SelectNativeLineage(ulong lineageId)
        {
            SelectFirstMatching(delegate(NativeWorldBibite bibite) { return bibite.LineageId == lineageId; });
        }

        internal void SelectNativeSnapshot(NativeWorldBibite bibite) { SelectBibite(bibite); }

        internal bool TrySelectNativeRectangle(Vector2 first, Vector2 second)
        {
            if (!_presentationActive || _lastSnapshot == null) return false;
            if (NativeUiVisibility.BlockingPanelOpen || !UserControl.AllowControl) return true;
            Camera camera = Camera.main;
            if (camera == null) return true;
            Rect rectangle = Rect.MinMaxRect(Mathf.Min(first.x, second.x), Mathf.Min(first.y, second.y),
                Mathf.Max(first.x, second.x), Mathf.Max(first.y, second.y));
            List<NativeWorldBibite> selected = new List<NativeWorldBibite>();
            HashSet<int> selectedSlots = new HashSet<int>();
            int visible = Math.Min(_lastSnapshot.VisibleBibiteCount,
                _lastSnapshot.VisibleBibites.Length);
            for (int i = 0; i < visible; i++)
            {
                NativeWorldBibite bibite = _lastSnapshot.VisibleBibites[i];
                Vector3 point = camera.WorldToScreenPoint(new Vector3(bibite.PositionX, bibite.PositionY, 0f));
                if (point.z > 0f && rectangle.Contains(new Vector2(point.x, point.y)))
                {
                    selected.Add(bibite);
                    selectedSlots.Add(bibite.Slot);
                }
            }
            int available = Math.Min(_lastSnapshot.BibiteCount, _lastSnapshot.Bibites.Length);
            for (int i = 0; i < available; i++)
            {
                NativeWorldBibite bibite = _lastSnapshot.Bibites[i];
                Vector3 point = camera.WorldToScreenPoint(new Vector3(bibite.PositionX, bibite.PositionY, 0f));
                if (point.z > 0f && rectangle.Contains(new Vector2(point.x, point.y)) &&
                    selectedSlots.Add(bibite.Slot)) selected.Add(bibite);
            }
            if (string.Equals(Environment.GetEnvironmentVariable("BIBITES_GPU_UI_TRACE"), "1", StringComparison.Ordinal))
                _logger.LogInfo("Native rectangle selection: " + selected.Count + " of " + available +
                    " snapshot Bibites, screen rectangle " + rectangle + ".");
            if (selected.Count == 1) { SelectBibite(selected[0]); return true; }
            ClearSelection();
            if (selected.Count == 0 && available >= _lastSnapshot.Stats.LivingBibites) return true;
            if (_stockInspector == null || !_stockInspector.Ready)
            {
                if (_stockInspector != null) _stockInspector.Dispose();
                _stockInspector = new GpuNativeInspector(this);
            }
            _stockInspector.ShowRectangleSelection(selected, available < _lastSnapshot.Stats.LivingBibites);
            return true;
        }

        internal void CloseNativeSpeciesPanel() { _nativeSpeciesPanelOpen = false; }

        internal bool SelectFirstBibiteForCapture()
        {
            if (!_presentationActive || _lastSnapshot == null ||
                _lastSnapshot.BibiteCount <= 0)
            {
                return false;
            }
            SelectBibite(_lastSnapshot.Bibites[0]);
            return true;
        }

        internal string StatusText
        {
            get
            {
                if (_runner == null)
                {
                    if (_retiredWorlds.Count > 0)
                        return "Finishing the previous GPU world safely...";
                    return "GPU visual world waiting for a new simulation";
                }
                if (!string.IsNullOrEmpty(_runner.Error))
                {
                    return "GPU visual world failed: " + FirstLine(_runner.Error);
                }
                if (!_presentationActive)
                {
                    return "Starting GPU visual world on " + _runner.DeviceName + "...";
                }
                if (_lastSnapshot == null)
                {
                    return "GPU visual world is waiting for its first frame";
                }
                return "GPU visual world | " + _runner.DeviceName + " | " +
                    _lastSnapshot.Stats.LivingBibites + " / " +
                    _populationCap +
                    " Bibites | " + _lastSnapshot.AchievedMultiplier.ToString("0.0") +
                    "x achieved | " + (_extremeMode ? "Extreme" : "Exact") + " profile";
            }
        }

        internal string FoodStatusText
        {
            get
            {
                if (!_presentationActive || _runner == null)
                {
                    return "Stock Dynamic Settings will control GPU food while a simulation runs.";
                }
                return "GPU plants: " + ActivePellets + " active / " +
                    _runner.CurrentFoodTarget + " cap (GPU ceiling " +
                    _configuredFoodCap + "); regrowth " +
                    _runner.CurrentFoodGrowthFactor.ToString("0.00") +
                    "x; pellet energy " +
                    _runner.CurrentPelletEnergy.ToString("0.0") + ".";
            }
        }

        internal int FoodTarget
        {
            get { return _runner != null ? _runner.CurrentFoodTarget : 0; }
        }

        internal void Update(
            bool enabled,
            bool allowLoadedSaves,
            int deviceIndex,
            int populationCap,
            int initialPopulation,
            int pelletCount,
            int renderFps,
            int detailedBibiteLimit,
            bool extremeMode,
            int brainKernel)
        {
            ReleaseStoppedWorldResources();
            if (_runner != null) _runner.PumpRenderThreadEvent();
            if (_applicationQuitRequested) return;
            _renderFps = Mathf.Clamp(renderFps, 10, 60);
            _detailedBibiteLimit = Mathf.Clamp(
                detailedBibiteLimit,
                0,
                BibiteDetailedSpritePool.MaximumDetailedBibites);
            if (!_loadedCheckpointActive && string.IsNullOrEmpty(_pendingCheckpointPath))
            {
                _extremeMode = extremeMode;
            }
            if (!GameManager.isSim)
            {
                if (_runner != null || _presentationActive)
                {
                    StopWorld();
                }
                _sceneAttempted = false;
                _simulationReadyAt = 0f;
                return;
            }

            // Avoid allocating a second large world while the prior worker is
            // still saving/releasing GPU memory. Unity menus remain responsive.
            if (_runner == null && _retiredWorlds.Count > 0) return;

            bool checkpointRequested = !string.IsNullOrEmpty(_pendingCheckpointPath);
            if (!enabled || (!checkpointRequested && !_loadedCheckpointActive &&
                !allowLoadedSaves && SimulationManager.gameWasLoaded))
            {
                return;
            }
            if (SimulationManager.Instance == null || WorldObjectsSpawner.Instance == null ||
                ZoneManager.instance == null || TimeController.Instance == null)
            {
                return;
            }
            if (_simulationReadyAt <= 0f)
            {
                _simulationReadyAt = Time.realtimeSinceStartup;
                return;
            }
            if (!_sceneAttempted)
            {
                _sceneAttempted = true;
                StartWorld(
                    deviceIndex,
                    populationCap,
                    initialPopulation,
                    pelletCount,
                    extremeMode,
                    brainKernel,
                    _pendingCheckpointPath);
            }
            if (_runner == null)
            {
                return;
            }
            if (!string.IsNullOrEmpty(_runner.Error))
            {
                // A driver failure may also stop simulation before the next
                // Update. Still retire a completed failed graphics handoff;
                // never leave an errored back buffer in the presentation path.
                string stoppedRenderError;
                if (_renderer != null && !_renderer.PumpDirectInterop(
                        _runner, false, out stoppedRenderError) && !_directInteropFaultReported)
                {
                    _directInteropFaultReported = true;
                    _logger.LogError("Stopped GPU graphics handoff failed safely: " + stoppedRenderError);
                    if (_lastSnapshot != null) _renderer.Render(_lastSnapshot);
                }
                return;
            }
            if (!_presentationActive && _runner.IsReady)
            {
                ActivatePresentation();
            }
            if (!_presentationActive)
            {
                return;
            }

            SyncLiveFoodSettings();
            _runner.SetVisibleViewport(Camera.main);
            _renderer.RefreshViewport();

            if (_runner.FailedPlacements > _lastReportedPlacementFailures)
            {
                _lastReportedPlacementFailures = _runner.FailedPlacements;
                PopupManager.DisplayError(
                    "GPU Bibite placement",
                    "The Bibite stayed out of the CPU world but could not enter CUDA: " +
                    (_runner.LastPlacementError ?? "placement failed") + ".");
            }

            float target = Mathf.Clamp(
                TimeWarpSpeeds.Snap(TimeController.targetTimeScale.val),
                1f,
                TimeWarpSpeeds.Maximum);
            _runner.SetControl(Mathf.RoundToInt(target), TimeController.paused, _renderFps);
            if (!TimeController.paused &&
                !Mathf.Approximately(TimeController.engineTimeScale.val, target))
            {
                // Keep the stock speed readout truthful. A Harmony prefix keeps
                // Unity's presentation clock at 1x while the native runner uses
                // this selected multiplier independently.
                TimeController.engineTimeScale.SetValue(target);
            }

            float renderNow = Time.realtimeSinceStartup;
            bool requestRenderFrame = renderNow >= _nextDirectRenderAt;
            string renderError;
            if (!_renderer.PumpDirectInterop(_runner, requestRenderFrame, out renderError) &&
                !_directInteropFaultReported)
            {
                _directInteropFaultReported = true;
                _lastRenderedSequence = -1;
                _logger.LogError("CUDA/Direct3D shared rendering was disabled safely; " +
                    "using separate CPU display buffers: " + renderError);
            }
            if (requestRenderFrame)
            {
                float renderInterval = 1f / Math.Max(1, _renderFps);
                _nextDirectRenderAt += renderInterval;
                if (_nextDirectRenderAt < renderNow - renderInterval)
                {
                    _nextDirectRenderAt = renderNow + renderInterval;
                }
            }
            _runner.WithLatest(_lastRenderedSequence, delegate(GpuWorldSnapshot snapshot)
            {
                _lastSnapshot = snapshot;
                _maximumSnapshotMilliseconds = Math.Max(
                    _maximumSnapshotMilliseconds,
                    snapshot.SnapshotMilliseconds);
                _lastRenderedSequence = snapshot.Sequence;
                TimeKeeper.simulatedTime = snapshot.Stats.SimulatedSeconds;
                SyncOriginalTimeDisplay(snapshot);
                if (_nativeSpeciesPanelOpen && InformationPanel.Instance == null)
                {
                    _lineageBreakdown = BuildBreakdown(
                        snapshot,
                        delegate(NativeWorldBibite bibite) { return bibite.LineageId; },
                        LineageName);
                }
                SyncOriginalInformation(snapshot);
                if (_lastChartSummarySequence != snapshot.SummarySequence)
                {
                    SyncHistoricalCharts(snapshot);
                    _lastChartSummarySequence = snapshot.SummarySequence;
                }
                _renderer.Render(snapshot);
                if (_hasSelection)
                {
                    RefreshSelected(snapshot);
                }
            });

            if (_hasSelection)
            {
                _runner.WithSelectedDetail(
                    _selectedBibite.Slot,
                    _selectedDetailSequence,
                    delegate(NativeWorldSelectedBibite selected)
                    {
                        _selectedDetailSequence = selected.Sequence;
                        if (_selectedInstanceToken != 0 &&
                            _selectedInstanceToken != selected.Detail.Reserved)
                        {
                            ClearSelection();
                            return;
                        }
                        _selectedDetail = selected.Detail;
                        _selectedInstanceToken = _selectedDetail.Reserved;
                        _selectedBrainNodes = selected.Nodes ??
                            new NativeWorldBrainNodeState[0];
                        _selectedBrainSynapses = selected.Synapses ??
                            new NativeWorldBrainSynapseState[0];
                        if (_selectedDetail.Alive != 1)
                        {
                            ClearSelection();
                            return;
                        }
                        ApplyDetailToSelection(_selectedDetail);
                    });
                UpdateSelectionControls();
            }

            if (_stockInspector != null) _stockInspector.Tick();
            UpdateNativeSelectionHotkeys();

        }

        internal void DrawOverlay(bool showDiagnostics)
        {
            if (_runner == null)
            {
                return;
            }
            if (!UserControl.AllowControl || NativeUiVisibility.BlockingPanelOpen ||
                (UserControl.Instance != null && UserControl.Instance.mainUI != null &&
                 !UserControl.Instance.mainUI.activeInHierarchy))
            {
                return;
            }
            if (showDiagnostics && !_hasSelection && !_nativeSpeciesPanelOpen)
            {
                GUI.Box(new Rect(12f, 12f, 600f, 28f), StatusText);
                if (_presentationActive && _lastSnapshot != null)
                {
                    string simulation = (_lastSnapshot.Metrics.Pipeline == 2
                        ? "FP32 physics / tensor FP16-FP32"
                        : "FP32 physics / FP16 genes") + " | requested " +
                        TimeController.targetTimeScale.val.ToString("0") + "x | graphics " +
                        _renderFps + " FPS | kernel share " +
                        _lastSnapshot.Metrics.GpuOffloadPercent.ToString("0.00") + "% | placed " +
                        _runner.SuccessfulPlacements + " | rejected " + _runner.FailedPlacements;
                    GUI.Box(new Rect(12f, 42f, 600f, 28f), simulation);
                    NativeWorldStepMetrics phases = _lastSnapshot.Metrics;
                    string phaseTiming = "GPU batch ms | prep " +
                        phases.PrepareMilliseconds.ToString("0.00") + " | spatial " +
                        phases.SpatialIndexMilliseconds.ToString("0.00") + " | contact " +
                        phases.ContactMilliseconds.ToString("0.00") + " | decision " +
                        phases.DecisionMilliseconds.ToString("0.00") + " | motion " +
                        phases.MotionMilliseconds.ToString("0.00") + " | lifecycle " +
                        phases.LifecycleMilliseconds.ToString("0.00");
                    GUI.Box(new Rect(12f, 72f, 850f, 28f), phaseTiming);
                    GUI.Box(new Rect(12f, 102f, 600f, 28f),
                        "Original menus/camera remain active. GPU saves use a .zip wrapper plus a .bgfgpu checkpoint.");
                    ulong classifiedDeaths = _lastSnapshot.Stats.StarvationDeaths +
                        _lastSnapshot.Stats.AgeDeaths +
                        _lastSnapshot.Stats.InvalidStateDeaths;
                    ulong damageDeaths = _lastSnapshot.Stats.Deaths > classifiedDeaths
                        ? _lastSnapshot.Stats.Deaths - classifiedDeaths : 0;
                    GUI.Box(new Rect(12f, 132f, 850f, 28f),
                        "Deaths | starvation " + _lastSnapshot.Stats.StarvationDeaths +
                        " | age " + _lastSnapshot.Stats.AgeDeaths +
                        " | damage/other " + damageDeaths +
                        " | invalid state " + _lastSnapshot.Stats.InvalidStateDeaths);
                    GUI.Box(new Rect(12f, 162f, 850f, 28f),
                        "Step call " + phases.WallMilliseconds.ToString("0.00") + " ms | GPU " +
                        phases.GpuMilliseconds.ToString("0.00") + " ms | other overhead " +
                        phases.HostOverheadMilliseconds.ToString("0.00") + " ms | snapshot " +
                        _lastSnapshot.SnapshotMilliseconds.ToString("0.00") + " ms");
                    if (phases.Pipeline != 0)
                    {
                        GUI.Box(new Rect(12f, 192f, 850f, 28f),
                            "Graph intervals ms | food/senses/imported brains " +
                            phases.FoodMilliseconds.ToString("0.00") + " | native brain " +
                            phases.BrainMilliseconds.ToString("0.00") + " | feeding/attack " +
                            phases.PostDecisionMilliseconds.ToString("0.00"));
                        GUI.Box(new Rect(12f, 222f, 850f, 28f),
                            "Graph build (CPU) " + phases.GraphBuildMilliseconds.ToString("0.00") +
                            " ms | graph cache hits " + phases.GraphCacheHits);
                    }
                }
            }
            if (_presentationActive && _hasSelection)
            {
                DrawSelectionMarker();
                if ((_stockInspector == null || !_stockInspector.Ready) && _selectedBibiteWindow.x <= 0f)
                {
                    // Leave the stock Information panel and its wide value rows unobscured.
                    _selectedBibiteWindow.x = Mathf.Clamp(
                        Screen.width - 1040f,
                        8f,
                        Mathf.Max(8f, Screen.width - _selectedBibiteWindow.width - 8f));
                }
                if (_stockInspector == null || !_stockInspector.Ready) _selectedBibiteWindow = GUILayout.Window(
                    164802,
                    _selectedBibiteWindow,
                    DrawSelectedBibiteWindow,
                    "GPU Bibite #" + _selectedBibite.Slot,
                    GUILayout.Width(620f),
                    GUILayout.Height(560f));
            }
            if (_presentationActive && _nativeSpeciesPanelOpen && (_stockInspector == null || !_stockInspector.SpeciesReady))
            {
                _nativeSpeciesWindow = GUILayout.Window(
                    164803,
                    _nativeSpeciesWindow,
                    DrawNativeSpeciesWindow,
                    "GPU Species",
                    GUILayout.Width(560f),
                    GUILayout.Height(520f));
            }
        }

        private void DrawSelectionMarker()
        {
            Camera camera = Camera.main;
            if (camera == null)
            {
                return;
            }
            Vector3 screen = camera.WorldToScreenPoint(new Vector3(
                _selectedBibite.PositionX,
                _selectedBibite.PositionY,
                0f));
            if (screen.z <= 0f)
            {
                return;
            }
            if (_stockInspector != null && _stockInspector.CoversScreenPoint(new Vector2(screen.x, screen.y))) return;
            float x = screen.x;
            float y = Screen.height - screen.y;
            const float half = 14f;
            const float thickness = 2f;
            Color old = GUI.color;
            GUI.color = Color.yellow;
            GUI.DrawTexture(new Rect(x - half, y - half, half * 0.65f, thickness), Texture2D.whiteTexture);
            GUI.DrawTexture(new Rect(x + half * 0.35f, y - half, half * 0.65f, thickness), Texture2D.whiteTexture);
            GUI.DrawTexture(new Rect(x - half, y + half, half * 0.65f, thickness), Texture2D.whiteTexture);
            GUI.DrawTexture(new Rect(x + half * 0.35f, y + half, half * 0.65f, thickness), Texture2D.whiteTexture);
            GUI.DrawTexture(new Rect(x - half, y - half, thickness, half * 0.65f), Texture2D.whiteTexture);
            GUI.DrawTexture(new Rect(x - half, y + half * 0.35f, thickness, half * 0.65f), Texture2D.whiteTexture);
            GUI.DrawTexture(new Rect(x + half, y - half, thickness, half * 0.65f), Texture2D.whiteTexture);
            GUI.DrawTexture(new Rect(x + half, y + half * 0.35f, thickness, half * 0.65f), Texture2D.whiteTexture);
            GUI.color = old;
        }

        private void DrawSelectedBibiteWindow(int windowId)
        {
            GUILayout.BeginHorizontal();
            GUILayout.Label(LineageName(_selectedBibite.LineageId), GUILayout.ExpandWidth(true));
            GUILayout.Label("Generation " + _selectedBibite.Generation, GUILayout.ExpandWidth(false));
            GUILayout.EndHorizontal();

            GUILayout.BeginHorizontal();
            DrawInspectorTabButton(InspectorTab.Stats, "Stats [1]");
            DrawInspectorTabButton(InspectorTab.Genes, "Genes [2]");
            DrawInspectorTabButton(InspectorTab.Biology, "Biology [3]");
            DrawInspectorTabButton(InspectorTab.Brain, "Brain [4]");
            DrawInspectorTabButton(InspectorTab.ExpandedBrain, "Expanded [5]");
            GUILayout.EndHorizontal();

            _selectedInspectorScroll = GUILayout.BeginScrollView(
                _selectedInspectorScroll,
                GUILayout.Height(390f));
            switch (_selectedInspectorTab)
            {
                case InspectorTab.Genes:
                    DrawSelectedGenes();
                    break;
                case InspectorTab.Biology:
                    DrawSelectedBiology();
                    break;
                case InspectorTab.Brain:
                    DrawSelectedBrain();
                    break;
                case InspectorTab.ExpandedBrain:
                    DrawExpandedSelectedBrain();
                    break;
                default:
                    DrawSelectedStats();
                    break;
            }
            GUILayout.EndScrollView();

            string commandStatus = _runner != null ? _runner.LastBibiteCommandStatus : null;
            if (!string.IsNullOrEmpty(_selectedInspectorNotice))
            {
                GUILayout.Label(_selectedInspectorNotice);
            }
            else if (!string.IsNullOrEmpty(commandStatus))
            {
                GUILayout.Label(commandStatus);
            }
            GUILayout.BeginHorizontal();
            if (GUILayout.Button(_followSelection ? "Stop following" : "Follow"))
            {
                _followSelection = !_followSelection;
            }
            if (GUILayout.Button("Species list"))
            {
                _nativeSpeciesPanelOpen = true;
            }
            if (GUILayout.Button("Close"))
            {
                ClearSelection();
            }
            GUILayout.EndHorizontal();
            GUI.DragWindow(new Rect(0f, 0f, 10000f, 24f));
        }

        private void DrawInspectorTabButton(InspectorTab tab, string label)
        {
            bool selected = _selectedInspectorTab == tab;
            bool previous = GUI.enabled;
            GUI.enabled = !selected;
            if (GUILayout.Button(label, GUILayout.Height(28f)))
            {
                SelectInspectorTab(tab);
            }
            GUI.enabled = previous;
        }

        private void DrawSelectedStats()
        {
            float speed = Mathf.Sqrt(
                _selectedBibite.VelocityX * _selectedBibite.VelocityX +
                _selectedBibite.VelocityY * _selectedBibite.VelocityY);
            GUILayout.Label("Live GPU state");
            GUILayout.Label("Species: " + LineageName(_selectedBibite.LineageId));
            GUILayout.Label("Tag: " + TagName(_selectedBibite.TagId));
            GUILayout.Label("Energy: " + _selectedBibite.Energy.ToString("0.00"));
            GUILayout.Label("Age: " + _selectedBibite.Age.ToString("0.0") +
                " simulated seconds");
            GUILayout.Label("Size: " + _selectedBibite.Size.ToString("0.00"));
            GUILayout.Label("Speed: " + speed.ToString("0.00"));
            GUILayout.Label("Heading: " +
                (_selectedBibite.Heading * Mathf.Rad2Deg).ToString("0.0") + " degrees");
            GUILayout.Label("Brain: " + _selectedBibite.BrainNodes + " nodes / " +
                _selectedBibite.BrainSynapses + " synapses");
            GUILayout.Label("Position: " + _selectedBibite.PositionX.ToString("0.0") + ", " +
                _selectedBibite.PositionY.ToString("0.0"));

            GUILayout.Space(8f);
            GUILayout.Label("Tag editor");
            GUILayout.BeginHorizontal();
            GUI.SetNextControlName("GpuBibiteTagEditor");
            _selectedTagEditor = GUILayout.TextField(
                _selectedTagEditor ?? string.Empty,
                100,
                GUILayout.ExpandWidth(true));
            if (GUILayout.Button("Apply", GUILayout.Width(90f)))
            {
                ApplySelectedTag();
            }
            if (GUILayout.Button("Copy", GUILayout.Width(80f)))
            {
                GUIUtility.systemCopyBuffer = TagName(_selectedBibite.TagId);
            }
            GUILayout.EndHorizontal();
            UserControl.SetKeyboardBlockFromSource(
                "GpuBibiteTagEditor",
                string.Equals(
                    GUI.GetNameOfFocusedControl(),
                    "GpuBibiteTagEditor",
                    StringComparison.Ordinal));

            GUILayout.Space(8f);
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Lay egg"))
            {
                QueueSelectedReproduction();
            }
            if (GUILayout.Button("Remove Bibite"))
            {
                QueueSelectedKill();
            }
            GUILayout.EndHorizontal();
        }

        private void DrawSelectedGenes()
        {
            GUILayout.Label("GPU-native inherited traits");
            GUILayout.Label("Body size: " + _selectedDetail.Size.ToString("0.000"));
            GUILayout.Label("Maximum speed: " + _selectedDetail.MaximumSpeed.ToString("0.000"));
            GUILayout.Label("Turn speed: " + _selectedDetail.TurnSpeed.ToString("0.000"));
            GUILayout.Label("Metabolism: " + _selectedDetail.Metabolism.ToString("0.0000") +
                " energy/s");
            GUILayout.Label("Lifespan: " + _selectedDetail.Lifespan.ToString("0.0") +
                " simulated seconds");
            GUILayout.Label("Gene mutation strength: " +
                _selectedDetail.GeneMutationStrength.ToString("0.0000"));
            GUILayout.Label("Brain mutation strength: " +
                _selectedDetail.BrainMutationStrength.ToString("0.0000"));
            GUILayout.Label("Colour: R " + _selectedDetail.ColorR.ToString("0.000") +
                "  G " + _selectedDetail.ColorG.ToString("0.000") +
                "  B " + _selectedDetail.ColorB.ToString("0.000"));
            Rect swatch = GUILayoutUtility.GetRect(80f, 28f, GUILayout.ExpandWidth(false));
            Color previous = GUI.color;
            GUI.color = new Color(
                Mathf.Clamp01(_selectedDetail.ColorR),
                Mathf.Clamp01(_selectedDetail.ColorG),
                Mathf.Clamp01(_selectedDetail.ColorB),
                1f);
            GUI.DrawTexture(swatch, Texture2D.whiteTexture);
            GUI.color = previous;
            GUILayout.Space(8f);
            GUILayout.Label(
                "Only traits represented by the CUDA evolution model are shown. " +
                "Stock-only organ genes are intentionally not fabricated.");
        }

        private void DrawSelectedBiology()
        {
            float speed = Mathf.Sqrt(
                _selectedDetail.VelocityX * _selectedDetail.VelocityX +
                _selectedDetail.VelocityY * _selectedDetail.VelocityY);
            float lifeRatio = _selectedDetail.Lifespan > 0f
                ? _selectedDetail.Age / _selectedDetail.Lifespan
                : 0f;
            GUILayout.Label("Lifecycle");
            GUILayout.Label("Energy: " + _selectedDetail.Energy.ToString("0.00"));
            GUILayout.Label("Age / lifespan: " + _selectedDetail.Age.ToString("0.0") + " / " +
                _selectedDetail.Lifespan.ToString("0.0") + " (" +
                (lifeRatio * 100f).ToString("0.0") + "%)");
            GUILayout.Label("Metabolic drain: " +
                _selectedDetail.Metabolism.ToString("0.0000") + " energy/s");
            GUILayout.Label("Reproduction cooldown: " +
                _selectedDetail.ReproductionCooldown.ToString("0.00") + " s");

            GUILayout.Space(8f);
            GUILayout.Label("Movement and perception");
            GUILayout.Label("Speed / maximum: " + speed.ToString("0.00") + " / " +
                _selectedDetail.MaximumSpeed.ToString("0.00"));
            GUILayout.Label("Acceleration output: " +
                _selectedDetail.AccelerationOutput.ToString("0.000"));
            GUILayout.Label("Rotation output: " +
                _selectedDetail.RotationOutput.ToString("0.000"));
            GUILayout.Label("Food target: " + (_selectedDetail.CachedFood >= 0
                ? "pellet #" + _selectedDetail.CachedFood + " at " +
                    Mathf.Sqrt(Mathf.Max(0f, _selectedDetail.FoodDistanceSquared)).ToString("0.0") + " units"
                : "none"));
            GUILayout.Label("Nearby Bibites seen: " + _selectedDetail.NeighboursSeen);
            GUILayout.Label("Nearest neighbour: " + (_selectedDetail.CachedNeighbour >= 0
                ? "#" + _selectedDetail.CachedNeighbour + " at " +
                    Mathf.Sqrt(Mathf.Max(0f, _selectedDetail.NeighbourDistanceSquared)).ToString("0.0") + " units"
                : "none"));

            GUILayout.Space(8f);
            GUILayout.Label("Pheromone outputs: " +
                _selectedDetail.Pheromone1Output.ToString("0.000") + " / " +
                _selectedDetail.Pheromone2Output.ToString("0.000") + " / " +
                _selectedDetail.Pheromone3Output.ToString("0.000"));
            GUILayout.Label("Reproduction output: " +
                _selectedDetail.ReproductionOutput.ToString("0.000"));
        }

        private void DrawSelectedBrain()
        {
            GUILayout.Label(_selectedDetail.TemplateBrain == 1
                ? "Placed/evolved stock topology"
                : _selectedDetail.TemplateBrain == 2
                    ? "Evolving CUDA 34 x 12 x 12 x 15 topology"
                    : "Legacy CUDA 16 x 12 x 12 x 6 topology");
            GUILayout.Label(_selectedDetail.BrainNodes + " nodes / " +
                _selectedDetail.BrainSynapses + " enabled synapses");
            GUILayout.Label("Weights: FP16 storage with FP32 accumulation");
            GUILayout.Space(8f);
            GUILayout.Label("Live action outputs");
            GUILayout.Label("Accelerate: " + _selectedDetail.AccelerationOutput.ToString("0.000"));
            GUILayout.Label("Rotate: " + _selectedDetail.RotationOutput.ToString("0.000"));
            GUILayout.Label("Pheromone 1: " + _selectedDetail.Pheromone1Output.ToString("0.000"));
            GUILayout.Label("Pheromone 2: " + _selectedDetail.Pheromone2Output.ToString("0.000"));
            GUILayout.Label("Pheromone 3: " + _selectedDetail.Pheromone3Output.ToString("0.000"));
            GUILayout.Label("Lay egg: " + _selectedDetail.ReproductionOutput.ToString("0.000"));
            GUILayout.Space(8f);
            int preview = Math.Min(12, _selectedBrainNodes.Length);
            GUILayout.Label("Active node preview");
            for (int index = 0; index < preview; index++)
            {
                NativeWorldBrainNodeState node = _selectedBrainNodes[index];
                GUILayout.Label(NodeLabel(index, node));
            }
            if (_selectedBrainNodes.Length > preview)
            {
                GUILayout.Label("Open Expanded [5] for every node and synapse.");
            }
        }

        private void DrawExpandedSelectedBrain()
        {
            GUILayout.BeginHorizontal();
            bool previous = GUI.enabled;
            GUI.enabled = _expandedBrainShowsSynapses;
            if (GUILayout.Button("Nodes"))
            {
                _expandedBrainShowsSynapses = false;
                _expandedBrainPage = 0;
            }
            GUI.enabled = !_expandedBrainShowsSynapses;
            if (GUILayout.Button("Synapses"))
            {
                _expandedBrainShowsSynapses = true;
                _expandedBrainPage = 0;
            }
            GUI.enabled = previous;
            GUILayout.EndHorizontal();

            const int pageSize = 24;
            int count = _expandedBrainShowsSynapses
                ? _selectedBrainSynapses.Length
                : _selectedBrainNodes.Length;
            int pages = Math.Max(1, (count + pageSize - 1) / pageSize);
            _expandedBrainPage = Mathf.Clamp(_expandedBrainPage, 0, pages - 1);
            GUILayout.BeginHorizontal();
            GUI.enabled = _expandedBrainPage > 0;
            if (GUILayout.Button("Previous")) _expandedBrainPage--;
            GUI.enabled = _expandedBrainPage + 1 < pages;
            if (GUILayout.Button("Next")) _expandedBrainPage++;
            GUI.enabled = previous;
            GUILayout.Label("Page " + (_expandedBrainPage + 1) + " / " + pages,
                GUILayout.Width(100f));
            GUILayout.EndHorizontal();

            int start = _expandedBrainPage * pageSize;
            int end = Math.Min(count, start + pageSize);
            if (_expandedBrainShowsSynapses)
            {
                for (int index = start; index < end; index++)
                {
                    NativeWorldBrainSynapseState synapse = _selectedBrainSynapses[index];
                    GUILayout.Label("#" + index + "  node " + synapse.NodeIn + " -> " +
                        synapse.NodeOut + "    weight " + synapse.Weight.ToString("0.00000"));
                }
            }
            else
            {
                for (int index = start; index < end; index++)
                {
                    GUILayout.Label(NodeLabel(index, _selectedBrainNodes[index]));
                }
            }
            if (count == 0)
            {
                GUILayout.Label("Waiting for the selected-brain download...");
            }
        }

        internal string NativeNodeLabel(int index, NativeWorldBrainNodeState node)
        {
            return NodeLabel(index, node);
        }

        private string NodeLabel(int index, NativeWorldBrainNodeState node)
        {
            string role;
            if (node.Sensor >= 0)
            {
                role = "sensor " + SensorName(node.Sensor, _selectedDetail.TemplateBrain != 0);
            }
            else if (node.Action >= 0)
            {
                role = "action " + ActionName(node.Action);
            }
            else
            {
                role = "hidden";
            }
            if (_selectedDetail.TemplateBrain == 0)
            {
                if (node.Sensor >= 0)
                    return "#" + index + "  " + role + "    value not retained by the compact engine";
                return "#" + index + "  " + role + "    out " + node.LastOutput.ToString("0.000") +
                    "  (pre-activation not retained)";
            }
            if (_selectedDetail.TemplateBrain == 2 && node.Sensor >= 0)
            {
                return "#" + index + "  " + role + "    live value " + node.LastOutput.ToString("0.000");
            }
            return "#" + index + "  " + NodeTypeName(node.Type) + "  " + role +
                "    bias " + node.BaseActivation.ToString("0.000") +
                "  in " + node.LastInput.ToString("0.000") +
                "  out " + node.LastOutput.ToString("0.000");
        }

        private static string NodeTypeName(int type)
        {
            string[] names =
            {
                "Input", "Sigmoid", "Linear", "TanH", "Sine", "ReLu", "Gaussian",
                "Latch", "Differential", "Abs", "Mult", "Integrator", "Inhibitory", "SoftLatch"
            };
            return type >= 0 && type < names.Length ? names[type] : "Node " + type;
        }

        private static string SensorName(int sensor, bool stockTopology)
        {
            string[] compact =
            {
                "Bias", "Energy", "Age", "Speed", "Food X", "Food Y", "Food closeness",
                "Neighbour X", "Neighbour Y", "Neighbour count", "Pheromone 1", "Pheromone 2",
                "Pheromone 3", "Random", "Heading sine", "Heading cosine"
            };
            string[] stock =
            {
                "Constant", "Energy ratio", "Maturity", "Life ratio", "Fullness", "Speed",
                "Rotation speed", "Is grabbing", "Attacked damage", "Egg stored",
                "Bibite closeness", "Bibite angle", "Bibite count", "Plant closeness",
                "Plant angle", "Plant count", "Meat closeness", "Meat angle", "Meat count",
                "Red Bibite", "Green Bibite", "Blue Bibite", "Tic", "Minute", "Time alive",
                "Pheromone sense 1", "Pheromone sense 2", "Pheromone sense 3",
                "Pheromone 1 angle", "Pheromone 2 angle", "Pheromone 3 angle",
                "Pheromone 1 heading", "Pheromone 2 heading", "Pheromone 3 heading"
            };
            string[] names = stockTopology ? stock : compact;
            return sensor >= 0 && sensor < names.Length ? names[sensor] : "Sensor " + sensor;
        }

        private static string ActionName(int action)
        {
            string[] names =
            {
                "Accelerate", "Rotate", "Pheromone 1", "Pheromone 2", "Pheromone 3", "Lay egg",
                "Herding", "Egg production", "Eat", "Digestion", "Grab", "Clock reset",
                "Grow", "Heal", "Attack"
            };
            return action >= 0 && action < names.Length ? names[action] : "Action " + action;
        }

        private void ApplySelectedTag()
        {
            if (_runner == null || !_hasSelection)
            {
                return;
            }
            string tagName = (_selectedTagEditor ?? string.Empty).Trim();
            if (tagName.Length > 100)
            {
                tagName = tagName.Substring(0, 100);
            }
            ulong tagId = string.IsNullOrEmpty(tagName)
                ? 0ul
                : NativeTemplateProjection.StableId("tag:" + tagName);
            if (_runner.QueueSetTag(_selectedBibite.Slot, tagId, tagName, _selectedDetail))
            {
                _tagNames[tagId] = string.IsNullOrEmpty(tagName) ? "Untagged" : tagName;
                _selectedBibite.TagId = tagId;
                _selectedDetail.TagId = tagId;
                _selectedInspectorNotice = null;
                GUI.FocusControl(string.Empty);
                UserControl.SetKeyboardBlockFromSource("GpuBibiteTagEditor", false);
            }
            else _selectedInspectorNotice = "Tag change was not queued. Wait for the live selected state or try again after the command queue clears.";
        }

        private void QueueSelectedKill()
        {
            if (_runner != null && _runner.QueueKill(_selectedBibite.Slot, _selectedDetail))
            {
                _selectedInspectorNotice = null;
            }
            else _selectedInspectorNotice = "Removal was not queued. Wait for the live selected state or try again after the command queue clears.";
        }

        private void QueueSelectedReproduction()
        {
            if (_runner != null && _runner.QueueForceReproduction(_selectedBibite.Slot, _selectedDetail))
            {
                _selectedInspectorNotice = null;
            }
            else _selectedInspectorNotice = "Lay egg was not queued. Wait for the live selected state or try again after the command queue clears.";
        }

        private void DrawNativeSpeciesWindow(int windowId)
        {
            GUILayout.Label(_lineageBreakdown.Count + " living species. Click one to select a member.");
            _nativeSpeciesScroll = GUILayout.BeginScrollView(_nativeSpeciesScroll);
            for (int index = 0; index < _lineageBreakdown.Count; index++)
            {
                BreakdownEntry entry = _lineageBreakdown[index];
                if (GUILayout.Button(
                    (index + 1) + ". " + entry.Name + "    " + entry.Count +
                    " Bibites    " + entry.Energy.ToString("0") + " energy"))
                {
                    ulong lineageId = entry.Id;
                    SelectFirstMatching(delegate(NativeWorldBibite bibite)
                    {
                        return bibite.LineageId == lineageId;
                    });
                }
            }
            GUILayout.EndScrollView();
            if (GUILayout.Button("Close species panel"))
            {
                _nativeSpeciesPanelOpen = false;
            }
            GUI.DragWindow(new Rect(0f, 0f, 10000f, 24f));
        }

        private void SyncLiveFoodSettings()
        {
            float now = Time.realtimeSinceStartup;
            if (now < _nextFoodSettingsPollAt || _runner == null ||
                _foodBaselineTarget < 0 || ScenarioSettings.Instance == null ||
                ScenarioIndependentSettings.Instance == null)
            {
                return;
            }
            _nextFoodSettingsPollAt = now + 0.25f;

            ScenarioSettings scenario = ScenarioSettings.Instance;
            ScenarioIndependentSettings independent = ScenarioIndependentSettings.Instance;
            // Stock's drag slider defaults to 5. Keep the native reference
            // damping (0.8/s) at that value and honour 0 as truly frictionless.
            float stockDrag = Mathf.Clamp(scenario.dragCoefficient.val, 0f, 25f);
            float nativeDrag = stockDrag * (0.8f / 5f);
            if (Mathf.Abs(nativeDrag - _lastQueuedLinearDrag) > 0.0001f)
            {
                _lastQueuedLinearDrag = nativeDrag;
                _runner.QueueLinearDrag(nativeDrag);
            }
            float collisionDamage = Mathf.Clamp(
                scenario.collisionDamageConstant.val, 0f, 2f);
            float collisionThreshold = Mathf.Clamp(
                scenario.collisionDamageThreshold.val, 0f, 200f);
            float bitingDamage = Mathf.Clamp(
                scenario.bitingDamageFactor.val, 0f, 20f);
            float bitingPressure = Mathf.Clamp(
                scenario.bitingPressure.val, 1f, 500f);
            if (Mathf.Abs(collisionDamage - _lastQueuedCollisionDamage) > 0.0001f ||
                Mathf.Abs(collisionThreshold - _lastQueuedCollisionThreshold) > 0.0001f ||
                Mathf.Abs(bitingDamage - _lastQueuedBitingDamage) > 0.0001f ||
                Mathf.Abs(bitingPressure - _lastQueuedBitingPressure) > 0.0001f)
            {
                _lastQueuedCollisionDamage = collisionDamage;
                _lastQueuedCollisionThreshold = collisionThreshold;
                _lastQueuedBitingDamage = bitingDamage;
                _lastQueuedBitingPressure = bitingPressure;
                _runner.QueueCombatSettings(collisionDamage, collisionThreshold,
                    bitingDamage, bitingPressure);
            }
            if (MatterMaterialManager.Plant != null &&
                MatterMaterialManager.Meat != null)
            {
                NativeWorldDigestionSettings digestion =
                    new NativeWorldDigestionSettings
                    {
                        PlantAffinityPower = Mathf.Clamp(
                            scenario.plantAffinityPowerFactor.val, 0.1f, 3f),
                        MeatAffinityPower = Mathf.Clamp(
                            scenario.meatAffinityPowerFactor.val, 0.1f, 3f),
                        PlantMinEfficiency = Mathf.Clamp(
                            MatterMaterialManager.Plant.MinConversionEfficiency, -1f, 1f),
                        PlantMaxEfficiency = Mathf.Clamp(
                            MatterMaterialManager.Plant.MaxConversionEfficiency, -1f, 1f),
                        MeatMinEfficiency = Mathf.Clamp(
                            MatterMaterialManager.Meat.MinConversionEfficiency, -1f, 1f),
                        MeatMaxEfficiency = Mathf.Clamp(
                            MatterMaterialManager.Meat.MaxConversionEfficiency, -1f, 1f)
                    };
                if (!_hasQueuedDigestionSettings ||
                    !SameDigestionSettings(digestion,
                        _lastQueuedDigestionSettings))
                {
                    _hasQueuedDigestionSettings = true;
                    _lastQueuedDigestionSettings = digestion;
                    _runner.QueueDigestionSettings(digestion);
                }
            }
            int omittedFoodZones;
            NativeWorldFoodZone[] foodZones = NativeFoodZoneProjection.Capture(
                scenario, _activeWorldHalfExtent, out omittedFoodZones);
            if (!NativeFoodZoneProjection.Same(foodZones, _lastQueuedFoodZones))
            {
                _lastQueuedFoodZones = foodZones;
                _runner.QueueFoodZones(foodZones);
            }
            double estimate = EstimateStockPellets(foodZones,
                scenario.pelletEnergy.val);
            // For a new simulation the stock biomass estimate is the actual
            // plant cap. The old ratio against a fixed 8192-pellet starting
            // count made low-food scenarios start with thousands of plants.
            int target;
            if (!_loadedCheckpointActive)
            {
                target = FoodPelletCap.Resolve(estimate,
                    _configuredFoodCap, _activePelletCount);
            }
            else
            {
                // Preserve a nonzero saved level proportionally. A zero-food
                // zone must stay zero, and subsequent growth from zero uses
                // the actual stock estimate rather than the default GPU cap.
                target = FoodPelletCap.ResolveLoaded(estimate,
                    _foodBaselineEstimate, _foodBaselineTarget,
                    _configuredFoodCap, _activePelletCount);
            }

            float growth = NativeFoodZoneProjection.TotalGrowth(foodZones);
            float growthRatio = _foodBaselineGrowth > 0f
                ? growth / _foodBaselineGrowth
                : Math.Max(0f, independent.pelletGrowth.val) /
                    Math.Max(0.00000001f, _foodBaselineGlobalGrowth);
            float growthFactor = _foodBaselineNativeGrowth > 0f
                ? _foodBaselineNativeGrowth * growthRatio
                : (Math.Abs(growth - _foodBaselineGrowth) < 0.000000001f
                    ? 0f
                    : Math.Max(0f, independent.pelletGrowth.val) / 0.00001f);
            growthFactor = Mathf.Clamp(growthFactor, 0f, 50f);
            if (float.IsNaN(growthFactor) || float.IsInfinity(growthFactor))
            {
                growthFactor = 0f;
            }
            if (growth <= 0f || foodZones.Length == 0)
                growthFactor = 0f;
            float pelletEnergy = Mathf.Clamp(
                _foodBaselineNativeEnergy *
                    Math.Max(0.001f, scenario.pelletEnergy.val) /
                    _foodBaselineStockEnergy,
                0.001f,
                100000f);
            if (float.IsNaN(pelletEnergy) || float.IsInfinity(pelletEnergy))
            {
                pelletEnergy = _foodBaselineNativeEnergy;
            }
            if (target == _lastQueuedFoodTarget &&
                Mathf.Abs(growthFactor - _lastQueuedFoodGrowth) < 0.0001f &&
                Mathf.Abs(pelletEnergy - _lastQueuedPelletEnergy) < 0.001f)
            {
                return;
            }
            _lastQueuedFoodTarget = target;
            _lastQueuedFoodGrowth = growthFactor;
            _lastQueuedPelletEnergy = pelletEnergy;
            _runner.QueueFoodSettings(target, growthFactor, pelletEnergy);
        }

        private static bool SameDigestionSettings(
            NativeWorldDigestionSettings a, NativeWorldDigestionSettings b)
        {
            return Mathf.Abs(a.PlantAffinityPower - b.PlantAffinityPower) < 0.0001f &&
                Mathf.Abs(a.MeatAffinityPower - b.MeatAffinityPower) < 0.0001f &&
                Mathf.Abs(a.PlantMinEfficiency - b.PlantMinEfficiency) < 0.0001f &&
                Mathf.Abs(a.PlantMaxEfficiency - b.PlantMaxEfficiency) < 0.0001f &&
                Mathf.Abs(a.MeatMinEfficiency - b.MeatMinEfficiency) < 0.0001f &&
                Mathf.Abs(a.MeatMaxEfficiency - b.MeatMaxEfficiency) < 0.0001f;
        }

        private static double EstimateStockPellets(
            NativeWorldFoodZone[] foodZones, float pelletEnergy)
        {
            double total = 0.0;
            if (foodZones == null) return total;
            foreach (NativeWorldFoodZone zone in foodZones)
            {
                double portion = FoodPelletCap.EstimateZonePellets(
                    zone.SeedWeight, zone.PelletSize, pelletEnergy);
                total = Math.Min(1000000000000.0,
                    total + portion);
            }
            return total;
        }

        private void StartWorld(
            int deviceIndex,
            int populationCap,
            int initialPopulation,
            int pelletCount,
            bool extremeMode,
            int brainKernel,
            string checkpointPath)
        {
            try
            {
                bool loadingCheckpoint = !string.IsNullOrEmpty(checkpointPath);
                NativeWorldConfig config = loadingCheckpoint
                    ? NativeWorldContext.ReadCheckpointConfig(
                        checkpointPath,
                        Math.Max(0, deviceIndex))
                    : NativeWorldContext.CreateDefaultConfig();
                config.DeviceIndex = Math.Max(0, deviceIndex);
                if (!loadingCheckpoint)
                {
                    config.DiagnosticMask = NativeKernelPolicy.Apply(config.DiagnosticMask, brainKernel);
                    config.MaxBibites = Mathf.Clamp(
                        populationCap,
                        64,
                        MaximumPopulation);
                    config.InitialBibites = Mathf.Clamp(initialPopulation, 1, config.MaxBibites);
                    // This is the allocation reserve, not the plant cap. The
                    // projected stock settings and GPU PelletCount setting
                    // jointly determine how many plant slots may be active.
                    config.PelletCount = 32768;
                    if (extremeMode)
                    {
                        config.ContactGridUpdateFactor = 4;
                        config.ContactSolveFactor = 8;
                        config.VisionLookupFactor = 160;
                        config.BrainUpdateFactor = 8;
                        config.LockFoodTarget = 1;
                    }
                    config.WorldHalfExtent = Mathf.Clamp(
                        ScenarioIndependentSettings.Instance.SimulationSize.val,
                        100f,
                        20000f);
                    uint integrationSeed;
                    string requestedSeed = Environment.GetEnvironmentVariable(
                        "BIBITES_GPU_TEST_SEED");
                    config.Seed = !string.IsNullOrEmpty(requestedSeed) &&
                        uint.TryParse(requestedSeed, out integrationSeed)
                        ? integrationSeed
                        : unchecked((uint)DateTime.UtcNow.Ticks);
                }
                ScenarioSettings scenario = ScenarioSettings.Instance;
                int omittedFoodZones;
                NativeWorldFoodZone[] foodZones = NativeFoodZoneProjection.Capture(
                    scenario, config.WorldHalfExtent, out omittedFoodZones);
                double stockFoodEstimate = EstimateStockPellets(foodZones,
                    scenario != null ? scenario.pelletEnergy.val : 200f);
                _populationCap = config.MaxBibites;
                _activePelletCount = config.PelletCount;
                _configuredFoodCap = Mathf.Clamp(pelletCount, 0,
                    config.PelletCount);
                _foodBaselineTarget = loadingCheckpoint
                    ? -1
                    : FoodPelletCap.Resolve(stockFoodEstimate,
                        _configuredFoodCap, config.PelletCount);
                _foodBaselineEstimate = stockFoodEstimate;
                float ecologyScale = Mathf.Clamp(
                    config.WorldHalfExtent / 500f *
                    Mathf.Sqrt(8192f / Mathf.Max(1, loadingCheckpoint
                        ? config.PelletCount : _foodBaselineTarget)),
                    1f,
                    32f);
                _foodBaselineGrowth = 0f;
                _foodBaselineGlobalGrowth = ScenarioIndependentSettings.Instance != null
                    ? Math.Max(0f, ScenarioIndependentSettings.Instance.pelletGrowth.val)
                    : 0.00001f;
                _foodBaselineStockEnergy = scenario != null
                    ? Math.Max(0.001f, scenario.pelletEnergy.val)
                    : 200f;
                _foodBaselineNativeEnergy = config.PelletEnergy;
                _foodBaselineNativeGrowth = 1f;
                _lastQueuedFoodTarget = -1;
                _lastQueuedFoodGrowth = -1f;
                _lastQueuedPelletEnergy = -1f;
                _lastQueuedLinearDrag = -1f;
                _lastQueuedCollisionDamage = -1f;
                _lastQueuedCollisionThreshold = -1f;
                _lastQueuedBitingDamage = -1f;
                _lastQueuedBitingPressure = -1f;
                _hasQueuedDigestionSettings = false;
                _nextFoodSettingsPollAt = 0f;
                _activeWorldHalfExtent = config.WorldHalfExtent;
                _loadedCheckpointActive = loadingCheckpoint;
                // The exact profile also batches vision (20 ticks) and brains
                // (2 ticks), so a generic "greater than one" test mislabeled
                // every reloaded exact checkpoint as Extreme. Match the actual
                // extreme preset that was written into the checkpoint.
                _extremeMode = config.ContactGridUpdateFactor == 4 &&
                    config.ContactSolveFactor == 8 &&
                    config.VisionLookupFactor == 160 &&
                    config.BrainUpdateFactor == 8 &&
                    config.LockFoodTarget != 0;
                _foodBaselineGrowth = NativeFoodZoneProjection.TotalGrowth(foodZones);
                if (omittedFoodZones > 0)
                    _logger.LogWarning("GPU plant zones capped at " +
                        NativeFoodZoneProjection.MaximumZones + "; " +
                        omittedFoodZones + " zone(s) omitted.");
                _lastQueuedFoodZones = foodZones;
                _runner = new GpuWorldRunner(
                    config, _foodBaselineTarget, foodZones,
                    _detailedBibiteLimit, checkpointPath);
                _pendingCheckpointPath = null;
                _logger.LogInfo((loadingCheckpoint
                        ? "Loading native GPU checkpoint: "
                        : "Starting native GPU visual world: ") +
                    "device " + config.DeviceIndex +
                    ", population " + config.InitialBibites + "/" + config.MaxBibites +
                    ", plant cap " + (loadingCheckpoint ? "checkpoint" :
                        _foodBaselineTarget.ToString()) + "/" + _configuredFoodCap +
                    " (" + config.PelletCount + " reserved)" +
                    ", plant zones " + foodZones.Length +
                    ", extent " + config.WorldHalfExtent +
                    ", ecology scale " + ecologyScale.ToString("0.00") + "x" +
                    ", profile " + (_extremeMode ? "Extreme" : "Exact") + ".");
            }
            catch (Exception ex)
            {
                _logger.LogError("Could not start native GPU visual world: " + ex);
            }
        }

        private void ActivatePresentation()
        {
            try
            {
                BibiteSpriteTemplate spriteTemplate = CaptureOriginalBibiteVisual();
                PelletSpriteTemplate pelletSpriteTemplate = CaptureOriginalPelletVisual();
                ClearStockWorld();
                float extent = Mathf.Clamp(_activeWorldHalfExtent, 100f, 20000f);
                _renderer = new GpuWorldMeshRenderer(
                    Mathf.Clamp(_populationCap, 64, MaximumPopulation),
                    Mathf.Clamp(_activePelletCount, 256, 32768),
                    extent,
                    spriteTemplate,
                    pelletSpriteTemplate,
                    _detailedBibiteLimit);
                if (GpuInteropProcessSafety.CircuitBreaker.IsOpen)
                {
                    _logger.LogWarning("CUDA/Direct3D shared rendering disabled until application restart " +
                        "after an earlier graphics error; GPU simulation remains active with CPU display buffers: " +
                        FirstLine(GpuInteropProcessSafety.CircuitBreaker.FirstFailure));
                }
                else if (!string.Equals(Environment.GetEnvironmentVariable("BIBITES_GPU_DISABLE_D3D_INTEROP"), "1", StringComparison.Ordinal))
                {
                    string interopError;
                    if (_renderer.BeginDirectInterop(_runner, out interopError))
                    {
                        _nextDirectRenderAt = 0f;
                        _logger.LogInfo(
                            "Queued asynchronous CUDA/Direct3D double-buffer registration; " +
                            "Unity will display only completed, unmapped GPU frames.");
                    }
                    else
                    {
                        _logger.LogWarning(
                            "CUDA/Direct3D 11 shared rendering was unavailable; using the " +
                            "FP32 CPU display fallback: " + interopError);
                    }
                }
                else
                {
                    _logger.LogInfo(
                        "CUDA/Direct3D shared rendering disabled for this run; using FP32 CPU display buffers.");
                }
                _presentationActive = true;
                // New multi-zone worlds start with their Bibites on fertile
                // islands. The stock camera starts at world origin, which can
                // be empty ocean in scenarios such as "3 Islands".
                if (!_loadedCheckpointActive && CameraManager.instance != null &&
                    _lastQueuedFoodZones != null && _lastQueuedFoodZones.Length > 0)
                {
                    int focus = -1;
                    float largestSeed = 0f;
                    for (int i = 0; i < _lastQueuedFoodZones.Length; i++)
                    {
                        if (_lastQueuedFoodZones[i].SeedWeight > largestSeed)
                        {
                            largestSeed = _lastQueuedFoodZones[i].SeedWeight;
                            focus = i;
                        }
                    }
                    if (focus >= 0)
                    {
                        Vector3 position = CameraManager.instance.transform.position;
                        position.x = _lastQueuedFoodZones[focus].CenterX;
                        position.y = _lastQueuedFoodZones[focus].CenterY;
                        CameraManager.instance.transform.position = position;
                    }
                }
                if (_foodBaselineTarget < 0)
                {
                    _foodBaselineTarget = _runner.CurrentFoodTarget;
                    _foodBaselineNativeEnergy = _runner.CurrentPelletEnergy;
                    _foodBaselineNativeGrowth = _runner.CurrentFoodGrowthFactor;
                }
                BindNativeDataLogger();
                float startupMilliseconds = Mathf.Max(
                    0f,
                    (Time.realtimeSinceStartup - _simulationReadyAt) * 1000f);
                _logger.LogInfo("Native GPU world is authoritative; original Unity graphics and GUI remain at " +
                    _renderFps + " FPS. Startup handoff took " +
                    startupMilliseconds.ToString("0") + " ms.");
            }
            catch (Exception ex)
            {
                _logger.LogError("Could not activate GPU visual presentation: " + ex);
                StopWorld();
            }
        }

        private BibiteSpriteTemplate CaptureOriginalBibiteVisual()
        {
            BibiteProceduralSpriter source = UnityEngine.Object.FindFirstObjectByType<BibiteProceduralSpriter>(
                FindObjectsInactive.Exclude);
            if (source == null && GameManager.defaultBibites.Count > 0 &&
                WorldObjectsSpawner.Instance != null)
            {
                try
                {
                    BibiteTemplate template = new BibiteTemplate(
                        GameManager.defaultBibites[0] + ".bb8template");
                    GameObject sample = WorldObjectsSpawner.Instance.SpawnBibiteFromTemplate(
                        template,
                        RandomizeGenes.No,
                        Tagging.NoTagging,
                        GrowthAtSpawn.Adult,
                        Vector3.zero,
                        0f);
                    source = sample != null
                        ? sample.GetComponent<BibiteProceduralSpriter>()
                        : null;
                }
                catch (Exception ex)
                {
                    _logger.LogWarning("Could not prepare original Bibite detail sprites: " + ex.Message);
                }
            }
            BibiteSpriteTemplate captured = BibiteSpriteTemplate.Capture(source);
            if (captured == null)
            {
                _logger.LogWarning("Original Bibite detail sprites were unavailable; using the batched world renderer only.");
            }
            else
            {
                _logger.LogInfo("Captured original Bibite sprites for the close-up GPU detail renderer.");
            }
            return captured;
        }

        private PelletSpriteTemplate CaptureOriginalPelletVisual()
        {
            PelletProceduralSpriter source = UnityEngine.Object.FindFirstObjectByType<PelletProceduralSpriter>(
                FindObjectsInactive.Exclude);
            if ((source == null || source.sr == null || source.sr.sprite == null) &&
                WorldObjectsSpawner.Instance != null)
            {
                try
                {
                    MatterPellet sample = WorldObjectsSpawner.Instance.SpawnPlantPellet(
                        Vector3.zero,
                        Mathf.Max(ScenarioSettings.Instance.pelletEnergy.val, 1f));
                    source = sample != null
                        ? sample.GetComponent<PelletProceduralSpriter>()
                        : null;
                }
                catch (Exception ex)
                {
                    _logger.LogWarning("Could not prepare the original plant pellet texture: " + ex.Message);
                }
            }
            SpriteRenderer renderer = source != null
                ? (source.sr != null ? source.sr : source.GetComponent<SpriteRenderer>())
                : null;
            PelletSpriteTemplate captured = PelletSpriteTemplate.Capture(renderer);
            if (captured == null)
            {
                _logger.LogWarning("Original plant pellet texture was unavailable; using the fallback pellet material.");
            }
            else
            {
                _logger.LogInfo("Captured the original plant pellet sprite, texture and material for GPU batching.");
            }
            return captured;
        }

        private void ClearStockWorld()
        {
            if (_stockWorldCleared)
            {
                return;
            }
            _stockWorldCleared = true;
            Stopwatch cleanupTimer = Stopwatch.StartNew();
            if (SimulationManager.Instance != null && SimulationManager.Instance.bibiteSpawner != null)
            {
                SimulationManager.Instance.bibiteSpawner.StopAllCoroutines();
                SimulationManager.Instance.bibiteSpawner.enabled = false;
            }
            if (ZoneManager.instance != null)
            {
                ZoneManager.instance.StopAllCoroutines();
                for (int index = 0; index < ZoneManager.instance.zones.Count; index++)
                {
                    Zone zone = ZoneManager.instance.zones[index];
                    if (zone == null) continue;
                    zone.StopAllCoroutines();
                    zone.enabled = false;
                }
            }

            BibiteBody[] bibites = UnityEngine.Object.FindObjectsByType<BibiteBody>(
                FindObjectsInactive.Include,
                FindObjectsSortMode.None);
            for (int index = 0; index < bibites.Length; index++)
            {
                if (bibites[index] == null) continue;
                bibites[index].gameObject.SetActive(false);
                UnityEngine.Object.Destroy(bibites[index].gameObject);
            }
            EggHatching[] eggs = UnityEngine.Object.FindObjectsByType<EggHatching>(
                FindObjectsInactive.Include,
                FindObjectsSortMode.None);
            for (int index = 0; index < eggs.Length; index++)
            {
                if (eggs[index] == null) continue;
                eggs[index].gameObject.SetActive(false);
                UnityEngine.Object.Destroy(eggs[index].gameObject);
            }
            MatterPellet[] pellets = UnityEngine.Object.FindObjectsByType<MatterPellet>(
                FindObjectsInactive.Include,
                FindObjectsSortMode.None);
            for (int index = 0; index < pellets.Length; index++)
            {
                if (pellets[index] == null) continue;
                pellets[index].gameObject.SetActive(false);
                UnityEngine.Object.Destroy(pellets[index].gameObject);
            }
            PlantCounter.ResetCount();
            MeatCounter.ResetCount();
            cleanupTimer.Stop();
            _logger.LogInfo("Cleared stock staging world: " + bibites.Length + " Bibites, " +
                eggs.Length + " eggs, and " + pellets.Length + " pellets in " +
                cleanupTimer.Elapsed.TotalMilliseconds.ToString("0.0") + " ms.");
        }

        private void RefreshSelected(GpuWorldSnapshot snapshot)
        {
            for (int index = 0; index < snapshot.VisibleBibiteCount; index++)
            {
                if (snapshot.VisibleBibites[index].Slot == _selectedBibite.Slot)
                {
                    _selectedBibite = snapshot.VisibleBibites[index];
                    return;
                }
            }
            for (int index = 0; index < snapshot.BibiteCount; index++)
            {
                if (snapshot.Bibites[index].Slot == _selectedBibite.Slot)
                {
                    _selectedBibite = snapshot.Bibites[index];
                    return;
                }
            }
            // A capped presentation sample may not contain the selected slot
            // in every refresh. Keep the last known inspector state instead of
            // closing it once per snapshot; refresh it whenever its slot is
            // present again.
        }

        private void SelectBibite(NativeWorldBibite bibite)
        {
            _selectedBibite = bibite;
            _selectedDetail = new NativeWorldBibiteDetail
            {
                Slot = bibite.Slot,
                Alive = 1,
                Generation = bibite.Generation,
                BrainNodes = bibite.BrainNodes,
                BrainSynapses = bibite.BrainSynapses,
                LineageId = bibite.LineageId,
                TagId = bibite.TagId,
                PositionX = bibite.PositionX,
                PositionY = bibite.PositionY,
                VelocityX = bibite.VelocityX,
                VelocityY = bibite.VelocityY,
                Heading = bibite.Heading,
                Energy = bibite.Energy,
                Age = bibite.Age,
                Size = bibite.Size,
                ColorR = bibite.ColorR,
                ColorG = bibite.ColorG,
                ColorB = bibite.ColorB
            };
            _selectedBrainNodes = new NativeWorldBrainNodeState[0];
            _selectedBrainSynapses = new NativeWorldBrainSynapseState[0];
            _selectedDetailSequence = -1;
            _selectedInstanceToken = 0;
            _selectedInspectorTab = InspectorTab.Stats;
            _selectedInspectorScroll = Vector2.zero;
            _expandedBrainPage = 0;
            _expandedBrainShowsSynapses = false;
            _selectedTagEditor = TagName(bibite.TagId);
            if (string.Equals(_selectedTagEditor, "Untagged", StringComparison.Ordinal))
            {
                _selectedTagEditor = string.Empty;
            }
            _selectedInspectorNotice = "Loading selected GPU state...";
            _hasSelection = true;
            if (_renderer != null) _renderer.SetSelectedSlot(bibite.Slot);
            if (_stockInspector == null || !_stockInspector.Ready)
            {
                if (_stockInspector != null) _stockInspector.Dispose();
                _stockInspector = new GpuNativeInspector(this);
            }
            _stockInspector.Show(_selectedInspectorTab);
            if (_runner != null)
            {
                _runner.SetSelectedInspectorNeedsBrain(false);
                _runner.SetSelectedSlot(bibite.Slot);
            }
        }

        private void ClearSelection()
        {
            _hasSelection = false;
            if (_renderer != null) _renderer.SetSelectedSlot(-1);
            if (_stockInspector != null) _stockInspector.Deselect();
            _followSelection = false;
            _selectedDetailSequence = -1;
            _selectedBrainNodes = new NativeWorldBrainNodeState[0];
            _selectedBrainSynapses = new NativeWorldBrainSynapseState[0];
            _selectedInspectorNotice = null;
            UserControl.SetKeyboardBlockFromSource("GpuBibiteTagEditor", false);
            if (_runner != null)
            {
                _runner.SetSelectedInspectorNeedsBrain(false);
                _runner.SetSelectedSlot(-1);
            }
        }

        private void ApplyDetailToSelection(NativeWorldBibiteDetail detail)
        {
            _selectedBibite.Slot = detail.Slot;
            _selectedBibite.Generation = detail.Generation;
            _selectedBibite.LineageId = detail.LineageId;
            _selectedBibite.TagId = detail.TagId;
            _selectedBibite.BrainNodes = detail.BrainNodes;
            _selectedBibite.BrainSynapses = detail.BrainSynapses;
            _selectedBibite.PositionX = detail.PositionX;
            _selectedBibite.PositionY = detail.PositionY;
            _selectedBibite.VelocityX = detail.VelocityX;
            _selectedBibite.VelocityY = detail.VelocityY;
            _selectedBibite.Heading = detail.Heading;
            _selectedBibite.Energy = detail.Energy;
            _selectedBibite.Age = detail.Age;
            _selectedBibite.Size = detail.Size;
            _selectedBibite.ColorR = detail.ColorR;
            _selectedBibite.ColorG = detail.ColorG;
            _selectedBibite.ColorB = detail.ColorB;
            if (string.Equals(_selectedInspectorNotice, "Loading selected GPU state...",
                StringComparison.Ordinal))
            {
                _selectedInspectorNotice = null;
            }
        }

        private void UpdateSelectionControls()
        {
            if (_followSelection && CameraManager.instance != null)
            {
                Vector3 cameraPosition = CameraManager.instance.transform.position;
                cameraPosition.x = _selectedBibite.PositionX;
                cameraPosition.y = _selectedBibite.PositionY;
                CameraManager.instance.transform.position = cameraPosition;
            }
            if (!UserControl.AllowControl || !UserControl.AllowKeyboardControl ||
                NativeUiVisibility.BlockingPanelOpen || (Plugin.Instance != null && !Plugin.Instance.CanUseSimulationHotkeys))
            {
                return;
            }
            if (Input.GetKeyDown(KeyCode.Alpha1)) SelectInspectorTab(InspectorTab.Stats);
            else if (Input.GetKeyDown(KeyCode.Alpha2)) SelectInspectorTab(InspectorTab.Genes);
            else if (Input.GetKeyDown(KeyCode.Alpha3)) SelectInspectorTab(InspectorTab.Biology);
            else if (Input.GetKeyDown(KeyCode.Alpha4)) SelectInspectorTab(InspectorTab.Brain);
            else if (Input.GetKeyDown(KeyCode.Alpha5)) SelectInspectorTab(InspectorTab.ExpandedBrain);
            if ((Input.GetKey(KeyCode.LeftControl) || Input.GetKey(KeyCode.LeftMeta)) && Input.GetKeyDown(KeyCode.C)) CopyNativeTag();
            if ((Input.GetKey(KeyCode.LeftControl) || Input.GetKey(KeyCode.LeftMeta)) && Input.GetKeyDown(KeyCode.V)) SetNativeTag(GUIUtility.systemCopyBuffer);
        }

        private void UpdateNativeSelectionHotkeys()
        {
            if (_lastSnapshot == null || _lastSnapshot.BibiteCount == 0 ||
                !UserControl.AllowControl || !UserControl.AllowKeyboardControl ||
                !UserControl.AllowSelection ||
                NativeUiVisibility.BlockingPanelOpen ||
                (Plugin.Instance != null && !Plugin.Instance.CanUseSimulationHotkeys)) return;
            bool random = Input.GetKeyDown(KeyCode.R);
            bool generation = Input.GetKeyDown(KeyCode.G);
            bool oldest = Input.GetKeyDown(KeyCode.O);
            if (!random && !generation && !oldest) return;
            int count = Math.Min(_lastSnapshot.BibiteCount, _lastSnapshot.Bibites.Length);
            int best = random ? UnityEngine.Random.Range(0, count) : 0;
            if (!random)
            {
                for (int i = 1; i < count; i++)
                {
                    if (generation ? _lastSnapshot.Bibites[i].Generation > _lastSnapshot.Bibites[best].Generation :
                        _lastSnapshot.Bibites[i].Age > _lastSnapshot.Bibites[best].Age) best = i;
                }
            }
            SelectBibite(_lastSnapshot.Bibites[best]);
            if (count < _lastSnapshot.Stats.LivingBibites)
                _selectedInspectorNotice = "Selection uses the presentation sample, not a full GPU population scan.";
        }

        private void SelectInspectorTab(InspectorTab tab)
        {
            if (_selectedInspectorTab == tab)
            {
                if (_stockInspector != null) _stockInspector.Show(tab);
                return;
            }
            _selectedInspectorTab = tab;
            _selectedInspectorScroll = Vector2.zero;
            _expandedBrainPage = 0;
            if (_stockInspector != null) _stockInspector.Show(tab);
            if (_runner != null)
            {
                _runner.SetSelectedInspectorNeedsBrain(
                    tab == InspectorTab.Brain || tab == InspectorTab.ExpandedBrain);
            }
        }

        private void SelectFirstMatching(Predicate<NativeWorldBibite> predicate)
        {
            if (_lastSnapshot == null || predicate == null)
            {
                return;
            }
            int count = Math.Min(_lastSnapshot.BibiteCount, _lastSnapshot.Bibites.Length);
            for (int index = 0; index < count; index++)
            {
                if (!predicate(_lastSnapshot.Bibites[index]))
                {
                    continue;
                }
                SelectBibite(_lastSnapshot.Bibites[index]);
                return;
            }
        }

        private void SyncOriginalInformation(GpuWorldSnapshot snapshot)
        {
            if (Time.realtimeSinceStartup < _nextInformationUpdateAt)
            {
                return;
            }
            _nextInformationUpdateAt = Time.realtimeSinceStartup + 0.25f;
            InformationPanel panel = InformationPanel.Instance;
            if (panel == null)
            {
                return;
            }
            float plantEnergy = snapshot.Stats.PlantEnergy;
            float meatEnergy = snapshot.Stats.MeatEnergy;
            try
            {
                panel.UpdateInformation(
                    snapshot.Stats.ActivePellets,
                    snapshot.Stats.ActiveMeat,
                    snapshot.Stats.LivingBibites,
                    0,
                    0f,
                    plantEnergy + meatEnergy,
                    snapshot.Stats.TotalEnergy,
                    0f,
                    plantEnergy,
                    meatEnergy);
                SyncNativeInformationBreakdown(panel, snapshot);
                _informationPanelFaulted = false;
            }
            catch (Exception ex)
            {
                if (!_informationPanelFaulted)
                    _logger.LogWarning("Could not synchronize the stock Information panel with the GPU world; will retry: " + ex.Message);
                _informationPanelFaulted = true;
                _nextInformationUpdateAt = Time.realtimeSinceStartup + 2f;
            }
        }

        private void SyncNativeInformationBreakdown(
            InformationPanel panel,
            GpuWorldSnapshot snapshot)
        {
            List<BreakdownEntry> lineages = BuildBreakdown(
                snapshot,
                delegate(NativeWorldBibite bibite) { return bibite.LineageId; },
                LineageName);
            bool sampledBreakdown = snapshot.Stats.LivingBibites >
                Math.Min(snapshot.BibiteCount, MaximumPresentationSamples);
            _lineageBreakdown = lineages;
            if (_stockInspector != null && _nativeSpeciesPanelOpen)
                _stockInspector.UpdateSpecies(lineages, snapshot.Stats.LivingBibites > Math.Min(snapshot.BibiteCount, MaximumPresentationSamples));
            IList speciesItems = InformationSpeciesItems != null
                ? InformationSpeciesItems.GetValue(panel) as IList
                : null;
            _speciesHandleLineages.Clear();
            int visibleSpecies = Math.Min(5, lineages.Count);
            if (speciesItems != null)
            {
                EnsureNativeSpeciesHandles(panel, speciesItems, visibleSpecies);
                visibleSpecies = Math.Min(visibleSpecies, speciesItems.Count);
                for (int index = 0; index < speciesItems.Count; index++)
                {
                    SpeciesInfoHandle handle = speciesItems[index] as SpeciesInfoHandle;
                    if (handle == null)
                    {
                        continue;
                    }
                    bool visible = index < visibleSpecies;
                    handle.gameObject.SetActive(visible);
                    if (!visible)
                    {
                        continue;
                    }
                    BreakdownEntry entry = lineages[index];
                    handle.UpdateIndex(index + 1);
                    SetTextField(handle, "speciesName", (sampledBreakdown ? "~ " : string.Empty) + entry.Name);
                    UpdateValueField(handle, "speciesCount", entry.Count);
                    UpdateValueField(handle, "speciesEnergy", entry.Energy);
                    _speciesHandleLineages[handle.GetInstanceID()] = entry.Id;
                }
            }
            UpdateValueHandle(InformationTopSpeciesCount, panel, visibleSpecies);
            UpdateValueHandle(InformationSpeciesCount, panel, lineages.Count);
            SetPropertyValue(
                InformationSpeciesSection != null ? InformationSpeciesSection.GetValue(panel) : null,
                "preferredHeight",
                15f + 30f * visibleSpecies);

            List<BreakdownEntry> tags = BuildBreakdown(
                snapshot,
                delegate(NativeWorldBibite bibite) { return bibite.TagId; },
                TagName);
            IList stockTagItems = InformationTagItems != null
                ? InformationTagItems.GetValue(panel) as IList
                : null;
            if (stockTagItems != null)
            {
                for (int index = 0; index < stockTagItems.Count; index++)
                {
                    TagElementHandle stock = stockTagItems[index] as TagElementHandle;
                    if (stock != null)
                    {
                        stock.gameObject.SetActive(false);
                    }
                }
            }
            _tagHandleTags.Clear();
            HashSet<ulong> visibleTags = new HashSet<ulong>();
            int visibleTagCount = Math.Min(7, tags.Count);
            for (int index = 0; index < visibleTagCount; index++)
            {
                BreakdownEntry entry = tags[index];
                TagElementHandle handle = EnsureNativeTagHandle(panel, entry.Id);
                if (handle == null)
                {
                    continue;
                }
                visibleTags.Add(entry.Id);
                handle.gameObject.SetActive(true);
                handle.transform.SetAsLastSibling();
                handle.UpdateIndex(index + 1);
                SetTextField(handle, "tagName", (sampledBreakdown ? "~ " : string.Empty) + entry.Name);
                UpdateValueField(handle, "tagCount", entry.Count);
                UpdateValueField(handle, "tagEnergy", entry.Energy);
                _tagHandleTags[handle.GetInstanceID()] = entry.Id;
            }
            List<ulong> retiredTags = new List<ulong>();
            foreach (KeyValuePair<ulong, TagElementHandle> item in _nativeTagItems)
            {
                if (item.Value != null && !visibleTags.Contains(item.Key))
                {
                    UnityEngine.Object.Destroy(item.Value.gameObject);
                    retiredTags.Add(item.Key);
                }
            }
            foreach (ulong id in retiredTags) _nativeTagItems.Remove(id);
            SetPropertyValue(
                InformationTagSection != null ? InformationTagSection.GetValue(panel) : null,
                "preferredHeight",
                15f + 30f * visibleTagCount);
        }

        private List<BreakdownEntry> BuildBreakdown(
            GpuWorldSnapshot snapshot,
            Func<NativeWorldBibite, ulong> idSelector,
            Func<ulong, string> nameSelector)
        {
            Dictionary<ulong, BreakdownEntry> byId = new Dictionary<ulong, BreakdownEntry>();
            int availableCount = Math.Min(snapshot.BibiteCount, snapshot.Bibites.Length);
            int populationCount = Math.Max(
                availableCount,
                snapshot.Stats.LivingBibites);
            int sampleCount = Math.Min(availableCount, MaximumPresentationSamples);
            for (int sample = 0; sample < sampleCount; sample++)
            {
                int index = PresentationSampleIndex(
                    snapshot,
                    sample,
                    availableCount,
                    sampleCount);
                NativeWorldBibite bibite = snapshot.Bibites[index];
                ulong id = idSelector(bibite);
                BreakdownEntry entry;
                if (!byId.TryGetValue(id, out entry))
                {
                    entry = new BreakdownEntry
                    {
                        Id = id,
                        Name = nameSelector(id)
                    };
                    byId.Add(id, entry);
                }
                entry.Count++;
                entry.Energy += bibite.Energy;
            }
            List<BreakdownEntry> result = new List<BreakdownEntry>(byId.Values);
            if (sampleCount > 0 && sampleCount < populationCount)
            {
                float scale = (float)populationCount / sampleCount;
                for (int index = 0; index < result.Count; index++)
                {
                    result[index].Count = Math.Max(
                        1,
                        Mathf.RoundToInt(result[index].Count * scale));
                    result[index].Energy *= scale;
                }
            }
            result.Sort(delegate(BreakdownEntry left, BreakdownEntry right)
            {
                int countOrder = right.Count.CompareTo(left.Count);
                return countOrder != 0 ? countOrder : right.Energy.CompareTo(left.Energy);
            });
            if (result.Count > 0)
            {
                int reportedCount = 0;
                for (int index = 0; index < result.Count; index++)
                {
                    reportedCount += result[index].Count;
                }
                result[0].Count = Math.Max(
                    1,
                    result[0].Count + populationCount - reportedCount);
            }
            return result;
        }

        private static int PresentationSampleIndex(
            GpuWorldSnapshot snapshot,
            int sample,
            int count,
            int sampleCount)
        {
            if (sampleCount >= count)
            {
                return sample;
            }
            int stride = Math.Max(1, count / sampleCount);
            int offset = (int)(snapshot.Sequence % stride);
            return (int)(((long)sample * count / sampleCount + offset) % count);
        }

        private TagElementHandle EnsureNativeTagHandle(InformationPanel panel, ulong id)
        {
            TagElementHandle handle;
            if (_nativeTagItems.TryGetValue(id, out handle) && handle != null)
            {
                return handle;
            }
            GameObject prefab = InformationTagPrefab != null
                ? InformationTagPrefab.GetValue(panel) as GameObject
                : null;
            Transform holder = InformationTagHolder != null
                ? InformationTagHolder.GetValue(panel) as Transform
                : null;
            if (prefab == null || holder == null)
            {
                return null;
            }
            GameObject instance = UnityEngine.Object.Instantiate(prefab, holder);
            handle = instance.GetComponent<TagElementHandle>();
            if (handle == null)
            {
                UnityEngine.Object.Destroy(instance);
                return null;
            }
            handle.InitTagElement(new BibiteTag(TagName(id)));
            _nativeTagItems[id] = handle;
            return handle;
        }

        private static void EnsureNativeSpeciesHandles(
            InformationPanel panel,
            IList speciesItems,
            int required)
        {
            if (panel == null || speciesItems == null || speciesItems.Count >= required)
            {
                return;
            }
            GameObject prefab = InformationSpeciesPrefab != null
                ? InformationSpeciesPrefab.GetValue(panel) as GameObject
                : null;
            Transform holder = InformationSpeciesHolder != null
                ? InformationSpeciesHolder.GetValue(panel) as Transform
                : null;
            if (prefab == null || holder == null)
            {
                return;
            }
            while (speciesItems.Count < required)
            {
                GameObject instance = UnityEngine.Object.Instantiate(prefab, holder);
                SpeciesInfoHandle handle = instance.GetComponent<SpeciesInfoHandle>();
                if (handle == null)
                {
                    UnityEngine.Object.Destroy(instance);
                    return;
                }
                handle.UpdateIndex(speciesItems.Count + 1);
                speciesItems.Add(handle);
            }
        }

        private string LineageName(ulong id)
        {
            if (id == 0ul)
            {
                return "GPU native population";
            }
            string name;
            return _lineageNames.TryGetValue(id, out name)
                ? name
                : "GPU lineage " + id.ToString("X8");
        }

        private string TagName(ulong id)
        {
            if (id == 0ul)
            {
                return "Untagged";
            }
            string name;
            return _tagNames.TryGetValue(id, out name)
                ? name
                : "GPU tag " + id.ToString("X8");
        }

        private static void SetTextField(object owner, string fieldName, string value)
        {
            if (owner == null)
            {
                return;
            }
            FieldInfo field = owner.GetType().GetField(
                fieldName,
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            object textHandle = field != null ? field.GetValue(owner) : null;
            // Names can come from imported templates or save metadata. Treat
            // their angle brackets literally, not as TextMesh Pro markup that
            // could change the original Information panel's layout or size.
            SetPropertyValue(textHandle, "richText", false);
            SetPropertyValue(textHandle, "text", value);
        }

        private static void UpdateValueField(object owner, string fieldName, float value)
        {
            if (owner == null)
            {
                return;
            }
            FieldInfo field = owner.GetType().GetField(
                fieldName,
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            InvokeUpdateValue(field != null ? field.GetValue(owner) : null, value);
        }

        private static void UpdateValueHandle(FieldInfo field, object owner, float value)
        {
            InvokeUpdateValue(field != null && owner != null ? field.GetValue(owner) : null, value);
        }

        private static void InvokeUpdateValue(object handle, float value)
        {
            if (handle == null)
            {
                return;
            }
            MethodInfo update = handle.GetType().GetMethod(
                "UpdateValue",
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
                null,
                new[] { typeof(float) },
                null);
            if (update != null)
            {
                update.Invoke(handle, new object[] { value });
            }
        }

        private static void SetPropertyValue(object target, string propertyName, object value)
        {
            if (target == null)
            {
                return;
            }
            PropertyInfo property = target.GetType().GetProperty(
                propertyName,
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            if (property != null && property.CanWrite)
            {
                property.SetValue(target, value, null);
            }
        }

        private void BindNativeDataLogger()
        {
            DataLogger logger = DataLogger.Instance;
            if (logger == null || logger == _boundDataLogger)
            {
                return;
            }
            RestoreStockDataLogger();
            if (logger.biomassDataStreamGroup == null || logger.countDataStreamGroup == null ||
                logger.brainSizeDataStreamGroup == null || logger.ageDataStreamGroup == null ||
                logger.deathAgeDataStreamGroup == null || logger.eggsLaidDataStreamGroup == null ||
                logger.birthDeathDataStreamGroup == null)
            {
                return;
            }
            _boundDataLogger = logger;
            _stockBiomassLogAction = logger.biomassDataStreamGroup.logAction;
            _stockCountLogAction = logger.countDataStreamGroup.logAction;
            _stockBrainSizeLogAction = logger.brainSizeDataStreamGroup.logAction;
            _stockAgeLogAction = logger.ageDataStreamGroup.logAction;
            _stockDeathAgeLogAction = logger.deathAgeDataStreamGroup.logAction;
            _stockEggsLaidLogAction = logger.eggsLaidDataStreamGroup.logAction;
            _stockBirthDeathLogAction = logger.birthDeathDataStreamGroup.logAction;
            logger.biomassDataStreamGroup.logAction = GetNativeBiomassData;
            logger.countDataStreamGroup.logAction = GetNativeCountData;
            logger.brainSizeDataStreamGroup.logAction = GetNativeBrainSizeData;
            logger.ageDataStreamGroup.logAction = GetNativeAgeData;
            logger.deathAgeDataStreamGroup.logAction = GetNativeDeathAgeData;
            logger.eggsLaidDataStreamGroup.logAction = GetNativeEggsLaidData;
            logger.birthDeathDataStreamGroup.logAction = GetNativeBirthDeathData;
            _nextHistorySampleSeconds = 0.0;
            _historyBirthBaseline = _lastSnapshot != null ? _lastSnapshot.Stats.Births : 0ul;
            _historyDeathBaseline = _lastSnapshot != null ? _lastSnapshot.Stats.Deaths : 0ul;
            _logger.LogInfo("Connected the original Information and historical chart data streams to the GPU world.");
        }

        private void SyncHistoricalCharts(GpuWorldSnapshot snapshot)
        {
            BindNativeDataLogger();
            if (_boundDataLogger == null || _boundDataLogger.parallelDataStreams == null ||
                snapshot.Stats.SimulatedSeconds + 0.0001 < _nextHistorySampleSeconds)
            {
                return;
            }
            int samples = 0;
            while (snapshot.Stats.SimulatedSeconds + 0.0001 >= _nextHistorySampleSeconds && samples < 4)
            {
                for (int index = 0; index < _boundDataLogger.parallelDataStreams.Count; index++)
                {
                    IDataPointStream stream = _boundDataLogger.parallelDataStreams[index];
                    if (stream != null)
                    {
                        stream.Log();
                    }
                }
                _nextHistorySampleSeconds += 60.0;
                samples++;
            }
            if (snapshot.Stats.SimulatedSeconds >= _nextHistorySampleSeconds + 240.0)
            {
                _nextHistorySampleSeconds =
                    Math.Floor(snapshot.Stats.SimulatedSeconds / 60.0 + 1.0) * 60.0;
            }
        }

        private DataPoint3 GetNativeCountData(bool peek)
        {
            if (_lastSnapshot == null) return new DataPoint3(0f, 0f, 0f);
            return new DataPoint3(
                _lastSnapshot.Stats.LivingBibites,
                _lastSnapshot.Stats.ActivePellets,
                _lastSnapshot.Stats.ActiveMeat);
        }

        private DataPoint4 GetNativeBiomassData(bool peek)
        {
            if (_lastSnapshot == null) return new DataPoint4(0f, 0f, 0f, 0f);
            return new DataPoint4(
                _lastSnapshot.Stats.TotalEnergy,
                _lastSnapshot.Stats.PlantEnergy,
                _lastSnapshot.Stats.MeatEnergy,
                0f);
        }

        private DataPoint6 GetNativeBrainSizeData(bool peek)
        {
            PreparePresentationSamples();
            return _cachedBrainSizeData;
        }

        private DataPoint4 GetNativeAgeData(bool peek)
        {
            PreparePresentationSamples();
            return _cachedAgeData;
        }

        private void PreparePresentationSamples()
        {
            if (_lastSnapshot == null || _lastSnapshot.BibiteCount <= 0)
            {
                _cachedBrainSizeData = new DataPoint6(0f, 0f, 0f, 0f, 0f, 0f);
                _cachedAgeData = new DataPoint4(0f, 0f, 0f, 0f);
                _presentationSampleSequence = _lastSnapshot != null
                    ? _lastSnapshot.SummarySequence
                    : -1;
                return;
            }
            if (_presentationSampleSequence == _lastSnapshot.SummarySequence)
            {
                return;
            }

            int count = Math.Min(_lastSnapshot.BibiteCount, _lastSnapshot.Bibites.Length);
            int sampleCount = Math.Min(count, MaximumPresentationSamples);
            float minimumNodes = float.MaxValue;
            float maximumNodes = 0f;
            float totalNodes = 0f;
            float minimumSynapses = float.MaxValue;
            float maximumSynapses = 0f;
            float totalSynapses = 0f;
            for (int sample = 0; sample < sampleCount; sample++)
            {
                int index = PresentationSampleIndex(
                    _lastSnapshot,
                    sample,
                    count,
                    sampleCount);
                NativeWorldBibite bibite = _lastSnapshot.Bibites[index];
                float nodes = Math.Max(0, bibite.BrainNodes);
                float synapses = Math.Max(0, bibite.BrainSynapses);
                minimumNodes = Math.Min(minimumNodes, nodes);
                maximumNodes = Math.Max(maximumNodes, nodes);
                totalNodes += nodes;
                minimumSynapses = Math.Min(minimumSynapses, synapses);
                maximumSynapses = Math.Max(maximumSynapses, synapses);
                totalSynapses += synapses;
                // The original Age Data stream/axis is measured in hours.
                // Native state stores simulated seconds, unlike BibiteBody.age.
                _presentationAgeSamples[sample] = bibite.Age / 3600f;
            }
            Array.Sort(_presentationAgeSamples, 0, sampleCount);
            _cachedBrainSizeData = new DataPoint6(
                minimumNodes,
                totalNodes / sampleCount,
                maximumNodes,
                minimumSynapses,
                totalSynapses / sampleCount,
                maximumSynapses);
            _cachedAgeData = new DataPoint4(
                _presentationAgeSamples[sampleCount / 4],
                _presentationAgeSamples[sampleCount / 2],
                _presentationAgeSamples[(3 * sampleCount) / 4],
                _presentationAgeSamples[sampleCount - 1]);
            _presentationSampleSequence = _lastSnapshot.SummarySequence;
        }

        private DataPoint4 GetNativeDeathAgeData(bool peek)
        {
            return new DataPoint4(0f, 0f, 0f, 0f);
        }

        private DataPoint6 GetNativeEggsLaidData(bool peek)
        {
            return new DataPoint6(0f, 0f, 0f, 0f, 0f, 0f);
        }

        private DataPoint2 GetNativeBirthDeathData(bool peek)
        {
            if (_lastSnapshot == null) return new DataPoint2(0f, 0f);
            ulong births = _lastSnapshot.Stats.Births;
            ulong deaths = _lastSnapshot.Stats.Deaths;
            float newBirths = (float)Math.Min(births - Math.Min(births, _historyBirthBaseline), 16777216ul);
            float newDeaths = (float)Math.Min(deaths - Math.Min(deaths, _historyDeathBaseline), 16777216ul);
            if (!peek)
            {
                _historyBirthBaseline = births;
                _historyDeathBaseline = deaths;
            }
            return new DataPoint2(newBirths, newDeaths);
        }

        private void RestoreStockDataLogger()
        {
            if (_boundDataLogger != null)
            {
                if (_boundDataLogger.biomassDataStreamGroup != null)
                    _boundDataLogger.biomassDataStreamGroup.logAction = _stockBiomassLogAction;
                if (_boundDataLogger.countDataStreamGroup != null)
                    _boundDataLogger.countDataStreamGroup.logAction = _stockCountLogAction;
                if (_boundDataLogger.brainSizeDataStreamGroup != null)
                    _boundDataLogger.brainSizeDataStreamGroup.logAction = _stockBrainSizeLogAction;
                if (_boundDataLogger.ageDataStreamGroup != null)
                    _boundDataLogger.ageDataStreamGroup.logAction = _stockAgeLogAction;
                if (_boundDataLogger.deathAgeDataStreamGroup != null)
                    _boundDataLogger.deathAgeDataStreamGroup.logAction = _stockDeathAgeLogAction;
                if (_boundDataLogger.eggsLaidDataStreamGroup != null)
                    _boundDataLogger.eggsLaidDataStreamGroup.logAction = _stockEggsLaidLogAction;
                if (_boundDataLogger.birthDeathDataStreamGroup != null)
                    _boundDataLogger.birthDeathDataStreamGroup.logAction = _stockBirthDeathLogAction;
            }
            _boundDataLogger = null;
            _stockBiomassLogAction = null;
            _stockCountLogAction = null;
            _stockBrainSizeLogAction = null;
            _stockAgeLogAction = null;
            _stockDeathAgeLogAction = null;
            _stockEggsLaidLogAction = null;
            _stockBirthDeathLogAction = null;
        }

        private void SyncOriginalTimeDisplay(GpuWorldSnapshot snapshot)
        {
            if (_timeKeeper == null)
            {
                _timeKeeper = UnityEngine.Object.FindFirstObjectByType<TimeKeeper>();
            }
            if (_timeKeeper == null)
            {
                return;
            }
            float achieved = (float)Math.Max(0.0, snapshot.AchievedMultiplier);
            if (TimeKeeperStps != null)
            {
                TimeKeeperStps.SetValue(_timeKeeper, achieved);
            }
            if (TimeKeeperTps != null)
            {
                TimeKeeperTps.SetValue(
                    _timeKeeper,
                    achieved / Math.Max(0.0001f, 0.025f));
            }
        }

        private void ReleaseStoppedWorldResources()
        {
            for (int index = _retiredWorlds.Count - 1; index >= 0; --index)
            {
                RetiredWorldResources retired = _retiredWorlds[index];
                retired.Runner.PumpRenderThreadEvent();
                if (!retired.Runner.IsStopped) continue;
                if (retired.Renderer != null)
                {
                    if (retired.Runner.RetainRenderResources)
                    {
                        GpuInteropProcessSafety.CircuitBreaker.Trip(
                            "a retired world's graphics buffers required quarantine");
                        QuarantinedRenderers.Add(retired.Renderer);
                        _logger.LogWarning("Retaining hidden graphics buffers until exit after a CUDA driver error.");
                    }
                    else retired.Renderer.Dispose();
                }
                _retiredWorlds.RemoveAt(index);
            }
        }

        private void StopWorld()
        {
            if (_stockInspector != null)
            {
                _stockInspector.Dispose();
                _stockInspector = null;
            }
            RestoreStockDataLogger();
            foreach (KeyValuePair<ulong, TagElementHandle> item in _nativeTagItems)
            {
                if (item.Value != null)
                {
                    UnityEngine.Object.Destroy(item.Value.gameObject);
                }
            }
            _nativeTagItems.Clear();
            _speciesHandleLineages.Clear();
            _tagHandleTags.Clear();
            _lineageNames.Clear();
            _tagNames.Clear();
            if (_runner != null)
            {
                _runner.Dispose();
                if (_renderer != null) _renderer.HideForRetirement();
                _retiredWorlds.Add(new RetiredWorldResources
                {
                    Runner = _runner,
                    Renderer = _renderer
                });
                _runner = null;
                _renderer = null;
            }
            if (_renderer != null)
            {
                _renderer.Dispose();
                _renderer = null;
            }
            ReleaseStoppedWorldResources();
            if (!_applicationQuitRequested && !GpuInteropProcessSafety.RenderingShuttingDown &&
                _autoSaveSuppressed && SaveController.Instance != null && UserSettings.AutoSave.val)
            {
                SaveController.Instance.ToggleAutoSave(true);
            }
            _autoSaveSuppressed = false;
            _presentationActive = false;
            _loadedCheckpointActive = false;
            _populationCap = 0;
            _activePelletCount = 0;
            _configuredFoodCap = 0;
            _foodBaselineTarget = 0;
            _foodBaselineEstimate = 0;
            _foodBaselineGrowth = 0f;
            _foodBaselineNativeGrowth = 1f;
            _lastQueuedFoodTarget = -1;
            _lastQueuedFoodZones = null;
            _hasQueuedDigestionSettings = false;
            _nextFoodSettingsPollAt = 0f;
            _activeWorldHalfExtent = 0f;
            _maximumSnapshotMilliseconds = 0.0;
            _stockWorldCleared = false;
            _lastSnapshot = null;
            _lastRenderedSequence = 0;
            _hasSelection = false;
            UserControl.SetKeyboardBlockFromSource("GpuBibiteTagEditor", false);
            _timeKeeper = null;
            _nextHistorySampleSeconds = 0.0;
            _historyBirthBaseline = 0ul;
            _historyDeathBaseline = 0ul;
            _nextInformationUpdateAt = 0f;
            _informationPanelFaulted = false;
            _directInteropFaultReported = false;
            _nextDirectRenderAt = 0f;
            _lastReportedPlacementFailures = 0;
            _nativeSpeciesPanelOpen = false;
            _lineageBreakdown.Clear();
            _presentationSampleSequence = -1;
            _lastChartSummarySequence = -1;
        }

        private static string FirstLine(string value)
        {
            if (string.IsNullOrEmpty(value)) return string.Empty;
            int line = value.IndexOfAny(new[] { '\r', '\n' });
            return line >= 0 ? value.Substring(0, line) : value;
        }

        public void Dispose()
        {
            StopWorld();
            // During final application shutdown there may be no next Update.
            // Keep unfinished DDOL render roots alive until process exit; do
            // not turn a slow save/kernel into a dangling native D3D pointer.
            ReleaseStoppedWorldResources();
        }
    }
}
