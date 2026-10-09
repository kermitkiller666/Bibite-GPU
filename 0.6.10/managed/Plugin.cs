using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using BepInEx;
using BepInEx.Configuration;
using BibitesGpuFork.Core;
using HarmonyLib;
using ManagementScripts;
using Newtonsoft.Json.Linq;
using OneUseScripts;
using ScriptHelpers;
using SettingScripts;
using SimulationScripts;
using SimulationScripts.BibiteScripts;
using Utility;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.SceneManagement;
using UIScripts.SettingHandles;
using UIScripts.UIReferences;

namespace BibitesGpuFork
{
    [BepInPlugin(PluginGuid, PluginName, PluginVersion)]
    [DefaultExecutionOrder(-32000)]
    public sealed class Plugin : BaseUnityPlugin
    {
        public const string PluginGuid = "local.bibites.gpu-fork";
        public const string PluginName = "Bibites GPU Fork";
        public const string PluginVersion = "0.6.10";

        internal static Plugin Instance { get; private set; }

        private Harmony _harmony;
        private ConfigEntry<bool> _showOverlay;
        private ConfigEntry<bool> _shadowValidation;
        private ConfigEntry<int> _shadowSampleLimit;
        private ConfigEntry<bool> _visionShadowValidation;
        private ConfigEntry<int> _visionShadowSampleLimit;
        private ConfigEntry<bool> _pheromoneShadowValidation;
        private ConfigEntry<bool> _authoritativeVisionGpu;
        private ConfigEntry<bool> _authoritativePheromoneGpu;
        private GpuShadowValidator _shadowValidator;
        private GpuVisionShadowValidator _visionShadowValidator;
        private ConfigEntry<bool> _authoritativeBrainGpu;
        private ConfigEntry<int> _gpuBrainMinimumSynapses;
        private ConfigEntry<int> _gpuDeviceIndex;
        private ConfigEntry<bool> _throughputMode;
        private ConfigEntry<int> _turboRenderFps;
        private ConfigEntry<bool> _multithreadedPhysics;
        private ConfigEntry<bool> _nativeGpuVisualWorld;
        private ConfigEntry<bool> _nativeGpuLoadedSaves;
        private ConfigEntry<int> _nativePopulationCap;
        private ConfigEntry<int> _nativeInitialPopulation;
        private ConfigEntry<int> _nativePelletCount;
        private ConfigEntry<int> _nativeRenderFps;
        private ConfigEntry<int> _nativeDetailedBibiteLimit;
        private ConfigEntry<bool> _nativeExtremeMode;
        private ConfigEntry<int> _nativeBrainKernel;
        private static readonly string[] NativeKernelLabels =
            { "Fused (default)", "Sparse brain graph", "Dense brain graph" };
        private string _nativePopulationCapInput;
        private string _nativeInitialPopulationInput;
        private string _nativeDetailedBibiteLimitInput;
        private string _nativePopulationInputStatus;
        private string _nativeGraphicsInputStatus;
        private GpuAuthoritativeBrainScheduler _authoritativeBrainScheduler;
        private GpuAuthoritativeVisionScheduler _authoritativeVisionScheduler;
        private GpuAuthoritativePheromoneScheduler _authoritativePheromoneScheduler;
        private GpuNativeWorldBridge _nativeWorldBridge;
        private readonly DeferredApplicationQuit _applicationQuit = new DeferredApplicationQuit();
        private float _nextQuitDrainStatusAt;
        private List<GpuDeviceInfo> _gpuDevices = new List<GpuDeviceInfo>();
        private bool _userSettingsOpen;
        private bool _gpuSettingsOpen;
        private Rect _gpuSettingsWindow = new Rect(0f, 0f, 560f, 660f);
        private Vector2 _gpuSettingsScroll;
        private IEscapable _gpuSettingsEscape;
        private UserSettingsManager _userSettingsPanel;
        private int _gpuSettingsLastScreenWidth;
        private int _gpuSettingsLastScreenHeight;
        private GUISkin _gpuSettingsSkin;
        private bool _performancePolicyApplied;
        private int _savedVSyncCount;
        private int _savedTargetFrameRate;
        private PhysicsJobOptions2D _savedPhysicsJobOptions;
        private int _fixedSteps;
        private float _sampleStarted;
        private float _achievedMultiplier;
        private string _integrationTestSave;
        private bool _integrationTestNewSimulation;
        private float _pluginStarted;
        private float _integrationSimulationStarted;
        private float _integrationBenchmarkStarted;
        private bool _integrationLaunchRequested;
        private bool _integrationResultReported;
        private bool _integrationEntitiesSpawned;
        private bool _integrationPlaceNativeBibite;
        private float _integrationFoodDensityMultiplier = -1f;
        private bool _integrationFoodDensityChanged;
        private float _integrationTestSpeed = 5f;
        private float _integrationWorldSize;
        private long _totalFixedSteps;
        private long _totalRenderedFrames;
        private long _integrationFixedStepsStarted;
        private long _integrationFramesStarted;
        private readonly List<float> _integrationFrameTimes = new List<float>();
        private int _integrationBibiteCount = 32;
        private int _integrationPelletCount = 64;
        private int _integrationPheromoneCount;
        private float _integrationReportSeconds = 10f;
        private bool _integrationValidation = true;
        private bool _integrationProfiling;
        private string _settingsCapturePath;
        private bool _settingsCaptureOpened;
        private bool _settingsCaptureCompleted;
        private float _settingsCaptureOpenedAt;
        private string _nativeWorldCapturePath;
        private bool _nativeWorldCaptureCompleted;
        private float _nativeWorldCaptureReadyAt;
        private float[] _nativeWorldCaptureDelays = { 15f, 30f };
        private int _nativeWorldCaptureIndex;
        private float _nativeWorldCaptureZoom = 500f;
        private bool _nativeWorldCaptureOpenInformation;
        private bool _nativeWorldCaptureSelectFirst;
        private bool _nativeWorldCaptureOpenSpecies;
        private bool _allowStockVision;
        private int _authoritativeVisionValidationSamples;
        private bool _allowStockPheromone;
        private int _authoritativePheromoneValidationSamples;
        private bool _nativeStartupBypassReported;
        private readonly GpuWorldSessionMode _worldSessionMode = new GpuWorldSessionMode();
        private string _pendingNativeSavePath;
        private string _pendingNativeSaveStagingPath;
        private SaveSystem _pendingNativeSaveSystem;
        private UnityAction _pendingNativeSaveFinalize;
        private float _pendingNativeSaveStarted;
        private float _nextNativeSaveStatusAt;
        private int _pendingNativeSaveBibites;
        private int _pendingNativeSavePellets;
        private double _pendingNativeSaveTime;
        private bool _nativeSaveStockContinuation;
        private bool _nativeSaveWrapperWriting;
        private GpuSaveTransaction _nativeSaveTransaction;
        private string _integrationSaveOutput;
        private int _integrationSaveCount;
        private int _integrationSaveCommittedCount;
        private bool _integrationSavePending;
        private float _integrationSaveStartedAt;
        private float _integrationSaveCompletedAt;
        private double _integrationSavedTime;

        private void Awake()
        {
            Instance = this;
            Application.runInBackground = true;
            _showOverlay = Config.Bind("Diagnostics", "ShowOverlay", true,
                "Show requested and actually achieved simulation speed.");
            _shadowValidation = Config.Bind("GPU", "ShadowValidation", false,
                "Batch live brain inputs on the GPU and compare them to CPU outputs without changing the simulation.");
            _shadowSampleLimit = Config.Bind("GPU", "ShadowSamplesPerFrame", 128,
                "Maximum live brain evaluations sampled in each rendered frame.");
            _visionShadowValidation = Config.Bind("GPU", "VisionShadowValidation", false,
                "Batch live vision calculations on the GPU and compare them to CPU outputs without changing the simulation.");
            _visionShadowSampleLimit = Config.Bind("GPU", "VisionShadowSamplesPerFrame", 64,
                "Maximum live vision calculations sampled in each rendered frame.");
            _pheromoneShadowValidation = Config.Bind("GPU", "PheromoneShadowValidation", false,
                "Compare authoritative CUDA pheromone sensing against the stock CPU path.");
            _authoritativeVisionGpu = Config.Bind("GPU", "AuthoritativeVisionGpu", false,
                "Replace stock Physics2D vision lookup and sensing with one world-wide CUDA batch.");
            _authoritativePheromoneGpu = Config.Bind("GPU", "AuthoritativePheromoneGpu", false,
                "Replace per-Bibite Physics2D pheromone sensing with one world-wide CUDA batch.");
            _authoritativeBrainGpu = Config.Bind("GPU", "AuthoritativeBrainGpu", false,
                "Use globally batched CUDA brain outputs. Experimental: synchronizes brain phases and changes update ordering.");
            _gpuBrainMinimumSynapses = Config.Bind("GPU", "BrainGpuMinimumBatchSynapses", 2048,
                "Use CPU brains below this total live-synapse count; set zero to force CUDA.");
            _gpuDeviceIndex = Config.Bind("GPU", "DeviceIndex", 0,
                "CUDA device used for accelerated simulation work.");
            _throughputMode = Config.Bind("Performance", "ThroughputMode", true,
                "At high time warp, prioritize simulation throughput over render frame rate and bypass the stock minimum-FPS governor.");
            _turboRenderFps = Config.Bind("Performance", "TurboRenderFps", 15,
                "Rendered frames per second while throughput mode is active. Simulation ticks continue between rendered frames.");
            _multithreadedPhysics = Config.Bind("Performance", "MultithreadedPhysics", true,
                "Use Unity's multithreaded Physics2D jobs while high-warp throughput mode is active.");
            _nativeGpuVisualWorld = Config.Bind("Native GPU World", "EnabledForNewSimulations", true,
                "Run new simulations in the fully GPU-resident world while retaining the Unity graphics, camera, and GUI.");
            _nativeGpuLoadedSaves = Config.Bind("Native GPU World", "ReplaceLoadedSaves", false,
                "Replace a loaded stock save with a fresh GPU world. Disabled by default to protect save compatibility.");
            _nativePopulationCap = Config.Bind("Native GPU World", "PopulationCap", 2048,
                "Player-selected population cap in the native GPU world (hard maximum 500000). Memory is allocated for this chosen cap only.");
            _nativeInitialPopulation = Config.Bind("Native GPU World", "InitialPopulation", 512,
                "Initial population in the native GPU world.");
            _nativePelletCount = Config.Bind("Native GPU World", "PelletCount", 8192,
                "Maximum GPU plant pellets (0-32768). The stock zone biomass settings can set a lower cap, including zero; this value no longer forces a starting count.");
            _nativeRenderFps = Config.Bind("Native GPU World", "GraphicsFps", 30,
                "Unity graphics and GUI frame rate. GPU simulation ticks independently of this value.");
            _nativeDetailedBibiteLimit = Config.Bind(
                "Native GPU World",
                "TexturedBibiteLimit",
                512,
                "Maximum on-screen Bibites drawn with original procedural textures (0-2048). Off-screen Bibites do not consume the limit. Higher values use more Unity CPU/render time and apply to the next simulation.");
            _nativeExtremeMode = Config.Bind("Native GPU World", "ExtremeThroughput", true,
                "Lower-refresh throughput profile: collisions every eight ticks, persistent food targets, vision lookup every 160 ticks, and brains every eight ticks. Achieved speed depends on the world and GPU.");
            _nativeBrainKernel = Config.Bind("Native GPU World", "BrainKernel", 0,
                "Next new simulation: 0 fused (default), 1 specialized sparse FP32 brain graph, 2 specialized dense FP32 brain graph. Checkpoints retain their saved pipeline; imported brains require a graph. Tensor mode is an experimental headless option only.");
            _nativeBrainKernel.Value = NativeKernelPolicy.Normalize(_nativeBrainKernel.Value);
            _nativePopulationCap.Value = Mathf.Clamp(
                _nativePopulationCap.Value,
                64,
                GpuNativeWorldBridge.MaximumPopulation);
            _nativeInitialPopulation.Value = Mathf.Clamp(
                _nativeInitialPopulation.Value,
                1,
                _nativePopulationCap.Value);
            _nativeDetailedBibiteLimit.Value = Mathf.Clamp(
                _nativeDetailedBibiteLimit.Value,
                0,
                BibiteDetailedSpritePool.MaximumDetailedBibites);
            _nativePopulationCapInput = _nativePopulationCap.Value.ToString(
                CultureInfo.InvariantCulture);
            _nativeInitialPopulationInput = _nativeInitialPopulation.Value.ToString(
                CultureInfo.InvariantCulture);
            _nativeDetailedBibiteLimitInput = _nativeDetailedBibiteLimit.Value.ToString(
                CultureInfo.InvariantCulture);
            try
            {
                _gpuDevices = NativeGpu.GetDevices();
                if (_gpuDevices.Count > 0)
                {
                    _gpuDeviceIndex.Value = Mathf.Clamp(_gpuDeviceIndex.Value, 0, _gpuDevices.Count - 1);
                    NativeGpu.SelectDevice(_gpuDeviceIndex.Value);
                }
            }
            catch (Exception ex)
            {
                Logger.LogError("CUDA device discovery failed: " + ex.Message);
                _gpuDevices.Clear();
                _gpuDeviceIndex.Value = 0;
            }
            _shadowValidator = new GpuShadowValidator(Logger, _gpuDeviceIndex.Value);
            _visionShadowValidator = new GpuVisionShadowValidator(Logger);
            _authoritativeBrainScheduler = new GpuAuthoritativeBrainScheduler(Logger, _gpuDeviceIndex.Value);
            _authoritativeVisionScheduler = new GpuAuthoritativeVisionScheduler(Logger, _gpuDeviceIndex.Value);
            _authoritativePheromoneScheduler = new GpuAuthoritativePheromoneScheduler(Logger, _gpuDeviceIndex.Value);
            _nativeWorldBridge = new GpuNativeWorldBridge(Logger);
            Application.wantsToQuit += OnApplicationWantsToQuit;
            Application.quitting += OnApplicationQuitting;
            _integrationTestSave = Environment.GetEnvironmentVariable("BIBITES_GPU_TEST_SAVE");
            _integrationTestNewSimulation = string.Equals(
                Environment.GetEnvironmentVariable("BIBITES_GPU_TEST_NEW_SIM"),
                "1",
                StringComparison.Ordinal);
            if (_integrationTestNewSimulation)
                _integrationSaveOutput = Environment.GetEnvironmentVariable("BIBITES_GPU_TEST_SAVE_OUTPUT");
            _integrationPlaceNativeBibite = string.Equals(
                Environment.GetEnvironmentVariable("BIBITES_GPU_TEST_PLACE_NATIVE"),
                "1",
                StringComparison.Ordinal);
            string integrationFoodMultiplier = Environment.GetEnvironmentVariable(
                "BIBITES_GPU_TEST_FOOD_DENSITY_MULTIPLIER");
            float parsedFoodMultiplier;
            if (!string.IsNullOrEmpty(integrationFoodMultiplier) &&
                float.TryParse(integrationFoodMultiplier, NumberStyles.Float,
                    CultureInfo.InvariantCulture, out parsedFoodMultiplier) &&
                !float.IsNaN(parsedFoodMultiplier) && !float.IsInfinity(parsedFoodMultiplier))
            {
                _integrationFoodDensityMultiplier = Mathf.Clamp(
                    parsedFoodMultiplier, 0f, 10f);
            }
            string integrationSpeed = Environment.GetEnvironmentVariable("BIBITES_GPU_TEST_SPEED");
            float parsedSpeed;
            if (!string.IsNullOrEmpty(integrationSpeed) &&
                float.TryParse(integrationSpeed, NumberStyles.Float, CultureInfo.InvariantCulture, out parsedSpeed))
            {
                _integrationTestSpeed = Mathf.Clamp(
                    parsedSpeed,
                    1f,
                    TimeWarpSpeeds.Maximum);
            }
            string integrationWorldSize = Environment.GetEnvironmentVariable(
                "BIBITES_GPU_TEST_WORLD_SIZE");
            float parsedWorldSize;
            if (!string.IsNullOrEmpty(integrationWorldSize) &&
                float.TryParse(
                    integrationWorldSize,
                    NumberStyles.Float,
                    CultureInfo.InvariantCulture,
                    out parsedWorldSize))
            {
                _integrationWorldSize = Mathf.Clamp(parsedWorldSize, 100f, 20000f);
            }
            _integrationBibiteCount = ReadIntegrationInt("BIBITES_GPU_TEST_BIBITES", 32, 1, 4096);
            _integrationPelletCount = ReadIntegrationInt("BIBITES_GPU_TEST_PELLETS", 64, 0, 16384);
            _integrationPheromoneCount = ReadIntegrationInt("BIBITES_GPU_TEST_PHEROMONES", 0, 0, 16384);
            _integrationValidation = !string.Equals(
                Environment.GetEnvironmentVariable("BIBITES_GPU_TEST_VALIDATION"),
                "0",
                StringComparison.Ordinal);
            _integrationProfiling = string.Equals(
                Environment.GetEnvironmentVariable("BIBITES_GPU_PROFILE"),
                "1",
                StringComparison.Ordinal);
            _settingsCapturePath = Environment.GetEnvironmentVariable("BIBITES_GPU_TEST_SETTINGS_SCREENSHOT");
            _nativeWorldCapturePath = Environment.GetEnvironmentVariable("BIBITES_GPU_TEST_WORLD_SCREENSHOT");
            _nativeWorldCaptureOpenInformation = string.Equals(
                Environment.GetEnvironmentVariable("BIBITES_GPU_TEST_OPEN_INFO"),
                "1",
                StringComparison.Ordinal);
            _nativeWorldCaptureSelectFirst = string.Equals(
                Environment.GetEnvironmentVariable("BIBITES_GPU_TEST_SELECT_FIRST"),
                "1",
                StringComparison.Ordinal);
            _nativeWorldCaptureOpenSpecies = string.Equals(
                Environment.GetEnvironmentVariable("BIBITES_GPU_TEST_OPEN_SPECIES"),
                "1",
                StringComparison.Ordinal);
            string captureDelays = Environment.GetEnvironmentVariable("BIBITES_GPU_TEST_CAPTURE_DELAYS");
            if (!string.IsNullOrEmpty(captureDelays))
            {
                List<float> parsedDelays = new List<float>();
                string[] values = captureDelays.Split(',');
                for (int index = 0; index < values.Length; index++)
                {
                    float delay;
                    if (float.TryParse(
                            values[index],
                            NumberStyles.Float,
                            CultureInfo.InvariantCulture,
                            out delay))
                    {
                        parsedDelays.Add(Mathf.Clamp(delay, 1f, 60f));
                    }
                }
                if (parsedDelays.Count > 0)
                {
                    parsedDelays.Sort();
                    _nativeWorldCaptureDelays = parsedDelays.ToArray();
                }
            }
            string captureZoom = Environment.GetEnvironmentVariable("BIBITES_GPU_TEST_CAPTURE_ZOOM");
            float parsedCaptureZoom;
            if (!string.IsNullOrEmpty(captureZoom) &&
                float.TryParse(captureZoom, NumberStyles.Float, CultureInfo.InvariantCulture, out parsedCaptureZoom))
            {
                _nativeWorldCaptureZoom = Mathf.Clamp(parsedCaptureZoom, 20f, 1000f);
            }
            string reportSeconds = Environment.GetEnvironmentVariable("BIBITES_GPU_TEST_REPORT_SECONDS");
            float parsedReportSeconds;
            if (!string.IsNullOrEmpty(reportSeconds) &&
                float.TryParse(reportSeconds, NumberStyles.Float, CultureInfo.InvariantCulture, out parsedReportSeconds))
            {
                _integrationReportSeconds = Mathf.Clamp(parsedReportSeconds, 2.5f, 60f);
            }
            if (!string.IsNullOrEmpty(_integrationTestSave) && !File.Exists(_integrationTestSave))
            {
                Logger.LogError("Integration-test save does not exist: " + _integrationTestSave);
                _integrationTestSave = null;
            }
            _harmony = new Harmony(PluginGuid);
            _harmony.PatchAll();
            GameManager.onSceneChange.AddListener(OnGameSceneChange);
            if (_integrationProfiling)
            {
                PerformanceDiagnostics.Install(_harmony);
            }
            _sampleStarted = Time.realtimeSinceStartup;
            _pluginStarted = _sampleStarted;

            Logger.LogInfo("GPU fork runtime loaded.");
            Logger.LogInfo("Unity GPU: " + SystemInfo.graphicsDeviceName);
            Logger.LogInfo("Compute shaders: " + SystemInfo.supportsComputeShaders);
            Logger.LogInfo(DescribeConfiguredGpu());
        }

        private void Update()
        {
            if (_applicationQuit.Requested)
            {
                // Keep ordinary Unity frames alive until the native worker has
                // returned all registered graphics resources. OnDestroy runs
                // too late: its last plugin event may never reach rendering.
                CompletePendingNativeSave();
                bool savePending = !string.IsNullOrEmpty(_pendingNativeSavePath);
                bool released = _nativeWorldBridge == null ||
                    _nativeWorldBridge.AdvanceApplicationQuit(!savePending);
                if (!released && Time.realtimeSinceStartup >= _nextQuitDrainStatusAt)
                {
                    Logger.LogWarning("Waiting for GPU shutdown: savePending=" + savePending +
                        ", " + _nativeWorldBridge.ApplicationQuitState);
                    _nextQuitDrainStatusAt = Time.realtimeSinceStartup + 5f;
                }
                if (_applicationQuit.TryComplete(!savePending && released))
                {
                    Logger.LogInfo("GPU world shutdown completed before Unity graphics teardown; continuing quit.");
                    Application.Quit();
                }
                return;
            }
            _totalRenderedFrames++;
            if (_userSettingsOpen && _userSettingsPanel != null &&
                !_userSettingsPanel.isActiveAndEnabled)
            {
                SetUserSettingsOpen(false);
            }
            CompletePendingNativeSave();
            if (_integrationBenchmarkStarted > 0f && !_integrationResultReported)
            {
                _integrationFrameTimes.Add(Time.unscaledDeltaTime * 1000f);
            }
            bool allowHotkeys = CanUseSimulationHotkeys;
            bool allowCompatibilityHotkeys = allowHotkeys && !IsNativeWorldActive;
            bool control = Input.GetKey(KeyCode.LeftControl) || Input.GetKey(KeyCode.RightControl);
            bool shift = Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift);
            if (allowHotkeys && control && Input.GetKeyDown(KeyCode.PageUp))
            {
                TimeController.targetTimeScale.SetValue(
                    TimeWarpSpeeds.Next(TimeController.targetTimeScale.val));
            }
            else if (allowHotkeys && control && Input.GetKeyDown(KeyCode.PageDown))
            {
                TimeController.targetTimeScale.SetValue(
                    TimeWarpSpeeds.Previous(TimeController.targetTimeScale.val));
            }

            if (allowCompatibilityHotkeys && control && shift && Input.GetKeyDown(KeyCode.F8))
            {
                _shadowValidation.Value = !_shadowValidation.Value;
                Logger.LogInfo("GPU shadow validation " + (_shadowValidation.Value ? "enabled" : "disabled") + ".");
            }
            if (allowCompatibilityHotkeys && control && shift && Input.GetKeyDown(KeyCode.F9))
            {
                _visionShadowValidation.Value = !_visionShadowValidation.Value;
                Logger.LogInfo("GPU vision validation " + (_visionShadowValidation.Value ? "enabled" : "disabled") + ".");
            }
            else if (allowHotkeys && !control && !shift && Input.GetKeyDown(KeyCode.F9) &&
                !ChallengeManager.isChallenge && SaveController.Instance != null)
            {
                SaveController.Instance.QuickLoad();
            }
            if (allowCompatibilityHotkeys && control && shift && Input.GetKeyDown(KeyCode.F10))
            {
                SetAuthoritativeBrainMode(!_authoritativeBrainGpu.Value);
            }

            float now = Time.realtimeSinceStartup;
            _nativeWorldBridge.Update(
                _worldSessionMode.ShouldRunNative(_nativeGpuVisualWorld.Value,
                    _nativeWorldBridge.OwnsSimulation, _nativeWorldBridge.IsStarting),
                _nativeGpuLoadedSaves.Value,
                _gpuDeviceIndex.Value,
                Mathf.Clamp(
                    _nativePopulationCap.Value,
                    64,
                    GpuNativeWorldBridge.MaximumPopulation),
                Mathf.Clamp(
                    _nativeInitialPopulation.Value,
                    1,
                    GpuNativeWorldBridge.MaximumPopulation),
                Mathf.Clamp(_nativePelletCount.Value, 0, 32768),
                Mathf.Clamp(_nativeRenderFps.Value, 10, 60),
                Mathf.Clamp(
                    _nativeDetailedBibiteLimit.Value,
                    0,
                    BibiteDetailedSpritePool.MaximumDetailedBibites),
                _nativeExtremeMode.Value,
                NativeKernelPolicy.Normalize(_nativeBrainKernel.Value));
            RunNativeWorldCapture();
            ApplyPerformancePolicy();
            RunSettingsCapture(now);
            RunIntegrationTest(now);
            RunIntegrationSaveProbe(now);
            float elapsed = now - _sampleStarted;
            if (elapsed >= 0.5f)
            {
                float fixedDelta = Time.fixedDeltaTime;
                _achievedMultiplier = elapsed > 0f ? (_fixedSteps * fixedDelta) / elapsed : 0f;
                _fixedSteps = 0;
                _sampleStarted = now;
            }
        }

        private void RunIntegrationTest(float now)
        {
            if (string.IsNullOrEmpty(_integrationTestSave) && !_integrationTestNewSimulation)
            {
                return;
            }

            if (!_integrationLaunchRequested && now - _pluginStarted >= 8f &&
                SceneManager.GetActiveScene().name == "Menu")
            {
                _integrationLaunchRequested = true;
                _shadowValidation.Value = _integrationValidation;
                _visionShadowValidation.Value = _integrationValidation;
                _pheromoneShadowValidation.Value = _integrationValidation;
                if (_integrationTestNewSimulation)
                {
                    if (_integrationWorldSize > 0f)
                    {
                        ScenarioIndependentSettings.Instance.SimulationSize.SetValue(
                            _integrationWorldSize);
                        Logger.LogInfo("Integration-test world extent set to " +
                            _integrationWorldSize.ToString("0", CultureInfo.InvariantCulture) + ".");
                    }
                    Logger.LogInfo("Starting isolated GPU integration test with a synthetic fresh simulation.");
                    GameManager.StartGame();
                }
                else
                {
                    Logger.LogInfo("Starting isolated GPU integration test from copied save: " + _integrationTestSave);
                    GameManager.StartGame(_integrationTestSave);
                }
                return;
            }

            if (_integrationLaunchRequested && GameManager.isSim &&
                SimulationManager.Instance != null && _integrationSimulationStarted <= 0f)
            {
                _integrationSimulationStarted = now;
                TimeController.targetTimeScale.SetValue(1f);
                TimeController.engineTimeScale.SetValue(1f);
                Logger.LogInfo("GPU integration simulation started; preparing benchmark entities at 1x.");
                if (!_integrationTestNewSimulation)
                {
                    StartIntegrationBenchmark(now);
                }
            }

            if (_integrationTestNewSimulation && !_integrationEntitiesSpawned &&
                _integrationSimulationStarted > 0f && now - _integrationSimulationStarted >= 2f &&
                WorldObjectsSpawner.Instance != null)
            {
                if (_nativeWorldBridge != null && _nativeWorldBridge.OwnsSimulation)
                {
                    if (_integrationPlaceNativeBibite)
                    {
                        SpawnIntegrationNativePlacement();
                    }
                    _integrationEntitiesSpawned = true;
                }
                else
                {
                    SpawnIntegrationEntities();
                }
                StartIntegrationBenchmark(Time.realtimeSinceStartup);
            }

            if (!_integrationFoodDensityChanged &&
                _integrationFoodDensityMultiplier >= 0f &&
                _integrationBenchmarkStarted > 0f &&
                now - _integrationBenchmarkStarted >= 2f &&
                _nativeWorldBridge != null && _nativeWorldBridge.OwnsSimulation)
            {
                _integrationFoodDensityChanged = true;
                int beforeTarget = _nativeWorldBridge.FoodTarget;
                int beforeActive = _nativeWorldBridge.ActivePellets;
                float density = ScenarioIndependentSettings.Instance.biomassDensity.val;
                float requested = density * _integrationFoodDensityMultiplier;
                ScenarioIndependentSettings.Instance.biomassDensity.SetValue(requested);
                Logger.LogInfo("BGF_FOOD_TEST_CHANGE | density " + density + " -> " +
                    requested + " | target " + beforeTarget +
                    " | active " + beforeActive);
            }

            if (!_integrationResultReported && _integrationBenchmarkStarted > 0f &&
                now - _integrationBenchmarkStarted >= _integrationReportSeconds)
            {
                _integrationResultReported = true;
                int bibiteCount = WorldObjectsSpawner.Instance != null ? WorldObjectsSpawner.Instance.nBibite : -1;
                int pelletCount = WorldObjectsSpawner.Instance != null ? WorldObjectsSpawner.Instance.allPellets.Count : -1;
                float elapsed = Mathf.Max(0.001f, now - _integrationBenchmarkStarted);
                long fixedSteps = _totalFixedSteps - _integrationFixedStepsStarted;
                long renderedFrames = _totalRenderedFrames - _integrationFramesStarted;
                float achieved = _nativeWorldBridge != null && _nativeWorldBridge.OwnsSimulation
                    ? (float)_nativeWorldBridge.AchievedMultiplier
                    : fixedSteps * Time.fixedDeltaTime / elapsed;
                float renderedFps = renderedFrames / elapsed;
                float fixedPerFrame = renderedFrames > 0 ? (float)fixedSteps / renderedFrames : 0f;
                float frameP95 = 0f;
                float frameMaximum = 0f;
                int framesOver50Milliseconds = 0;
                if (_integrationFrameTimes.Count > 0)
                {
                    float[] sortedFrameTimes = _integrationFrameTimes.ToArray();
                    Array.Sort(sortedFrameTimes);
                    int p95Index = Mathf.Clamp(
                        Mathf.CeilToInt(sortedFrameTimes.Length * 0.95f) - 1,
                        0,
                        sortedFrameTimes.Length - 1);
                    frameP95 = sortedFrameTimes[p95Index];
                    frameMaximum = sortedFrameTimes[sortedFrameTimes.Length - 1];
                    for (int index = 0; index < sortedFrameTimes.Length; index++)
                    {
                        if (sortedFrameTimes[index] > 50f)
                        {
                            framesOver50Milliseconds++;
                        }
                    }
                }
                Logger.LogInfo("BGF_INTEGRATION_RESULT | bibites " + bibiteCount + " | pellets " + pelletCount +
                    " | target " + TimeController.targetTimeScale.val.ToString("0.0") + "x" +
                    " | engine " + TimeController.engineTimeScale.val.ToString("0.0") + "x" +
                    " | achieved " + achieved.ToString("0.0") + "x" +
                    " | render " + renderedFps.ToString("0.0") + " fps" +
                    " | frame p95/max " + frameP95.ToString("0.0") + "/" +
                    frameMaximum.ToString("0.0") + " ms" +
                    " | frames over 50 ms " + framesOver50Milliseconds +
                    " | fixed/frame " + fixedPerFrame.ToString("0.00") +
                    " | maxDelta " + Time.maximumDeltaTime.ToString("0.0000") +
                    " | " + (_shadowValidation.Value
                        ? _shadowValidator.StatusText
                        : "GPU brain validation off") +
                    " | " + (_visionShadowValidation.Value
                        ? _visionShadowValidator.StatusText
                        : "GPU vision validation off") +
                    " | " + _authoritativeVisionScheduler.StatusText +
                    " | " + _authoritativePheromoneScheduler.StatusText +
                    " | " + _authoritativeBrainScheduler.StatusText +
                    " | native living " + (_nativeWorldBridge != null ? _nativeWorldBridge.LivingBibites : 0) +
                    " | native food " + (_nativeWorldBridge != null ? _nativeWorldBridge.ActivePellets : 0) +
                    " active / " +
                    (_nativeWorldBridge != null ? _nativeWorldBridge.FoodTarget : 0) +
                    " target" +
                    " / " + (_nativeWorldBridge != null ? _nativeWorldBridge.PelletsEaten : 0ul) +
                    " eaten" +
                    " | CPU snapshot " + (_nativeWorldBridge != null
                        ? _nativeWorldBridge.LatestSnapshotMilliseconds.ToString("0.00")
                        : "0.00") + " ms / " + (_nativeWorldBridge != null
                        ? _nativeWorldBridge.PresentationBibites
                        : 0) + " records (max " + (_nativeWorldBridge != null
                        ? _nativeWorldBridge.MaximumSnapshotMilliseconds.ToString("0.00")
                        : "0.00") + " ms)" +
                    " | render max gap " + (_nativeWorldBridge != null
                        ? _nativeWorldBridge.MaximumRenderGapMilliseconds.ToString("0.00")
                        : "0.00") + " ms / " + (_nativeWorldBridge != null
                        ? _nativeWorldBridge.CompletedRenderUpdates
                        : 0) + " updates" +
                    " | max simulation batch " + (_nativeWorldBridge != null
                        ? _nativeWorldBridge.MaximumStepMilliseconds.ToString("0.00")
                        : "0.00") + " ms" +
                    " | native placed " + (_nativeWorldBridge != null ? _nativeWorldBridge.SuccessfulPlacements : 0));
                if (_integrationProfiling)
                {
                    Logger.LogInfo(PerformanceDiagnostics.StopAndReport(fixedSteps, elapsed));
                }
            }
        }

        private void RunSettingsCapture(float now)
        {
            if (string.IsNullOrEmpty(_settingsCapturePath) || _settingsCaptureCompleted)
            {
                return;
            }

            if (!_settingsCaptureOpened && now - _pluginStarted >= 10f)
            {
                UserSettingsManager settings = UnityEngine.Object.FindFirstObjectByType<UserSettingsManager>(
                    FindObjectsInactive.Include);
                if (settings == null)
                {
                    return;
                }
                settings.OpenPanel();
                SetUserSettingsOpen(true, settings);
                SetGpuSettingsOpen(true);
                _settingsCaptureOpened = true;
                _settingsCaptureOpenedAt = now;
                return;
            }

            if (_settingsCaptureOpened && now - _settingsCaptureOpenedAt >= 2f)
            {
                ScreenCapture.CaptureScreenshot(_settingsCapturePath);
                _settingsCaptureCompleted = true;
                Logger.LogInfo("Captured GPU settings test screenshot: " + _settingsCapturePath);
            }
        }

        private void RunNativeWorldCapture()
        {
            if (string.IsNullOrEmpty(_nativeWorldCapturePath) || _nativeWorldCaptureCompleted ||
                _nativeWorldBridge == null || !_nativeWorldBridge.OwnsSimulation)
            {
                return;
            }
            if (_nativeWorldCaptureReadyAt <= 0f)
            {
                _nativeWorldCaptureReadyAt = Time.realtimeSinceStartup;
                if (CameraManager.instance != null)
                {
                    CameraManager.instance.transform.position = new Vector3(0f, 0f,
                        CameraManager.instance.transform.position.z);
                    CameraManager.instance.SetCamSize(_nativeWorldCaptureZoom);
                }
                if (_nativeWorldCaptureOpenInformation && UIScripts.InformationPanel.Instance != null)
                {
                    UIScripts.InformationPanel.Instance.OpenPanel();
                }
                if (_nativeWorldCaptureSelectFirst)
                {
                    _nativeWorldBridge.SelectFirstBibiteForCapture();
                }
                if (_nativeWorldCaptureOpenSpecies)
                {
                    _nativeWorldBridge.OpenNativeSpeciesPanel();
                }
                return;
            }
            if (_nativeWorldCaptureIndex >= _nativeWorldCaptureDelays.Length ||
                Time.realtimeSinceStartup - _nativeWorldCaptureReadyAt <
                    _nativeWorldCaptureDelays[_nativeWorldCaptureIndex])
            {
                return;
            }
            string capturePath = _nativeWorldCapturePath;
            if (_nativeWorldCaptureDelays.Length > 1)
            {
                string folder = Path.GetDirectoryName(_nativeWorldCapturePath) ?? string.Empty;
                string name = Path.GetFileNameWithoutExtension(_nativeWorldCapturePath);
                string extension = Path.GetExtension(_nativeWorldCapturePath);
                capturePath = Path.Combine(
                    folder,
                    name + "-" + _nativeWorldCaptureDelays[_nativeWorldCaptureIndex].ToString(
                        "0",
                        CultureInfo.InvariantCulture) + "s" + extension);
            }
            ScreenCapture.CaptureScreenshot(capturePath);
            _nativeWorldCaptureIndex++;
            _nativeWorldCaptureCompleted = _nativeWorldCaptureIndex >= _nativeWorldCaptureDelays.Length;
            Logger.LogInfo("Captured native GPU visual world screenshot: " + capturePath);
        }

        private void SpawnIntegrationEntities()
        {
            _integrationEntitiesSpawned = true;
            if (GameManager.defaultBibites.Count == 0)
            {
                Logger.LogError("No built-in Bibite templates were available for the integration test.");
                return;
            }

            string templateFile = GameManager.defaultBibites[0] + ".bb8template";
            BibiteTemplate template = new BibiteTemplate(templateFile);
            int bibiteCount = _integrationBibiteCount;
            int pelletCount = _integrationPelletCount;
            int registeredBrainCount = 0;
            for (int i = 0; i < bibiteCount; i++)
            {
                float angle = 2f * Mathf.PI * i / bibiteCount;
                float radius = 35f + 12f * (i % 3);
                Vector3 position = new Vector3(Mathf.Cos(angle) * radius, Mathf.Sin(angle) * radius, 0f);
                GameObject spawned = WorldObjectsSpawner.Instance.SpawnBibiteFromTemplate(
                    template,
                    RandomizeGenes.No,
                    Tagging.NoTagging,
                    GrowthAtSpawn.Adult,
                    position,
                    angle * Mathf.Rad2Deg);
                NEATBrain brain = spawned != null ? spawned.GetComponent<NEATBrain>() : null;
                if (brain != null)
                {
                    RegisterAuthoritativeBrain(brain);
                    registeredBrainCount++;
                }
                if (_integrationPheromoneCount > 0)
                {
                    Pherosense sensor = spawned != null ? spawned.GetComponent<Pherosense>() : null;
                    if (sensor != null)
                    {
                        sensor.StartPherosensing();
                    }
                }
            }
            for (int i = 0; i < pelletCount; i++)
            {
                float angle = 2f * Mathf.PI * i / pelletCount;
                float radius = 18f + 55f * ((i % 7) / 6f);
                Vector3 position = new Vector3(Mathf.Cos(angle) * radius, Mathf.Sin(angle) * radius, 0f);
                WorldObjectsSpawner.Instance.SpawnPlantPellet(position, amount: 20f);
            }
            for (int i = 0; i < _integrationPheromoneCount; i++)
            {
                float angle = 2f * Mathf.PI * i / Math.Max(1, _integrationPheromoneCount);
                float radius = 12f + 70f * ((i % 9) / 8f);
                Vector3 position = new Vector3(Mathf.Cos(angle) * radius, Mathf.Sin(angle) * radius, 0f);
                PheromoneSpot spot = WorldObjectsSpawner.Instance
                    .GenerateNewPheromonesSource(position)
                    .GetComponent<PheromoneSpot>();
                spot.SetPhero(
                    1f + i % 3,
                    0.5f + i % 5,
                    0.25f + i % 7,
                    new Vector2(Mathf.Cos(angle), Mathf.Sin(angle)));
                spot.dissipate = false;
            }
            Logger.LogInfo("Spawned " + bibiteCount + " Bibites and " + pelletCount +
                " pellets plus " + _integrationPheromoneCount +
                " pheromone spots for isolated live GPU validation; resolved " + registeredBrainCount +
                " brain components directly.");
        }

        private void SpawnIntegrationNativePlacement()
        {
            if (GameManager.defaultBibites.Count == 0 || WorldObjectsSpawner.Instance == null)
            {
                Logger.LogError("BGF_NATIVE_PLACEMENT_TEST | no built-in Bibite template was available.");
                return;
            }
            string templateFile = GameManager.defaultBibites[0] + ".bb8template";
            BibiteTemplate template = new BibiteTemplate(templateFile);
            GameObject stockObject = WorldObjectsSpawner.Instance.SpawnBibiteFromTemplate(
                template,
                RandomizeGenes.No,
                Tagging.NoTagging,
                GrowthAtSpawn.Adult,
                new Vector3(0f, 0f, 0f),
                0f);
            Logger.LogInfo("BGF_NATIVE_PLACEMENT_TEST | request sent through the stock placer | stock object " +
                (stockObject == null ? "suppressed" : "unexpectedly created") + ".");
        }

        private void StartIntegrationBenchmark(float now)
        {
            _integrationBenchmarkStarted = now;
            _integrationFixedStepsStarted = _totalFixedSteps;
            _integrationFramesStarted = _totalRenderedFrames;
            _integrationFrameTimes.Clear();
            if (_integrationProfiling)
            {
                PerformanceDiagnostics.Start();
            }
            TimeController.targetTimeScale.SetValue(_integrationTestSpeed);
            TimeController.engineTimeScale.SetValue(_integrationTestSpeed);
            Logger.LogInfo("GPU integration benchmark started at requested " +
                _integrationTestSpeed.ToString("0", CultureInfo.InvariantCulture) + "x.");
        }

        private void LateUpdate()
        {
            if (_nativeWorldBridge != null && _nativeWorldBridge.OwnsSimulation)
            {
                _shadowValidator.ClearPending();
                _visionShadowValidator.ClearPending();
                return;
            }
            if (_shadowValidation != null && _shadowValidation.Value)
            {
                _shadowValidator.Flush();
            }
            else
            {
                _shadowValidator.ClearPending();
            }
            if (_visionShadowValidation != null && _visionShadowValidation.Value)
            {
                _visionShadowValidator.Flush();
            }
            else
            {
                _visionShadowValidator.ClearPending();
            }
            _authoritativeVisionValidationSamples = 0;
            _authoritativePheromoneValidationSamples = 0;
        }

        private void FixedUpdate()
        {
            _fixedSteps++;
            _totalFixedSteps++;
            if (ShouldSuppressStockWorld)
            {
                return;
            }
            if (_authoritativePheromoneGpu != null && _authoritativePheromoneGpu.Value &&
                !_authoritativePheromoneScheduler.FixedStep())
            {
                _authoritativePheromoneGpu.Value = false;
            }
            if (_authoritativeVisionGpu != null && _authoritativeVisionGpu.Value &&
                !_authoritativeVisionScheduler.FixedStep())
            {
                _authoritativeVisionGpu.Value = false;
            }
            if (_authoritativeBrainGpu != null && _authoritativeBrainGpu.Value &&
                !_authoritativeBrainScheduler.FixedStep(Math.Max(0, _gpuBrainMinimumSynapses.Value)))
            {
                _authoritativeBrainGpu.Value = false;
            }
        }

        private void OnGUI()
        {
            bool showDiagnostics = GameManager.isSim && !_userSettingsOpen &&
                _showOverlay != null && _showOverlay.Value;
            if (_nativeWorldBridge != null &&
                (_nativeWorldBridge.OwnsSimulation || _nativeWorldBridge.IsStarting))
            {
                _nativeWorldBridge.DrawOverlay(showDiagnostics);
            }
            else if (showDiagnostics)
            {
                string text = "Bibites GPU Fork  |  requested " +
                    TimeController.targetTimeScale.val.ToString("0") + "x  |  achieved " +
                    _achievedMultiplier.ToString("0.0") + "x";
                GUI.Box(new Rect(12f, 12f, 520f, 28f), text);
                string shadowText = _shadowValidation != null && _shadowValidation.Value
                    ? _shadowValidator.StatusText
                    : "GPU brain validation off (Ctrl+Shift+F8 to toggle)";
                GUI.Box(new Rect(12f, 42f, 520f, 28f), shadowText);
                string visionText = _visionShadowValidation != null && _visionShadowValidation.Value
                    ? _visionShadowValidator.StatusText
                    : "GPU vision validation off (Ctrl+Shift+F9 to toggle)";
                GUI.Box(new Rect(12f, 72f, 520f, 28f), visionText);
                string fastText = _authoritativeBrainGpu != null && _authoritativeBrainGpu.Value
                    ? _authoritativeBrainScheduler.StatusText
                    : "GPU brain fast mode off (Ctrl+Shift+F10 to toggle)";
                GUI.Box(new Rect(12f, 102f, 520f, 28f), fastText);
                string visionFastText = _authoritativeVisionGpu != null && _authoritativeVisionGpu.Value
                    ? _authoritativeVisionScheduler.StatusText
                    : "GPU vision fast mode off (Settings > GPU settings)";
                GUI.Box(new Rect(12f, 132f, 520f, 28f), visionFastText);
                string pheromoneFastText = _authoritativePheromoneGpu != null && _authoritativePheromoneGpu.Value
                    ? _authoritativePheromoneScheduler.StatusText
                    : "GPU pheromone mode off (Settings > GPU settings)";
                GUI.Box(new Rect(12f, 162f, 520f, 28f), pheromoneFastText);
            }

            DrawGpuSettingsTab();
        }

        private void DrawGpuSettingsTab()
        {
            if (!_userSettingsOpen)
            {
                return;
            }

            Rect tabRect = GpuSettingsTabRect;
            if (!_gpuSettingsOpen && !GpuMenuIntegration.HasSettingsButton(_userSettingsPanel) &&
                GUI.Button(tabRect, "GPU settings"))
            {
                SetGpuSettingsOpen(!_gpuSettingsOpen);
            }

            if (_gpuSettingsOpen)
            {
                FitGpuSettingsWindow(false);
                Color previousColor = GUI.color;
                GUI.color = new Color(0f, 0f, 0f, 0.55f);
                GUI.DrawTexture(new Rect(0f, 0f, Screen.width, Screen.height), Texture2D.whiteTexture);
                GUI.color = new Color(0.12f, 0.13f, 0.15f, 1f);
                GUI.DrawTexture(_gpuSettingsWindow, Texture2D.whiteTexture);
                GUI.color = previousColor;
                GUISkin previousSkin = GUI.skin;
                if (_gpuSettingsSkin == null)
                {
                    _gpuSettingsSkin = UnityEngine.Object.Instantiate(previousSkin);
                    _gpuSettingsSkin.label.wordWrap = true;
                    _gpuSettingsSkin.label.fontSize = 13;
                    _gpuSettingsSkin.label.normal.textColor = Color.white;
                    _gpuSettingsSkin.toggle.wordWrap = true;
                }
                GUI.skin = _gpuSettingsSkin;
                _gpuSettingsWindow = GUI.Window(
                    73031,
                    _gpuSettingsWindow,
                    DrawGpuSettingsWindow,
                    "Bibites GPU acceleration");
                GUI.skin = previousSkin;
            }
        }

        private void DrawGpuSettingsWindow(int windowId)
        {
            GUILayout.Label("GPU-specific options. The original Graphics, Game and Hotkeys tabs remain available.");
            _gpuSettingsScroll = GUILayout.BeginScrollView(_gpuSettingsScroll,
                GUILayout.Height(Mathf.Max(80f, _gpuSettingsWindow.height - 100f)));
            GUILayout.Label("CUDA device");
            if (_gpuDevices.Count == 0)
            {
                GUILayout.Label("No compatible NVIDIA CUDA device was found.");
            }
            else
            {
                for (int i = 0; i < _gpuDevices.Count; i++)
                {
                    GpuDeviceInfo device = _gpuDevices[i];
                    string marker = device.Index == _gpuDeviceIndex.Value ? "Selected: " : "Use: ";
                    if (GUILayout.Button(marker + device.Name, GUILayout.Height(32f)) &&
                        device.Index != _gpuDeviceIndex.Value)
                    {
                        SelectGpuDevice(device.Index);
                    }
                }
            }

            GUILayout.Space(10f);
            GUILayout.Label("Fully GPU-resident visual world");
            bool nativeWorld = GUILayout.Toggle(
                _nativeGpuVisualWorld.Value,
                "Use native GPU world for new simulations");
            if (nativeWorld != _nativeGpuVisualWorld.Value)
            {
                _nativeGpuVisualWorld.Value = nativeWorld;
            }
            GUILayout.Label(_nativeWorldBridge.StatusText);
            GUILayout.Label(_nativeWorldBridge.FoodStatusText);
            GUILayout.Label("Zone biomass sets the plant-pellet cap; the GPU PelletCount ceiling also applies. Fertility and pellet energy update live.");
            GUILayout.Label("Compact-model limits: plant spawning uses projected zones, but original towers and some stock biology/physics settings remain unsupported. GPU menus do not imply full simulation parity.");
            if (_nativeWorldBridge.OwnsSimulation)
            {
            GUILayout.Label("Population, texture limit and Extreme mode apply to the next simulation. Graphics FPS applies live.");
            }
            else
            {
                GUILayout.Label("Keeps the original menus, camera, world graphics, and GUI.");
            }
            GUILayout.Label("Population cap");
            GUILayout.BeginHorizontal();
            int[] populationCaps = { 256, 512, 2048, 8192, 32768 };
            for (int i = 0; i < populationCaps.Length; i++)
            {
                int cap = populationCaps[i];
                if (GUILayout.Button(
                    (_nativePopulationCap.Value == cap ? "[" : string.Empty) +
                    cap.ToString("N0", CultureInfo.InvariantCulture) +
                    (_nativePopulationCap.Value == cap ? "]" : string.Empty)))
                {
                    _nativePopulationCap.Value = cap;
                    _nativeInitialPopulation.Value = Mathf.Min(_nativeInitialPopulation.Value, cap);
                    _nativePopulationCapInput = cap.ToString(CultureInfo.InvariantCulture);
                    _nativeInitialPopulationInput = _nativeInitialPopulation.Value.ToString(
                        CultureInfo.InvariantCulture);
                }
            }
            GUILayout.EndHorizontal();
            GUILayout.BeginHorizontal();
            int[] largePopulationCaps = { 65536, 131072, 262144, 500000 };
            for (int i = 0; i < largePopulationCaps.Length; i++)
            {
                int cap = largePopulationCaps[i];
                if (GUILayout.Button(
                    (_nativePopulationCap.Value == cap ? "[" : string.Empty) +
                    cap.ToString("N0", CultureInfo.InvariantCulture) +
                    (_nativePopulationCap.Value == cap ? "]" : string.Empty)))
                {
                    _nativePopulationCap.Value = cap;
                    _nativeInitialPopulation.Value = Mathf.Min(_nativeInitialPopulation.Value, cap);
                    _nativePopulationCapInput = cap.ToString(CultureInfo.InvariantCulture);
                    _nativeInitialPopulationInput = _nativeInitialPopulation.Value.ToString(
                        CultureInfo.InvariantCulture);
                }
            }
            GUILayout.EndHorizontal();
            GUILayout.BeginHorizontal();
            GUILayout.Label("Custom cap", GUILayout.Width(90f));
            _nativePopulationCapInput = GUILayout.TextField(
                _nativePopulationCapInput ?? string.Empty,
                GUILayout.Width(150f));
            if (GUILayout.Button("Apply cap", GUILayout.Width(100f)))
            {
                int requestedCap;
                if (int.TryParse(
                    (_nativePopulationCapInput ?? string.Empty).Replace(",", string.Empty),
                    NumberStyles.Integer,
                    CultureInfo.InvariantCulture,
                    out requestedCap))
                {
                    requestedCap = Mathf.Clamp(
                        requestedCap,
                        64,
                        GpuNativeWorldBridge.MaximumPopulation);
                    _nativePopulationCap.Value = requestedCap;
                    _nativeInitialPopulation.Value = Mathf.Min(
                        _nativeInitialPopulation.Value,
                        requestedCap);
                    _nativePopulationCapInput = requestedCap.ToString(
                        CultureInfo.InvariantCulture);
                    _nativeInitialPopulationInput = _nativeInitialPopulation.Value.ToString(
                        CultureInfo.InvariantCulture);
                    _nativePopulationInputStatus = "Population cap set to " +
                        requestedCap.ToString("N0", CultureInfo.InvariantCulture) + ".";
                }
                else
                {
                    _nativePopulationInputStatus = "Enter a whole number from 64 to 500,000.";
                }
            }
            GUILayout.EndHorizontal();
            double gibibyte = 1024.0 * 1024.0 * 1024.0;
            double compactGpuGiB = (
                _nativePopulationCap.Value * 1846.0 +
                Math.Min(_nativePopulationCap.Value, 8192) * 80.0 +
                64.0 * 1024.0 * 1024.0) / gibibyte;
            double placedBrainReserveGiB =
                _nativePopulationCap.Value * 4608.0 / gibibyte;
            GUILayout.Label("Compact world allocation is approximately " +
                compactGpuGiB.ToString("0.00", CultureInfo.InvariantCulture) +
                " GiB plus CUDA/Unity overhead.");
            GUILayout.Label("The first placed stock brain lazily reserves up to " +
                placedBrainReserveGiB.ToString("0.00", CultureInfo.InvariantCulture) +
                " GiB for its mutable descendant state; topology is shared.");
            GUILayout.Label("Initial population");
            GUILayout.BeginHorizontal();
            int[] initialPopulations = { 128, 256, 512, 2048, 8192 };
            for (int i = 0; i < initialPopulations.Length; i++)
            {
                int population = initialPopulations[i];
                if (population > _nativePopulationCap.Value) continue;
                if (GUILayout.Button(
                    (_nativeInitialPopulation.Value == population ? "[" : string.Empty) +
                    population.ToString("N0", CultureInfo.InvariantCulture) +
                    (_nativeInitialPopulation.Value == population ? "]" : string.Empty)))
                {
                    _nativeInitialPopulation.Value = population;
                    _nativeInitialPopulationInput = population.ToString(
                        CultureInfo.InvariantCulture);
                }
            }
            GUILayout.EndHorizontal();
            GUILayout.BeginHorizontal();
            GUILayout.Label("Custom start", GUILayout.Width(90f));
            _nativeInitialPopulationInput = GUILayout.TextField(
                _nativeInitialPopulationInput ?? string.Empty,
                GUILayout.Width(150f));
            if (GUILayout.Button("Apply start", GUILayout.Width(100f)))
            {
                int requestedInitial;
                if (int.TryParse(
                    (_nativeInitialPopulationInput ?? string.Empty).Replace(",", string.Empty),
                    NumberStyles.Integer,
                    CultureInfo.InvariantCulture,
                    out requestedInitial))
                {
                    requestedInitial = Mathf.Clamp(
                        requestedInitial,
                        1,
                        _nativePopulationCap.Value);
                    _nativeInitialPopulation.Value = requestedInitial;
                    _nativeInitialPopulationInput = requestedInitial.ToString(
                        CultureInfo.InvariantCulture);
                    _nativePopulationInputStatus = "Initial population set to " +
                        requestedInitial.ToString("N0", CultureInfo.InvariantCulture) + ".";
                }
                else
                {
                    _nativePopulationInputStatus = "Enter a whole starting population within the chosen cap.";
                }
            }
            GUILayout.EndHorizontal();
            if (!string.IsNullOrEmpty(_nativePopulationInputStatus))
            {
                GUILayout.Label(_nativePopulationInputStatus);
            }
            GUILayout.Label("Graphics / GUI rate (simulation remains independent)");
            GUILayout.BeginHorizontal();
            int[] nativeRenderRates = { 15, 20, 30, 60 };
            for (int i = 0; i < nativeRenderRates.Length; i++)
            {
                int rate = nativeRenderRates[i];
                if (GUILayout.Button(
                    (_nativeRenderFps.Value == rate ? "[" : string.Empty) + rate + " FPS" +
                    (_nativeRenderFps.Value == rate ? "]" : string.Empty)))
                {
                    _nativeRenderFps.Value = rate;
                }
            }
            GUILayout.EndHorizontal();
            GUILayout.Label("Original textured Bibites on screen (next simulation)");
            GUILayout.BeginHorizontal();
            int[] detailedBibiteLimits = { 0, 96, 256, 512, 1024, 2048 };
            for (int i = 0; i < detailedBibiteLimits.Length; i++)
            {
                int limit = detailedBibiteLimits[i];
                string label = limit == 0
                    ? "Off"
                    : limit.ToString("N0", CultureInfo.InvariantCulture);
                if (GUILayout.Button(
                    (_nativeDetailedBibiteLimit.Value == limit ? "[" : string.Empty) +
                    label +
                    (_nativeDetailedBibiteLimit.Value == limit ? "]" : string.Empty)))
                {
                    _nativeDetailedBibiteLimit.Value = limit;
                    _nativeDetailedBibiteLimitInput = limit.ToString(
                        CultureInfo.InvariantCulture);
                    _nativeGraphicsInputStatus = "Textured Bibite limit set to " +
                        label + ". Restart the simulation to apply it.";
                }
            }
            GUILayout.EndHorizontal();
            GUILayout.BeginHorizontal();
            GUILayout.Label("Custom textures", GUILayout.Width(110f));
            _nativeDetailedBibiteLimitInput = GUILayout.TextField(
                _nativeDetailedBibiteLimitInput ?? string.Empty,
                GUILayout.Width(130f));
            if (GUILayout.Button("Apply", GUILayout.Width(90f)))
            {
                int requestedTextures;
                if (int.TryParse(
                    (_nativeDetailedBibiteLimitInput ?? string.Empty).Replace(",", string.Empty),
                    NumberStyles.Integer,
                    CultureInfo.InvariantCulture,
                    out requestedTextures))
                {
                    requestedTextures = Mathf.Clamp(
                        requestedTextures,
                        0,
                        BibiteDetailedSpritePool.MaximumDetailedBibites);
                    _nativeDetailedBibiteLimit.Value = requestedTextures;
                    _nativeDetailedBibiteLimitInput = requestedTextures.ToString(
                        CultureInfo.InvariantCulture);
                    _nativeGraphicsInputStatus = "Textured Bibite limit set to " +
                        requestedTextures.ToString("N0", CultureInfo.InvariantCulture) +
                        ". Restart the simulation to apply it.";
                }
                else
                {
                    _nativeGraphicsInputStatus = "Enter a whole number from 0 to 2,048.";
                }
            }
            GUILayout.EndHorizontal();
            GUILayout.Label(
                "The remaining population keeps the lightweight GPU silhouette. " +
                "Large texture limits can reduce frame rate.");
            if (!string.IsNullOrEmpty(_nativeGraphicsInputStatus))
            {
                GUILayout.Label(_nativeGraphicsInputStatus);
            }
            bool extremeNative = GUILayout.Toggle(
                _nativeExtremeMode.Value,
                "Extreme GPU throughput profile (lower refresh cadence)");
            if (extremeNative != _nativeExtremeMode.Value)
            {
                _nativeExtremeMode.Value = extremeNative;
            }
            GUILayout.Label(extremeNative
                ? "Extreme: collisions are solved every 8 ticks, targets are retained, vision refreshes every 160 ticks, and brains every 8 ticks. Changes apply to the next simulation."
                : "Exact profile: collisions run every tick and sensing/brains update at the standard cadence.");
            GUILayout.Label("Native brain kernel (next new simulation)");
            _nativeBrainKernel.Value = GUILayout.SelectionGrid(
                NativeKernelPolicy.Normalize(_nativeBrainKernel.Value), NativeKernelLabels, 3);
            GUILayout.Label("Keep Fused unless a comparison on your GPU shows a graph is faster. " +
                "All three retain FP32 accumulation. Imported brains use a graph; loaded checkpoints retain their saved kernel.");
            GUILayout.Label("Stock saves stay in compatibility mode; loaded saves are not replaced.");

            GUILayout.Space(10f);
            bool restoreEnabled = GUI.enabled;
            bool nativeWorldInUse = _nativeWorldBridge.OwnsSimulation || _nativeWorldBridge.IsStarting;
            GUILayout.Label("Stock-world compatibility acceleration");
            GUILayout.Label(nativeWorldInUse
                ? "These experimental options do not apply to the GPU-resident world and are disabled while it is running."
                : "The options below accelerate the original CPU world. They are separate from native GPU-world mode.");
            GUI.enabled = restoreEnabled && !nativeWorldInUse;
            if (GUILayout.Button("Enable maximum GPU offload", GUILayout.Height(34f)))
            {
                _shadowValidation.Value = false;
                _visionShadowValidation.Value = false;
                _pheromoneShadowValidation.Value = false;
                if (!_authoritativeVisionGpu.Value) SetAuthoritativeVisionMode(true);
                if (!_authoritativePheromoneGpu.Value) SetAuthoritativePheromoneMode(true);
                if (!_authoritativeBrainGpu.Value) SetAuthoritativeBrainMode(true);
                _throughputMode.Value = true;
            }
            bool gpuBrains = GUILayout.Toggle(
                _authoritativeBrainGpu.Value,
                "Authoritative GPU brains (experimental)");
            if (gpuBrains != _authoritativeBrainGpu.Value)
            {
                SetAuthoritativeBrainMode(gpuBrains);
            }
            GUILayout.Label("GPU brain auto threshold (total live synapses)");
            GUILayout.BeginHorizontal();
            int[] brainThresholds = { 0, 512, 2048, 8192 };
            for (int i = 0; i < brainThresholds.Length; i++)
            {
                int threshold = brainThresholds[i];
                string label = threshold == 0 ? "Force GPU" : threshold.ToString();
                if (GUILayout.Button(
                    (_gpuBrainMinimumSynapses.Value == threshold ? "[" : string.Empty) + label +
                    (_gpuBrainMinimumSynapses.Value == threshold ? "]" : string.Empty)))
                {
                    _gpuBrainMinimumSynapses.Value = threshold;
                }
            }
            GUILayout.EndHorizontal();

            bool gpuVision = GUILayout.Toggle(
                _authoritativeVisionGpu.Value,
                "Authoritative GPU vision (experimental)");
            if (gpuVision != _authoritativeVisionGpu.Value)
            {
                SetAuthoritativeVisionMode(gpuVision);
            }

            bool gpuPheromones = GUILayout.Toggle(
                _authoritativePheromoneGpu.Value,
                "Authoritative GPU pheromone sensing (experimental)");
            if (gpuPheromones != _authoritativePheromoneGpu.Value)
            {
                SetAuthoritativePheromoneMode(gpuPheromones);
            }

            bool throughput = GUILayout.Toggle(
                _throughputMode.Value,
                "Throughput mode above 25x");
            if (throughput != _throughputMode.Value)
            {
                _throughputMode.Value = throughput;
                if (!throughput)
                {
                    RestorePerformancePolicy();
                }
            }

            bool multithreadedPhysics = GUILayout.Toggle(
                _multithreadedPhysics.Value,
                "Multithread Unity Physics2D");
            if (multithreadedPhysics != _multithreadedPhysics.Value)
            {
                _multithreadedPhysics.Value = multithreadedPhysics;
                if (_performancePolicyApplied)
                {
                    ApplyPhysicsJobPolicy();
                }
            }

            GUILayout.Label("Turbo render rate (simulation continues between frames)");
            GUILayout.BeginHorizontal();
            int[] renderRates = { 5, 10, 15, 30 };
            for (int i = 0; i < renderRates.Length; i++)
            {
                int rate = renderRates[i];
                if (GUILayout.Button(
                    (_turboRenderFps.Value == rate ? "[" : string.Empty) + rate + " FPS" +
                    (_turboRenderFps.Value == rate ? "]" : string.Empty)))
                {
                    _turboRenderFps.Value = rate;
                }
            }
            GUILayout.EndHorizontal();

            GUI.enabled = restoreEnabled;
            GUILayout.Space(10f);
            GUILayout.Label("Diagnostics");
            _showOverlay.Value = GUILayout.Toggle(_showOverlay.Value, "Show simulation performance overlay");
            GUI.enabled = restoreEnabled && !nativeWorldInUse;
            _shadowValidation.Value = GUILayout.Toggle(_shadowValidation.Value, "Validate GPU brains against CPU");
            _visionShadowValidation.Value = GUILayout.Toggle(_visionShadowValidation.Value, "Validate GPU vision against CPU");
            _pheromoneShadowValidation.Value = GUILayout.Toggle(
                _pheromoneShadowValidation.Value,
                "Validate GPU pheromones against CPU");
            GUI.enabled = restoreEnabled;
            GUILayout.Label("Requested: " + TimeController.targetTimeScale.val.ToString("0") + "x");
            GUILayout.Label("Engine setting: " + TimeController.engineTimeScale.val.ToString("0.0") + "x");
            GUILayout.Label("Measured: " + (nativeWorldInUse
                ? _nativeWorldBridge.AchievedMultiplier : _achievedMultiplier).ToString("0.0") + "x");
            GUILayout.Label("Note: validation modes duplicate work and reduce speed.");
            GUILayout.Label("For small/simple brains, CPU can be faster than a synchronized GPU batch.");
            GUILayout.Label("GPU vision bypasses stock Physics2D sight queries and may alter edge-case ordering.");

            GUILayout.EndScrollView();
            GUILayout.Space(4f);
            if (GUILayout.Button("Close GPU settings"))
            {
                SetGpuSettingsOpen(false);
            }
            GUI.DragWindow(new Rect(0f, 0f, 10000f, 28f));
        }

        private string DescribeConfiguredGpu()
        {
            if (_gpuDevices.Count == 0)
            {
                return "No CUDA devices available.";
            }
            string names = string.Empty;
            for (int i = 0; i < _gpuDevices.Count; i++)
            {
                names += (i == 0 ? string.Empty : ", ") + _gpuDevices[i].Name;
            }
            return _gpuDevices.Count + " CUDA device(s): " + names +
                " | selected GPU " + _gpuDeviceIndex.Value + ": " +
                _gpuDevices[_gpuDeviceIndex.Value].Name;
        }

        private void SelectGpuDevice(int deviceIndex)
        {
            if (deviceIndex < 0 || deviceIndex >= _gpuDevices.Count)
            {
                return;
            }
            try
            {
                bool nativeWorldActive = _nativeWorldBridge != null &&
                    _nativeWorldBridge.OwnsSimulation;
                NativeGpu.SelectDevice(deviceIndex);
                _gpuDeviceIndex.Value = deviceIndex;
                _shadowValidator.SetDevice(deviceIndex);
                _authoritativeBrainScheduler.SetDevice(deviceIndex);
                _authoritativeVisionScheduler.SetDevice(deviceIndex);
                _authoritativePheromoneScheduler.SetDevice(deviceIndex);
                if (!nativeWorldActive &&
                    (_authoritativeBrainGpu.Value || _authoritativeVisionGpu.Value ||
                        _authoritativePheromoneGpu.Value))
                {
                    RegisterExistingBrains();
                    RegisterExistingPheromoneSpots();
                }
                Logger.LogInfo("Selected CUDA GPU " + deviceIndex + ": " +
                    _gpuDevices[deviceIndex].Name + (nativeWorldActive
                        ? ". The current native world remains on its original GPU; " +
                            "the new selection applies when the next world starts."
                        : string.Empty));
            }
            catch (Exception ex)
            {
                Logger.LogError("Could not switch CUDA GPU: " + ex);
            }
        }

        private void SetAuthoritativeBrainMode(bool enabled)
        {
            if (enabled && _gpuDevices.Count == 0)
            {
                Logger.LogWarning("Authoritative GPU brains cannot start because no CUDA GPU is available.");
                _authoritativeBrainGpu.Value = false;
                return;
            }

            _authoritativeBrainGpu.Value = enabled;
            _authoritativeBrainScheduler.Reset();
            if (enabled)
            {
                RegisterExistingBrains();
            }
            Logger.LogWarning("Authoritative GPU brain mode " +
                (enabled ? "enabled" : "disabled") +
                ". This mode synchronizes brain phases and can change evolution outcomes.");
        }

        private void SetAuthoritativeVisionMode(bool enabled)
        {
            if (enabled && _gpuDevices.Count == 0)
            {
                Logger.LogWarning("Authoritative GPU vision cannot start because no CUDA GPU is available.");
                _authoritativeVisionGpu.Value = false;
                return;
            }

            _authoritativeVisionGpu.Value = enabled;
            _authoritativeVisionScheduler.Reset();
            if (enabled)
            {
                RegisterExistingBrains();
            }
            Logger.LogWarning("Authoritative GPU vision " +
                (enabled ? "enabled" : "disabled") +
                ". This bypasses stock Physics2D sight lookup and can change edge-case target ordering.");
        }

        private void SetAuthoritativePheromoneMode(bool enabled)
        {
            if (enabled && _gpuDevices.Count == 0)
            {
                Logger.LogWarning("Authoritative GPU pheromone sensing cannot start because no CUDA GPU is available.");
                _authoritativePheromoneGpu.Value = false;
                return;
            }

            _authoritativePheromoneGpu.Value = enabled;
            _authoritativePheromoneScheduler.Reset();
            if (enabled)
            {
                RegisterExistingBrains();
                RegisterExistingPheromoneSpots();
            }
            Logger.LogWarning("Authoritative GPU pheromone sensing " +
                (enabled ? "enabled" : "disabled") +
                ". It replaces per-Bibite Physics2D overlap queries with a CUDA world batch.");
        }

        private void ApplyPerformancePolicy()
        {
            bool nativePresentation = _nativeWorldBridge != null && _nativeWorldBridge.OwnsSimulation;
            bool shouldApply = nativePresentation ||
                (ShouldBypassFpsGovernor && !TimeController.paused);
            if (shouldApply)
            {
                if (!_performancePolicyApplied)
                {
                    _savedVSyncCount = QualitySettings.vSyncCount;
                    _savedTargetFrameRate = Application.targetFrameRate;
                    _savedPhysicsJobOptions = Physics2D.jobOptions;
                    _performancePolicyApplied = true;
                    ApplyPhysicsJobPolicy();
                }
                QualitySettings.vSyncCount = 0;
                Application.targetFrameRate = nativePresentation
                    ? Mathf.Clamp(_nativeWorldBridge.RenderFps, 10, 60)
                    : Mathf.Clamp(_turboRenderFps.Value, 5, 60);
            }
            else
            {
                RestorePerformancePolicy();
            }
        }

        private void RestorePerformancePolicy()
        {
            if (!_performancePolicyApplied)
            {
                return;
            }
            QualitySettings.vSyncCount = _savedVSyncCount;
            Application.targetFrameRate = _savedTargetFrameRate;
            Physics2D.jobOptions = _savedPhysicsJobOptions;
            _performancePolicyApplied = false;
        }

        private void ApplyPhysicsJobPolicy()
        {
            PhysicsJobOptions2D options = _performancePolicyApplied
                ? _savedPhysicsJobOptions
                : Physics2D.jobOptions;
            if (_multithreadedPhysics != null && _multithreadedPhysics.Value)
            {
                options.useMultithreading = true;
                options.useConsistencySorting = false;
            }
            Physics2D.jobOptions = options;
        }

        internal bool ShouldBypassFpsGovernor
        {
            get
            {
                return _throughputMode != null && _throughputMode.Value &&
                    GameManager.isSim && TimeController.targetTimeScale.val >= 25f ||
                    (_nativeWorldBridge != null && _nativeWorldBridge.OwnsSimulation);
            }
        }

        internal bool ShouldSuppressStockWorld
        {
            get
            {
                return _worldSessionMode.ShouldSuppressStock(GameManager.isSim,
                    _nativeGpuVisualWorld != null && _nativeGpuVisualWorld.Value,
                    _nativeWorldBridge != null && _nativeWorldBridge.OwnsSimulation,
                    _nativeWorldBridge != null && _nativeWorldBridge.IsStarting,
                    SimulationManager.loadedGame || SimulationManager.gameWasLoaded,
                    _nativeGpuLoadedSaves != null && _nativeGpuLoadedSaves.Value);
            }
        }

        internal void ReportNativeStartupBypass()
        {
            if (_nativeStartupBypassReported)
            {
                return;
            }
            _nativeStartupBypassReported = true;
            Logger.LogInfo(
                "Native startup fast path active: skipped stock pellet seeding, stock Bibite " +
                "spawning, and compatibility GPU batches.");
        }

        internal bool CanUseSimulationHotkeys
        {
            get
            {
                return GameManager.isSim && !_userSettingsOpen && !_gpuSettingsOpen &&
                    UserControl.AllowControl && UserControl.AllowKeyboardControl &&
                    !NativeUiVisibility.BlockingPanelOpen && !GpuMenuIntegration.HasFocusedTextInput();
            }
        }

        internal bool IsGpuSettingsOpen
        {
            get { return _userSettingsOpen && _gpuSettingsOpen; }
        }

        internal bool IsUserSettingsOpen
        {
            get { return _userSettingsOpen; }
        }

        private Rect GpuSettingsTabRect
        {
            get { return new Rect(20f, Mathf.Max(12f, Screen.height - 54f), 152f, 36f); }
        }

        internal bool IsPointerOverGpuSettings(Vector3 screenPoint)
        {
            if (!_userSettingsOpen)
            {
                return false;
            }
            Vector2 guiPoint = new Vector2(screenPoint.x, Screen.height - screenPoint.y);
            return (_gpuSettingsOpen && _gpuSettingsWindow.Contains(guiPoint)) ||
                GpuSettingsTabRect.Contains(guiPoint);
        }

        internal void SetGpuSettingsOpen(bool open)
        {
            if (_gpuSettingsOpen == open)
            {
                return;
            }
            _gpuSettingsOpen = open;
            GpuMenuIntegration.SetModalBlocker(open);
            UserControl.SetKeyboardBlockFromSource("GpuForkSettings", open);
            if (open)
            {
                FitGpuSettingsWindow(true);
                _gpuSettingsEscape = new EscapableAction(delegate { SetGpuSettingsOpen(false); });
                UINavigationManager.AddEscapableToStack(_gpuSettingsEscape);
            }
            else if (_gpuSettingsEscape != null)
            {
                UINavigationManager.RemoveEscapableFromStack(_gpuSettingsEscape);
                _gpuSettingsEscape = null;
            }
        }

        private void FitGpuSettingsWindow(bool center)
        {
            float margin = 12f;
            float maximumWidth = Mathf.Max(260f, Screen.width - margin * 2f);
            float maximumHeight = Mathf.Max(180f, Screen.height - margin * 2f);
            _gpuSettingsWindow.width = Mathf.Min(560f, maximumWidth);
            _gpuSettingsWindow.height = Mathf.Min(700f, maximumHeight);
            if (center || _gpuSettingsLastScreenWidth != Screen.width ||
                _gpuSettingsLastScreenHeight != Screen.height)
            {
                _gpuSettingsWindow.x = (Screen.width - _gpuSettingsWindow.width) * 0.5f;
                _gpuSettingsWindow.y = (Screen.height - _gpuSettingsWindow.height) * 0.5f;
            }
            _gpuSettingsWindow.x = Mathf.Clamp(_gpuSettingsWindow.x, margin,
                Mathf.Max(margin, Screen.width - _gpuSettingsWindow.width - margin));
            _gpuSettingsWindow.y = Mathf.Clamp(_gpuSettingsWindow.y, margin,
                Mathf.Max(margin, Screen.height - _gpuSettingsWindow.height - margin));
            _gpuSettingsLastScreenWidth = Screen.width;
            _gpuSettingsLastScreenHeight = Screen.height;
        }

        internal void SetUserSettingsOpen(bool open, UserSettingsManager panel = null)
        {
            _userSettingsOpen = open;
            UserControl.SetKeyboardBlockFromSource("GpuForkUserSettings", open);
            if (open && panel != null)
            {
                _userSettingsPanel = panel;
            }
            if (!open)
            {
                SetGpuSettingsOpen(false);
                _userSettingsPanel = null;
                if (ShouldBypassFpsGovernor && !TimeController.paused &&
                    (_nativeWorldBridge == null || !_nativeWorldBridge.OwnsSimulation))
                {
                    TimeController.engineTimeScale.SetValue(TimeController.targetTimeScale.val);
                }
            }
        }

        private void OnGameSceneChange()
        {
            SetUserSettingsOpen(false);
            _worldSessionMode.ResetForSceneChange(_nativeGpuVisualWorld != null && _nativeGpuVisualWorld.Value);
            if (_nativeWorldBridge != null) _nativeWorldBridge.ResetForSceneChange();
            GpuMenuIntegration.Dispose();
            if (_gpuSettingsSkin != null)
            {
                UnityEngine.Object.Destroy(_gpuSettingsSkin);
                _gpuSettingsSkin = null;
            }
            _nativeStartupBypassReported = false;
            _achievedMultiplier = 0f;
            _fixedSteps = 0;
            _sampleStarted = Time.realtimeSinceStartup;
        }

        private bool OnApplicationWantsToQuit()
        {
            bool wasRequested = _applicationQuit.Requested;
            bool resourcesPending = !string.IsNullOrEmpty(_pendingNativeSavePath) ||
                (_nativeWorldBridge != null && _nativeWorldBridge.HasResourcesForApplicationQuit);
            bool allowed = _applicationQuit.Request(resourcesPending);
            if (!allowed && !wasRequested)
            {
                // Alt+F4 can remove focus. Cleanup must continue pumping even
                // when the game normally pauses presentation in the background.
                Application.runInBackground = true;
                _nextQuitDrainStatusAt = Time.realtimeSinceStartup + 5f;
                Logger.LogInfo("Deferring application quit until pending GPU saves and graphics cleanup finish.");
            }
            return allowed;
        }

        private void OnApplicationQuitting()
        {
            // No P/Invoke or graphics commands from this final notification.
            GpuInteropProcessSafety.RenderingShuttingDown = true;
        }

        private void OnDestroy()
        {
            Application.wantsToQuit -= OnApplicationWantsToQuit;
            Application.quitting -= OnApplicationQuitting;
            GameManager.onSceneChange.RemoveListener(OnGameSceneChange);
            SetUserSettingsOpen(false);
            ClearPendingNativeSave();
            GpuMenuIntegration.Dispose();
            if (_gpuSettingsSkin != null) UnityEngine.Object.Destroy(_gpuSettingsSkin);
            if (_nativeWorldBridge != null)
            {
                _nativeWorldBridge.Dispose();
            }
            RestorePerformancePolicy();
            if (_harmony != null)
            {
                _harmony.UnpatchSelf();
            }
            if (_shadowValidator != null)
            {
                _shadowValidator.Dispose();
            }
            if (_authoritativeBrainScheduler != null)
            {
                _authoritativeBrainScheduler.Dispose();
            }
            if (_authoritativeVisionScheduler != null)
            {
                _authoritativeVisionScheduler.Dispose();
            }
            if (_authoritativePheromoneScheduler != null)
            {
                _authoritativePheromoneScheduler.Dispose();
            }
            Instance = null;
        }

        internal bool BeginBrainShadow(NEATBrain brain)
        {
            return _shadowValidation != null && _shadowValidation.Value &&
                _shadowValidator.Begin(brain, Mathf.Max(1, _shadowSampleLimit.Value));
        }

        internal void CaptureBrainShadow(NEATBrain brain)
        {
            _shadowValidator.CaptureAfterSenses(brain);
        }

        internal void CompleteBrainShadow(NEATBrain brain)
        {
            _shadowValidator.CompleteAfterCpu(brain);
        }

        internal VisionShadowSample BeginVisionShadow(FieldOfView vision)
        {
            if (_visionShadowValidation == null || !_visionShadowValidation.Value)
            {
                return null;
            }
            return _visionShadowValidator.Begin(
                vision,
                Mathf.Max(1, _visionShadowSampleLimit.Value));
        }

        internal void CompleteVisionShadow(FieldOfView vision, VisionShadowSample sample)
        {
            _visionShadowValidator.Complete(vision, sample);
        }

        internal bool ValidateAuthoritativeVision(FieldOfView vision, GpuVisionResult actual)
        {
            if (_visionShadowValidation == null || !_visionShadowValidation.Value ||
                _authoritativeVisionValidationSamples >= Math.Max(1, _visionShadowSampleLimit.Value))
            {
                return false;
            }

            _authoritativeVisionValidationSamples++;
            _allowStockVision = true;
            try
            {
                vision.FindSeenEntities();
                vision.ComputeSenses();
                _visionShadowValidator.CompareAuthoritative(
                    GpuVisionShadowValidator.Capture(vision),
                    actual);
            }
            finally
            {
                _allowStockVision = false;
            }
            return true;
        }

        internal bool AllowStockVision
        {
            get { return _allowStockVision; }
        }

        internal bool ValidateAuthoritativePheromone(Pherosense sensor, GpuPheromoneResult actual)
        {
            if (_pheromoneShadowValidation == null || !_pheromoneShadowValidation.Value ||
                _authoritativePheromoneValidationSamples >= Math.Max(1, _visionShadowSampleLimit.Value))
            {
                return false;
            }

            _authoritativePheromoneValidationSamples++;
            _allowStockPheromone = true;
            try
            {
                sensor.PherosenseAround();
                _authoritativePheromoneScheduler.CompareStock(sensor, actual);
            }
            finally
            {
                _allowStockPheromone = false;
            }
            return true;
        }

        internal bool AllowStockPheromone
        {
            get { return _allowStockPheromone; }
        }

        internal bool IsAuthoritativeBrainMode
        {
            get
            {
                return _authoritativeBrainGpu != null && _authoritativeBrainGpu.Value &&
                    _authoritativeBrainScheduler.Active;
            }
        }

        internal bool IsAuthoritativeVisionMode
        {
            get
            {
                return _authoritativeVisionGpu != null && _authoritativeVisionGpu.Value &&
                    _authoritativeVisionScheduler.Active;
            }
        }

        internal bool IsAuthoritativePheromoneMode
        {
            get
            {
                return _authoritativePheromoneGpu != null && _authoritativePheromoneGpu.Value &&
                    _authoritativePheromoneScheduler.Active;
            }
        }

        internal bool IsNativeWorldActive
        {
            get { return _nativeWorldBridge != null && _nativeWorldBridge.OwnsSimulation; }
        }

        internal bool TryPlaceNativeBibite(
            BibiteTemplate template,
            RandomizeGenes randomizeGenes,
            Tagging tagging,
            GrowthAtSpawn growth,
            Vector3? position,
            float? angleDegrees,
            string customTag)
        {
            if (!position.HasValue || _nativeWorldBridge == null || !_nativeWorldBridge.OwnsSimulation)
            {
                return false;
            }
            return _nativeWorldBridge.QueuePlacedBibite(
                template,
                randomizeGenes,
                tagging,
                growth,
                position.Value,
                angleDegrees,
                customTag);
        }

        private void RunIntegrationSaveProbe(float now)
        {
            // Opt-in only for a separate automated integration process. Normal
            // games never request a save or quit through this test hook.
            if (string.IsNullOrEmpty(_integrationSaveOutput) ||
                !_integrationEntitiesSpawned || _nativeWorldBridge == null ||
                !_nativeWorldBridge.OwnsSimulation || SaveController.Instance == null)
                return;
            if (!_integrationSavePending)
            {
                if (now - _integrationSimulationStarted < 5f ||
                    (_integrationSaveCompletedAt > 0f && now - _integrationSaveCompletedAt < 2f))
                    return;
                SaveController.Instance.ToggleAutoSave(false);
                _integrationSavePending = true;
                _integrationSaveStartedAt = now;
                _integrationSaveCompletedAt = 0f;
                _integrationSavedTime = TimeKeeper.simulatedTime;
                ++_integrationSaveCount;
                Logger.LogInfo("BGF_SAVE_PROBE begin cycle " + _integrationSaveCount);
                SaveController.Instance.SaveWorld(_integrationSaveOutput);
                return;
            }
            if (now - _integrationSaveStartedAt > 45f)
            {
                Logger.LogError("BGF_SAVE_PROBE failed: save/resume timeout; " +
                    _nativeWorldBridge.PendingSaveState);
                _integrationSaveOutput = null;
                Application.Quit();
                return;
            }
            if (!string.IsNullOrEmpty(_pendingNativeSavePath)) return;
            if (!File.Exists(_integrationSaveOutput) ||
                !File.Exists(GpuNativeWorldBridge.CheckpointPathForWorld(_integrationSaveOutput)) ||
                _integrationSaveCommittedCount != _integrationSaveCount)
            {
                Logger.LogError("BGF_SAVE_PROBE failed: committed save pair is missing");
                _integrationSaveOutput = null;
                Application.Quit();
                return;
            }
            if (_integrationSaveCompletedAt <= 0f)
            {
                _integrationSaveCompletedAt = now;
                _integrationSavedTime = TimeKeeper.simulatedTime;
                Logger.LogInfo("BGF_SAVE_PROBE committed cycle " + _integrationSaveCount +
                    " in " + (now - _integrationSaveStartedAt).ToString("0.000") + " s");
            }
            if (now - _integrationSaveCompletedAt < 1f ||
                TimeKeeper.simulatedTime <= _integrationSavedTime) return;
            Logger.LogInfo("BGF_SAVE_PROBE resumed cycle " + _integrationSaveCount +
                " at " + TimeKeeper.simulatedTime.ToString("0.000") + " simulated seconds");
            _integrationSavePending = false;
            if (_integrationSaveCount >= 2)
            {
                Logger.LogInfo("BGF_SAVE_PROBE passed: two full wrapper/checkpoint saves and continued simulation");
                _integrationSaveOutput = null;
                Application.Quit();
            }
        }

        internal bool TrySaveNativeWorld(string worldPath)
        {
            if (_nativeSaveStockContinuation && string.Equals(
                    _pendingNativeSaveStagingPath,
                    worldPath,
                    StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
            if (_nativeWorldBridge == null || !_nativeWorldBridge.OwnsSimulation)
            {
                return true;
            }
            if (string.IsNullOrEmpty(worldPath))
            {
                return true;
            }
            if (!string.IsNullOrEmpty(_pendingNativeSavePath))
            {
                if (SaveController.Instance != null)
                {
                    SaveController.Instance.ShowSaveStatus("Saving GPU world...");
                }
                return false;
            }
            if (SaveSystem.instance == null)
            {
                ReportNativeSaveFailure("the stock save system is unavailable");
                return false;
            }
            try
            {
                _nativeSaveTransaction = new GpuSaveTransaction(worldPath);
                _pendingNativeSavePath = _nativeSaveTransaction.WorldPath;
                _pendingNativeSaveStagingPath = _nativeSaveTransaction.StagingWorldPath;
            }
            catch (Exception ex)
            {
                ClearPendingNativeSave();
                ReportNativeSaveFailure(ex.Message);
                return false;
            }
            if (SaveController.Instance != null)
            {
                SaveController.Instance.ShowSaveStatus("Saving GPU world...");
            }
            string error;
            if (_nativeWorldBridge.BeginSaveWorldCheckpoint(_pendingNativeSaveStagingPath, out error))
            {
                _pendingNativeSaveBibites = _nativeWorldBridge.LivingBibites;
                _pendingNativeSavePellets = _nativeWorldBridge.ActivePellets;
                _pendingNativeSaveTime = TimeKeeper.simulatedTime;
                _pendingNativeSaveStarted = Time.realtimeSinceStartup;
                _nextNativeSaveStatusAt = _pendingNativeSaveStarted + 5f;
                // Register before the original AutoSave coroutine can attach
                // ReloadAfterAutoSave; the pair must be committed before reload.
                ScheduleNativeSaveMetadata(_pendingNativeSaveStagingPath,
                    _pendingNativeSaveBibites, _pendingNativeSavePellets,
                    _pendingNativeSaveTime);
                return false;
            }
            ClearPendingNativeSave();
            ReportNativeSaveFailure(error);
            return false;
        }

        private void CompletePendingNativeSave()
        {
            if (string.IsNullOrEmpty(_pendingNativeSavePath) || _nativeWorldBridge == null)
            {
                return;
            }
            if (_nativeSaveWrapperWriting)
            {
                if (Time.realtimeSinceStartup - _pendingNativeSaveStarted > 120f)
                {
                    FailPendingNativeSave(_pendingNativeSaveStagingPath,
                        "the stock save wrapper did not finish within two minutes");
                }
                return;
            }
            bool completed;
            string completedPath;
            string error;
            if (!_nativeWorldBridge.PollSaveWorldCheckpoint(
                    out completed,
                    out completedPath,
                    out error))
            {
                ClearPendingNativeSave();
                ReportNativeSaveFailure("the GPU checkpoint request was lost");
                return;
            }
            if (!completed)
            {
                if (Time.realtimeSinceStartup >= _nextNativeSaveStatusAt)
                {
                    string state = _nativeWorldBridge.PendingSaveState;
                    Logger.LogWarning("GPU save pending: " + state);
                    if (SaveController.Instance != null)
                        SaveController.Instance.ShowSaveStatus("Saving GPU world: " + state);
                    _nextNativeSaveStatusAt = Time.realtimeSinceStartup + 5f;
                }
                return;
            }

            string worldPath = _pendingNativeSaveStagingPath;
            if (!string.IsNullOrEmpty(completedPath) && !string.Equals(
                completedPath, worldPath, StringComparison.OrdinalIgnoreCase))
            {
                error = "the GPU checkpoint completed for an unexpected destination";
            }

            if (!string.IsNullOrEmpty(error))
            {
                ClearPendingNativeSave();
                ReportNativeSaveFailure(error);
                return;
            }
            if (SaveController.Instance == null)
            {
                ClearPendingNativeSave();
                ReportNativeSaveFailure("the stock save controller is unavailable");
                return;
            }

            _nativeSaveWrapperWriting = true;
            _pendingNativeSaveStarted = Time.realtimeSinceStartup;
            _nativeSaveStockContinuation = true;
            try
            {
                // Re-enter the original save only after CUDA has produced a
                // complete sidecar. The Harmony prefix recognizes this one
                // continuation and lets the normal ZIP wrapper be written.
                SaveController.Instance.SaveWorld(worldPath);
            }
            catch (Exception ex)
            {
                ClearPendingNativeSave();
                ReportNativeSaveFailure(ex.Message);
                Logger.LogError("Could not write the stock GPU save wrapper: " + ex);
            }
            finally
            {
                _nativeSaveStockContinuation = false;
            }
        }

        internal bool IsStagedNativeSave(string savePath)
        {
            return _nativeSaveWrapperWriting && !string.IsNullOrEmpty(savePath) &&
                string.Equals(savePath, _pendingNativeSaveStagingPath,
                    StringComparison.OrdinalIgnoreCase);
        }

        internal void FailPendingNativeSave(string savePath, string error)
        {
            if (!IsStagedNativeSave(savePath)) return;
            Logger.LogError("GPU world save failed: " + error);
            ClearPendingNativeSave();
            ReportNativeSaveFailure(error);
        }

        private void ClearPendingNativeSave()
        {
            if (_pendingNativeSaveSystem != null && _pendingNativeSaveFinalize != null)
            {
                _pendingNativeSaveSystem.onSavingDone.RemoveListener(_pendingNativeSaveFinalize);
            }
            _pendingNativeSaveSystem = null;
            _pendingNativeSaveFinalize = null;
            if (_nativeSaveTransaction != null)
            {
                _nativeSaveTransaction.DiscardStaging();
                _nativeSaveTransaction = null;
            }
            _pendingNativeSavePath = null;
            _pendingNativeSaveStagingPath = null;
            _pendingNativeSaveBibites = 0;
            _pendingNativeSavePellets = 0;
            _pendingNativeSaveTime = 0.0;
            _nativeSaveWrapperWriting = false;
        }

        private void ReportNativeSaveFailure(string error)
        {
            if (SaveController.Instance != null)
            {
                MethodInfo hideStatus = AccessTools.Method(
                    typeof(SaveController),
                    "HideSaveStatus");
                if (hideStatus != null)
                {
                    hideStatus.Invoke(SaveController.Instance, null);
                }
            }
            PopupManager.DisplayError(
                "GPU save failed",
                "The save did not complete. Keep any recovery files mentioned below.\n" +
                (error ?? "Unknown checkpoint error."));
        }

        private void ScheduleNativeSaveMetadata(
            string worldPath,
            int livingBibites,
            int activePellets,
            double simulatedSeconds)
        {
            if (SaveSystem.instance == null)
            {
                return;
            }
            UnityAction finalize = null;
            finalize = delegate
            {
                if (!_nativeSaveWrapperWriting || _nativeSaveTransaction == null)
                {
                    return;
                }
                try
                {
                    using (ZipArchive zip = ZipFile.Open(worldPath, ZipArchiveMode.Update))
                    {
                        ZipArchiveEntry sceneEntry = zip.GetEntry("scene.bb8scene") ??
                            zip.GetEntry("scene.json");
                        if (sceneEntry == null)
                        {
                            throw new InvalidDataException("The stock save wrapper has no scene record.");
                        }
                        string sceneName = sceneEntry.FullName;
                        JObject scene = SaveSystem.ReadJObjectFromArchive(sceneEntry);
                        sceneEntry.Delete();
                        scene["nBibites"] = livingBibites;
                        scene["nPellets"] = activePellets;
                        scene["simulatedTime"] = simulatedSeconds;
                        scene["gpuForkVersion"] = PluginVersion;
                        scene["gpuPresentation"] = _nativeWorldBridge.ExportPresentationMetadata();
                        scene["gpuCheckpoint"] = Path.GetFileName(
                            GpuNativeWorldBridge.CheckpointPathForWorld(_pendingNativeSavePath));
                        SaveSystem.WriteJObjectToArchive(zip, sceneName, scene);
                    }
                    string destination = _pendingNativeSavePath;
                    _nativeSaveTransaction.Commit();
                    if (string.Equals(destination, _integrationSaveOutput,
                            StringComparison.OrdinalIgnoreCase))
                        ++_integrationSaveCommittedCount;
                    ClearPendingNativeSave();
                    Logger.LogInfo("Saved GPU checkpoint and original-game wrapper together: " + destination);
                }
                catch (Exception ex)
                {
                    FailPendingNativeSave(worldPath, ex.Message);
                }
            };
            _pendingNativeSaveSystem = SaveSystem.instance;
            _pendingNativeSaveFinalize = finalize;
            _pendingNativeSaveSystem.onSavingDone.AddListener(finalize);
        }

        internal bool PrepareNativeWorldLoad(string worldPath)
        {
            if (!ValidateWorldForLoad(worldPath)) return false;
            if (_nativeWorldBridge == null)
            {
                return true;
            }
            if (string.IsNullOrEmpty(worldPath) || !File.Exists(worldPath))
            {
                return true;
            }
            string checkpointPath = GpuNativeWorldBridge.CheckpointPathForWorld(worldPath);
            if (!string.IsNullOrEmpty(checkpointPath) && File.Exists(checkpointPath))
            {
                // Optional display metadata must be read before stopping the
                // old runner, then applied after its dictionaries are cleared.
                JObject presentation = null;
                try
                {
                    using (ZipArchive zip = ZipFile.OpenRead(worldPath))
                        presentation = SaveSystem.GetSceneOfSave(zip)["gpuPresentation"] as JObject;
                }
                catch (Exception ex)
                {
                    Logger.LogWarning("GPU save display names could not be read: " + ex.Message);
                }
                string error;
                if (!_nativeWorldBridge.PrepareCheckpointLoad(worldPath, out error))
                {
                    PopupManager.DisplayError(
                        "GPU load failed",
                        "The GPU checkpoint could not be prepared for loading.\n" +
                        (error ?? "Unknown checkpoint error."));
                    return false;
                }
                _nativeWorldBridge.ImportPresentationMetadata(presentation);
                _worldSessionMode.SelectCheckpointLoad();
                SimulationManager.gameWasLoaded = true;
                Logger.LogInfo("Prepared GPU-native checkpoint load for " + worldPath + ".");
                return true;
            }

            // QuickLoad does not run SimulationManager.Start, so its loaded
            // flags cannot be relied on to disable native seeding/spawn hooks.
            // This explicit session choice also prevents ReplaceLoadedSaves
            // from silently replacing a stock save chosen through the menu.
            _worldSessionMode.SelectStockLoad();
            SimulationManager.gameWasLoaded = true;
            _nativeWorldBridge.PrepareStockWorldLoad();
            Logger.LogInfo(
                "Loading this stock save in compatibility mode; new-world GPU preferences were preserved.");
            return true;
        }

        internal bool CanChangeWorld()
        {
            if (string.IsNullOrEmpty(_pendingNativeSavePath)) return true;
            PopupManager.DisplayError("GPU save in progress",
                "Wait for the current save to finish before leaving or replacing this world.");
            return false;
        }

        internal bool ValidateWorldForLoad(string worldPath)
        {
            if (!CanChangeWorld()) return false;
            if (string.IsNullOrEmpty(worldPath) || !File.Exists(worldPath))
            {
                PopupManager.DisplayError("Load game", "The selected save file no longer exists.");
                return false;
            }
            try
            {
                bool expectsGpuCheckpoint;
                using (ZipArchive zip = ZipFile.OpenRead(worldPath))
                {
                    JObject scene = SaveSystem.GetSceneOfSave(zip);
                    if (scene == null)
                        throw new InvalidDataException("This save has no readable scene information.");
                    if (SaveSystem.GetSettingsOfSave(zip) == null)
                        throw new InvalidDataException("This save has no readable world settings.");
                    Utility.Version version = SaveSystem.GetVersionOfFile(scene);
                    if (version == Utility.Version.Null || !VersionTracker.CanUpdateFromVersion(version))
                        throw new InvalidDataException("This save's game version is not compatible with the supplied original game.");
                    expectsGpuCheckpoint = scene["gpuForkVersion"] != null || scene["gpuCheckpoint"] != null;
                }
                string checkpointPath = GpuNativeWorldBridge.CheckpointPathForWorld(worldPath);
                if (!File.Exists(checkpointPath))
                {
                    if (expectsGpuCheckpoint)
                        throw new InvalidDataException("This is a GPU-world save, but its .bgfgpu companion is missing. " +
                            "Keep the .zip and .zip.bgfgpu files together. An empty stock world will not be substituted.");
                }
                else
                {
                    NativeWorldContext.ReadCheckpointConfig(checkpointPath, _gpuDeviceIndex.Value);
                }
                return true;
            }
            catch (Exception ex)
            {
                Logger.LogWarning("Refused unreadable save before changing the world: " + ex.Message);
                PopupManager.DisplayError("Load game", ex.Message + "\nThe current world was not replaced.");
                return false;
            }
        }

        internal void RefreshNativeSavePreview(UIScripts.SaveGamePanel panel)
        {
            if (_nativeWorldBridge == null || !_nativeWorldBridge.OwnsSimulation) return;
            GpuMenuIntegration.SetPanelText(panel, "nBibites", _nativeWorldBridge.LivingBibites.ToString("N0"));
            GpuMenuIntegration.SetPanelText(panel, "nPellets", _nativeWorldBridge.ActivePellets.ToString("N0"));
        }

        internal static void DeleteCheckpointSidecarIfWrapperGone(string worldPath)
        {
            if (string.IsNullOrEmpty(worldPath) || File.Exists(worldPath))
            {
                return;
            }
            string checkpointPath = GpuNativeWorldBridge.CheckpointPathForWorld(worldPath);
            if (string.IsNullOrEmpty(checkpointPath) || !File.Exists(checkpointPath))
            {
                return;
            }
            try
            {
                File.Delete(checkpointPath);
                if (Instance != null)
                {
                    Instance.Logger.LogInfo(
                        "Removed GPU checkpoint companion for deleted save: " + checkpointPath);
                }
            }
            catch (Exception ex)
            {
                if (Instance != null)
                {
                    Instance.Logger.LogWarning(
                        "Could not remove GPU checkpoint companion: " + ex.Message);
                }
            }
        }

        internal static void LogNativeCheckpointWarning(string message)
        {
            if (Instance != null)
            {
                Instance.Logger.LogWarning(message);
            }
        }

        internal bool TrySelectNativeSpeciesHandle(SpeciesInfoHandle handle)
        {
            return _nativeWorldBridge != null &&
                _nativeWorldBridge.TrySelectSpeciesHandle(handle);
        }

        internal bool TrySelectNativeTagHandle(TagElementHandle handle)
        {
            return _nativeWorldBridge != null &&
                _nativeWorldBridge.TrySelectTagHandle(handle);
        }

        internal bool TrySelectNativeBibiteAtScreenPoint(Vector3 screenPoint)
        {
            return _nativeWorldBridge != null &&
                _nativeWorldBridge.TrySelectAtScreenPoint(screenPoint);
        }

        internal bool TrySelectNativeRectangle(Vector2 first, Vector2 second)
        {
            return _nativeWorldBridge != null &&
                _nativeWorldBridge.TrySelectNativeRectangle(first, second);
        }

        internal bool OpenNativeSpeciesPanel()
        {
            return _nativeWorldBridge != null &&
                _nativeWorldBridge.OpenNativeSpeciesPanel();
        }

        internal void RegisterAuthoritativeBrain(NEATBrain brain)
        {
            if (brain == null)
            {
                return;
            }
            if (_authoritativeBrainGpu != null && _authoritativeBrainGpu.Value)
            {
                _authoritativeBrainScheduler.Register(brain);
            }
            if (_authoritativeVisionGpu != null && _authoritativeVisionGpu.Value)
            {
                _authoritativeVisionScheduler.Register(brain.GetComponent<FieldOfView>());
            }
            if (_authoritativePheromoneGpu != null && _authoritativePheromoneGpu.Value)
            {
                _authoritativePheromoneScheduler.RegisterSensor(brain.GetComponent<Pherosense>());
            }
        }

        internal void RegisterPheromoneSpot(PheromoneSpot spot)
        {
            if (_authoritativePheromoneGpu != null && _authoritativePheromoneGpu.Value)
            {
                _authoritativePheromoneScheduler.RegisterSpot(spot);
            }
        }

        private void RegisterExistingBrains()
        {
            NEATBrain[] brains = UnityEngine.Object.FindObjectsByType<NEATBrain>(
                FindObjectsInactive.Include,
                FindObjectsSortMode.None);
            for (int i = 0; i < brains.Length; i++)
            {
                RegisterAuthoritativeBrain(brains[i]);
            }
            Logger.LogInfo("Discovered " + brains.Length + " existing brains for authoritative GPU mode.");
        }

        private void RegisterExistingPheromoneSpots()
        {
            PheromoneSpot[] spots = UnityEngine.Object.FindObjectsByType<PheromoneSpot>(
                FindObjectsInactive.Include,
                FindObjectsSortMode.None);
            for (int i = 0; i < spots.Length; i++)
            {
                _authoritativePheromoneScheduler.RegisterSpot(spots[i]);
            }
            Logger.LogInfo("Discovered " + spots.Length + " existing pheromone spots for GPU sensing.");
        }

        private static int ReadIntegrationInt(string variable, int fallback, int minimum, int maximum)
        {
            int value;
            return int.TryParse(Environment.GetEnvironmentVariable(variable), out value)
                ? Mathf.Clamp(value, minimum, maximum)
                : fallback;
        }
    }

    [HarmonyPatch(typeof(Zone), "InitialSeeding")]
    internal static class NativeWorldInitialSeedingSkipPatch
    {
        [HarmonyPrefix]
        [HarmonyPriority(Priority.First)]
        private static bool Prefix()
        {
            if (Plugin.Instance == null || !Plugin.Instance.ShouldSuppressStockWorld)
            {
                return true;
            }
            Plugin.Instance.ReportNativeStartupBypass();
            return false;
        }
    }

    [HarmonyPatch(typeof(Zone), "FixedUpdate")]
    internal static class NativeWorldZoneUpdateSkipPatch
    {
        [HarmonyPrefix]
        [HarmonyPriority(Priority.First)]
        private static bool Prefix()
        {
            return Plugin.Instance == null || !Plugin.Instance.ShouldSuppressStockWorld;
        }
    }

    [HarmonyPatch(typeof(BibiteSpawner), "StartSpawner")]
    internal static class NativeWorldBibiteSpawnerSkipPatch
    {
        [HarmonyPrefix]
        [HarmonyPriority(Priority.First)]
        private static bool Prefix()
        {
            if (Plugin.Instance == null || !Plugin.Instance.ShouldSuppressStockWorld)
            {
                return true;
            }
            Plugin.Instance.ReportNativeStartupBypass();
            return false;
        }
    }

    [HarmonyPatch(typeof(WorldObjectsSpawner), "SpawnBibiteFromTemplate")]
    internal static class SpawnedBibiteRegistrationPatch
    {
        private static bool Prefix(
            BibiteTemplate template,
            RandomizeGenes randomizeGenes,
            Tagging taggingChoice,
            GrowthAtSpawn targetGrowth,
            Vector3? pos,
            float? angle,
            string customTag,
            ref GameObject __result)
        {
            if (Plugin.Instance != null && Plugin.Instance.TryPlaceNativeBibite(
                    template,
                    randomizeGenes,
                    taggingChoice,
                    targetGrowth,
                    pos,
                    angle,
                    customTag))
            {
                __result = null;
                return false;
            }
            return true;
        }

        private static void Postfix(GameObject __result)
        {
            RegisterResult(__result);
        }

        internal static void RegisterResult(GameObject result)
        {
            if (Plugin.Instance == null || result == null)
            {
                return;
            }
            Plugin.Instance.RegisterAuthoritativeBrain(
                result.GetComponentInChildren<NEATBrain>(true));
        }
    }

    [HarmonyPatch(typeof(WorldObjectsSpawner), "GenerateNewBibite")]
    internal static class GeneratedBibiteRegistrationPatch
    {
        private static void Postfix(GameObject __result)
        {
            SpawnedBibiteRegistrationPatch.RegisterResult(__result);
        }
    }

    [HarmonyPatch(typeof(SpeciesInfoHandle), "OnClick")]
    internal static class NativeSpeciesInfoClickPatch
    {
        private static bool Prefix(SpeciesInfoHandle __instance)
        {
            return Plugin.Instance == null ||
                !Plugin.Instance.TrySelectNativeSpeciesHandle(__instance);
        }
    }

    [HarmonyPatch(typeof(TagElementHandle), "ClickOnTag")]
    internal static class NativeTagInfoClickPatch
    {
        private static bool Prefix(TagElementHandle __instance)
        {
            return Plugin.Instance == null ||
                !Plugin.Instance.TrySelectNativeTagHandle(__instance);
        }
    }

    [HarmonyPatch(typeof(UserControl), "SelectClosestBibiteAroundCursor")]
    internal static class NativeBibiteClickSelectionPatch
    {
        [HarmonyPrefix]
        [HarmonyPriority(Priority.First)]
        private static bool Prefix()
        {
            if (Plugin.Instance != null && Plugin.Instance.IsNativeWorldActive &&
                GpuMenuIntegration.BlocksWorldPointer())
            {
                return false;
            }
            return Plugin.Instance == null ||
                !Plugin.Instance.TrySelectNativeBibiteAtScreenPoint(Input.mousePosition);
        }
    }

    [HarmonyPatch(
        typeof(UIScripts.UIReferences.LineagePanel.SpeciesPanel),
        "OpenPanel")]
    internal static class NativeSpeciesPanelPatch
    {
        [HarmonyPrefix]
        [HarmonyPriority(Priority.First)]
        private static bool Prefix()
        {
            return Plugin.Instance == null || !Plugin.Instance.OpenNativeSpeciesPanel();
        }
    }

    [HarmonyPatch(typeof(WorldObjectsSpawner), "GenerateNewPheromonesSource")]
    internal static class SpawnedPheromoneRegistrationPatch
    {
        private static void Postfix(GameObject __result)
        {
            if (Plugin.Instance != null && __result != null)
            {
                Plugin.Instance.RegisterPheromoneSpot(__result.GetComponent<PheromoneSpot>());
            }
        }
    }

    [HarmonyPatch(typeof(NEATBrain), "CopyBrain")]
    internal static class CopiedBrainRegistrationPatch
    {
        private static void Postfix(NEATBrain __instance)
        {
            if (Plugin.Instance != null)
            {
                Plugin.Instance.RegisterAuthoritativeBrain(__instance);
            }
        }
    }

    [HarmonyPatch(typeof(NEATBrain), "ResumeBrain")]
    internal static class ResumedBrainRegistrationPatch
    {
        private static void Postfix(NEATBrain __instance)
        {
            if (Plugin.Instance != null)
            {
                Plugin.Instance.RegisterAuthoritativeBrain(__instance);
            }
        }
    }

    [HarmonyPatch(typeof(NEATBrain), "Thinker")]
    internal static class BrainThinkerSkipPatch
    {
        [HarmonyPrefix]
        [HarmonyPriority(Priority.First)]
        private static bool Prefix()
        {
            return Plugin.Instance == null || !Plugin.Instance.IsAuthoritativeBrainMode;
        }
    }

    [HarmonyPatch(typeof(NEATBrain), "Thinker")]
    internal static class BrainThinkerShadowPatch
    {
        private static void Prefix(NEATBrain __instance, out bool __state)
        {
            if (Plugin.Instance != null && Plugin.Instance.IsAuthoritativeBrainMode)
            {
                __state = false;
                return;
            }
            __state = Plugin.Instance != null && Plugin.Instance.BeginBrainShadow(__instance);
        }

        private static void Postfix(NEATBrain __instance, bool __state)
        {
            if (__state && Plugin.Instance != null)
            {
                Plugin.Instance.CompleteBrainShadow(__instance);
            }
        }
    }

    [HarmonyPatch(typeof(NEATBrain), "UpdateSenses")]
    internal static class BrainSensesShadowPatch
    {
        private static void Postfix(NEATBrain __instance)
        {
            if (Plugin.Instance != null)
            {
                Plugin.Instance.CaptureBrainShadow(__instance);
            }
        }
    }

    [HarmonyPatch(typeof(FieldOfView), "ComputeSenses")]
    internal static class VisionSensesShadowPatch
    {
        [HarmonyPrefix]
        [HarmonyPriority(Priority.First)]
        private static bool Prefix(FieldOfView __instance, out VisionShadowSample __state)
        {
            if (Plugin.Instance != null && Plugin.Instance.AllowStockVision)
            {
                __state = null;
                return true;
            }
            if (Plugin.Instance != null && Plugin.Instance.IsAuthoritativeVisionMode)
            {
                __state = null;
                return false;
            }
            __state = Plugin.Instance != null
                ? Plugin.Instance.BeginVisionShadow(__instance)
                : null;
            return true;
        }

        private static void Postfix(FieldOfView __instance, VisionShadowSample __state)
        {
            if (__state != null && Plugin.Instance != null)
            {
                Plugin.Instance.CompleteVisionShadow(__instance, __state);
            }
        }
    }

    [HarmonyPatch(typeof(FieldOfView), "FindSeenEntities")]
    internal static class VisionLookupSkipPatch
    {
        [HarmonyPrefix]
        [HarmonyPriority(Priority.First)]
        private static bool Prefix()
        {
            return Plugin.Instance == null || Plugin.Instance.AllowStockVision ||
                !Plugin.Instance.IsAuthoritativeVisionMode;
        }
    }

    [HarmonyPatch(typeof(Pherosense), "PherosenseAround")]
    internal static class PheromoneSenseSkipPatch
    {
        [HarmonyPrefix]
        [HarmonyPriority(Priority.First)]
        private static bool Prefix()
        {
            return Plugin.Instance == null || Plugin.Instance.AllowStockPheromone ||
                !Plugin.Instance.IsAuthoritativePheromoneMode;
        }
    }

    [HarmonyPatch(typeof(TimeController), "Awake")]
    internal static class TimeControllerAwakePatch
    {
        private static void Postfix(TimeController __instance)
        {
            TimeController.targetTimeScale.minValue = 1f;
            TimeController.targetTimeScale.maxValue = TimeWarpSpeeds.Maximum;
            TimeController.engineTimeScale.maxValue = TimeWarpSpeeds.Maximum;
            __instance.timeScaleSlider.SetMinMax(1f, TimeWarpSpeeds.Maximum, false);
            TimeController.targetTimeScale.SetValue(
                TimeWarpSpeeds.Snap(TimeController.targetTimeScale.val));
        }
    }

    [HarmonyPatch(typeof(SaveController), "SaveWorld")]
    internal static class NativeWorldSaveCheckpointPatch
    {
        [HarmonyPrefix]
        [HarmonyPriority(Priority.First)]
        private static bool Prefix(string worldPath)
        {
            if (Plugin.Instance == null || !Plugin.Instance.IsNativeWorldActive)
            {
                return true;
            }
            return Plugin.Instance.TrySaveNativeWorld(worldPath);
        }
    }

    [HarmonyPatch(typeof(SaveController), "LoadWorld")]
    internal static class NativeWorldLoadCheckpointPatch
    {
        [HarmonyPrefix]
        [HarmonyPriority(Priority.First)]
        private static bool Prefix(string worldPath)
        {
            return Plugin.Instance == null ||
                Plugin.Instance.PrepareNativeWorldLoad(worldPath);
        }
    }

    [HarmonyPatch(typeof(UIScripts.SaveGamePanel), "DeleteSelectedSave")]
    internal static class NativeWorldSavePanelDeletePatch
    {
        [HarmonyPrefix]
        private static bool Prefix(UIScripts.SaveGamePanel __instance, out string __state)
        {
            FileInfo info = GpuMenuIntegration.GetSelectedSave(__instance);
            __state = info != null ? info.FullName : null;
            return info != null && File.Exists(info.FullName);
        }

        [HarmonyPostfix]
        private static void Postfix(string __state)
        {
            Plugin.DeleteCheckpointSidecarIfWrapperGone(__state);
        }
    }

    [HarmonyPatch(typeof(UIScripts.LoadGamePanel), "DeleteSelectedSave")]
    internal static class NativeWorldLoadPanelDeletePatch
    {
        [HarmonyPrefix]
        private static bool Prefix(UIScripts.LoadGamePanel __instance, out string __state)
        {
            FileInfo info = GpuMenuIntegration.GetSelectedSave(__instance);
            __state = info != null ? info.FullName : null;
            return info != null && File.Exists(info.FullName);
        }

        [HarmonyPostfix]
        private static void Postfix(string __state)
        {
            Plugin.DeleteCheckpointSidecarIfWrapperGone(__state);
        }
    }

    [HarmonyPatch(typeof(SaveController), "CleanUpAutosaveFolder")]
    internal static class NativeWorldAutosaveCleanupPatch
    {
        [HarmonyPostfix]
        private static void Postfix(string folder)
        {
            if (string.IsNullOrEmpty(folder) || !Directory.Exists(folder))
            {
                return;
            }
            try
            {
                string[] checkpoints = Directory.GetFiles(folder, "*.zip.bgfgpu");
                for (int index = 0; index < checkpoints.Length; index++)
                {
                    string checkpoint = checkpoints[index];
                    string wrapper = checkpoint.Substring(
                        0,
                        checkpoint.Length - ".bgfgpu".Length);
                    Plugin.DeleteCheckpointSidecarIfWrapperGone(wrapper);
                }
            }
            catch (Exception ex)
            {
                Plugin.LogNativeCheckpointWarning(
                    "Could not clean orphaned GPU autosave checkpoints: " + ex.Message);
            }
        }
    }

    [HarmonyPatch(typeof(DataLogger), "FixedUpdate")]
    internal static class NativeWorldDataLoggerPatch
    {
        [HarmonyPrefix]
        private static bool Prefix()
        {
            return Plugin.Instance == null || !Plugin.Instance.IsNativeWorldActive;
        }
    }

    [HarmonyPatch(typeof(TimeController), "CheckMinFPS")]
    internal static class TimeControllerFpsGovernorPatch
    {
        private static bool Prefix()
        {
            return Plugin.Instance == null || !Plugin.Instance.ShouldBypassFpsGovernor;
        }
    }

    [HarmonyPatch(typeof(TimeKeeper), "FixedUpdate")]
    internal static class NativeWorldTimeKeeperPatch
    {
        [HarmonyPrefix]
        [HarmonyPriority(Priority.First)]
        private static bool Prefix()
        {
            // Native snapshots own simulated time and the achieved-speed
            // fields. The stock FixedUpdate measures Unity's intentionally
            // 1x presentation clock and otherwise overwrites them each second.
            return Plugin.Instance == null || !Plugin.Instance.IsNativeWorldActive;
        }
    }

    [HarmonyPatch(typeof(TimeController), "UpdateTimeScale")]
    internal static class NativeWorldPresentationTimePatch
    {
        [HarmonyPrefix]
        [HarmonyPriority(Priority.First)]
        private static bool Prefix(float val)
        {
            if (Plugin.Instance == null || !Plugin.Instance.IsNativeWorldActive)
            {
                return true;
            }

            bool paused = Mathf.Approximately(val, 0f);
            TimeController.paused = paused;
            Time.timeScale = paused ? 0f : 1f;
            if (!paused)
            {
                Time.fixedDeltaTime = 1f / Mathf.Max(
                    1,
                    ScenarioIndependentSettings.Instance.simTPS.val);
            }
            Time.maximumDeltaTime = Time.fixedDeltaTime;
            return false;
        }
    }

    [HarmonyPatch(typeof(TimeController), "SetTarget")]
    internal static class NativeWorldTargetTimePatch
    {
        [HarmonyPostfix]
        [HarmonyPriority(Priority.Last)]
        private static void Postfix(float val)
        {
            if (Plugin.Instance != null && Plugin.Instance.IsNativeWorldActive && val > 0f)
            {
                float target = TimeWarpSpeeds.Snap(val);
                if (!Mathf.Approximately(TimeController.engineTimeScale.val, target))
                {
                    TimeController.engineTimeScale.SetValue(target);
                }
            }
        }
    }

    [HarmonyPatch(typeof(UserSettingsManager), "OpenPanel")]
    internal static class UserSettingsOpenedPatch
    {
        private static void Postfix(UserSettingsManager __instance)
        {
            if (Plugin.Instance != null)
            {
                GpuMenuIntegration.EnsureSettingsButton(__instance);
                Plugin.Instance.SetUserSettingsOpen(true, __instance);
            }
        }
    }

    [HarmonyPatch(typeof(UserSettingsManager), "ClosePanel")]
    internal static class UserSettingsClosedPatch
    {
        private static void Postfix()
        {
            if (Plugin.Instance != null)
            {
                Plugin.Instance.SetUserSettingsOpen(false);
            }
        }
    }

    [HarmonyPatch(typeof(NumericSetting<float>), "SetValue")]
    internal static class TimeWarpSnapPatch
    {
        private static void Prefix(NumericSetting<float> __instance, ref float _value)
        {
            if (ReferenceEquals(__instance, TimeController.targetTimeScale))
            {
                _value = TimeWarpSpeeds.Snap(_value);
            }
        }
    }
}
