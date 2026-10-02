using System;
using System.Collections.Generic;
using System.Reflection;
using HarmonyLib;
using ManagementScripts;
using TMPro;
using UIScripts;
using UIScripts.UIPanels;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace BibitesGpuFork
{
    internal static class NativeUiVisibility
    {
        internal static bool BlockingPanelOpen
        {
            get
            {
                if (Plugin.Instance != null && Plugin.Instance.IsUserSettingsOpen) return true;
                GUIManager gui = GUIManager.Instance;
                if (gui == null) return false;
                return IsOpen(gui.escapePanel) || IsOpen(gui.savePanel) || IsOpen(gui.loadPanel) ||
                    IsOpen(gui.selectorPanel) || IsOpen(gui.workshopItemsPanel) ||
                    (gui.rightmostPanelsManager != null && IsOpen(gui.rightmostPanelsManager.graphsPanel));
            }
        }
        private static bool IsOpen(UIPanel panel) { return panel != null && panel.gameObject.activeInHierarchy; }
    }

    // The GPU owns simulation state, not menus. Reuse the shipped panel roots,
    // fonts and art, but do not create fake CPU BibiteBody/organ components just
    // to satisfy the original inspectors. Original children are restored on exit.
    internal sealed class GpuNativeInspector : IDisposable
    {
        internal static GpuNativeInspector Active;
        private readonly GpuNativeWorldBridge _bridge;
        private readonly GUIManager _gui;
        private readonly Dictionary<GpuNativeWorldBridge.InspectorTab, NativePanelSurface> _panels =
            new Dictionary<GpuNativeWorldBridge.InspectorTab, NativePanelSurface>();
        private NativePanelSurface _current;
        private float _nextRefresh;
        private int _brainPage;
        private bool _synapses;
        private TMP_InputField _tagInput;
        private GpuBrainGraphic _brainGraph;
        private int _selectedSlot = -1;
        private readonly List<TextMeshProUGUI> _rows = new List<TextMeshProUGUI>();
        private TextMeshProUGUI _notice;
        private TextMeshProUGUI _followLabel;
        private TextMeshProUGUI _brainPageLabel;
        private readonly EscapableAction _escape;
        private bool _registeredEscape;
        private NativePanelSurface _speciesSurface;
        private readonly List<TextMeshProUGUI> _speciesRows = new List<TextMeshProUGUI>();
        private readonly List<ulong> _speciesIds = new List<ulong>();
        private TextMeshProUGUI _speciesSummary;
        private readonly EscapableAction _speciesEscape;
        private bool _speciesOpen;
        private bool _suspended;
        private NativePanelSurface _selectionSurface;
        private bool _selectionOpen;
        private readonly EscapableAction _selectionEscape;
        private static readonly bool TraceEnabled = string.Equals(Environment.GetEnvironmentVariable("BIBITES_GPU_UI_TRACE"), "1", StringComparison.Ordinal);

        internal bool Ready { get { return _gui != null; } }
        internal bool IsOpen { get { return _current != null && _current.Visible; } }
        internal bool SpeciesReady { get { return _speciesSurface != null; } }
        internal GpuNativeWorldBridge Bridge { get { return _bridge; } }
        internal bool CoversScreenPoint(Vector2 point)
        {
            return (_current != null && _current.ContainsScreenPoint(point)) ||
                (_speciesOpen && _speciesSurface != null && _speciesSurface.ContainsScreenPoint(point)) ||
                (_selectionOpen && _selectionSurface != null && _selectionSurface.ContainsScreenPoint(point));
        }

        internal GpuNativeInspector(GpuNativeWorldBridge bridge)
        {
            _bridge = bridge;
            _gui = GUIManager.Instance;
            _escape = new EscapableAction(ClosePanels);
            _speciesEscape = new EscapableAction(HideSpecies);
            _selectionEscape = new EscapableAction(HideRectangleSelection);
            Active = this;
        }

        internal void Show(GpuNativeWorldBridge.InspectorTab tab)
        {
            if (!Ready || !_bridge.HasNativeSelection) return;
            Trace("show " + tab + ": close old panels");
            HideSpecies();
            HideRectangleSelection();
            ClosePanels();
            NativePanelSurface surface;
            if (!_panels.TryGetValue(tab, out surface))
            {
                UIPanel original = tab == GpuNativeWorldBridge.InspectorTab.Genes ? (UIPanel)_gui.GenesPanel :
                    tab == GpuNativeWorldBridge.InspectorTab.Biology ? (UIPanel)_gui.BiologyPanel :
                    tab == GpuNativeWorldBridge.InspectorTab.Brain ? (UIPanel)_gui.BrainPanel :
                    tab == GpuNativeWorldBridge.InspectorTab.ExpandedBrain ? (UIPanel)_gui.ExpandedBrainPanel :
                    _gui.StatsPanel;
                if (original == null) return;
                Trace("show " + tab + ": adapt original root");
                surface = new NativePanelSurface(original, _gui.StatsPanel, ClosePanels);
                _panels.Add(tab, surface);
            }
            _current = surface;
            Trace("show " + tab + ": activate root");
            _current.Show("Bibite #" + _bridge.SelectedBibite.Slot + " — " + TabTitle(tab));
            _current.ClearBody();
            _rows.Clear();
            _tagInput = null;
            _brainGraph = null;
            _brainPage = 0;
            _selectedSlot = _bridge.SelectedBibite.Slot;
            _gui.TopLeftButtonHolder.SetActive(true);
            if (_gui.OpenBiologyPanelButton != null) _gui.OpenBiologyPanelButton.SetActive(true);
            BuildBody(tab);
            _bridge.SetNativeInspectorBrainVisible(tab == GpuNativeWorldBridge.InspectorTab.Brain || tab == GpuNativeWorldBridge.InspectorTab.ExpandedBrain);
            Trace("show " + tab + ": body built");
            if (!_registeredEscape)
            {
                UINavigationManager.AddEscapableToStack(_escape);
                _registeredEscape = true;
            }
            _nextRefresh = 0f;
            Tick();
            Trace("show " + tab + ": complete");
        }

        private static void Trace(string message)
        {
            if (TraceEnabled) UnityEngine.Debug.Log("[GPU inspector] " + message);
        }

        private static string TabTitle(GpuNativeWorldBridge.InspectorTab tab)
        {
            return tab == GpuNativeWorldBridge.InspectorTab.ExpandedBrain ? "Expanded Brain" : tab.ToString();
        }

        private void BuildBody(GpuNativeWorldBridge.InspectorTab tab)
        {
            _current.AddText("Live GPU state · menus run on the CPU", 14f);
            if (tab == GpuNativeWorldBridge.InspectorTab.Brain || tab == GpuNativeWorldBridge.InspectorTab.ExpandedBrain)
            {
                if (!string.Equals(Environment.GetEnvironmentVariable("BIBITES_GPU_UI_DISABLE_GRAPH"), "1", StringComparison.Ordinal))
                    _brainGraph = _current.AddBrainGraph(tab == GpuNativeWorldBridge.InspectorTab.ExpandedBrain ? 420f : 320f);
                _current.AddText("Blue: positive weight · red: negative weight. Sensor → hidden → action.", 13f);
                _current.AddButton("Nodes / Synapses", delegate { _synapses = !_synapses; _brainPage = 0; _nextRefresh = 0f; });
                _current.AddButton("Previous page", delegate { _brainPage = Math.Max(0, _brainPage - 1); _nextRefresh = 0f; });
                _current.AddButton("Next page", delegate { _brainPage++; _nextRefresh = 0f; });
                _brainPageLabel = _current.AddText(string.Empty, 14f);
                for (int i = 0; i < 26; i++) _rows.Add(_current.AddText(string.Empty));
            }
            else
            {
                int count = tab == GpuNativeWorldBridge.InspectorTab.Stats ? 10 :
                    tab == GpuNativeWorldBridge.InspectorTab.Genes ? 10 : 15;
                for (int i = 0; i < count; i++) _rows.Add(_current.AddText(string.Empty));
            }
            if (tab == GpuNativeWorldBridge.InspectorTab.Stats)
            {
                _current.AddText("Tag", 16f);
                _tagInput = _current.AddInput(_bridge.SelectedBibite.TagId == 0 ? string.Empty : _bridge.SelectedTagName);
                _current.AddButton("Apply tag", delegate { _bridge.SetNativeTag(_tagInput.text); _tagInput.DeactivateInputField(); });
                _current.AddButton("Copy tag", _bridge.CopyNativeTag);
                _current.AddButton("Lay egg", _bridge.ReproduceNativeSelection);
                _current.AddButton("Remove Bibite", _bridge.KillNativeSelection);
                _current.AddDisabledButton("Save Bibite — unavailable in GPU mode");
                _current.AddDisabledButton("Edit Bibite — unavailable in GPU mode");
                _current.AddText("Use Save World to preserve this individual. The compact GPU genome cannot be exported or edited as a full stock Bibite without inventing missing genes.", 13f);
                _current.AddText("Parents / children / eggs laid: not recorded by this GPU model.", 13f);
                _current.AddText("Organ, combat, parent and egg-history fields are not stored by this GPU model.", 13f);
            }
            else if (tab == GpuNativeWorldBridge.InspectorTab.Genes)
            {
                _current.AddText("Only inherited traits represented by the GPU simulation are shown. Original organ genes and species-distance values are unavailable.", 13f);
            }
            _notice = _current.AddStatusNotice();
            if (tab == GpuNativeWorldBridge.InspectorTab.Brain)
                _current.AddButton("Expanded brain [5]", delegate { _bridge.ShowNativeInspectorTab(GpuNativeWorldBridge.InspectorTab.ExpandedBrain); });
            else if (tab == GpuNativeWorldBridge.InspectorTab.ExpandedBrain)
                _current.AddButton("Return to brain [4]", delegate { _bridge.ShowNativeInspectorTab(GpuNativeWorldBridge.InspectorTab.Brain); });
            _followLabel = _current.AddFooterButton("Follow", delegate { _bridge.ToggleSelectionFollow(); _nextRefresh = 0f; });
            _current.AddFooterButton("Species list", delegate { _bridge.OpenNativeSpeciesPanel(); });
            _current.AddFooterButton("Deselect", _bridge.CloseNativeSelection);
        }

        internal void Tick()
        {
            bool suppress = !UserControl.AllowControl || NativeUiVisibility.BlockingPanelOpen ||
                (UserControl.Instance != null && UserControl.Instance.mainUI != null && !UserControl.Instance.mainUI.activeInHierarchy);
            if (suppress != _suspended)
            {
                _suspended = suppress;
                if (suppress)
                {
                    _bridge.SetNativeInspectorBrainVisible(false);
                    if (_current != null) _current.Hide();
                    if (_speciesSurface != null) _speciesSurface.Hide();
                    if (_selectionSurface != null) _selectionSurface.Hide();
                    if (_gui != null && _gui.TopLeftButtonHolder != null) _gui.TopLeftButtonHolder.SetActive(false);
                    if (_registeredEscape) UINavigationManager.RemoveEscapableFromStack(_escape);
                    if (_speciesOpen) UINavigationManager.RemoveEscapableFromStack(_speciesEscape);
                    if (_selectionOpen) UINavigationManager.RemoveEscapableFromStack(_selectionEscape);
                }
                else
                {
                    if (_current != null && _bridge.HasNativeSelection)
                    {
                        _bridge.SetNativeInspectorBrainVisible(_bridge.SelectedInspectorTab == GpuNativeWorldBridge.InspectorTab.Brain || _bridge.SelectedInspectorTab == GpuNativeWorldBridge.InspectorTab.ExpandedBrain);
                        _current.Show("Bibite #" + _bridge.SelectedBibite.Slot + " — " + TabTitle(_bridge.SelectedInspectorTab));
                        if (_registeredEscape) UINavigationManager.AddEscapableToStack(_escape);
                    }
                    if (_speciesOpen && _speciesSurface != null)
                    {
                        _speciesSurface.Show("Species / living lineages");
                        UINavigationManager.AddEscapableToStack(_speciesEscape);
                    }
                    if (_selectionOpen && _selectionSurface != null)
                    {
                        _selectionSurface.Show("Selection");
                        UINavigationManager.AddEscapableToStack(_selectionEscape);
                    }
                    if (_gui != null && _gui.TopLeftButtonHolder != null) _gui.TopLeftButtonHolder.SetActive(_bridge.HasNativeSelection);
                }
            }
            bool editing = _tagInput != null && _tagInput.isActiveAndEnabled && _tagInput.isFocused;
            UserControl.SetKeyboardBlockFromSource("GpuNativeInspectorTag", editing);
            if (!IsOpen || !_bridge.HasNativeSelection || Time.realtimeSinceStartup < _nextRefresh) return;
            _nextRefresh = Time.realtimeSinceStartup + 0.2f;
            if (_selectedSlot != _bridge.SelectedBibite.Slot) { Show(_bridge.SelectedInspectorTab); return; }
            NativeWorldBibite b = _bridge.SelectedBibite;
            NativeWorldBibiteDetail d = _bridge.SelectedDetail;
            float speed = Mathf.Sqrt(b.VelocityX * b.VelocityX + b.VelocityY * b.VelocityY);
            string[] values;
            switch (_bridge.SelectedInspectorTab)
            {
                case GpuNativeWorldBridge.InspectorTab.Genes:
                    values = new[] {
                        "Lineage: " + _bridge.SelectedLineageName,
                        "Generation: " + b.Generation,
                        "Body size: " + d.Size.ToString("0.000"),
                        "Maximum speed: " + d.MaximumSpeed.ToString("0.000"),
                        "Turn speed: " + d.TurnSpeed.ToString("0.000"),
                        "Metabolism: " + d.Metabolism.ToString("0.0000") + " energy/s",
                        "Lifespan: " + d.Lifespan.ToString("0.0") + " simulated seconds",
                        "Gene mutation strength: " + d.GeneMutationStrength.ToString("0.0000"),
                        "Brain mutation strength: " + d.BrainMutationStrength.ToString("0.0000"),
                        "Colour RGB: " + d.ColorR.ToString("0.000") + " / " + d.ColorG.ToString("0.000") + " / " + d.ColorB.ToString("0.000") };
                    break;
                case GpuNativeWorldBridge.InspectorTab.Biology:
                    values = new[] {
                        "Energy: " + d.Energy.ToString("0.00"),
                        "Age: " + d.Age.ToString("0.0") + " / " + d.Lifespan.ToString("0.0") + " s lifespan",
                        "Metabolic drain: " + d.Metabolism.ToString("0.0000") + " energy/s",
                        "Reproduction cooldown: " + d.ReproductionCooldown.ToString("0.00") + " s",
                        "Speed: " + speed.ToString("0.00") + " / " + d.MaximumSpeed.ToString("0.00"),
                        "Acceleration output: " + d.AccelerationOutput.ToString("0.000"),
                        "Rotation output: " + d.RotationOutput.ToString("0.000"),
                        "Food target: " + (d.CachedFood < 0 ? "none" : "pellet #" + d.CachedFood),
                        "Food distance: " + (d.CachedFood < 0 ? "—" : Mathf.Sqrt(Mathf.Max(0f, d.FoodDistanceSquared)).ToString("0.0")),
                        "Nearby Bibites sensed: " + d.NeighboursSeen,
                        "Nearest neighbour: " + (d.CachedNeighbour < 0 ? "none" : "#" + d.CachedNeighbour),
                        "Pheromone outputs: " + d.Pheromone1Output.ToString("0.000") + " / " + d.Pheromone2Output.ToString("0.000") + " / " + d.Pheromone3Output.ToString("0.000"),
                        "Reproduction output: " + d.ReproductionOutput.ToString("0.000"),
                        "GPU lifecycle uses direct offspring creation; no separate incubating egg body.",
                        "Original stomach, organs and damage model are not simulated here." };
                    break;
                case GpuNativeWorldBridge.InspectorTab.Brain:
                case GpuNativeWorldBridge.InspectorTab.ExpandedBrain:
                    UpdateBrain(d);
                    values = null;
                    break;
                default:
                    values = new[] {
                        "Lineage: " + _bridge.SelectedLineageName, "Tag: " + _bridge.SelectedTagName,
                        "Generation: " + b.Generation, "Energy: " + b.Energy.ToString("0.00"),
                        "Age: " + b.Age.ToString("0.0") + " simulated seconds", "Body size: " + b.Size.ToString("0.00"),
                        "Speed: " + speed.ToString("0.00"), "Heading: " + (b.Heading * Mathf.Rad2Deg).ToString("0.0") + "°",
                        "Brain: " + b.BrainNodes + " nodes / " + b.BrainSynapses + " synapses",
                        "Position: " + b.PositionX.ToString("0.0") + ", " + b.PositionY.ToString("0.0") };
                    break;
            }
            if (values != null)
                for (int i = 0; i < _rows.Count; i++) SetText(_rows[i], i < values.Length ? values[i] : string.Empty);
            SetText(_notice, _bridge.SelectedNotice ?? string.Empty);
            SetText(_followLabel, _bridge.FollowingSelection ? "Stop following" : "Follow");
        }

        private void UpdateBrain(NativeWorldBibiteDetail detail)
        {
            NativeWorldBrainNodeState[] nodes = _bridge.SelectedBrainNodes;
            NativeWorldBrainSynapseState[] synapses = _bridge.SelectedBrainSynapses;
            if (_brainGraph != null) _brainGraph.SetBrain(nodes, synapses);
            SetText(_rows[0], (detail.TemplateBrain != 0 ? "Imported stock topology" : "Compact 16 × 6 topology") + " · FP16 weights, FP32 arithmetic");
            SetText(_rows[1], nodes.Length + " nodes / " + synapses.Length + " enabled synapses" +
                (detail.TemplateBrain == 0 ? " · sensor/pre-activation values are not retained" : string.Empty));
            int count = _synapses ? synapses.Length : nodes.Length;
            int pages = Math.Max(1, (count + 23) / 24);
            _brainPage = Mathf.Clamp(_brainPage, 0, pages - 1);
            SetText(_brainPageLabel, (_synapses ? "Synapses" : "Nodes") + " · page " + (_brainPage + 1) + " / " + pages);
            for (int row = 0; row < 24; row++)
            {
                int i = _brainPage * 24 + row;
                string value = string.Empty;
                if (i < count)
                {
                    if (_synapses) value = "#" + i + "  node " + synapses[i].NodeIn + " → " + synapses[i].NodeOut + "  weight " + synapses[i].Weight.ToString("0.00000");
                    else value = _bridge.NativeNodeLabel(i, nodes[i]);
                }
                else if (row == 0 && count == 0) value = "Waiting for selected-brain readback…";
                SetText(_rows[row + 2], value);
            }
        }

        private static void SetText(TextMeshProUGUI text, string value)
        {
            if (text != null && text.text != value) text.text = value;
        }

        internal void ClosePanels()
        {
            _bridge.SetNativeInspectorBrainVisible(false);
            UserControl.SetKeyboardBlockFromSource("GpuNativeInspectorTag", false);
            if (_tagInput != null) _tagInput.DeactivateInputField();
            foreach (NativePanelSurface surface in _panels.Values) surface.Hide();
            _current = null;
            if (_registeredEscape) UINavigationManager.RemoveEscapableFromStack(_escape);
            _registeredEscape = false;
        }

        internal void Deselect()
        {
            ClosePanels();
            HideRectangleSelection();
            if (_gui != null && _gui.TopLeftButtonHolder != null) _gui.TopLeftButtonHolder.SetActive(false);
        }

        internal void ShowSpecies(List<GpuNativeWorldBridge.BreakdownEntry> entries, bool estimated)
        {
            if (!Ready || UIScripts.UIReferences.LineagePanel.SpeciesPanel.instance == null) return;
            ClosePanels();
            HideRectangleSelection();
            if (_speciesSurface == null)
            {
                _speciesSurface = new NativePanelSurface(UIScripts.UIReferences.LineagePanel.SpeciesPanel.instance, _gui.StatsPanel, HideSpecies);
                _speciesSummary = _speciesSurface.AddText(string.Empty, 18f);
                _speciesSurface.AddText("The GPU model preserves imported lineages. It does not create automatic stock species trees, so no invented genealogy is displayed.", 14f);
            }
            _speciesSurface.Show("Species / living lineages");
            if (!_speciesOpen) UINavigationManager.AddEscapableToStack(_speciesEscape);
            _speciesOpen = true;
            UpdateSpecies(entries, estimated);
        }

        internal void UpdateSpecies(List<GpuNativeWorldBridge.BreakdownEntry> entries, bool estimated)
        {
            if (_speciesSurface == null || !_speciesOpen) return;
            int count = Math.Min(256, entries.Count);
            SetText(_speciesSummary, entries.Count + " living lineages" +
                (estimated ? " observed in the presentation sample. Counts and energy below are estimates." : ". Counts and energy below are exact for the current snapshot.") +
                (entries.Count > count ? " Showing the 256 largest." : "") + " Click a row to select a member.");
            while (_speciesRows.Count < count)
            {
                int row = _speciesRows.Count;
                _speciesIds.Add(0);
                _speciesRows.Add(_speciesSurface.AddButton(string.Empty, delegate { _bridge.SelectNativeLineage(_speciesIds[row]); }));
            }
            for (int i = 0; i < _speciesRows.Count; i++)
            {
                _speciesRows[i].transform.parent.gameObject.SetActive(i < count);
                if (i >= count) continue;
                _speciesIds[i] = entries[i].Id;
                SetText(_speciesRows[i], entries[i].Name + "  ·  " + (estimated ? "~" : "") + entries[i].Count +
                    " Bibites  ·  " + entries[i].Energy.ToString("0") + " energy");
            }
        }

        private void HideSpecies()
        {
            if (_speciesSurface != null) _speciesSurface.Hide();
            if (_speciesOpen) UINavigationManager.RemoveEscapableFromStack(_speciesEscape);
            _speciesOpen = false;
            _bridge.CloseNativeSpeciesPanel();
        }

        internal void ShowRectangleSelection(List<NativeWorldBibite> selection, bool sampled)
        {
            if (!Ready || _gui.selectionPanel == null) return;
            Trace("show rectangle selection: " + selection.Count + " Bibites, sampled=" + sampled);
            ClosePanels(); HideSpecies();
            if (_selectionSurface == null)
                _selectionSurface = new NativePanelSurface(_gui.selectionPanel, _gui.StatsPanel, HideRectangleSelection);
            _selectionSurface.ClearBody();
            double energy = 0.0;
            foreach (NativeWorldBibite bibite in selection) energy += bibite.Energy;
            _selectionSurface.AddText(selection.Count + " Bibites in this selection snapshot", 18f);
            _selectionSurface.AddText("Combined selected energy: " + energy.ToString("0.0"));
            _selectionSurface.AddText(sampled
                ? "This is a presentation sample, not every Bibite in the rectangle. Food and un-sampled Bibites are not included."
                : "All Bibites whose centres were inside the rectangle at capture are included. Food is not included.", 14f);
            _selectionSurface.AddText("Click a member to inspect its live state. Draw a new rectangle to refresh the selection.", 14f);
            _selectionSurface.AddDisabledButton("Bulk lay / remove / tag — unavailable in GPU mode");
            _selectionSurface.AddText("Bulk mutation is disabled because sampled, reusable slots are not a safe persistent group. Select an individual to use its guarded actions.", 13f);
            int count = Math.Min(256, selection.Count);
            if (count < selection.Count) _selectionSurface.AddText("Showing the first 256 sampled members.", 13f);
            for (int i = 0; i < count; i++)
            {
                NativeWorldBibite bibite = selection[i];
                _selectionSurface.AddButton("Bibite #" + bibite.Slot + " · generation " + bibite.Generation + " · energy " + bibite.Energy.ToString("0.0"),
                    delegate { _bridge.SelectNativeSnapshot(bibite); });
            }
            _selectionSurface.AddFooterButton("Close selection", HideRectangleSelection);
            _selectionSurface.Show("Selection");
            if (!_selectionOpen) UINavigationManager.AddEscapableToStack(_selectionEscape);
            _selectionOpen = true;
        }

        private void HideRectangleSelection()
        {
            if (_selectionSurface != null) _selectionSurface.Hide();
            if (_selectionOpen) UINavigationManager.RemoveEscapableFromStack(_selectionEscape);
            _selectionOpen = false;
        }

        public void Dispose()
        {
            Deselect();
            HideSpecies();
            HideRectangleSelection();
            if (_selectionSurface != null) { _selectionSurface.Dispose(); _selectionSurface = null; }
            if (_speciesSurface != null) { _speciesSurface.Dispose(); _speciesSurface = null; }
            foreach (NativePanelSurface surface in _panels.Values) surface.Dispose();
            _panels.Clear();
            if (Active == this) Active = null;
        }
    }

    internal sealed class NativePanelSurface : IDisposable
    {
        private readonly UIPanel _original;
        private readonly bool _wasEnabled;
        private readonly List<GameObject> _children = new List<GameObject>();
        private readonly List<bool> _childStates = new List<bool>();
        private readonly List<Behaviour> _disabledRootBehaviours = new List<Behaviour>();
        private readonly GameObject _view;
        private readonly RectTransform _content;
        private readonly RectTransform _footer;
        private readonly RectTransform _viewport;
        private readonly RectTransform _scrollbarRect;
        private GameObject _statusNotice;
        private readonly TextMeshProUGUI _fontSource;
        private readonly Image _buttonStyle;
        private readonly Image _inputStyle;
        private readonly TextMeshProUGUI _title;
        private readonly RectTransform _root;
        private readonly Vector2 _savedSize;
        private readonly Vector2 _savedPosition;
        private readonly ScrollRect _scroll;
        internal bool Visible { get { return _original != null && _original.gameObject.activeInHierarchy; } }
        internal bool ContainsScreenPoint(Vector2 point)
        {
            if (!Visible) return false;
            Canvas canvas = _root.GetComponentInParent<Canvas>();
            Camera camera = canvas != null && canvas.renderMode != RenderMode.ScreenSpaceOverlay ? canvas.worldCamera : null;
            return RectTransformUtility.RectangleContainsScreenPoint(_root, point, camera);
        }

        internal NativePanelSurface(UIPanel original, BibiteStatsPanel styleSource, Action close)
        {
            _original = original;
            _root = original.GetComponent<RectTransform>();
            _savedSize = _root.sizeDelta;
            _savedPosition = _root.anchoredPosition;
            _fontSource = styleSource.speciesText;
            foreach (Button button in styleSource.GetComponentsInChildren<Button>(true))
            {
                Image candidate = button.GetComponent<Image>();
                if (candidate != null && candidate.sprite != null) { _buttonStyle = candidate; break; }
            }
            _inputStyle = styleSource.tagInputField != null ? styleSource.tagInputField.GetComponent<Image>() : null;
            Image background = original.GetComponent<Image>();
            if (background == null) background = original.GetComponentInChildren<Image>(true);
            _wasEnabled = original.enabled;
            original.enabled = false;
            foreach (Behaviour component in original.GetComponents<Behaviour>())
            {
                if (component != original && !(component is Graphic) && component.enabled)
                {
                    component.enabled = false;
                    _disabledRootBehaviours.Add(component);
                }
            }
            if (!(original is ExpandedBrainPanel))
            {
                _root.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, Mathf.Max(450f, _root.rect.width));
                _root.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, Mathf.Max(600f, _root.rect.height));
            }
            foreach (Transform child in original.transform)
            {
                _children.Add(child.gameObject);
                _childStates.Add(child.gameObject.activeSelf);
                child.gameObject.SetActive(false);
            }
            _view = new GameObject("GPU state adapter", typeof(RectTransform), typeof(Image));
            RectTransform viewRect = _view.GetComponent<RectTransform>();
            viewRect.SetParent(original.transform, false);
            // Keep the shipped border visible around the opaque content.
            Stretch(viewRect, new Vector2(4f, 4f), new Vector2(-4f, -4f));
            Image panelImage = _view.GetComponent<Image>();
            // A solid backplate is intentional: the old IMGUI skin was nearly
            // transparent, so thousands of sprites rendered through the text.
            float ink = _fontSource.color.r + _fontSource.color.g + _fontSource.color.b;
            panelImage.color = ink < 1.5f ? new Color(0.82f, 0.82f, 0.82f, 1f) : new Color(0.14f, 0.15f, 0.18f, 1f);
            panelImage.raycastTarget = true;

            GameObject header = MakeRect("Title", _view.transform);
            RectTransform headerRect = header.GetComponent<RectTransform>();
            headerRect.anchorMin = new Vector2(0f, 1f); headerRect.anchorMax = Vector2.one;
            headerRect.pivot = new Vector2(0.5f, 1f); headerRect.sizeDelta = new Vector2(-16f, 42f);
            Image headerImage = header.AddComponent<Image>(); headerImage.color = new Color(0f, 0f, 0f, 0.10f);
            GpuPanelDrag drag = header.AddComponent<GpuPanelDrag>(); drag.Target = _root;
            _title = MakeText(header.transform, string.Empty, 18f);
            Stretch(_title.rectTransform, new Vector2(10f, 4f), new Vector2(-48f, -4f));
            GameObject closeButton = MakeRect("Close", header.transform);
            RectTransform closeRect = closeButton.GetComponent<RectTransform>();
            closeRect.anchorMin = closeRect.anchorMax = new Vector2(1f, 0.5f);
            closeRect.sizeDelta = new Vector2(32f, 30f); closeRect.anchoredPosition = new Vector2(-20f, 0f);
            SetupButton(closeButton, "×", close);

            GameObject viewport = MakeRect("Scroll viewport", _view.transform);
            RectTransform viewportRect = viewport.GetComponent<RectTransform>();
            _viewport = viewportRect;
            Stretch(viewportRect, new Vector2(10f, 54f), new Vector2(-28f, -48f));
            Image viewportImage = viewport.AddComponent<Image>(); viewportImage.color = new Color(1f, 1f, 1f, 0.001f);
            viewport.AddComponent<RectMask2D>();
            _scroll = viewport.AddComponent<ScrollRect>();
            _scroll.viewport = viewportRect;
            _scroll.horizontal = false; _scroll.vertical = true; _scroll.scrollSensitivity = 28f;
            _scroll.movementType = ScrollRect.MovementType.Clamped;
            _scroll.inertia = true;
            GameObject scrollBarObject = MakeRect("Vertical scrollbar", _view.transform);
            RectTransform scrollBarRect = scrollBarObject.GetComponent<RectTransform>();
            _scrollbarRect = scrollBarRect;
            scrollBarRect.anchorMin = new Vector2(1f, 0f); scrollBarRect.anchorMax = Vector2.one;
            scrollBarRect.offsetMin = new Vector2(-23f, 54f); scrollBarRect.offsetMax = new Vector2(-7f, -48f);
            Image track = scrollBarObject.AddComponent<Image>(); track.color = new Color(0f, 0f, 0f, 0.16f);
            GameObject handleObject = MakeRect("Handle", scrollBarObject.transform);
            RectTransform handleRect = handleObject.GetComponent<RectTransform>(); Stretch(handleRect, new Vector2(2f, 2f), new Vector2(-2f, -2f));
            Image handleImage = handleObject.AddComponent<Image>();
            handleImage.sprite = _buttonStyle != null ? _buttonStyle.sprite : null;
            handleImage.type = _buttonStyle != null ? _buttonStyle.type : Image.Type.Simple;
            if (_buttonStyle != null) handleImage.pixelsPerUnitMultiplier = _buttonStyle.pixelsPerUnitMultiplier;
            handleImage.color = new Color(0.5f, 0.5f, 0.5f, 1f);
            Scrollbar scrollBar = scrollBarObject.AddComponent<Scrollbar>();
            scrollBar.handleRect = handleRect; scrollBar.targetGraphic = handleImage; scrollBar.direction = Scrollbar.Direction.BottomToTop;
            _scroll.verticalScrollbar = scrollBar;
            _scroll.verticalScrollbarVisibility = ScrollRect.ScrollbarVisibility.AutoHide;
            _content = MakeRect("Live values", viewport.transform).GetComponent<RectTransform>();
            _content.anchorMin = new Vector2(0f, 1f); _content.anchorMax = Vector2.one;
            _content.pivot = new Vector2(0.5f, 1f); _content.sizeDelta = Vector2.zero;
            VerticalLayoutGroup layout = _content.gameObject.AddComponent<VerticalLayoutGroup>();
            layout.childControlHeight = true; layout.childControlWidth = true;
            layout.childForceExpandHeight = false; layout.childForceExpandWidth = true;
            layout.spacing = 4f; layout.padding = new RectOffset(4, 10, 4, 8);
            ContentSizeFitter fitter = _content.gameObject.AddComponent<ContentSizeFitter>();
            fitter.verticalFit = ContentSizeFitter.FitMode.PreferredSize;
            _scroll.content = _content;
            _footer = MakeRect("Actions", _view.transform).GetComponent<RectTransform>();
            _footer.anchorMin = Vector2.zero; _footer.anchorMax = new Vector2(1f, 0f);
            _footer.pivot = new Vector2(0.5f, 0f); _footer.anchoredPosition = new Vector2(0f, 10f);
            _footer.sizeDelta = new Vector2(-20f, 32f);
            HorizontalLayoutGroup footerLayout = _footer.gameObject.AddComponent<HorizontalLayoutGroup>();
            footerLayout.spacing = 5f; footerLayout.childControlWidth = true; footerLayout.childControlHeight = true;
            footerLayout.childForceExpandWidth = true; footerLayout.childForceExpandHeight = true;
        }

        internal void Show(string title)
        {
            _title.text = title;
            _original.gameObject.SetActive(true);
            _view.SetActive(true);
            _root.SetAsLastSibling();
            GpuPanelDrag.Clamp(_root);
        }
        internal void Hide() { if (_original != null) _original.gameObject.SetActive(false); }
        internal void ClearBody()
        {
            foreach (Transform child in _content) { child.gameObject.SetActive(false); UnityEngine.Object.Destroy(child.gameObject); }
            foreach (Transform child in _footer) { child.gameObject.SetActive(false); UnityEngine.Object.Destroy(child.gameObject); }
            if (_statusNotice != null)
            {
                _statusNotice.SetActive(false);
                UnityEngine.Object.Destroy(_statusNotice);
                _statusNotice = null;
            }
            _viewport.offsetMin = new Vector2(_viewport.offsetMin.x, 54f);
            _scrollbarRect.offsetMin = new Vector2(_scrollbarRect.offsetMin.x, 54f);
            _content.anchoredPosition = Vector2.zero;
            _scroll.StopMovement();
        }
        internal TextMeshProUGUI AddStatusNotice()
        {
            // Action results must remain visible while the user is looking at
            // the action buttons, not disappear below the scrolling body.
            _statusNotice = MakeRect("Action status", _view.transform);
            RectTransform rect = _statusNotice.GetComponent<RectTransform>();
            rect.anchorMin = Vector2.zero; rect.anchorMax = new Vector2(1f, 0f);
            rect.pivot = new Vector2(0.5f, 0f); rect.anchoredPosition = new Vector2(0f, 49f);
            rect.sizeDelta = new Vector2(-20f, 64f);
            Image background = _statusNotice.AddComponent<Image>();
            background.color = new Color(0f, 0f, 0f, 0.08f); background.raycastTarget = false;
            TextMeshProUGUI label = MakeText(_statusNotice.transform, string.Empty, 14f);
            label.alignment = TextAlignmentOptions.MidlineLeft;
            label.enableAutoSizing = true; label.fontSizeMin = 11f; label.fontSizeMax = 14f;
            Stretch(label.rectTransform, new Vector2(6f, 3f), new Vector2(-6f, -3f));
            _viewport.offsetMin = new Vector2(_viewport.offsetMin.x, 119f);
            _scrollbarRect.offsetMin = new Vector2(_scrollbarRect.offsetMin.x, 119f);
            return label;
        }
        internal TextMeshProUGUI AddText(string text, float size = 16f)
        {
            TextMeshProUGUI label = MakeText(_content, text, size);
            LayoutElement layout = label.gameObject.AddComponent<LayoutElement>(); layout.minHeight = size + 6f;
            return label;
        }
        internal TextMeshProUGUI AddButton(string title, Action clicked)
        {
            GameObject button = MakeRect(title, _content);
            LayoutElement layout = button.AddComponent<LayoutElement>(); layout.preferredHeight = 30f;
            return SetupButton(button, title, clicked);
        }
        internal void AddDisabledButton(string title)
        {
            TextMeshProUGUI text = AddButton(title, delegate { });
            text.transform.parent.GetComponent<Button>().interactable = false;
        }
        internal TextMeshProUGUI AddFooterButton(string title, Action clicked)
        {
            GameObject button = MakeRect(title, _footer);
            LayoutElement layout = button.AddComponent<LayoutElement>(); layout.flexibleWidth = 1f; layout.minWidth = 70f;
            TextMeshProUGUI label = SetupButton(button, title, clicked); label.fontSize = 14f;
            return label;
        }
        private TextMeshProUGUI SetupButton(GameObject obj, string title, Action clicked)
        {
            Image image = obj.AddComponent<Image>();
            image.sprite = _buttonStyle != null ? _buttonStyle.sprite : null;
            image.type = _buttonStyle != null ? _buttonStyle.type : Image.Type.Simple;
            if (_buttonStyle != null)
            {
                image.pixelsPerUnitMultiplier = _buttonStyle.pixelsPerUnitMultiplier;
                image.material = _buttonStyle.material;
                image.fillCenter = _buttonStyle.fillCenter;
            }
            image.color = _buttonStyle != null ? _buttonStyle.color : new Color(0.38f, 0.39f, 0.40f, 1f);
            Button button = obj.AddComponent<Button>(); button.targetGraphic = image;
            button.onClick.AddListener(delegate { clicked(); });
            TextMeshProUGUI label = MakeText(obj.transform, title, 16f);
            label.alignment = TextAlignmentOptions.Center;
            Stretch(label.rectTransform, new Vector2(4f, 0f), new Vector2(-4f, 0f));
            return label;
        }
        internal TMP_InputField AddInput(string text)
        {
            GameObject input = MakeRect("Tag editor", _content);
            LayoutElement layout = input.AddComponent<LayoutElement>(); layout.preferredHeight = 32f;
            Image image = input.AddComponent<Image>();
            image.sprite = _inputStyle != null ? _inputStyle.sprite : null;
            image.type = _inputStyle != null ? _inputStyle.type : Image.Type.Simple;
            if (_inputStyle != null)
            {
                image.pixelsPerUnitMultiplier = _inputStyle.pixelsPerUnitMultiplier;
                image.material = _inputStyle.material;
                image.fillCenter = _inputStyle.fillCenter;
            }
            image.color = _inputStyle != null ? _inputStyle.color : Color.white;
            GameObject area = MakeRect("Text area", input.transform);
            RectTransform areaRect = area.GetComponent<RectTransform>(); Stretch(areaRect, new Vector2(6f, 2f), new Vector2(-6f, -2f));
            area.AddComponent<RectMask2D>();
            TextMeshProUGUI label = MakeText(area.transform, string.Empty, 16f);
            Stretch(label.rectTransform, Vector2.zero, Vector2.zero);
            TMP_InputField field = input.AddComponent<TMP_InputField>();
            field.textViewport = areaRect; field.textComponent = label; field.targetGraphic = image;
            field.characterLimit = 100; field.lineType = TMP_InputField.LineType.SingleLine;
            field.text = text;
            return field;
        }
        internal GpuBrainGraphic AddBrainGraph(float height)
        {
            GameObject obj = MakeRect("Native neural network", _content);
            LayoutElement layout = obj.AddComponent<LayoutElement>(); layout.preferredHeight = height;
            GpuBrainGraphic graph = obj.AddComponent<GpuBrainGraphic>(); graph.raycastTarget = false;
            return graph;
        }
        private TextMeshProUGUI MakeText(Transform parent, string text, float size)
        {
            GameObject obj = MakeRect("Label", parent);
            TextMeshProUGUI label = obj.AddComponent<TextMeshProUGUI>();
            label.font = _fontSource.font; label.fontSharedMaterial = _fontSource.fontSharedMaterial;
            label.color = _fontSource.color; label.fontSize = size; label.richText = false;
            label.textWrappingMode = TextWrappingModes.Normal; label.raycastTarget = false;
            label.text = text;
            return label;
        }
        private static GameObject MakeRect(string name, Transform parent)
        {
            GameObject obj = new GameObject(name, typeof(RectTransform)); obj.transform.SetParent(parent, false); return obj;
        }
        private static void Stretch(RectTransform rect, Vector2 min, Vector2 max)
        {
            rect.anchorMin = Vector2.zero; rect.anchorMax = Vector2.one; rect.offsetMin = min; rect.offsetMax = max;
        }
        public void Dispose()
        {
            if (_original == null) return;
            Hide();
            UnityEngine.Object.Destroy(_view);
            for (int i = 0; i < _children.Count; i++) if (_children[i] != null) _children[i].SetActive(_childStates[i]);
            _original.enabled = _wasEnabled;
            foreach (Behaviour component in _disabledRootBehaviours) if (component != null) component.enabled = true;
            _root.sizeDelta = _savedSize; _root.anchoredPosition = _savedPosition;
        }
    }

    internal sealed class GpuPanelDrag : MonoBehaviour, IDragHandler
    {
        internal RectTransform Target;
        public void OnDrag(PointerEventData eventData)
        {
            Canvas canvas = Target.GetComponentInParent<Canvas>();
            Target.anchoredPosition += eventData.delta / (canvas != null ? Mathf.Max(0.001f, canvas.scaleFactor) : 1f);
            Clamp(Target);
        }
        private void LateUpdate() { Clamp(Target); }
        internal static void Clamp(RectTransform target)
        {
            if (target == null || !target.gameObject.activeInHierarchy) return;
            Canvas canvas = target.GetComponentInParent<Canvas>();
            float scale = canvas != null ? Mathf.Max(0.001f, canvas.scaleFactor) : 1f;
            float maxWidth = Mathf.Max(220f, (Screen.width - 16f) / scale);
            float maxHeight = Mathf.Max(220f, (Screen.height - 16f) / scale);
            if (target.rect.width > maxWidth) target.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, maxWidth);
            if (target.rect.height > maxHeight) target.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, maxHeight);
            Vector3[] corners = new Vector3[4]; target.GetWorldCorners(corners);
            Camera camera = canvas != null && canvas.renderMode != RenderMode.ScreenSpaceOverlay ? canvas.worldCamera : null;
            Vector2 lo = RectTransformUtility.WorldToScreenPoint(camera, corners[0]);
            Vector2 hi = RectTransformUtility.WorldToScreenPoint(camera, corners[2]);
            float x = lo.x < 8f ? 8f - lo.x : hi.x > Screen.width - 8f ? Screen.width - 8f - hi.x : 0f;
            float y = lo.y < 8f ? 8f - lo.y : hi.y > Screen.height - 8f ? Screen.height - 8f - hi.y : 0f;
            target.anchoredPosition += new Vector2(x, y) / scale;
        }
    }

    internal sealed class GpuBrainGraphic : MaskableGraphic
    {
        private NativeWorldBrainNodeState[] _nodes = new NativeWorldBrainNodeState[0];
        private NativeWorldBrainSynapseState[] _edges = new NativeWorldBrainSynapseState[0];
        internal void SetBrain(NativeWorldBrainNodeState[] nodes, NativeWorldBrainSynapseState[] edges)
        {
            _nodes = nodes; _edges = edges; SetVerticesDirty();
        }
        protected override void OnPopulateMesh(VertexHelper vh)
        {
            vh.Clear();
            if (_nodes == null || _nodes.Length == 0) return;
            Rect r = rectTransform.rect;
            Vector2[] positions = new Vector2[_nodes.Length];
            int[] counts = new int[3]; int[] offsets = new int[3];
            for (int i = 0; i < _nodes.Length; i++) counts[Column(_nodes[i])]++;
            for (int i = 0; i < _nodes.Length; i++)
            {
                int c = Column(_nodes[i]);
                positions[i] = new Vector2(r.xMin + 15f + c * (r.width - 30f) / 2f,
                    r.yMax - 10f - (++offsets[c]) * (r.height - 20f) / (counts[c] + 1f));
            }
            for (int i = 0; i < _edges.Length && i < 4096; i++)
            {
                NativeWorldBrainSynapseState edge = _edges[i];
                if (edge.NodeIn < 0 || edge.NodeOut < 0 || edge.NodeIn >= positions.Length || edge.NodeOut >= positions.Length) continue;
                Vector2 start = positions[edge.NodeIn], end = positions[edge.NodeOut];
                Vector2 direction = end - start;
                Vector2 side = direction.sqrMagnitude > 0.001f ? new Vector2(-direction.y, direction.x).normalized * 0.65f : Vector2.right;
                Color color = edge.Weight >= 0f ? new Color(0.2f, 0.55f, 1f, 0.65f) : new Color(1f, 0.3f, 0.25f, 0.65f);
                Quad(vh, start - side, start + side, end + side, end - side, color);
            }
            for (int i = 0; i < positions.Length; i++)
            {
                Vector2 p = positions[i]; const float radius = 3f;
                Color color = Color.Lerp(new Color(0.35f, 0.38f, 0.42f), new Color(0.4f, 1f, 0.5f), Mathf.Clamp01(Mathf.Abs(_nodes[i].LastOutput)));
                Quad(vh, p + new Vector2(-radius, -radius), p + new Vector2(-radius, radius), p + new Vector2(radius, radius), p + new Vector2(radius, -radius), color);
            }
        }
        private static int Column(NativeWorldBrainNodeState node) { return node.Sensor >= 0 ? 0 : node.Action >= 0 ? 2 : 1; }
        private static void Quad(VertexHelper vh, Vector2 a, Vector2 b, Vector2 c, Vector2 d, Color color)
        {
            int start = vh.currentVertCount;
            vh.AddVert(a, color, Vector2.zero); vh.AddVert(b, color, Vector2.zero); vh.AddVert(c, color, Vector2.zero); vh.AddVert(d, color, Vector2.zero);
            vh.AddTriangle(start, start + 1, start + 2); vh.AddTriangle(start, start + 2, start + 3);
        }
    }

    [HarmonyPatch(typeof(GUIManager), "OpenBibitePanel", new[] { typeof(BibitePanels), typeof(bool) })]
    internal static class NativeInspectorNavigationPatch
    {
        private static bool Prefix(BibitePanels panel, bool toggle)
        {
            GpuNativeInspector ui = GpuNativeInspector.Active;
            if (ui == null || !ui.Bridge.OwnsSimulation || !ui.Bridge.HasNativeSelection) return true;
            GpuNativeWorldBridge.InspectorTab tab = panel == BibitePanels.GenesPanel ? GpuNativeWorldBridge.InspectorTab.Genes :
                panel == BibitePanels.BiologyPanel ? GpuNativeWorldBridge.InspectorTab.Biology :
                panel == BibitePanels.BrainPanel ? GpuNativeWorldBridge.InspectorTab.Brain :
                panel == BibitePanels.ExpendedBrainPanel ? GpuNativeWorldBridge.InspectorTab.ExpandedBrain : GpuNativeWorldBridge.InspectorTab.Stats;
            if (panel == BibitePanels.None || (toggle && ui.IsOpen && ui.Bridge.SelectedInspectorTab == tab)) ui.ClosePanels();
            else ui.Bridge.ShowNativeInspectorTab(tab);
            return false;
        }
    }

    [HarmonyPatch]
    internal static class NativeInspectorActionPatch
    {
        private static IEnumerable<MethodBase> TargetMethods()
        {
            yield return AccessTools.Method(typeof(GUIManager), "CopyTagOfTarget");
            yield return AccessTools.Method(typeof(GUIManager), "TagTarget");
            yield return AccessTools.Method(typeof(GUIManager), "LayEggIfBibite");
            yield return AccessTools.Method(typeof(GUIManager), "TryKillTarget");
        }
        private static bool Prefix(MethodBase __originalMethod)
        {
            GpuNativeInspector ui = GpuNativeInspector.Active;
            if (ui == null || !ui.Bridge.OwnsSimulation || !ui.Bridge.HasNativeSelection) return true;
            switch (__originalMethod.Name)
            {
                case "CopyTagOfTarget": ui.Bridge.CopyNativeTag(); break;
                case "TagTarget": ui.Bridge.SetNativeTag(GUIUtility.systemCopyBuffer); break;
                case "LayEggIfBibite": ui.Bridge.ReproduceNativeSelection(); break;
                case "TryKillTarget": ui.Bridge.KillNativeSelection(); break;
            }
            return false;
        }
    }

    [HarmonyPatch]
    internal static class NativeStockSelectorGuardPatch
    {
        private static IEnumerable<MethodBase> TargetMethods()
        {
            yield return AccessTools.Method(typeof(UserControl), "SelectRandomBibite");
            yield return AccessTools.Method(typeof(UserControl), "SelectOldestBibite");
            yield return AccessTools.Method(typeof(UserControl), "SelectHighGenBibite");
        }
        private static bool Prefix()
        {
            // Native hotkeys are handled by the bridge. The original oldest
            // selector otherwise runs MaxBy over an intentionally empty CPU list.
            return Plugin.Instance == null || !Plugin.Instance.IsNativeWorldActive;
        }
    }

    [HarmonyPatch(typeof(UserControl), "SelectObjectsInRectangle")]
    internal static class NativeRectangleSelectionPatch
    {
        private static bool Prefix(Vector2 diagonalCorner1, Vector2 diagonalCorner2)
        {
            if (Plugin.Instance == null || !Plugin.Instance.IsNativeWorldActive) return true;
            return !Plugin.Instance.TrySelectNativeRectangle(diagonalCorner1, diagonalCorner2);
        }
    }

    [HarmonyPatch(typeof(UserControl), "SelectRandomEgg")]
    internal static class NativeEggSelectionPatch
    {
        private static bool Prefix()
        {
            if (Plugin.Instance == null || !Plugin.Instance.IsNativeWorldActive) return true;
            if (NativeUiVisibility.BlockingPanelOpen) return false;
            PopupManager.DisplayDialog("Egg selection", "The compact GPU model creates offspring directly and has no separate incubating eggs to select.");
            return false;
        }
    }
}
