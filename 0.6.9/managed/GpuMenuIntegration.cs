using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using HarmonyLib;
using ManagementScripts;
using TMPro;
using UIScripts;
using UIScripts.SettingHandles;
using UIScripts.UIPanels;
using UIScripts.UIReferences;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace BibitesGpuFork
{
    internal static class GpuMenuIntegration
    {
        private static GameObject _modalBlocker;
        private static readonly HashSet<string> _shownNativeNotices = new HashSet<string>();
        private static readonly Dictionary<string, IEscapable> _nativeNoticeBlocks =
            new Dictionary<string, IEscapable>();
        private static int _noticeSequence;
        private const string GpuButtonName = "GPU Settings Button";

        internal static bool HasFocusedTextInput()
        {
            GameObject selected = EventSystem.current != null
                ? EventSystem.current.currentSelectedGameObject : null;
            if (selected == null) return false;
            TMP_InputField tmp = selected.GetComponentInParent<TMP_InputField>();
            if (tmp != null && tmp.isFocused) return true;
            InputField legacy = selected.GetComponentInParent<InputField>();
            return legacy != null && legacy.isFocused;
        }

        internal static bool BlocksWorldPointer()
        {
            return (EventSystem.current != null && EventSystem.current.IsPointerOverGameObject()) ||
                (Plugin.Instance != null && Plugin.Instance.IsPointerOverGpuSettings(Input.mousePosition));
        }

        internal static void SetModalBlocker(bool active)
        {
            if (!active)
            {
                if (_modalBlocker != null) _modalBlocker.SetActive(false);
                return;
            }
            if (_modalBlocker == null)
            {
                _modalBlocker = new GameObject("GPU settings pointer blocker",
                    typeof(RectTransform), typeof(Canvas), typeof(GraphicRaycaster));
                _modalBlocker.hideFlags = HideFlags.DontSave;
                Canvas canvas = _modalBlocker.GetComponent<Canvas>();
                canvas.renderMode = RenderMode.ScreenSpaceOverlay;
                canvas.overrideSorting = true;
                canvas.sortingOrder = short.MaxValue;
                GameObject surface = new GameObject("Pointer capture", typeof(RectTransform), typeof(Image));
                surface.transform.SetParent(_modalBlocker.transform, false);
                RectTransform rect = surface.GetComponent<RectTransform>();
                rect.anchorMin = Vector2.zero;
                rect.anchorMax = Vector2.one;
                rect.offsetMin = Vector2.zero;
                rect.offsetMax = Vector2.zero;
                Image image = surface.GetComponent<Image>();
                image.color = new Color(0f, 0f, 0f, 0f);
                image.raycastTarget = true;
            }
            _modalBlocker.SetActive(true);
            if (EventSystem.current != null) EventSystem.current.SetSelectedGameObject(null);
        }

        internal static bool HasSettingsButton(UserSettingsManager panel)
        {
            return panel != null && panel.transform.Find("SelectorPanel/List/" + GpuButtonName) != null;
        }

        internal static void EnsureSettingsButton(UserSettingsManager panel)
        {
            if (panel == null || HasSettingsButton(panel)) return;
            Transform stockButton = panel.transform.Find("SelectorPanel/List/GraphicsButton");
            if (stockButton == null || stockButton.GetComponent<Button>() == null) return;
            GameObject clone = UnityEngine.Object.Instantiate(stockButton.gameObject, stockButton.parent, false);
            clone.name = GpuButtonName;
            Transform hotkeys = stockButton.parent.Find("Controls Button");
            clone.transform.SetSiblingIndex(hotkeys != null ? hotkeys.GetSiblingIndex() + 1 : 3);
            Button button = clone.GetComponent<Button>();
            // Remove persistent stock OpenTab callbacks as well as runtime ones.
            button.onClick = new Button.ButtonClickedEvent();
            button.onClick.AddListener(delegate
            {
                if (Plugin.Instance != null) Plugin.Instance.SetGpuSettingsOpen(true);
            });
            TMP_Text label = clone.GetComponentInChildren<TMP_Text>(true);
            if (label != null) label.text = "GPU Settings";
            clone.SetActive(true);

            Transform hotkeyPage = panel.transform.Find("PagesPanels/Hotkeys");
            if (hotkeyPage != null)
            {
                foreach (TMP_Text text in hotkeyPage.GetComponentsInChildren<TMP_Text>(true))
                {
                    if (text.text.IndexOf("Space =>", StringComparison.Ordinal) >= 0 &&
                        text.text.IndexOf("Ctrl + Page", StringComparison.Ordinal) < 0)
                    {
                        text.text += "\n\nGPU fork controls:\nCtrl + PageUp / PageDown => increase / decrease requested time warp" +
                            "\nCtrl + Shift + F8 / F9 / F10 => compatibility-mode diagnostics / brain mode" +
                            "\nGPU Settings => simulation device, population, graphics and performance options";
                    }
                }
            }
        }

        internal static void SetPanelText(object panel, string field, string value)
        {
            FieldInfo info = AccessTools.Field(panel.GetType(), field);
            TMP_Text text = info != null ? info.GetValue(panel) as TMP_Text : null;
            if (text != null) text.text = value;
        }

        internal static void ExplainNativePanel(UIPanel panel)
        {
            if (Plugin.Instance == null || !Plugin.Instance.IsNativeWorldActive) return;
            if (panel is DynamicSettingsPanel || panel is ZonesEditorPanel || panel is SettingsChangersEditor)
            {
                if (!_shownNativeNotices.Add("Dynamic settings")) return;
                ShowNativeDialog("GPU world - Dynamic Settings",
                    "Live GPU controls: food density, fertility (regrowth) and pellet energy.\n\n" +
                    "Use GPU Settings for the next world's population, textures and Extreme profile; graphics FPS is live.\n\n" +
                    "The compact GPU model does not implement the original radiation/pheromone towers, spatial zone effects, " +
                    "or all the original physics, biology and mutation controls. Those settings remain available for stock CPU worlds, " +
                    "but changing them here does not change the running GPU simulation.");
            }
            else if (panel is BibiteEditorPanel && _shownNativeNotices.Add("Template editor"))
            {
                ShowNativeDialog("GPU world - Template Editor",
                    "This is the original template editor. Its changes apply to templates you save and place later, " +
                    "not to the selected live GPU Bibite. Placement imports the supported genes and neural-network topology; " +
                    "the compact GPU model does not implement every original organ or gene.");
            }
        }

        internal static void ShowNativeDialog(string title, string message)
        {
            string keyboardSource = "GpuForkNotice" + (++_noticeSequence);
            UserControl.SetKeyboardBlockFromSource(keyboardSource, true);
            _nativeNoticeBlocks[keyboardSource] = null;
            try
            {
                DialogHandle dialog = PopupManager.DisplayDialog(title, message,
                    delegate { ReleaseNativeNotice(keyboardSource); }, true);
                IEscapable escape = new EscapableAction(delegate { if (dialog != null) dialog.Dismiss(); });
                _nativeNoticeBlocks[keyboardSource] = escape;
                UINavigationManager.AddEscapableToStack(escape);
            }
            catch (Exception ex)
            {
                ReleaseNativeNotice(keyboardSource);
                Plugin.LogNativeCheckpointWarning("Could not display the GPU-world notice: " + ex.Message);
            }
        }

        private static void ReleaseNativeNotice(string keyboardSource)
        {
            UserControl.SetKeyboardBlockFromSource(keyboardSource, false);
            IEscapable escape;
            if (_nativeNoticeBlocks.TryGetValue(keyboardSource, out escape) && escape != null)
                UINavigationManager.RemoveEscapableFromStack(escape);
            _nativeNoticeBlocks.Remove(keyboardSource);
        }

        internal static FileInfo GetSelectedSave(object panel)
        {
            FieldInfo itemField = AccessTools.Field(panel.GetType(), "selectedSaveItem");
            SaveItemReference item = itemField != null ? itemField.GetValue(panel) as SaveItemReference : null;
            if (item == null) return null;
            FieldInfo info = AccessTools.Field(panel.GetType(), "selectedSaveInfo");
            return info != null ? info.GetValue(panel) as FileInfo : null;
        }

        internal static bool HasValidSaveName(SaveGamePanel panel)
        {
            FieldInfo field = AccessTools.Field(typeof(SaveGamePanel), "saveName");
            TMP_InputField input = field != null ? field.GetValue(panel) as TMP_InputField : null;
            if (input == null) return false;
            return GpuMenuSafety.IsValidSaveName(input.text);
        }

        internal static void ClearStaleSaveSelection(object panel, SaveItemReference item)
        {
            if (item != null) return;
            FieldInfo field = AccessTools.Field(panel.GetType(), "selectedSaveInfo");
            if (field != null) field.SetValue(panel, null);
            FieldInfo buttonField = AccessTools.Field(panel.GetType(), "loadButton");
            Button loadButton = buttonField != null ? buttonField.GetValue(panel) as Button : null;
            if (loadButton != null) loadButton.interactable = false;
        }

        internal static void Dispose()
        {
            if (_modalBlocker != null) UnityEngine.Object.Destroy(_modalBlocker);
            _modalBlocker = null;
            foreach (KeyValuePair<string, IEscapable> notice in _nativeNoticeBlocks)
            {
                UserControl.SetKeyboardBlockFromSource(notice.Key, false);
                UINavigationManager.RemoveEscapableFromStack(notice.Value);
            }
            _nativeNoticeBlocks.Clear();
            _shownNativeNotices.Clear();
        }
    }

    // Unity catches exceptions thrown by coroutines outside SaveWorld's caller.
    // Intercept the iterator so a failed stock wrapper releases the save lock,
    // removes only our staging files, and reports a useful original-style popup.
    internal sealed class GpuSaveGuardEnumerator : IEnumerator, IDisposable
    {
        private readonly IEnumerator _inner;
        private readonly string _savePath;
        private bool _finished;
        internal GpuSaveGuardEnumerator(IEnumerator inner, string savePath)
        {
            _inner = inner;
            _savePath = savePath;
        }
        public object Current { get { return _inner.Current; } }
        public bool MoveNext()
        {
            try
            {
                bool hasNext = _inner.MoveNext();
                if (!hasNext)
                {
                    _finished = true;
                    if (Plugin.Instance != null)
                        Plugin.Instance.FailPendingNativeSave(_savePath,
                            "the stock save finished without committing its GPU checkpoint");
                }
                return hasNext;
            }
            catch (Exception ex)
            {
                _finished = true;
                string error = ex.Message;
                try
                {
                    IDisposable disposable = _inner as IDisposable;
                    if (disposable != null) disposable.Dispose();
                }
                catch (Exception cleanupError) { error += " (save cleanup: " + cleanupError.Message + ")"; }
                if (Plugin.Instance != null) Plugin.Instance.FailPendingNativeSave(_savePath, error);
                return false;
            }
        }
        public void Reset() { throw new NotSupportedException(); }
        public void Dispose()
        {
            try
            {
                IDisposable disposable = _inner as IDisposable;
                if (disposable != null) disposable.Dispose();
            }
            finally
            {
                if (!_finished && Plugin.Instance != null)
                    Plugin.Instance.FailPendingNativeSave(_savePath, "the save operation was cancelled before completion");
                _finished = true;
            }
        }
    }

    [HarmonyPatch(typeof(SaveSystem), "CreateSave")]
    internal static class NativeSaveCoroutineGuardPatch
    {
        private static void Postfix(string saveFileName, ref IEnumerator __result)
        {
            if (Plugin.Instance != null && Plugin.Instance.IsStagedNativeSave(saveFileName))
                __result = new GpuSaveGuardEnumerator(__result, saveFileName);
        }
    }

    [HarmonyPatch(typeof(GameManager), "StartGame", new Type[] { typeof(string) })]
    internal static class NativeMenuLoadPreflightPatch
    {
        private static bool Prefix(string saveToLoadPath)
        {
            return Plugin.Instance == null || Plugin.Instance.ValidateWorldForLoad(saveToLoadPath);
        }
    }

    [HarmonyPatch]
    internal static class NativeSaveSceneChangeGuardPatch
    {
        private static IEnumerable<MethodBase> TargetMethods()
        {
            yield return AccessTools.Method(typeof(GameManager), "StartGame", Type.EmptyTypes);
            yield return AccessTools.Method(typeof(GameManager), "OpenMenu");
            yield return AccessTools.Method(typeof(GameManager), "QuitGame");
        }
        private static bool Prefix()
        {
            return Plugin.Instance == null || Plugin.Instance.CanChangeWorld();
        }
    }

    [HarmonyPatch(typeof(LoadGamePanel), "LoadSelectedSave")]
    internal static class NativeLoadPanelPreflightPatch
    {
        private static bool Prefix(LoadGamePanel __instance)
        {
            FileInfo selected = GpuMenuIntegration.GetSelectedSave(__instance);
            if (selected == null) return false;
            return Plugin.Instance == null || Plugin.Instance.ValidateWorldForLoad(selected.FullName);
        }
    }

    [HarmonyPatch]
    internal static class NativeSaveSelectionClearPatch
    {
        private static IEnumerable<MethodBase> TargetMethods()
        {
            yield return AccessTools.Method(typeof(LoadGamePanel), "SelectSaveItem");
            yield return AccessTools.Method(typeof(SaveGamePanel), "SelectSaveItem");
        }
        private static void Postfix(object __instance, SaveItemReference item)
        {
            GpuMenuIntegration.ClearStaleSaveSelection(__instance, item);
        }
    }

    [HarmonyPatch(typeof(SaveGamePanel), "OpenPanel")]
    internal static class NativeSavePanelPreviewPatch
    {
        private static void Postfix(SaveGamePanel __instance)
        {
            if (Plugin.Instance != null) Plugin.Instance.RefreshNativeSavePreview(__instance);
        }
    }

    [HarmonyPatch(typeof(SaveGamePanel), "OnSaveNameChanged")]
    internal static class SafeSaveNameButtonPatch
    {
        private static void Postfix(SaveGamePanel __instance)
        {
            if (GpuMenuIntegration.HasValidSaveName(__instance)) return;
            FieldInfo field = AccessTools.Field(typeof(SaveGamePanel), "saveButton");
            Button button = field != null ? field.GetValue(__instance) as Button : null;
            if (button != null) button.interactable = false;
        }
    }

    [HarmonyPatch(typeof(SaveGamePanel), "Save")]
    internal static class SafeSaveNameActionPatch
    {
        private static bool Prefix(SaveGamePanel __instance)
        {
            if (GpuMenuIntegration.HasValidSaveName(__instance)) return true;
            PopupManager.DisplayError("Save game", "Enter a valid file name without trailing spaces, dots or reserved device names.");
            return false;
        }
    }

    [HarmonyPatch(typeof(CameraManager), "Update")]
    internal static class NativeWarpCameraShortcutPatch
    {
        private static bool Prefix()
        {
            // Bare PageUp/Down retain the original camera-speed controls.
            return Plugin.Instance == null || !Plugin.Instance.CanUseSimulationHotkeys ||
                !(Input.GetKey(KeyCode.LeftControl) || Input.GetKey(KeyCode.RightControl)) ||
                !(Input.GetKeyDown(KeyCode.PageUp) || Input.GetKeyDown(KeyCode.PageDown));
        }
    }

    [HarmonyPatch(typeof(BibitePlacer), "UpdatePanel")]
    internal static class NativePlacerModalInputPatch
    {
        private static bool Prefix()
        {
            return Plugin.Instance == null || !Plugin.Instance.IsGpuSettingsOpen;
        }
    }

    [HarmonyPatch(typeof(UIPanel), "OpenPanel")]
    internal static class NativeDynamicSettingsNoticePatch
    {
        private static void Prefix(UIPanel __instance, out bool __state)
        {
            __state = __instance.isActiveAndEnabled;
        }
        private static void Postfix(UIPanel __instance, bool __state)
        {
            if (!__state && __instance.isActiveAndEnabled)
                GpuMenuIntegration.ExplainNativePanel(__instance);
        }
    }

    [HarmonyPatch]
    internal static class NativeUnsupportedTowerActionPatch
    {
        private static IEnumerable<MethodBase> TargetMethods()
        {
            yield return AccessTools.Method(typeof(WorldObjectsSpawner), "GenerateColorSelector");
            yield return AccessTools.Method(typeof(WorldObjectsSpawner), "GeneratePheromoneTower");
            yield return AccessTools.Method(typeof(WorldObjectsSpawner), "GenerateRadioTower");
        }
        private static bool Prefix()
        {
            if (Plugin.Instance == null || !Plugin.Instance.IsNativeWorldActive) return true;
            GpuMenuIntegration.ShowNativeDialog("Original-world tool",
                "Radiation towers, pheromone towers and colour selectors operate on the original CPU simulation. " +
                "They do not affect GPU-resident Bibites, so no inactive tool was placed.\n\n" +
                "To use these tools, disable native GPU worlds in GPU Settings and start a new stock world.");
            return false;
        }
    }
}
