using HarmonyLib;
using ManagementScripts;
using TMPro;
using UIScripts;
using UIScripts.UIReferences;
using UnityEngine;
using UnityEngine.UI;
using Utility;

namespace BibitesGpuFork
{
    // Keep the stock chart navigation/renderer, but distinguish measured native
    // streams from CPU-only metrics. Zero is a value, not a missing-data marker.
    internal sealed class GpuNativeCharts : MonoBehaviour
    {
        private GraphsPanel _panel;
        private GameObject _notice;
        private TextMeshProUGUI _text;
        private TextMeshProUGUI _samplingNote;

        internal static bool NativeActive
        {
            get { return Plugin.Instance != null && Plugin.Instance.IsNativeWorldActive; }
        }

        internal void Bind(GraphsPanel panel)
        {
            _panel = panel;
            if (_notice == null)
            {
                _notice = new GameObject("GPU chart availability", typeof(RectTransform), typeof(Image));
                RectTransform rect = _notice.GetComponent<RectTransform>(); rect.SetParent(panel.transform, false);
                rect.anchorMin = new Vector2(0.15f, 0.25f); rect.anchorMax = new Vector2(0.85f, 0.75f);
                rect.offsetMin = rect.offsetMax = Vector2.zero;
                _notice.GetComponent<Image>().color = new Color(0.14f, 0.15f, 0.18f, 1f);
                _text = CreateText(_notice.transform, panel.dataDropdown.label, 20f);
                _text.color = Color.white; _text.alignment = TextAlignmentOptions.Center;
                RectTransform textRect = _text.rectTransform; textRect.anchorMin = Vector2.zero; textRect.anchorMax = Vector2.one;
                textRect.offsetMin = new Vector2(22f, 22f); textRect.offsetMax = new Vector2(-22f, -22f);
                _samplingNote = CreateText(panel.transform, panel.dataDropdown.label, 14f);
                _samplingNote.color = new Color(0.95f, 0.9f, 0.65f, 1f);
                _samplingNote.alignment = TextAlignmentOptions.BottomLeft;
                RectTransform noteRect = _samplingNote.rectTransform;
                noteRect.anchorMin = new Vector2(0f, 0f); noteRect.anchorMax = new Vector2(1f, 0f);
                noteRect.pivot = new Vector2(0.5f, 0f); noteRect.anchoredPosition = new Vector2(0f, 10f); noteRect.sizeDelta = new Vector2(-40f, 40f);
            }
            Refresh();
        }

        private static TextMeshProUGUI CreateText(Transform parent, TextMeshProUGUI source, float size)
        {
            GameObject obj = new GameObject("Native chart note", typeof(RectTransform)); obj.transform.SetParent(parent, false);
            TextMeshProUGUI text = obj.AddComponent<TextMeshProUGUI>();
            text.font = source.font; text.fontSharedMaterial = source.fontSharedMaterial;
            text.fontSize = size; text.richText = false; text.raycastTarget = false;
            text.textWrappingMode = TextWrappingModes.Normal;
            return text;
        }

        private void Update() { Refresh(); }

        private void Refresh()
        {
            if (_panel == null || _notice == null) return;
            bool active = NativeActive;
            int stream = _panel.dataDropdown.value;
            bool genes = _panel.genesGraphPage.activeSelf;
            bool unsupported = active && (genes || stream == 4 || stream == 5);
            _notice.SetActive(unsupported);
            _samplingNote.gameObject.SetActive(active && !unsupported);
            if (unsupported)
            {
                string subject = genes ? "Stock gene-distribution history" : stream == 4 ? "Age-at-death distribution" : "Per-Bibite eggs-laid history";
                _text.text = subject + " is not recorded by the compact GPU simulation.\n\nNo data is shown here; zeros would be misleading.\n\nUse Historical Data for live counts, energy, sampled age/brain size and birth/death totals.";
            }
            else if (active)
            {
                _samplingNote.text = "GPU history: age and brain-size distributions use a presentation sample; history is sampled at render snapshots. Unsupported biological series are explicitly unavailable.";
            }
            // The blank CPU buckets contain 0/0 ratios. Do not leave those bars
            // or fabricated zero curves visible behind the missing-data notice.
            if (_panel.genesGraph != null) _panel.genesGraph.gameObject.SetActive(!active || !genes);
            if (_panel.historicGraph != null) _panel.historicGraph.gameObject.SetActive(!unsupported || genes);
        }
    }

    [HarmonyPatch(typeof(GraphsPanel), "OpenPanel")]
    internal static class NativeChartAvailabilityPatch
    {
        private static void Postfix(GraphsPanel __instance)
        {
            GpuNativeCharts adapter = __instance.GetComponent<GpuNativeCharts>();
            if (adapter == null) adapter = __instance.gameObject.AddComponent<GpuNativeCharts>();
            adapter.Bind(__instance);
        }
    }

    [HarmonyPatch(typeof(BarGraph), "UpdateGraphData")]
    internal static class NativeGeneHistogramGuardPatch
    {
        private static bool Prefix() { return !GpuNativeCharts.NativeActive; }
    }
}
