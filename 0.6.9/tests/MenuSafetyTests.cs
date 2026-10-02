using System;
using System.Collections.Generic;
using System.Globalization;
using System.Threading;
using BibitesGpuFork;
using Newtonsoft.Json.Linq;

internal static class MenuSafetyTests
{
    internal static void Run()
    {
        TestSaveNames();
        TestSessionTransitions();
        TestPresentationRoundTrip();
        TestMalformedPresentation();
        TestPresentationBounds();
        TestBibiteActionMessages();
    }

    private static void TestSaveNames()
    {
        string[] valid = { "World 1", "My world 2026.10", "Audit_Aurora", "conifer", "COM0",
            "COM10", "LPT10", "two..dots", "evolved α 🧬" };
        foreach (string name in valid)
            Check(GpuMenuSafety.IsValidSaveName(name), "valid save name: " + name);
        string[] invalid = { null, "", "   ", ".", "..", "world.", "world ", "a/b", "a\\b",
            "a:b", "a\"b", "a?b", "a*b", "a<b", "a>b", "a|b", "a\0b", "a\nb", "a\tb",
            "CON", "con.txt", "PRN", "aux", "NUL.backup", "CLOCK$", "CONIN$", "CONOUT$",
            "COM1", "com9.old", "LPT1", "lpt9", "COM1 .old", "COM¹", "LPT²", "COM³" };
        foreach (string name in invalid)
            Check(!GpuMenuSafety.IsValidSaveName(name), "invalid save name: " + (name ?? "<null>"));
    }

    private static void TestSessionTransitions()
    {
        GpuWorldSessionMode mode = new GpuWorldSessionMode();
        Check(mode.ShouldRunNative(true, false, false), "new GPU world follows preference");
        Check(mode.ShouldSuppressStock(true, true, false, false, false, false), "new GPU world suppresses stock seeding");
        Check(!mode.ShouldSuppressStock(false, true, true, true, false, true), "menus never suppress stock simulation");

        // Changing a NEXT-world preference must not freeze the current native
        // runner or restart the CPU food/spawn systems underneath it.
        Check(mode.ShouldRunNative(false, true, false), "active GPU world survives next-world checkbox off");
        Check(mode.ShouldSuppressStock(true, false, true, false, false, false), "active GPU retains stock suppression");
        Check(mode.ShouldRunNative(false, false, true), "starting GPU world survives next-world checkbox off");
        Check(mode.ShouldSuppressStock(true, false, false, true, false, false), "starting GPU retains stock suppression");

        mode.SelectStockLoad();
        foreach (bool loadedFlag in new[] { false, true })
        foreach (bool replaceLoaded in new[] { false, true })
        {
            Check(!mode.ShouldRunNative(true, false, false), "stock load never starts a native replacement");
            Check(!mode.ShouldRunNative(true, true, true), "stock load decision overrides stale native flags");
            Check(!mode.ShouldSuppressStock(true, true, true, true, loadedFlag, replaceLoaded),
                "stock QuickLoad unblocks zones and spawners regardless of stale gameWasLoaded/ReplaceLoadedSaves");
        }

        // Same-scene loading a GPU save again is an explicit session switch.
        mode.SelectCheckpointLoad();
        Check(mode.ShouldRunNative(false, false, false), "GPU checkpoint works with new-world GPU preference off");
        Check(mode.ShouldSuppressStock(true, false, false, false, true, false), "GPU checkpoint suppresses stock seeding");
        mode.SelectStockLoad();
        Check(!mode.ShouldRunNative(true, false, false), "GPU-to-stock second roundtrip remains stock");
        mode.ResetForSceneChange(true);
        Check(mode.ShouldRunNative(true, false, false), "new scene clears stock compatibility override");
        mode.ResetForSceneChange(false);
        Check(!mode.ShouldRunNative(false, false, false), "new scene with disabled preference remains stock");
        Check(!mode.ShouldRunNative(true, false, false), "enabling next-world GPU preference cannot replace current CPU world");
        Check(!mode.ShouldSuppressStock(true, true, false, false, false, true), "enabling next-world GPU preference cannot freeze current CPU food/spawners");
        mode.ResetForSceneChange(true);
        Check(mode.ShouldRunNative(false, false, false), "disabling next-world GPU preference cannot cancel current scene startup");
        Check(!mode.ShouldSuppressStock(true, true, false, false, true, false), "unprepared stock load protected by stock loaded flag");
        mode.SelectCheckpointLoad();
        mode.ResetForSceneChange(false);
        Check(!mode.ShouldRunNative(false, false, false), "scene reset clears old checkpoint intent");
    }

    private static void TestPresentationRoundTrip()
    {
        Dictionary<ulong, string> lineages = new Dictionary<ulong, string>
        {
            { 0, "GPU native population" }, { 18446744073709551615UL, "Aurora lineage 🌌" }
        };
        Dictionary<ulong, string> tags = new Dictionary<ulong, string>
        {
            { 42, "Audit Aurora128" }, { 18446744073709551615UL, "α tag" }
        };
        CultureInfo previousCulture = Thread.CurrentThread.CurrentCulture;
        try
        {
            Thread.CurrentThread.CurrentCulture = CultureInfo.GetCultureInfo("ar-SA");
            JObject exported = GpuPresentationMetadata.Export(lineages, tags);
            Check(exported["lineages"]["18446744073709551615"] != null, "ID serialization uses invariant decimal");
            JObject serializedAndRead = JObject.Parse(exported.ToString());
            Dictionary<ulong, string> loadedLineages = new Dictionary<ulong, string> { { 9, "stale lineage" } };
            Dictionary<ulong, string> loadedTags = new Dictionary<ulong, string> { { 9, "stale tag" } };
            GpuPresentationMetadata.Import(serializedAndRead, loadedLineages, loadedTags);
            EqualNames(lineages, loadedLineages, "lineage save-load roundtrip");
            EqualNames(tags, loadedTags, "tag save-load roundtrip");
            lineages[0] = "changed after export";
            Check((string)exported["lineages"]["0"] == "GPU native population", "export does not alias source dictionary");
            serializedAndRead["tags"]["42"] = "changed after import";
            Check(loadedTags[42] == "Audit Aurora128", "import does not alias metadata");
            GpuPresentationMetadata.Import(null, loadedLineages, loadedTags);
            Check(loadedLineages.Count == 0 && loadedTags.Count == 0, "legacy save without metadata clears previous names");
        }
        finally { Thread.CurrentThread.CurrentCulture = previousCulture; }
    }

    private static void TestMalformedPresentation()
    {
        JObject metadata = new JObject
        {
            ["lineages"] = new JObject
            {
                ["1"] = "kept", ["-1"] = "negative", ["+2"] = "signed", [" 3"] = "whitespace",
                ["bad"] = "not an ID", ["18446744073709551616"] = "overflow", ["7"] = 77,
                ["8"] = JValue.CreateNull(), ["9"] = new JObject { ["nested"] = "not a name" }
            },
            ["tags"] = new JArray("not", "an", "object")
        };
        Dictionary<ulong, string> lineages = new Dictionary<ulong, string> { { 99, "stale" } };
        Dictionary<ulong, string> tags = new Dictionary<ulong, string> { { 99, "stale" } };
        GpuPresentationMetadata.Import(metadata, lineages, tags);
        Check(lineages.Count == 1 && lineages[1] == "kept", "malformed metadata skips bad entries without losing valid names");
        Check(tags.Count == 0, "malformed tag group cannot retain previous-world names");
    }

    private static void TestPresentationBounds()
    {
        string splitPair = new string('x', 99) + "🧬" + "suffix";
        string retainedPair = new string('x', 98) + "🧬" + "suffix";
        Dictionary<ulong, string> names = new Dictionary<ulong, string>
        {
            { 1, splitPair }, { 2, retainedPair }, { 3, new string('x', 150) }, { 4, null }
        };
        JObject exported = GpuPresentationMetadata.Export(names, new Dictionary<ulong, string>());
        Check(((string)exported["lineages"]["1"]).Length == 99, "export truncation does not split UTF-16 pair");
        Check(((string)exported["lineages"]["2"]).Length == 100, "complete UTF-16 pair fits name limit");
        Check(((string)exported["lineages"]["3"]).Length == 100, "export name length bounded");
        Check(exported["lineages"]["4"] == null, "null export name omitted");
        Dictionary<ulong, string> lineages = new Dictionary<ulong, string>();
        Dictionary<ulong, string> tags = new Dictionary<ulong, string>();
        GpuPresentationMetadata.Import(new JObject { ["lineages"] = new JObject { ["1"] = splitPair } }, lineages, tags);
        Check(lineages[1].Length == 99, "import truncation does not split UTF-16 pair");

        JObject many = new JObject();
        Dictionary<ulong, string> manyNames = new Dictionary<ulong, string>();
        for (ulong i = 0; i < (ulong)GpuPresentationMetadata.MaximumNames + 2; ++i)
        {
            many[i.ToString(CultureInfo.InvariantCulture)] = "name";
            manyNames[i] = "name";
        }
        GpuPresentationMetadata.Import(new JObject { ["lineages"] = many }, lineages, tags);
        Check(lineages.Count == GpuPresentationMetadata.MaximumNames &&
            !lineages.ContainsKey((ulong)GpuPresentationMetadata.MaximumNames), "import entry count bounded");
        JObject bounded = GpuPresentationMetadata.Export(manyNames, tags);
        Check(((JObject)bounded["lineages"]).Count == GpuPresentationMetadata.MaximumNames, "export entry count bounded");
    }

    private static void EqualNames(Dictionary<ulong, string> expected, Dictionary<ulong, string> actual, string name)
    {
        Check(expected.Count == actual.Count, name + " count");
        foreach (KeyValuePair<ulong, string> entry in expected)
            Check(actual.ContainsKey(entry.Key) && actual[entry.Key] == entry.Value, name + " value");
    }

    private static void TestBibiteActionMessages()
    {
        Check(GpuBibiteActionNotice.MatchesSlot("GPU Bibite #12: insufficient energy", 12), "colon failure reaches selected actor");
        Check(GpuBibiteActionNotice.MatchesSlot("GPU Bibite #12. Removed.", 12), "period completion reaches selected actor");
        Check(GpuBibiteActionNotice.MatchesSlot("GPU Bibite #12", 12), "end-of-string actor accepted");
        Check(GpuBibiteActionNotice.MatchesSlot("GPU Bibite #12 produced offspring #128.", 12), "parent receives offspring result");
        Check(!GpuBibiteActionNotice.MatchesSlot("GPU Bibite #128: failed", 12), "slot prefix cannot match longer ID");
        Check(!GpuBibiteActionNotice.MatchesSlot("GPU Bibite #12 produced offspring #128.", 128), "offspring reference cannot steal actor result");
        Check(!GpuBibiteActionNotice.MatchesSlot("GPU Bibite #128 tag set to GPU Bibite #12.", 12), "tag reference cannot steal actor result");
        Check(!GpuBibiteActionNotice.MatchesSlot("GPU Bibite #12oops", 12), "actor ID requires delimiter");
        Check(!GpuBibiteActionNotice.MatchesSlot("GPU Bibite #2147483648: too large", int.MaxValue), "overflowing actor ID rejected");
        Check(!GpuBibiteActionNotice.MatchesSlot(null, 12), "missing status ignored");
        Check(!GpuBibiteActionNotice.MatchesSlot("", 12), "empty status ignored");
        Check(!GpuBibiteActionNotice.MatchesSlot("GPU Bibite #-12: invalid", -12), "negative actor ID rejected");
    }

    private static void Check(bool condition, string name)
    {
        if (!condition) throw new InvalidOperationException(name);
    }
}
