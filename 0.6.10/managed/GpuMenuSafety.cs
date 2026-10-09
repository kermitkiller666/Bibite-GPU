using System;

namespace BibitesGpuFork
{
    // Kept independent of Unity so the menu's Windows file-name rules are
    // exercised by the same small executable as the save transaction tests.
    internal static class GpuMenuSafety
    {
        internal static bool IsValidSaveName(string name)
        {
            if (string.IsNullOrWhiteSpace(name) || name.EndsWith(".", StringComparison.Ordinal) ||
                name.EndsWith(" ", StringComparison.Ordinal)) return false;
            foreach (char character in name)
            {
                if (character < 32 || "<>:\"/\\|?*".IndexOf(character) >= 0) return false;
            }
            // Windows reserves device stems even when an extension is present.
            string stem = name.Split('.')[0].TrimEnd(' ').ToUpperInvariant();
            if (stem == "CON" || stem == "PRN" || stem == "AUX" || stem == "NUL" ||
                stem == "CLOCK$" || stem == "CONIN$" || stem == "CONOUT$") return false;
            if (stem.Length == 4 && (stem.StartsWith("COM", StringComparison.Ordinal) ||
                stem.StartsWith("LPT", StringComparison.Ordinal)))
            {
                char digit = stem[3];
                if ((digit >= '1' && digit <= '9') || digit == '\u00b9' ||
                    digit == '\u00b2' || digit == '\u00b3') return false;
            }
            return true;
        }
    }

    // Loading a stock save is a session decision, not an edit to the player's
    // new-world preference. In particular QuickLoad does not enter a new scene.
    internal sealed class GpuWorldSessionMode
    {
        private bool _stockCompatibility;
        private bool _checkpointRequested;
        private bool? _newWorldNativeChoice;

        internal void ResetForSceneChange(bool enabledForNewWorlds)
        {
            _stockCompatibility = false;
            _checkpointRequested = false;
            _newWorldNativeChoice = enabledForNewWorlds;
        }

        internal void SelectStockLoad()
        {
            _stockCompatibility = true;
            _checkpointRequested = false;
        }

        internal void SelectCheckpointLoad()
        {
            _stockCompatibility = false;
            _checkpointRequested = true;
        }

        internal bool ShouldRunNative(bool enabledForNewWorlds, bool ownsWorld, bool starting)
        {
            return !_stockCompatibility &&
                ((_newWorldNativeChoice ?? enabledForNewWorlds) || _checkpointRequested || ownsWorld || starting);
        }

        internal bool ShouldSuppressStock(bool isSimulation, bool enabledForNewWorlds,
            bool ownsWorld, bool starting, bool loadedOrLoading, bool replaceLoadedSaves)
        {
            if (!isSimulation || _stockCompatibility) return false;
            if (_checkpointRequested || ownsWorld || starting) return true;
            if (!(_newWorldNativeChoice ?? enabledForNewWorlds)) return false;
            return !loadedOrLoading || replaceLoadedSaves;
        }
    }
}
