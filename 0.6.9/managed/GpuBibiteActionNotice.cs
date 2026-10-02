using System;
using System.Globalization;

namespace BibitesGpuFork
{
    internal static class GpuBibiteActionNotice
    {
        internal static bool MatchesSlot(string status, int slot)
        {
            if (string.IsNullOrEmpty(status) || slot < 0) return false;
            const string marker = "GPU Bibite #";
            // Only the first actor is authoritative. Later text may contain an
            // offspring ID or a player-supplied tag which mentions another ID.
            int markerStart = status.IndexOf(marker, StringComparison.Ordinal);
            if (markerStart < 0) return false;
            int start = markerStart + marker.Length;
            int end = start;
            while (end < status.Length && status[end] >= '0' && status[end] <= '9') ++end;
            if (end == start) return false;
            if (end < status.Length && !char.IsWhiteSpace(status[end]) &&
                status[end] != '.' && status[end] != ':' && status[end] != ';' && status[end] != ',')
                return false;
            int actor;
            return int.TryParse(status.Substring(start, end - start), NumberStyles.None,
                CultureInfo.InvariantCulture, out actor) && actor == slot;
        }
    }
}
