using System;
using System.Collections.Generic;
using System.Globalization;
using Newtonsoft.Json.Linq;

namespace BibitesGpuFork
{
    internal static class GpuPresentationMetadata
    {
        internal const int MaximumNames = 65536;
        internal const int MaximumNameCharacters = 100;

        internal static JObject Export(IDictionary<ulong, string> lineages,
            IDictionary<ulong, string> tags)
        {
            return new JObject
            {
                ["lineages"] = ExportNames(lineages),
                ["tags"] = ExportNames(tags)
            };
        }

        internal static void Import(JObject metadata, IDictionary<ulong, string> lineages,
            IDictionary<ulong, string> tags)
        {
            if (lineages == null) throw new ArgumentNullException("lineages");
            if (tags == null) throw new ArgumentNullException("tags");
            lineages.Clear();
            tags.Clear();
            if (metadata == null) return;
            ImportNames(metadata["lineages"] as JObject, lineages);
            ImportNames(metadata["tags"] as JObject, tags);
        }

        private static JObject ExportNames(IDictionary<ulong, string> source)
        {
            if (source == null) throw new ArgumentNullException("source");
            JObject result = new JObject();
            int count = 0;
            foreach (KeyValuePair<ulong, string> item in source)
            {
                if (++count > MaximumNames) break;
                if (item.Value == null) continue;
                result[item.Key.ToString(CultureInfo.InvariantCulture)] = LimitName(item.Value);
            }
            return result;
        }

        private static void ImportNames(JObject source, IDictionary<ulong, string> target)
        {
            if (source == null) return;
            int count = 0;
            foreach (JProperty item in source.Properties())
            {
                if (++count > MaximumNames) break;
                ulong id;
                if (!ulong.TryParse(item.Name, NumberStyles.None, CultureInfo.InvariantCulture, out id) ||
                    item.Value.Type != JTokenType.String) continue;
                string name = (string)item.Value;
                if (name != null) target[id] = LimitName(name);
            }
        }

        private static string LimitName(string name)
        {
            if (name.Length <= MaximumNameCharacters) return name;
            int count = MaximumNameCharacters;
            if (char.IsHighSurrogate(name[count - 1]) && char.IsLowSurrogate(name[count])) --count;
            return name.Substring(0, count);
        }
    }
}
