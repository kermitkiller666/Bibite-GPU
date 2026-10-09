using System;

namespace BibitesGpuFork.Core
{
    public static class BrainDiagramMath
    {
        public const int NativeNodeCount = 46;
        public const int FullNativeNodeCount = 73;

        public static int NativeColumn(int nodeIndex)
        {
            if (nodeIndex < 0 || nodeIndex >= NativeNodeCount) return -1;
            if (nodeIndex < 16) return 0;
            if (nodeIndex < 28) return 1;
            if (nodeIndex < 40) return 2;
            return 3;
        }

        public static int FullNativeColumn(int nodeIndex)
        {
            if (nodeIndex < 0 || nodeIndex >= FullNativeNodeCount) return -1;
            if (nodeIndex < 34) return 0;
            if (nodeIndex < 46) return 1;
            if (nodeIndex < 58) return 2;
            return 3;
        }

        // The 85th percentile avoids one exceptional weight dimming every
        // other connection. The floor keeps tiny near-zero weights faint.
        public static float ReferenceMagnitude(float[] weights)
        {
            if (weights == null || weights.Length == 0) return 1f;
            float[] magnitudes = new float[weights.Length];
            int count = 0;
            for (int i = 0; i < weights.Length; i++)
            {
                float value = weights[i];
                if (float.IsNaN(value) || float.IsInfinity(value)) continue;
                float magnitude = Math.Abs(value);
                if (magnitude > 0f) magnitudes[count++] = magnitude;
            }
            if (count == 0) return 1f;
            Array.Sort(magnitudes, 0, count);
            int percentile = (int)Math.Floor((count - 1) * 0.85);
            return Math.Max(0.25f, magnitudes[percentile]);
        }

        public static float StrengthBrightness(float weight, float referenceMagnitude)
        {
            if (float.IsNaN(weight) || float.IsInfinity(weight)) return 0f;
            float reference = float.IsNaN(referenceMagnitude) ||
                float.IsInfinity(referenceMagnitude) || referenceMagnitude <= 0f
                ? 1f : referenceMagnitude;
            double fraction = Math.Min(1d, Math.Abs((double)weight) / reference);
            return 0.12f + 0.88f * (float)Math.Sqrt(fraction);
        }
    }
}
