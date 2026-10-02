using System;

namespace BibitesGpuFork.Core
{
    public static class TimeWarpSpeeds
    {
        public const int Maximum = 10000;

        public static readonly float[] Values =
        {
            1f, 2f, 3f, 5f, 10f, 25f, 50f, 100f, 250f, 500f, 1000f,
            2500f, 5000f, Maximum
        };

        public static float Snap(float requested)
        {
            if (float.IsNaN(requested) || requested <= Values[0])
            {
                return Values[0];
            }
            // Clamp before subtracting: Infinity and large finite floats can
            // make every candidate distance identical and otherwise select 1x.
            if (requested >= Values[Values.Length - 1])
            {
                return Values[Values.Length - 1];
            }

            float closest = Values[0];
            float closestDistance = Math.Abs(requested - closest);
            for (int i = 1; i < Values.Length; i++)
            {
                float distance = Math.Abs(requested - Values[i]);
                if (distance < closestDistance)
                {
                    closest = Values[i];
                    closestDistance = distance;
                }
            }

            return closest;
        }

        public static float Next(float current)
        {
            float snapped = Snap(current);
            for (int i = 0; i < Values.Length - 1; i++)
            {
                if (snapped == Values[i])
                {
                    return Values[i + 1];
                }
            }

            return Values[Values.Length - 1];
        }

        public static float Previous(float current)
        {
            float snapped = Snap(current);
            for (int i = 1; i < Values.Length; i++)
            {
                if (snapped == Values[i])
                {
                    return Values[i - 1];
                }
            }

            return Values[0];
        }
    }
}
