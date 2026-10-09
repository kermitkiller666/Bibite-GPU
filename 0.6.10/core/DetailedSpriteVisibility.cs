using System;

namespace BibitesGpuFork.Core
{
    public struct DetailedSpriteViewport
    {
        public readonly float MinX;
        public readonly float MinY;
        public readonly float MaxX;
        public readonly float MaxY;

        public DetailedSpriteViewport(float minX, float minY, float maxX, float maxY)
        {
            MinX = minX;
            MinY = minY;
            MaxX = maxX;
            MaxY = maxY;
        }

        public bool Contains(DetailedSpriteViewport other)
        {
            return IsValid && other.IsValid &&
                MinX <= other.MinX && MinY <= other.MinY &&
                MaxX >= other.MaxX && MaxY >= other.MaxY;
        }

        private bool IsValid
        {
            get
            {
                return Finite(MinX) && Finite(MinY) &&
                    Finite(MaxX) && Finite(MaxY) && MinX < MaxX && MinY < MaxY;
            }
        }

        private static bool Finite(float value)
        {
            return !float.IsNaN(value) && !float.IsInfinity(value);
        }
    }

    public static class DetailedSpriteVisibility
    {
        public const float ApproximateBibiteDiameter = 12f;
        public const float MinimumDetailedDiameterPixels = 3.5f;
        public const float EdgePadding = 12f;

        public static bool ShouldRender(float orthographicSize, int screenHeight)
        {
            if (float.IsNaN(orthographicSize) ||
                float.IsInfinity(orthographicSize) ||
                orthographicSize <= 0f || screenHeight <= 0)
                return false;

            double pixelsPerWorldUnit = screenHeight /
                (Math.Max(0.001, orthographicSize) * 2.0);
            return pixelsPerWorldUnit * ApproximateBibiteDiameter >=
                MinimumDetailedDiameterPixels;
        }

        public static bool CanReplaceLowDetailMesh(
            int visibleCount,
            int totalCount,
            int detailedCapacity)
        {
            if (visibleCount <= 0 || totalCount <= 0 || detailedCapacity <= 0 ||
                visibleCount > detailedCapacity)
                return false;

            // The native visible list is truncated at detailedCapacity. A count
            // below the capacity therefore proves every on-screen Bibite fit.
            // When it exactly fills the list, it is only complete if the entire
            // living population also fits in that list.
            return visibleCount < detailedCapacity || totalCount <= visibleCount;
        }
    }
}
