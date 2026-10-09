using System;

namespace BibitesGpuFork.Core
{
    public static class FoodPelletCap
    {
        // Stock zones express their maximum biomass as energy. A pellet's
        // expected energy is its size factor times the scenario pellet energy.
        public static double EstimateZonePellets(
            double maxBiomass, double pelletSize, double pelletEnergy)
        {
            if (double.IsNaN(maxBiomass) || double.IsInfinity(maxBiomass) ||
                maxBiomass <= 0.0 || double.IsNaN(pelletSize) ||
                double.IsInfinity(pelletSize) || pelletSize <= 0.0 ||
                double.IsNaN(pelletEnergy) || double.IsInfinity(pelletEnergy) ||
                pelletEnergy <= 0.0)
                return 0.0;

            return Math.Ceiling(Math.Min(1000000000000.0,
                maxBiomass / (pelletSize * pelletEnergy)));
        }

        // Both the stock biomass setting and the GPU PelletCount setting are
        // ceilings, not multipliers or a request to spawn 8192 plants.
        public static int Resolve(double stockEstimate, int gpuCap, int reserve)
        {
            int maximum = Math.Max(0, Math.Min(gpuCap, reserve));
            if (maximum == 0 || double.IsNaN(stockEstimate) ||
                stockEstimate <= 0.0)
                return 0;
            if (double.IsInfinity(stockEstimate) || stockEstimate >= maximum)
                return maximum;
            return Math.Min(maximum, (int)Math.Ceiling(stockEstimate));
        }

        public static int ResolveLoaded(
            double stockEstimate, double baselineEstimate, int savedTarget,
            int gpuCap, int reserve)
        {
            // A growth-only zone is still present when its biomass is zero.
            // Do not replace a saved zero target with the configured GPU cap.
            if (savedTarget == 0 && stockEstimate == baselineEstimate)
                return 0;
            double desiredTarget = stockEstimate;
            if (stockEstimate > 0.0 && baselineEstimate > 0.0 &&
                !double.IsInfinity(baselineEstimate) && savedTarget > 0)
                desiredTarget = savedTarget * (stockEstimate / baselineEstimate);
            return Resolve(desiredTarget, gpuCap, reserve);
        }
    }
}
