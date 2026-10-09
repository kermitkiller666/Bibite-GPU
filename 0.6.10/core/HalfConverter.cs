using System;

namespace BibitesGpuFork.Core
{
    public static class HalfConverter
    {
        public static ushort ToHalfBits(float value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            uint bits = BitConverter.ToUInt32(bytes, 0);
            uint sign = (bits >> 16) & 0x8000u;
            uint exponent = (bits >> 23) & 0xffu;
            uint mantissa = bits & 0x7fffffu;

            if (exponent == 0xffu)
            {
                return (ushort)(sign | (mantissa == 0 ? 0x7c00u : 0x7e00u));
            }

            int halfExponent = (int)exponent - 127 + 15;
            if (halfExponent >= 31)
            {
                return (ushort)(sign | 0x7c00u);
            }
            if (halfExponent <= 0)
            {
                if (halfExponent < -10)
                {
                    return (ushort)sign;
                }
                mantissa |= 0x800000u;
                int shift = 14 - halfExponent;
                uint rounded = (mantissa + ((1u << (shift - 1)) - 1u) + ((mantissa >> shift) & 1u)) >> shift;
                return (ushort)(sign | rounded);
            }

            uint halfMantissa = (mantissa + 0xfffu + ((mantissa >> 13) & 1u)) >> 13;
            if (halfMantissa == 0x400u)
            {
                halfMantissa = 0;
                halfExponent++;
                if (halfExponent >= 31)
                {
                    return (ushort)(sign | 0x7c00u);
                }
            }
            return (ushort)(sign | ((uint)halfExponent << 10) | halfMantissa);
        }
    }
}
