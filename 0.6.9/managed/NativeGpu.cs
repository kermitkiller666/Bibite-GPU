using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace BibitesGpuFork
{
    internal sealed class GpuDeviceInfo
    {
        internal int Index;
        internal string Name;

        public override string ToString()
        {
            return "GPU " + Index + " - " + Name;
        }
    }

    internal static class NativeGpu
    {
        private const string LibraryName = "BibitesGpuNative";
        private static bool _loadAttempted;
        private static IntPtr _libraryHandle;

        [DllImport("kernel32", SetLastError = true, CharSet = CharSet.Unicode)]
        private static extern IntPtr LoadLibrary(string fileName);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_get_device_count(out int count);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        private static extern int bgf_get_device_name(int deviceIndex, StringBuilder buffer, int bufferLength);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int bgf_set_device(int deviceIndex);

        private static void EnsureLoaded()
        {
            if (_loadAttempted)
            {
                return;
            }

            _loadAttempted = true;
            string assemblyFolder = Path.GetDirectoryName(typeof(NativeGpu).Assembly.Location);
            string nativePath = Path.Combine(assemblyFolder ?? string.Empty, LibraryName + ".dll");
            if (File.Exists(nativePath))
            {
                _libraryHandle = LoadLibrary(nativePath);
            }
        }

        internal static string DescribeDevices()
        {
            try
            {
                List<GpuDeviceInfo> devices = GetDevices();
                StringBuilder result = new StringBuilder();
                result.Append(devices.Count).Append(" CUDA device(s)");
                for (int i = 0; i < devices.Count; i++)
                {
                    result.Append(i == 0 ? ": " : ", ").Append(devices[i].Name);
                }

                return result.ToString();
            }
            catch (DllNotFoundException)
            {
                int error = Marshal.GetLastWin32Error();
                return "native GPU backend not installed or failed to load (Windows error " + error + ")";
            }
            catch (EntryPointNotFoundException)
            {
                return "native GPU backend is incompatible";
            }
        }

        internal static List<GpuDeviceInfo> GetDevices()
        {
            EnsureLoaded();
            int count;
            int status = bgf_get_device_count(out count);
            Check(status, "enumerate CUDA devices");
            List<GpuDeviceInfo> devices = new List<GpuDeviceInfo>(count);
            for (int i = 0; i < count; i++)
            {
                StringBuilder name = new StringBuilder(256);
                Check(bgf_get_device_name(i, name, name.Capacity), "read CUDA device name");
                devices.Add(new GpuDeviceInfo { Index = i, Name = name.ToString() });
            }
            return devices;
        }

        internal static void SelectDevice(int deviceIndex)
        {
            EnsureLoaded();
            Check(bgf_set_device(deviceIndex), "select CUDA device " + deviceIndex);
        }

        private static void Check(int status, string operation)
        {
            if (status != 0)
            {
                throw new InvalidOperationException("Failed to " + operation + " (CUDA error " + status + ").");
            }
        }
    }
}
