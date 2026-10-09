using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Windows.Forms;

internal static class PortableLauncher
{
    private const string PayloadResource = "BibitesPayload.zip";
    private const string PayloadSha256 = "__PAYLOAD_SHA256__";
    private const string CoreSha256 = "__CORE_SHA256__";
    private const string PluginSha256 = "__PLUGIN_SHA256__";
    private const string NativeSha256 = "__NATIVE_SHA256__";
    private const string MainExe = "The Bibites.exe";

    [STAThread]
    private static void Main()
    {
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        Application.Run(new LaunchWindow());
    }

    private sealed class LaunchWindow : Form
    {
        private readonly Label message;
        private readonly ProgressBar progress;
        private readonly BackgroundWorker worker;

        internal LaunchWindow()
        {
            Text = "Bibites GPU Fork 0.6.10 PREVIEW";
            ClientSize = new Size(430, 98);
            FormBorderStyle = FormBorderStyle.FixedDialog;
            MaximizeBox = false;
            MinimizeBox = false;
            StartPosition = FormStartPosition.CenterScreen;
            ShowInTaskbar = true;

            message = new Label();
            message.Text = "Preparing The Bibites...";
            message.Location = new Point(17, 16);
            message.Size = new Size(396, 26);
            Controls.Add(message);

            progress = new ProgressBar();
            progress.Location = new Point(17, 51);
            progress.Size = new Size(396, 23);
            progress.Minimum = 0;
            progress.Maximum = 100;
            Controls.Add(progress);

            worker = new BackgroundWorker();
            worker.WorkerReportsProgress = true;
            worker.DoWork += Prepare;
            worker.ProgressChanged += OnProgress;
            worker.RunWorkerCompleted += OnPrepared;
            Shown += delegate { worker.RunWorkerAsync(); };
        }

        private void OnProgress(object sender, ProgressChangedEventArgs e)
        {
            progress.Value = Math.Max(0, Math.Min(100, e.ProgressPercentage));
            if (e.UserState != null)
                message.Text = e.UserState.ToString();
        }

        private void OnPrepared(object sender, RunWorkerCompletedEventArgs e)
        {
            if (e.Error != null)
            {
                MessageBox.Show(this, e.Error.Message, "Bibites GPU Fork could not start",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
                Close();
                return;
            }

            string gameDirectory = (string)e.Result;
            try
            {
                ProcessStartInfo info = new ProcessStartInfo();
                info.FileName = Path.Combine(gameDirectory, MainExe);
                info.WorkingDirectory = gameDirectory;
                info.UseShellExecute = true;
                Process.Start(info);
                Close();
            }
            catch (Exception ex)
            {
                MessageBox.Show(this, "The game files are ready, but Windows could not start the game.\n\n" +
                    ex.Message + "\n\nGame folder: " + gameDirectory,
                    "Bibites GPU Fork could not start", MessageBoxButtons.OK, MessageBoxIcon.Error);
                Close();
            }
        }

        private void Prepare(object sender, DoWorkEventArgs e)
        {
            string launcherDirectory = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            string gameDirectory = Path.Combine(launcherDirectory,
                "Bibites GPU Fork 0.6.10 - preview-data-" + PayloadSha256.Substring(0, 12).ToLowerInvariant());
            string marker = Path.Combine(gameDirectory, ".portable-payload.sha256");

            if (Directory.Exists(gameDirectory))
            {
                if (!File.Exists(marker) || !String.Equals(File.ReadAllText(marker).Trim(),
                    PayloadSha256, StringComparison.OrdinalIgnoreCase) || !HasRequiredFiles(gameDirectory))
                {
                    throw new IOException("The extracted game folder is incomplete or has been changed. " +
                        "It was left untouched to protect your data.\n\nFolder: " + gameDirectory);
                }
                worker.ReportProgress(100, "Launching The Bibites...");
                e.Result = gameDirectory;
                return;
            }

            worker.ReportProgress(0, "Checking the bundled game files...");
            VerifyPayload();

            string temporaryDirectory = gameDirectory + ".extracting-" + Guid.NewGuid().ToString("N");
            Directory.CreateDirectory(temporaryDirectory);
            try
            {
                ExtractPayload(temporaryDirectory);
                if (!HasRequiredFiles(temporaryDirectory))
                    throw new InvalidDataException("The bundled game is missing a required file.");
                File.WriteAllText(Path.Combine(temporaryDirectory, ".portable-payload.sha256"),
                    PayloadSha256 + Environment.NewLine, Encoding.ASCII);
                Directory.Move(temporaryDirectory, gameDirectory);
            }
            catch
            {
                // Keep a failed extraction for diagnosis; never delete possible user data.
                throw;
            }

            worker.ReportProgress(100, "Launching The Bibites...");
            e.Result = gameDirectory;
        }

        private static bool HasRequiredFiles(string directory)
        {
            return File.Exists(Path.Combine(directory, MainExe)) &&
                File.Exists(Path.Combine(directory, "UnityPlayer.dll")) &&
                File.Exists(Path.Combine(directory, "winhttp.dll")) &&
                File.Exists(Path.Combine(directory, "BepInEx", "plugins", "BibitesGpuFork", "BibitesGpuFork.Core.dll")) &&
                File.Exists(Path.Combine(directory, "BepInEx", "plugins", "BibitesGpuFork", "BibitesGpuFork.dll")) &&
                File.Exists(Path.Combine(directory, "BepInEx", "plugins", "BibitesGpuFork", "BibitesGpuNative.dll")) &&
                MatchesFileHash(Path.Combine(directory, "BepInEx", "plugins", "BibitesGpuFork", "BibitesGpuFork.Core.dll"), CoreSha256) &&
                MatchesFileHash(Path.Combine(directory, "BepInEx", "plugins", "BibitesGpuFork", "BibitesGpuFork.dll"), PluginSha256) &&
                MatchesFileHash(Path.Combine(directory, "BepInEx", "plugins", "BibitesGpuFork", "BibitesGpuNative.dll"), NativeSha256);
        }

        private static bool MatchesFileHash(string path, string expected)
        {
            using (FileStream stream = File.OpenRead(path))
            using (SHA256 sha = SHA256.Create())
                return String.Equals(BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", ""),
                    expected, StringComparison.OrdinalIgnoreCase);
        }

        private static Stream OpenPayload()
        {
            Stream stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(PayloadResource);
            if (stream == null)
                throw new InvalidDataException("The game payload is missing from this launcher.");
            return stream;
        }

        private static void VerifyPayload()
        {
            using (Stream stream = OpenPayload())
            using (SHA256 sha = SHA256.Create())
            {
                byte[] digest = sha.ComputeHash(stream);
                string actual = BitConverter.ToString(digest).Replace("-", "");
                if (!String.Equals(actual, PayloadSha256, StringComparison.OrdinalIgnoreCase))
                    throw new InvalidDataException("The bundled game files failed their integrity check. " +
                        "Please obtain a fresh copy of this launcher.");
            }
        }

        private void ExtractPayload(string destination)
        {
            string root = Path.GetFullPath(destination).TrimEnd(Path.DirectorySeparatorChar) +
                Path.DirectorySeparatorChar;
            long totalBytes = 0;

            using (Stream payload = OpenPayload())
            using (ZipArchive archive = new ZipArchive(payload, ZipArchiveMode.Read, false))
            {
                if (archive.Entries.Count > 10000)
                    throw new InvalidDataException("The game payload has too many files.");

                for (int i = 0; i < archive.Entries.Count; i++)
                {
                    ZipArchiveEntry entry = archive.Entries[i];
                    string relative = entry.FullName.Replace('/', Path.DirectorySeparatorChar);
                    if (String.IsNullOrEmpty(relative) || Path.IsPathRooted(relative) ||
                        relative.IndexOf(':') >= 0)
                        throw new InvalidDataException("The game payload has an unsafe path.");

                    string fullPath = Path.GetFullPath(Path.Combine(root, relative));
                    if (!fullPath.StartsWith(root, StringComparison.OrdinalIgnoreCase))
                        throw new InvalidDataException("The game payload has an unsafe path.");

                    if (entry.Name.Length != 0)
                    {
                        totalBytes += entry.Length;
                        if (totalBytes > 512L * 1024L * 1024L)
                            throw new InvalidDataException("The game payload exceeds its expected size.");

                        Directory.CreateDirectory(Path.GetDirectoryName(fullPath));
                        using (Stream input = entry.Open())
                        using (FileStream output = new FileStream(fullPath, FileMode.CreateNew, FileAccess.Write))
                            input.CopyTo(output);
                    }

                    int percent = (int)((i + 1L) * 100L / archive.Entries.Count);
                    worker.ReportProgress(percent, "Unpacking The Bibites... " + percent + "%");
                }
            }
        }
    }
}
