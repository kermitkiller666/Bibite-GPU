using System;
using System.IO;

namespace BibitesGpuFork
{
    // Both files are generated beside the destination before either old file is
    // replaced. The wrapper is the commit point; if it cannot be replaced, the
    // prior checkpoint is put back so the original save remains a matching pair.
    internal sealed class GpuSaveTransaction
    {
        internal string WorldPath { get; private set; }
        internal string StagingWorldPath { get; private set; }
        internal string CheckpointPath { get { return WorldPath + ".bgfgpu"; } }
        internal string StagingCheckpointPath { get { return StagingWorldPath + ".bgfgpu"; } }
        private readonly string _rollbackCheckpointPath;

        internal GpuSaveTransaction(string worldPath)
        {
            WorldPath = Path.GetFullPath(worldPath);
            string directory = Path.GetDirectoryName(WorldPath);
            if (string.IsNullOrEmpty(directory))
            {
                throw new ArgumentException("The save destination has no parent directory.", "worldPath");
            }
            Directory.CreateDirectory(directory);
            string token = Guid.NewGuid().ToString("N");
            StagingWorldPath = WorldPath + ".saving-" + token;
            _rollbackCheckpointPath = WorldPath + ".checkpoint-rollback-" + token;
        }

        internal void Commit()
        {
            if (!File.Exists(StagingWorldPath) || !File.Exists(StagingCheckpointPath))
            {
                throw new IOException("The staged wrapper or GPU checkpoint is missing; the previous save was not replaced.");
            }
            bool hadCheckpoint = File.Exists(CheckpointPath);
            bool installedCheckpoint = false;
            try
            {
                if (hadCheckpoint)
                {
                    File.Replace(StagingCheckpointPath, CheckpointPath, _rollbackCheckpointPath);
                }
                else
                {
                    File.Move(StagingCheckpointPath, CheckpointPath);
                }
                installedCheckpoint = true;
                if (File.Exists(WorldPath))
                {
                    File.Replace(StagingWorldPath, WorldPath, null);
                }
                else
                {
                    File.Move(StagingWorldPath, WorldPath);
                }
            }
            catch (Exception commitError)
            {
                if (installedCheckpoint)
                {
                    try
                    {
                        if (hadCheckpoint)
                        {
                            File.Replace(_rollbackCheckpointPath, CheckpointPath, null);
                        }
                        else
                        {
                            File.Delete(CheckpointPath);
                        }
                    }
                    catch (Exception rollbackError)
                    {
                        throw new IOException("The save could not be committed and the checkpoint rollback also failed. " +
                            "Keep the recovery file at " + _rollbackCheckpointPath + ". " +
                            rollbackError.Message, commitError);
                    }
                }
                throw;
            }
            // Failure to delete an obsolete recovery copy must not turn an
            // otherwise complete, loadable save into a reported save failure.
            TryDelete(_rollbackCheckpointPath);
        }

        internal void DiscardStaging()
        {
            TryDelete(StagingWorldPath);
            TryDelete(StagingCheckpointPath);
        }

        private static void TryDelete(string path)
        {
            try
            {
                if (File.Exists(path)) File.Delete(path);
            }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }
    }
}
