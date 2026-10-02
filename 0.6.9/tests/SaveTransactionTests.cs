using System;
using System.IO;
using BibitesGpuFork;

internal static class SaveTransactionTests
{
    internal static void Run()
    {
        string root = Path.Combine(Path.GetTempPath(), "BibitesGpuSaveTests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            string path = Path.Combine(root, "world.zip");
            var first = new GpuSaveTransaction(path);
            Stage(first, "wrapper-one", "checkpoint-one");
            first.Commit();
            AssertPair(path, "wrapper-one", "checkpoint-one", "initial save");
            AssertNoStaging(first);

            var replacement = new GpuSaveTransaction(path);
            Stage(replacement, "wrapper-two", "checkpoint-two");
            replacement.Commit();
            AssertPair(path, "wrapper-two", "checkpoint-two", "replacement save");
            AssertNoStaging(replacement);

            var incomplete = new GpuSaveTransaction(path);
            File.WriteAllText(incomplete.StagingWorldPath, "incomplete-wrapper");
            ExpectIoFailure(incomplete.Commit, "missing GPU checkpoint");
            AssertPair(path, "wrapper-two", "checkpoint-two", "failed staging preserves old pair");
            incomplete.DiscardStaging();
            AssertNoStaging(incomplete);

            // The checkpoint is installed first. Force the wrapper replacement
            // to fail and prove that the prior matching checkpoint is restored.
            var blocked = new GpuSaveTransaction(path);
            Stage(blocked, "blocked-wrapper", "blocked-checkpoint");
            using (File.Open(path, FileMode.Open, FileAccess.Read, FileShare.Read))
                ExpectIoFailure(blocked.Commit, "locked wrapper");
            AssertPair(path, "wrapper-two", "checkpoint-two", "wrapper failure rolls checkpoint back");
            blocked.DiscardStaging();
            AssertNoStaging(blocked);

            // A stock-only save has no prior GPU checkpoint. Rollback must not
            // leave a stray sidecar that would make it appear to be a GPU save.
            string stockPath = Path.Combine(root, "stock.zip");
            File.WriteAllText(stockPath, "stock-wrapper");
            var stockBlocked = new GpuSaveTransaction(stockPath);
            Stage(stockBlocked, "native-wrapper", "native-checkpoint");
            using (File.Open(stockPath, FileMode.Open, FileAccess.Read, FileShare.Read))
                ExpectIoFailure(stockBlocked.Commit, "locked stock wrapper");
            if (File.ReadAllText(stockPath) != "stock-wrapper" || File.Exists(stockPath + ".bgfgpu"))
                throw new InvalidOperationException("Failed conversion changed the stock save or left a sidecar.");
            stockBlocked.DiscardStaging();
            AssertNoStaging(stockBlocked);

            if (Directory.GetFiles(root).Length != 3)
                throw new InvalidOperationException("Save transactions left unexpected staging or recovery files.");
        }
        finally
        {
            // Only the unique test directory created above is eligible for
            // cleanup; no player save directory is used by these tests.
            Directory.Delete(root, true);
        }
    }

    private static void Stage(GpuSaveTransaction transaction, string wrapper, string checkpoint)
    {
        File.WriteAllText(transaction.StagingWorldPath, wrapper);
        File.WriteAllText(transaction.StagingCheckpointPath, checkpoint);
    }

    private static void AssertPair(string path, string wrapper, string checkpoint, string label)
    {
        if (File.ReadAllText(path) != wrapper || File.ReadAllText(path + ".bgfgpu") != checkpoint)
            throw new InvalidOperationException(label + ": save files are not the expected matching pair.");
    }

    private static void AssertNoStaging(GpuSaveTransaction transaction)
    {
        if (File.Exists(transaction.StagingWorldPath) || File.Exists(transaction.StagingCheckpointPath))
            throw new InvalidOperationException("Save transaction left a staged wrapper/checkpoint.");
    }

    private static void ExpectIoFailure(Action action, string label)
    {
        try { action(); }
        catch (IOException) { return; }
        throw new InvalidOperationException(label + ": expected an I/O failure.");
    }
}
