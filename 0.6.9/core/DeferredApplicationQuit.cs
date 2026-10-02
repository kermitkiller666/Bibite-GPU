namespace BibitesGpuFork.Core
{
    // Pure state machine for Application.wantsToQuit. Pending work is drained
    // by normal Update calls, never by blocking Unity's shutdown callback.
    public sealed class DeferredApplicationQuit
    {
        public bool Requested { get; private set; }
        public bool Allowed { get; private set; }

        public bool Request(bool resourcesPending)
        {
            if (Allowed) return true;
            if (!resourcesPending)
            {
                Allowed = true;
                return true;
            }
            Requested = true;
            return false;
        }

        public bool TryComplete(bool resourcesReleased)
        {
            if (!Requested || Allowed || !resourcesReleased) return false;
            Allowed = true;
            return true;
        }
    }
}
