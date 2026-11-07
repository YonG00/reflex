"""Optional wall budget beginning after the first completed Actor update."""


class JointBudget:
    def __init__(self, seconds=None):
        if seconds is not None and seconds <= 0:
            raise ValueError("Joint budget must be positive")
        self.seconds = seconds
        self.started = None

    def actor_completed(self, now):
        if self.started is None:
            self.started = now
            return True
        return False

    def elapsed(self, now):
        return 0.0 if self.started is None else now - self.started

    def expired(self, now):
        return (
            self.seconds is not None
            and self.started is not None
            and self.elapsed(now) >= self.seconds
        )
