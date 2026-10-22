class FakeClock:
    """Advances by `dt` on every call; a training step reads it twice, so each step lasts dt."""

    def __init__(self, dt: float = 0.5):
        self.t = 0.0
        self.dt = dt

    def __call__(self) -> float:
        self.t += self.dt
        return self.t
