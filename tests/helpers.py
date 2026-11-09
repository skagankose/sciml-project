class FakeClock:
    """Advances by `dt` on every call.

    A training step reads it three times (before fetching the batch, when the batch is ready, after the
    update), so each step waits dt for data and computes for dt: dt per step under budget_clock=compute,
    2 dt under wall.
    """

    def __init__(self, dt: float = 0.5):
        self.t = 0.0
        self.dt = dt

    def __call__(self) -> float:
        self.t += self.dt
        return self.t
