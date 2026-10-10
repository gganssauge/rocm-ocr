"""Dynamic concurrency limiter for managing concurrent inference tasks."""

import asyncio
from types import TracebackType


class DynamicConcurrencyLimiter:
    """Manages dynamic concurrency limits for asynchronous OCR tasks."""

    def __init__(self, initial_concurrency: int = 1) -> None:
        """Initialize the limiter with an initial concurrency capacity.

        Args:
            initial_concurrency: Maximum number of simultaneous tasks. Must be >= 1.
        """
        if initial_concurrency < 1:
            raise ValueError("Concurrency must be greater than or equal to 1")

        self._capacity = initial_concurrency
        self._active_tasks = 0
        self._condition = asyncio.Condition()

    @property
    def capacity(self) -> int:
        """Return the current maximum concurrency limit."""
        return self._capacity

    @property
    def active_tasks(self) -> int:
        """Return the number of currently executing tasks."""
        return self._active_tasks

    async def update_concurrency(self, new_concurrency: int) -> None:
        """Dynamically update maximum concurrency and notify waiting tasks.

        Args:
            new_concurrency: The new capacity limit (must be >= 1).
        """
        if new_concurrency < 1:
            raise ValueError("Concurrency must be greater than or equal to 1")

        async with self._condition:
            self._capacity = new_concurrency
            self._condition.notify_all()

    async def acquire(self) -> None:
        """Wait until a slot is available and increment the active task counter."""
        async with self._condition:
            while self._active_tasks >= self._capacity:
                await self._condition.wait()
            self._active_tasks += 1

    async def release(self) -> None:
        """Release a slot and notify waiting tasks."""
        async with self._condition:
            self._active_tasks = max(0, self._active_tasks - 1)
            self._condition.notify_all()

    async def __aenter__(self) -> "DynamicConcurrencyLimiter":
        """Acquire a concurrency permit upon entering context."""
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Release the concurrency permit upon exiting context."""
        await self.release()
