"""Dynamic concurrency limiter for managing concurrent inference tasks."""

import asyncio
from collections.abc import Callable
from types import TracebackType
from typing import Self


async def run_in_thread[**P, T](
    function: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
) -> T:
    """Wait for native work to finish before propagating caller cancellation.

    Keeping this await inside a limiter context retains the permit until the
    thread exits. Worker exceptions propagate normally, including after cancellation.
    """
    worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    cancelled = False
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            if worker.cancelled():
                raise
            cancelled = True
    result = worker.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


class DynamicConcurrencyLimiter:
    """Manages dynamic concurrency limits for asynchronous OCR tasks."""

    def __init__(self, initial_concurrency: int = 1) -> None:
        """Initialize with a maximum concurrency of at least one."""
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
        """Set capacity (at least one) and notify waiting tasks."""
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

    async def __aenter__(self) -> Self:
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
