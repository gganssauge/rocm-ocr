"""Unit tests for the dynamic concurrency limiter."""

import asyncio
import threading

import pytest

from rocm_ocr.concurrency import DynamicConcurrencyLimiter, run_in_thread


@pytest.mark.asyncio
async def test_concurrency_limiter_initialization() -> None:
    """Test valid and invalid limiter initializations."""
    limiter = DynamicConcurrencyLimiter(initial_concurrency=2)
    assert limiter.capacity == 2
    assert limiter.active_tasks == 0

    with pytest.raises(ValueError, match="Concurrency must be greater"):
        DynamicConcurrencyLimiter(initial_concurrency=0)


@pytest.mark.asyncio
async def test_concurrency_limiter_context_manager() -> None:
    """Test acquiring and releasing via async context manager."""
    limiter = DynamicConcurrencyLimiter(initial_concurrency=1)
    assert limiter.active_tasks == 0

    async with limiter:
        assert limiter.active_tasks == 1

    assert limiter.active_tasks == 0


@pytest.mark.asyncio
async def test_concurrency_limiter_update() -> None:
    """Test dynamically adjusting the concurrency capacity."""
    limiter = DynamicConcurrencyLimiter(initial_concurrency=1)
    await limiter.update_concurrency(4)
    assert limiter.capacity == 4

    with pytest.raises(ValueError, match="Concurrency must be greater"):
        await limiter.update_concurrency(0)


@pytest.mark.asyncio
async def test_concurrency_limiter_blocks_excess_tasks() -> None:
    """Verify that extra tasks wait until slots become available."""
    limiter = DynamicConcurrencyLimiter(initial_concurrency=1)
    execution_order: list[str] = []

    async def task(name: str, delay: float) -> None:
        async with limiter:
            execution_order.append(f"{name}_start")
            await asyncio.sleep(delay)
            execution_order.append(f"{name}_end")

    await asyncio.gather(task("t1", 0.05), task("t2", 0.01))

    assert execution_order == ["t1_start", "t1_end", "t2_start", "t2_end"]


async def test_worker_failure_after_cancellation_is_observed() -> None:
    """Do not lose a native worker exception while draining an aborted request."""
    entered = threading.Event()
    release = threading.Event()
    limiter = DynamicConcurrencyLimiter(1)

    def failing_worker() -> None:
        entered.set()
        release.wait(timeout=2)
        raise RuntimeError("native worker failure")

    async def request() -> None:
        async with limiter:
            await run_in_thread(failing_worker)

    task = asyncio.create_task(request())
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        assert limiter.active_tasks == 1
    finally:
        release.set()
        with pytest.raises(RuntimeError, match="native worker failure"):
            await task
    assert limiter.active_tasks == 0
