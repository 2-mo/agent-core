"""Bounded standard-library workers for independent tasks and tools."""

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing
from itertools import islice


def iter_bounded_map[T, R](
    func: Callable[[T], R], items: Iterable[T], concurrency: int,
) -> Iterator[tuple[int, R]]:
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    indexed = iter(enumerate(items))
    pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="agent-core")
    pending = {}
    try:
        for index, item in islice(indexed, concurrency):
            pending[pool.submit(func, item)] = index
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                index = pending.pop(future)
                yield index, future.result()
                for next_index, item in islice(indexed, 1):
                    pending[pool.submit(func, item)] = next_index
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def bounded_map(func, items, concurrency, *, on_result=None):
    answers = {}
    with closing(iter_bounded_map(func, items, concurrency)) as completed:
        for index, result in completed:
            answers[index] = result
            if on_result is not None:
                on_result(index, result)
    return [answers[index] for index in range(len(answers))]
