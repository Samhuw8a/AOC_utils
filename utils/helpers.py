__all__ = [
    "autorange",
    "bin2int",
    "count_if",
    "cat",
    "digits",
    "DIGITS",
    "filterl",
    "integers",
    "mapl",
    "mapt",
    "parse_multiline_string",
    "read_input",
    "read_input_line",
    "sign",
    "transpose",
    "wait",
]

import sys
import re
from typing import (
    Collection,
    Type,
    TypeVar,
    Callable,
    Union,
    Sequence,
    List,
    Iterable,
    Any,
    Optional,
)
from typing import AnyStr
from itertools import chain


T = TypeVar("T")
D = TypeVar("D")
DIGITS: tuple = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
cat = "".join


def transpose(matrix: list) -> list:
    return list(zip(*matrix))


def maxval(d: dict) -> Any:
    return max(d.values())


def mapl(f: Callable, iterable: Iterable) -> list:
    return list(map(f, iterable))


def mapt(f: Callable, iterable: Iterable) -> tuple:
    return tuple(map(f, iterable))


def filterl(f: Callable, iterable: Iterable) -> list:
    return list(filter(f, iterable))


def parse_multiline_string(s: str, datatype: Type = str, sep: str = "\n") -> List:
    return mapl(datatype, s.split(sep))


def read_input(
    filename: Union[str, int], datatype: Type = str, sep: str = "\n"
) -> List[str]:
    filename = f"{filename:02d}" if isinstance(filename, int) else filename
    with open(f"inputs/{filename}.txt") as f:
        return parse_multiline_string(f.read(), datatype, sep)


def read_input_line(filename: Union[str, int], sep: str = "") -> Sequence[str]:
    filename = f"{filename:02d}" if isinstance(filename, int) else filename
    with open(f"inputs/{filename}.txt") as f:
        contents = f.read().strip()
        return contents if not sep else contents.split(sep)


def wait(msg: str = "Press [ENTER] to continue...") -> None:
    """Wait for user interaction by printing a message to standard error and
    waiting for input.
    """
    print(msg, end=" ")

    try:
        input()
    except KeyboardInterrupt:
        print(" keyboard interrupt, exiting...\n")
        sys.exit(0)


def digits(line: str) -> list:
    return mapl(int, line)


def integers(
    str_or_bytes: AnyStr, container: Type[Collection] = list, negatives: bool = True
) -> Collection[int]:
    """Extract integers within a string or a bytes object using a regular
    expression and return a list of int.

    With negative=False, discard any leading hypen, effectively extracting
    positive integer values even when perceded by hypens.

    The container= class is instantiated to hold the ints.
    """
    exp = r"-?\d+" if negatives else r"\d+"
    if isinstance(str_or_bytes, bytes):
        exp = exp.encode()  # type: ignore
    return container(map(int, re.findall(exp, str_or_bytes)))  # type: ignore


def count_if(iterable: Iterable, pred: Callable[[Any], bool] = bool) -> int:
    return sum(1 for item in iterable if pred(item))


def sign(n: int) -> int:
    if n > 0:
        return 1
    elif n < 0:
        return -1
    else:
        return 0


def bin2int(s: Any) -> int:
    return int(s, 2)


def autorange(start: int, end: int, step: int = 1) -> range:
    """Range from start to end (end is INCLUDED) in steps of +/- step regardless
    if start > end or end > start.

    autorange(1, 3) -> 1, 2, 3
    autorange(3, 1) -> 3, 2, 1
    autorange(10, 1, 2) -> 10, 8, 6, 4, 2
    """
    if start > end:
        return range(start, end - 1, -step)
    return range(start, end + 1, step)
