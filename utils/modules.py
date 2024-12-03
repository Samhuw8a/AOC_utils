import re
import io
import sys
import heapq
import string

from importlib import find_loader
from copy import deepcopy
from functools import lru_cache, reduce, partial
from collections import defaultdict, deque, namedtuple, Counter
from operator import itemgetter, attrgetter, methodcaller
from itertools import product, permutations, combinations, filterfalse, starmap, count
from math import pi as PI, inf as INFINITY
from math import floor, ceil, sqrt, sin, cos, tan, asin, acos, atan, factorial
from enum import Enum, auto as enum_auto

from .polyfill import *  # type: ignore

if find_loader("z3") is not None:
    import z3  # type: ignore

if find_loader("blist") is not None:
    try:
        import blist  # type: ignore
    except ImportError:
        pass

if find_loader("sortedcontainers") is not None:
    import sortedcontainers  # type: ignore


if find_loader("more_itertools") is not None:
    try:
        from more_itertools import first, chunked, exactly_n, padded, interleave, flatten  # type: ignore
        from more_itertools import divide as idvide  # type:ignore
    except ImportError:
        pass

if find_loader("numpy") is not None:
    try:
        import numpy as np  # type: ignore
    except ImportError:
        pass

if find_loader("networkx") is not None:
    import networkx as nx  # type: ignore
