__all__ = [
    "INFINITY",
    "grid_neighbors_gen",
    "grid_neighbors_values_gen",
    "neighbors4",
    "neighbors4x",
    "neighbors8",
    "neighbors4_values",
    "neighbors4x_values",
    "neighbors8_values",
    "neighbors4_coords",
    "neighbors4x_coords",
    "neighbors8_coords",
    "grid_find_adjacent",
    "graph_from_grid",
    "grid_bfs",
    "grid_bfs_lru",
    "bfs",
    "connected_components",
    "dijkstra",
    "dijkstra_lru",
    "dijkstra_path",
    "dijkstra_path_lru",
    "dijkstra_all",
    "dijkstra_all_paths",
    "bisection",
    "binary_search",
]

import heapq
from collections import deque, defaultdict
from functools import lru_cache
from bisect import bisect_left
from math import inf as INFINITY
from itertools import filterfalse
from operator import itemgetter
from typing import (
    TypeVar,
    Any,
    Iterable,
    Iterator,
    Callable,
    Union,
    Optional,
    Set,
    Generator,
)
from typing import List, Tuple, Dict, DefaultDict
from collections.abc import Sequence, Container

T = TypeVar("T")

Grid2D = Sequence[Sequence[Any]]
Coord2D = Tuple[int, int]
# Coord2D = Vector
WeightedGraphDict = Dict[T, List[Tuple[T, int]]]
UnweightedGraphDict = Dict[T, List[T]]
GraphDict = Union[WeightedGraphDict, UnweightedGraphDict]
GridNeighborsFunc = Callable[[Grid2D, int, int, Container], Iterator[Coord2D]]
GraphNeighborsFunc = Callable[[GraphDict, T], Iterator[T]]
Distance = Union[int, float]
IntOrFloat = Union[int, float]

# Maximum cache size used for memoization with lru_cache
MAX_CACHE_SIZE = 256 * 2**20  # 256Mi entries -> ~8GiB if one entry is 32 bytes


def grid_neighbors_gen(deltas: Iterable[Coord2D]) -> GridNeighborsFunc:
    """Create a generator function for finding coordinates of neighbors in a
    grid (2D matrix) given a list of deltas to apply to the source coordinates
    to get neighboring cells.
    """

    def g(grid: Grid2D, r: int, c: int, avoid: Container = ()) -> Iterator[Coord2D]:
        """Get neighbors of a cell in a 2D grid (matrix) i.e. list of lists or
        similar. Performs bounds checking. Grid is assumed to be rectangular.
        """
        maxr = len(grid) - 1
        maxc = len(grid[0]) - 1
        check = r == 0 or r == maxr or c == 0 or c == maxc

        if check:
            for dr, dc in deltas:
                rr, cc = (r + dr, c + dc)
                if 0 <= rr <= maxr and 0 <= cc <= maxc:
                    if grid[rr][cc] not in avoid:
                        yield (rr, cc)
        else:
            for dr, dc in deltas:
                rr, cc = (r + dr, c + dc)
                if grid[rr][cc] not in avoid:
                    yield (rr, cc)

    return g


def grid_neighbors_values_gen(
    deltas: Iterable[Coord2D],
) -> Callable[[Grid2D, int, int, Container], Iterator[Any]]:
    """Create a generator function for finding values of neighbors in a grid
    (2D matrix) given a list of deltas to apply to the source coordinates
    to get neighboring cells.
    """
    g = grid_neighbors_gen(deltas)

    def v(grid: Grid2D, r: int, c: int, avoid: Container = ()) -> Iterator[Any]:
        for rr, cc in g(grid, r, c, avoid):
            yield grid[rr][cc]

    return v


def neighbors_coords_gen(
    deltas: Iterable[Coord2D],
) -> Callable[[int, int], Iterator[Coord2D]]:
    """Create a generator function yielding coordinates of neighbors of a given
    cell in a hypotetical grid (2D matrix), given a list of deltas to apply to
    the source coordinates.

    Useful for cases where bound-checking is not needed, like sparse matrices.
    """

    def neighbor_coords(r: int, c: int) -> Iterator[Coord2D]:
        for dr, dc in deltas:
            yield (r + dr, c + dc)

    return neighbor_coords


neighbors4 = grid_neighbors_gen(((-1, 0), (0, -1), (0, 1), (1, 0)))
neighbors4x = grid_neighbors_gen(((-1, -1), (-1, 1), (1, -1), (1, 1)))
neighbors8 = grid_neighbors_gen(
    ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
)

neighbors4_values = grid_neighbors_values_gen(((-1, 0), (0, -1), (0, 1), (1, 0)))
neighbors4x_values = grid_neighbors_values_gen(((-1, -1), (-1, 1), (1, -1), (1, 1)))
neighbors8_values = grid_neighbors_values_gen(
    ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
)

neighbors4_coords = neighbors_coords_gen(((-1, 0), (0, -1), (0, 1), (1, 0)))
neighbors4x_coords = neighbors_coords_gen(((-1, -1), (-1, 1), (1, -1), (1, 1)))
neighbors8_coords = neighbors_coords_gen(
    ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
)


def grid_find_adjacent(
    grid: Grid2D,
    src: Coord2D,
    find: Container,
    avoid: Container = (),
    coords: bool = False,
    get_neighbors: GridNeighborsFunc = neighbors4,
) -> Generator[Union[Tuple[Tuple[int, int, Any], int], Tuple[Any, int]], None, None]:
    """Find and yield edges to reachable nodes in grid (2D matrix) from src,
    considering the values in find as nodes, and the values in avoid as walls.

    src: (row, col) to start searching from
    find: values to consider nodes
    avoid: values to consider walls
    anyting else is considered open space
    get_neighbors(grid, r, c, avoid) is called to determine cell neighbors

    If coords=False (default), yields edges in the form (node, dist), otherwise
    in the form ((row, col, node), dist). For a character grid node will be a
    single character.

    Uses breadth-first search.
    """
    visited = {src}
    queue: deque = deque()

    for n in neighbors4(grid, *src, avoid):
        queue.append((1, n))

    while queue:
        dist, node = queue.popleft()

        if node not in visited:
            visited.add(node)
            r, c = node

            if grid[r][c] in find:
                if coords:
                    yield ((r, c, grid[r][c]), dist)
                else:
                    yield (grid[r][c], dist)

                continue

            for neighbor in filterfalse(
                visited.__contains__, get_neighbors(grid, *node)
            ):
                queue.append((dist + 1, neighbor))


def graph_from_grid(
    grid: Grid2D,
    find: Container,
    avoid: Container = (),
    coords: bool = False,
    get_neighbors: GridNeighborsFunc = neighbors4,
) -> WeightedGraphDict:
    """Reduce a grid (2D matrix) to an undirected graph by finding all nodes and
    calculating their distance to others. Note: can return a disconnected graph.

    find: values to consider nodes of the graph
    avoid: values to consider walls
    anything else is considered open space
    get_neighbors(grid, r, c, avoid) is called to determine cell neighbors

    If coord=False (default), nodes of the graph will be represented by the
    found value only, otherwise by a tuple of the form (row, col, char).

    Returns a "graph dictionary" of the form {node: [(node, dist)]}.
    """
    graph: GraphDict = {}  # type: ignore

    for r, row in enumerate(grid):
        for c, char in enumerate(row):
            if char in find:
                node = (r, c, char) if coords else char
                graph[node] = list(
                    grid_find_adjacent(
                        grid, (r, c), find, avoid, coords, get_neighbors
                    )  # type: ignore
                )

    return graph


def grid_bfs(
    grid: Grid2D,
    src: Coord2D,
    dst: Coord2D,
    avoid: Container = (),
    get_neighbors: GridNeighborsFunc = neighbors4,
) -> Distance:
    """Find the length of any path from src to dst in grid using breadth-first
    search. Returns INFINITY if no path is found.

    grid is a 2D matrix i.e. list of lists or similar.
    src and dst are tuples in the form (row, col)
    get_neighbors(grid, r, c, avoid) is called to determine cell neighbors

    For memoization, use: bfs = bfs_grid_lru(grid); bfs(src, dst).
    """
    queue = deque([(0, src)])
    visited: set = set()

    while queue:
        dist, rc = queue.popleft()

        if rc == dst:
            return dist

        if rc not in visited:
            visited.add(rc)

            for n in filterfalse(visited.__contains__, get_neighbors(grid, *rc, avoid)):
                queue.append((dist + 1, n))

    return INFINITY


def grid_bfs_lru(
    grid: Grid2D, avoid: Container = (), get_neighbors: GridNeighborsFunc = neighbors4
) -> Callable[[Coord2D, Coord2D], Distance]:
    @lru_cache(MAX_CACHE_SIZE)
    def wrapper(src: Coord2D, dst: Coord2D) -> Distance:
        nonlocal grid, get_neighbors
        return grid_bfs(grid, src, dst, avoid, get_neighbors)

    return wrapper


def bfs(
    G: GraphDict,
    src: Any,
    weighted: bool = False,
    get_neighbors: Optional[GraphNeighborsFunc] = None,
) -> Set[Any]:
    """Find and return the set of all nodes reachable from src in G using
    breadth-first search.

    G is a "graph dictionary" of the form {src: [dst]} or {src: [(dst, weight)]}
    if weighted=True, in which case weights are ignored.

    get_neighbors(node) is called to determine node neighbors (default is G.get)

    NOTE: for correct results in case of an undirected graph, all nodes must be
    present in G as keys.
    """
    if get_neighbors is None:
        get_neighbors = G.get  # type: ignore

    queue = deque([src])
    visited: set = set()

    while queue:
        node = queue.popleft()
        if node in visited:
            continue

        visited.add(node)
        neighbors = get_neighbors(node) or ()  # type: ignore
        neighbors = map(itemgetter(0), neighbors) if weighted else neighbors
        queue.extend(filterfalse(visited.__contains__, neighbors))

    return visited


def connected_components(
    G: GraphDict,
    weighted: bool = False,
    get_neighbors: Optional[GraphNeighborsFunc] = None,
) -> List[Set[Any]]:
    """Find and return a list of all the connected components of G.

    G is a "graph dictionary" of the form {src: [dst]} or {src: [(dst, weight)]}
    if weighted=True, in which case weights are ignored.

    get_neighbors(node) is called to determine node neighbors (default is G.get)

    Returns a list of sets each representing the nodes of a connected component.

    NOTE: for correct results in case of an undirected graph, all nodes must be
    present in G as keys.
    """
    visited: set = set()
    components = []

    for node in filterfalse(visited.__contains__, G):
        component = bfs(G, node, weighted, get_neighbors)
        visited |= component
        components.append(component)

    return components


def dijkstra(
    G: WeightedGraphDict,
    src: Any,
    dst: Any,
    get_neighbors: Optional[GraphNeighborsFunc] = None,
) -> Distance:
    """Find the length of the shortest path from src to dst in G using
    Dijkstra's algorithm.

    G is a weighted "graph dictionary" of the form {src: [(dst, weight)]}.

    For memoization, use: djk = dijkstra_lru(G); djk(src, dst).
    """
    if get_neighbors is None:
        get_neighbors = G.get  # type: ignore

    distance = defaultdict(lambda: INFINITY, {src: 0})
    queue = [(0, src)]
    visited: set = set()

    while queue:
        dist, node = heapq.heappop(queue)

        if node == dst:
            return dist

        if node not in visited:
            visited.add(node)
            neighbors = get_neighbors(node)  # type: ignore

            if not neighbors:
                continue

            for neighbor, weight in filter(lambda n: n[0] not in visited, neighbors):
                new_dist = dist + weight

                if new_dist < distance[neighbor]:
                    distance[neighbor] = new_dist
                    heapq.heappush(queue, (new_dist, neighbor))

    return INFINITY


def dijkstra_lru(
    G: WeightedGraphDict, get_neighbors: Optional[GraphNeighborsFunc] = None
) -> Callable[[Any, Any], Distance]:
    """Memoized version of dijkstra(): djk = dijkstra_lru(G); djk(src, dst)."""

    @lru_cache(MAX_CACHE_SIZE)
    def wrapper(src: Any, dst: Any) -> Distance:
        nonlocal G, get_neighbors
        return dijkstra(G, src, dst, get_neighbors)

    return wrapper


def dijkstra_path(
    G: WeightedGraphDict,
    src: Any,
    dst: Any,
    get_neighbors: Optional[GraphNeighborsFunc] = None,
) -> Tuple[Tuple[Any], Distance]:
    """Find the shortest path from src to dst in G using Dijkstra's algorithm.

    G is a weighted "graph dictionary" of the form {src: [(dst, weight)]}
    get_neighbors(node) is called to determine node neighbors (default is G.get)

    Returns a tuple (shortest_path, length) where shortest_path is a tuple of
    the form (src, ..., dst). If no path is found, the result is ((), INFINITY).

    NOTE that the returned length is the length of the shortest path (i.e. sum
    of edge weights along the path), not the number of nodes in the path.

    For memoization, use: djk = dijkstra_path_lru(G); djk(src, dst).
    """
    if get_neighbors is None:
        get_neighbors = G.get  # type: ignore

    distance = defaultdict(lambda: INFINITY, {src: 0})
    previous = {src: None}
    queue = [(0, src)]
    visited: set = set()

    while queue:
        dist, node = heapq.heappop(queue)

        if node == dst:
            path = []

            while node is not None:
                path.append(node)
                node = previous[node]

            return tuple(reversed(path)), dist  # type: ignore

        if node not in visited:
            visited.add(node)
            neighbors = get_neighbors(node)  # type: ignore

            if not neighbors:
                continue

            for neighbor, weight in filter(lambda n: n[0] not in visited, neighbors):
                new_dist = dist + weight

                if new_dist < distance[neighbor]:
                    distance[neighbor] = new_dist
                    previous[neighbor] = node
                    heapq.heappush(queue, (new_dist, neighbor))

    return (), INFINITY  # type: ignore


def dijkstra_path_lru(
    G: WeightedGraphDict, get_neighbors: Optional[GraphNeighborsFunc] = None
) -> Callable[[Any, Any], Tuple[Tuple[Any], Distance]]:
    """Memoized version of dijkstra_path():
    djk = dijkstra_path_lru(G); djk(src, dst).
    """

    @lru_cache(MAX_CACHE_SIZE)
    def wrapper(src: Any, dst: Any) -> Tuple[Tuple[Any], Distance]:
        nonlocal G, get_neighbors
        return dijkstra_path(G, src, dst, get_neighbors)

    return wrapper


def dijkstra_all(
    G: WeightedGraphDict, src: Any, get_neighbors: Optional[GraphNeighborsFunc] = None
) -> DefaultDict[Any, Distance]:
    """Find the length of all the shortest paths from src to any reachable node
    in G using Dijkstra's algorithm.

    G is a weighted "graph dictionary" of the form {src: [(dst, weight)]}
    get_neighbors(node) is called to determine node neighbors (default is G.get)

    Reurns a defaultdict {node: distance}, where unreachable nodes have
    distance=INFINITY.
    """
    if get_neighbors is None:
        get_neighbors = G.get  # type: ignore

    distance = defaultdict(lambda: INFINITY, {src: 0})
    queue = [(0, src)]
    visited: set = set()

    while queue:
        dist, node = heapq.heappop(queue)

        if node not in visited:
            visited.add(node)
            neighbors = get_neighbors(node)  # type: ignore

            if not neighbors:
                continue

            for neighbor, weight in filter(lambda n: n[0] not in visited, neighbors):
                new_dist = dist + weight

                if new_dist < distance[neighbor]:
                    distance[neighbor] = new_dist
                    heapq.heappush(queue, (new_dist, neighbor))

    return distance


def dijkstra_all_paths(
    G: WeightedGraphDict, src: Any, get_neighbors: Optional[GraphNeighborsFunc] = None
) -> DefaultDict[Any, Tuple[Tuple[Any], Distance]]:
    """Find all the shortest paths from src to any reachable node in G and their
    using Dijkstra's algorithm.

    G is a "graph dictionary" of the form {src: [(dst, weight)]}
    get_neighbors(node) is called to determine node neighbors (default is G.get)

    Returns a defaultdict {node: (path, length)}, where unreachable nodes have
    path=() and length=INFINITY. NOTE that src is always present in the result
    with path=(src,) and length=0.
    """
    if get_neighbors is None:
        get_neighbors = G.get  # type: ignore

    pd = defaultdict(lambda: ((), INFINITY), {src: ((src,), 0)})
    queue = [(0, src, (src,))]
    visited: set = set()

    while queue:
        dist, node, path = heapq.heappop(queue)

        if node not in visited:
            visited.add(node)
            neighbors = get_neighbors(node)  # type: ignore

            if not neighbors:
                continue

            for neighbor, weight in filter(lambda n: n[0] not in visited, neighbors):
                new_dist = dist + weight

                if new_dist < pd[neighbor][1]:
                    new_path = path + (neighbor,)
                    pd[neighbor] = (new_path, new_dist)
                    # type: ignore
                    heapq.heappush(queue, (new_dist, neighbor, new_path))

    return pd


def bisection(
    fn: Callable[[IntOrFloat], IntOrFloat],
    y: IntOrFloat,
    lo: Optional[IntOrFloat] = None,
    hi: Optional[IntOrFloat] = None,
    tolerance: IntOrFloat = 1e-9,
    upper: bool = False,
) -> Optional[IntOrFloat]:
    """Find a value x in the range [lo, hi] such that fn(x) == y.

    NOTE: fn(x) must be a monotinically increasing function for this to work! In
    case f(x) is monotonically decreasing, -fn(x) can be provided. In case f(x)
    is not monotonous, this method cannot possibly work.

    If y is an int, find an exact match for x such that fn(x) == y; return None
    on failure.

    If y is a float, find a lower bound for x instead (or upper bound if
    upper=True) stopping when the size of the range of values to search gets
    below tolerance; in this case, the search cannot fail. It is up to the
    caller to supply meaningful values for lo and hi.

    If not supplied, lo and hi are found through exponential search.

    ```
                     * * *
                   * |
                 *   |
        y ------*    |
               *     |
             *       |
           *         |
        *  |         |
           lo        hi
    ```
    """
    if type(y) not in (int, float):
        raise TypeError("y must be int or float, got {}".format(type(y).__name__))

    if lo is not None and hi is not None and fn(lo) <= fn(hi):
        raise TypeError(
            "fn(x) must be a monotonically increasing function, but have fn(lo) > fn(hi)"
        )

    if lo is None:
        # Optimistic check
        if fn(0) <= y:
            lo = 0
        else:
            lo = -1
            while fn(lo) > y:
                lo *= 2

    if hi is None:
        hi = 1
        while fn(hi) < y:
            hi *= 2

    if type(y) is int:
        while lo <= hi:
            x = (lo + hi) // 2
            v = fn(x)

            if v > y:
                hi = x - 1
            elif v < y:
                lo = x + 1
            else:
                return x

        return None

    # y is float
    if upper:
        while hi - lo >= tolerance:
            x = (lo + hi) / 2

            if fn(x) < y:
                lo = x
            else:
                hi = x

        return hi

    while hi - lo >= tolerance:
        x = (lo + hi) / 2

        if fn(x) > y:
            hi = x
        else:
            lo = x

    return lo


def binary_search(seq: Sequence[Any], x: Any) -> int:
    """Find the index of x in the sorted sequence seq using binary search.
    Returns -1 if x not in seq.

    NOTE: seq must be indexable.
    """
    # https://docs.python.org/3/library/bisect.html#searching-sorted-lists
    i = bisect_left(seq, x)
    if i != len(seq) and seq[i] == x:
        return i
    return -1
