# -*- coding: utf-8 -*-
"""
Collection of datatypes and tools used to implement grid-based path planning
"""
from __future__ import print_function

import math
import heapq

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Empty(Exception):
    """A simple extension to signal an empty heap queue"""


class HeapQueue(object):
    """Heap-based implementation of a priority queue

    This class is a wrapper around Python's heapq module.

    Methods
    -------
        put(item)
            Put an item in the priority queue
        get()
            Pop and return the top-most element of the priority queue
        peak()
            Look at the top-most element without popping it
        empty()
            Check if the priority queue is empty
    """

    def __init__(self):
        self._heap = []

    def put(self, item):
        """Put item in the priority queue"""
        heapq.heappush(self._heap, item)

    def get(self):
        """Pop and return the top-most element of the priority queue"""
        try:
            return heapq.heappop(self._heap)
        except IndexError:
            # pylint: disable=raise-missing-from
            raise Empty("Priority queue is empty!")

    def peak(self):
        """Look at the top-most element without popping it"""
        return self._heap[0]

    def empty(self):
        """Check if the priority queue is empty"""
        return len(self._heap) == 0


def prepare_map(my_img):
    """Apply threshold to map.

    Swappes axes and applies threshold.

    Parameters
    ----------
    my_img : np.array
        Image of a Map were each cell is either 0 (free) or 1 (occupied).

    Returns
    -------
    np.array
        Array like my_img with threshold applied.

    Notes
    -----
    The IPA map is mapped in a false color schema, so the colors need to be
    inverted!
    """
    swapped_img = np.swapaxes(np.flipud(np.swapaxes(np.flipud(my_img), 0, 1)),
                              0, 1)
    swapped_img = swapped_img.copy()
    swapped_img[swapped_img < 254] = 1
    swapped_img[swapped_img >= 254] = 0
    return swapped_img


class Vertex(object):
    """The container class for a vertex.

    Attributes
    ----------
    x, y : int
        Coordinates.

    Parameters
    ----------
    x, y : int
        Coordinates.

    Examples
    --------
    >>> v = Vertex(1, 2)
    >>> print("x = {}, y = {}".format(v.x, v.y))
    x = 1, y = 2
    >>> print(v)
    Vertex(1, 2)

    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "Vertex({}, {})".format(self.x, self.y)

    def __repr__(self):
        return str(self)

    def get_coordinates(self):
        """Convenience Function.

        Examples
        --------
        >>> v = Vertex(1, 2)
        >>> x, y = v.get_coordinates()
        >>> print("x = {}, y = {}".format(x, y))
        x = 1, y = 2
        """
        return self.x, self.y

    def __eq__(self, rhs):
        """Overriding equality operator.

        Returns
        -------
        bool
            True when equal, False when not.

        Examples
        --------
        Equal:

        >>> v1 = Vertex(1, 1)
        >>> v2 = Vertex(1, 1)
        >>> v1 == v2
        True

        Not equal:

        >>> v1 = Vertex(1, 1)
        >>> v2 = Vertex(1, 2)
        >>> v3 = Vertex(2, 1)
        >>> v1 == v2
        False
        >>> v1 == v3
        False
        >>> v2 == v3
        False
        """
        if self.x == rhs.x and self.y == rhs.y:
            return True
        return False

    def __ne__(self, rhs):
        # pylint: disable=superfluous-parens
        # actually == has precedence over not, but it looks clearer with the
        # parens around the subexpression
        # see also https://docs.python.org/3/reference/expressions.html#operator-precedence
        return not (self == rhs)


class WeightedDirection(object):
    """Used to manage neighborhood

    Attributes
    ----------
    dx : int
        Difference (delta) in x direction.
    dy : int
        Difference (delta) in y direction.
    w : float
        Weight (cost) of this edge.
    n : str
        Name of that particular direction.

    Parameters
    ----------
    dx : int
        Difference (delta) in x direction.
    dy : int
        Difference (delta) in y direction.
    weight : float
        Weight (cost) of this edge.
    name : str, optional
        Name of that particular direction.

    Examples
    --------
    Create movement to the right.

    >>> w = WeightedDirection(1,0,1,'right')
    >>> print("w.n: {}, w.dx: {}, w.dy: {}, w.w: {}".format(w.n, w.dx, w.dy, w.w))
    w.n: right, w.dx: 1, w.dy: 0, w.w: 1
    >>> print(w)
    Direction.right(x:1, y:0, w:1)
    """
    def __init__(self, dx=0, dy=0, weight=0, name=None):
        self.dx = dx
        self.dy = dy
        self.w = weight
        self.n = name

    def __str__(self):
        return "Direction.{}(x:{}, y:{}, w:{})".format(self.n, self.dx,
                                                       self.dy, self.w)

    def __repr__(self):
        return str(self)


def heuristic(position, goal, mode='dijkstra'):
    """Computes the euclidean distance between two vertices.

    Parameters
    ----------
    position : Vertex
        The position from which the heuristic cost is computed.
    goal : Vertex
        The position we want to reach.
    mode : str
        Determines whether the function returns 0 (dijkstra) or the heuristic
        cost (astar).

    Returns
    -------
    float
        Value of the heuristic between two vertices.

    Examples
    --------

    >>> position = Vertex(1, 1)
    >>> goal = Vertex(3, 3)
    >>> print(heuristic(position, goal, 'dijkstra'))
    0
    >>> print(heuristic(position, goal, 'astar'))
    2.82842712475

    """
    result = 0
    if mode == 'astar':
        result = math.hypot(position.x - goal.x, position.y - goal.y)
    return result


def get_neighborhood(n=8):
    """Creates a list of directions for searching or new neighbors.

    Parameters
    ----------
    n : int, optional (default: 8)
        Number of neighbors (only 4 and 8 supported)

    Returns
    -------
    List of WeightedDirection
        Returns a list with either a four or eight direction in a neighborhood.

    Examples
    --------
    >>> four_neighborhood = get_neighborhood(4)
    >>> for n in four_neighborhood:
    ...     print(n)
    ...
    Direction.down(x:0, y:-1, w:1)
    Direction.left(x:-1, y:0, w:1)
    Direction.right(x:1, y:0, w:1)
    Direction.up(x:0, y:1, w:1)

    >>> eight_neighborhood = get_neighborhood(8)
    >>> for n in eight_neighborhood:
    ...     print(n)
    ...
    Direction.lower_left(x:-1, y:-1, w:1.41421356237)
    Direction.lower(x:0, y:-1, w:1)
    Direction.lower_right(x:1, y:-1, w:1.41421356237)
    Direction.left(x:-1, y:0, w:1)
    Direction.right(x:1, y:0, w:1)
    Direction.upper_left(x:-1, y:1, w:1.41421356237)
    Direction.upper(x:0, y:1, w:1)
    Direction.upper_right(x:1, y:1, w:1.41421356237)
    """
    neighbors = []
    one = 1
    sqrt_two = math.sqrt(2)
    if n == 8:
        neighbors = [WeightedDirection(-1, -1, sqrt_two, 'lower_left'),
                     WeightedDirection(0, -1, one, 'lower'),
                     WeightedDirection(1, -1, sqrt_two, 'lower_right'),
                     WeightedDirection(-1, 0, one, 'left'),
                     WeightedDirection(1, 0, one, 'right'),
                     WeightedDirection(-1, 1, sqrt_two, 'upper_left'),
                     WeightedDirection(0, 1, one, 'upper'),
                     WeightedDirection(1, 1, sqrt_two, 'upper_right')]
    else:
        neighbors = [WeightedDirection(0, -1, one, 'down'),
                     WeightedDirection(-1, 0, one, 'left'),
                     WeightedDirection(1, 0, one, 'right'),
                     WeightedDirection(0, 1, one, 'up')]
    return neighbors


class PredecessorMatrix(object):
    """This object is a container for a predecessor matrix.

    Parameters
    ----------
    nav_map : np.array
        The size at which the predecessor matrix is initialized.

    Attributes
    ----------
    p_map : np.array of Vertex
        The matrix where each cell corresponds to an existing Vertex and  the
        value of each cell is the predecessor vertex. Vertices without
        predecessor have None as value (the start vertex also has None as
        predecessor!)

    Methods
    -------
    :meth:`~PredecessorMatrix.insert`
        Inserts an Predecessor v at the position of s.
    :meth:`~PredecessorMatrix.get_predecessor`
        Returns the predecessor vertex at the position of v.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[0, 0], [0, 0]])
    >>> pm = PredecessorMatrix(a)
    >>> print(pm.p_map)
    [[None None]
     [None None]]
    """
    def __init__(self, nav_map):
        self.p_map = np.zeros_like(nav_map, dtype=Vertex)
        self.p_map.fill(None)

    def insert(self, successor, vertex):
        """Writes the vertex at the position of successor in the matrix.

        Parameters
        ----------
        successor : Vertex
            The vertex at whose position the value is updated.
        vertex : Vertex
            The predecessor vertex.

        Examples
        --------
        Add an entry to the table

        >>> import numpy as np
        >>> a = np.array([[0, 0], [0, 0]])
        >>> pm = PredecessorMatrix(a)
        >>> print(pm.p_map)
        [[None None]
         [None None]]
        >>> v0 = Vertex(0, 0)
        >>> v1 = Vertex(1, 0)
        >>> pm.insert(v1, v0)
        >>> print(pm.p_map)
        [[None None]
         [Vertex(0, 0) None]]
        >>> print(pm.p_map[v1.x, v1.y])
        Vertex(0, 0)
        """
        x, y = successor.get_coordinates()
        self.p_map[x, y] = vertex

    def get_predecessor(self, vertex):
        """Returns the predecessor of the given vertex from the table

        Parameters
        ----------
        vertex : Vertex
            The vertex object whose position will be used for table lookup.

        Returns
        -------
        Vertex
            The predecessor vertex.

        Examples
        --------
        Retrieve a predecessor

        >>> import numpy as np
        >>> a = np.array([[0, 0], [0, 0]])
        >>> pm = PredecessorMatrix(a)
        >>> v0 = Vertex(0, 0)
        >>> v1 = Vertex(1, 0)
        >>> pm.insert(v1, v0)
        >>> print(pm.get_predecessor(v1))
        Vertex(0, 0)
        """
        x, y = vertex.get_coordinates()
        return self.p_map[x, y]


def plot_map_debug(nav_map, cost_map, combined_cost_map, start=None, goal=None,
                   path=None, pathmatrix=None, mode=None):
    """This will plot the same visualization as plot_map but you can add
    the predecessor matrix to draw all current graphs (this computation is
    rather expensive)

    Parameters
    ----------
    nav_map : np.array
        Map in which our path planning takes place.
    cost_map : np_array
        Layer for the nav_map in which the the cost of the distance from
        start to goal are mapped.
    combined_cost_map : np_array
        Layer for the nav_map in which the costs from distance and heuristic
        are added.
    start : Vertex, optional
        Position from which the planning began.
    goal : Vertex, optional
        Position were the planning ends.
    path : list of Vertex, optional
        The Representation of the Path as sequence of vertices.
    pathmatrix : np.array
        The matrix which contains vertices.

    """
    plt.clf()
    plot_map(nav_map, cost_map, combined_cost_map, start, goal, path, mode)

    if pathmatrix is not None:
        pathmatrix = pathmatrix.p_map
        x_array = []
        y_array = []
        for x in range(pathmatrix.shape[0]):
            for y in range(pathmatrix.shape[1]):
                if pathmatrix[x, y] is not None:
                    pre = pathmatrix[x, y]
                    x_array.extend([x, pre.x, None])
                    y_array.extend([y, pre.y, None])
        left_plot = plt.gcf().get_axes()[0]
        left_plot.plot(x_array, y_array, 'b', mew=1)

    plt.show(block=True)


def plot_map(nav_map, cost_map, combined_cost_map, start=None, goal=None,
             path=None, mode=None):
    """This will plot two visualizations into one window.

    On the left side it will show the map, robot position (circle) and goal
    (cross) as well as the content form the cost map.
    On the right side it will show the map, robot position (circle) and goal
    (cross) as well as the content form the combined cost map.

    Parameters
    ----------
    nav_map : np.array
        Map in which our path planning takes place.
    cost_map : np_array
        Layer for the nav_map in which the the cost of the distance from
        start to goal are mapped.
    combined_cost_map : np_array
        Layer for the nav_map in which the costs from distance and heuristic
        are added.
    start : Vertex, optional
        Position from which the planning began.
    goal : Vertex, optional
        Position were the planning ends.
    path : list of Vertex, optional
        representation of the Path as sequence of vertices.
    """
    if mode is not None:
        if mode == 'astar':
            plt.suptitle("A* Algorithm")
        else:
            plt.suptitle("Dijkstra Algorithm")

    fig = plt.gcf()
    if not fig.get_axes():
        left_plot = fig.add_subplot(1, 2, 1)
        right_plot = fig.add_subplot(1, 2, 2)
    else:
        left_plot, right_plot = fig.get_axes()[:2]

    if left_plot.images:
        min_, max_ = np.min(cost_map), np.max(cost_map)
        left_im = left_plot.images[1]
        left_im.set_data(np.swapaxes(cost_map, 0, 1))
        left_im.set_clim(min_, max_)
        # update colorbar
        left_cbar = left_im.colorbar
        left_cbar.set_ticks([min_, (min_ + max_)/2, max_], update_ticks=False)
        left_cbar.set_ticklabels(['Low', 'Medium', 'High'])
    else:
        left_plot.set_title("Path Cost (cell to goal)")
        left_plot.imshow(np.swapaxes(nav_map, 0, 1),
                         interpolation='none', cmap='gray', origin='lower')
        left_im = left_plot.imshow(np.swapaxes(cost_map, 0, 1), alpha=0.7,
                                   interpolation='none', cmap='gist_heat_r',
                                   origin='lower')
        left_plot.axis('off')
        left_divider = make_axes_locatable(left_plot)
        left_cax = left_divider.append_axes("right", size="5%", pad=0.05)
        min_, max_ = np.min(cost_map), np.max(cost_map)
        left_im.set_clim(min_, max_)
        left_cbar = fig.colorbar(left_im, cax=left_cax,
                                 ticks=[min_, (min_ + max_) / 2, max_])
        left_cbar.ax.set_yticklabels(['Low', 'Medium', 'High'])
        celloffset = 0.5
        left_plot.set_xlim(-celloffset, nav_map.shape[0] - celloffset)
        left_plot.set_ylim(-celloffset, nav_map.shape[1] - celloffset)

    if right_plot.images:
        min_, max_ = np.min(combined_cost_map), np.max(combined_cost_map)
        right_im = right_plot.images[1]
        right_im.set_data(np.swapaxes(combined_cost_map, 0, 1))
        right_im.set_clim(min_, max_)
        # update colorbar
        right_cbar = right_im.colorbar
        right_cbar.set_ticks([min_, (min_ + max_) / 2, max_], update_ticks=False)
        right_cbar.set_ticklabels(['Low', 'Medium', 'High'])
    else:
        right_plot.set_title("Combined Cost \n"
                             + "path cost (cell to goal) + "
                             + "heuristic cost (cell to robot)")
        right_plot.imshow(np.swapaxes(nav_map, 0, 1),
                          interpolation='none', cmap='gray', origin='lower')
        right_im = right_plot.imshow(np.swapaxes(combined_cost_map, 0, 1),
                                     alpha=0.7, interpolation='none',
                                     cmap='gist_heat_r', origin='lower')
        right_plot.axis('off')
        right_divider = make_axes_locatable(right_plot)
        right_cax = right_divider.append_axes("right", size="5%", pad=0.05)
        min_, max_ = np.min(combined_cost_map), np.max(combined_cost_map)
        right_im.set_clim(min_, max_)
        right_cbar = fig.colorbar(right_im, cax=right_cax,
                                  ticks=[min_, (min_ + max_) / 2, max_])
        right_cbar.ax.set_yticklabels(['Low', 'Medium', 'High'])
        celloffset = 0.5
        right_plot.set_xlim(-celloffset, nav_map.shape[0] - celloffset)
        right_plot.set_ylim(-celloffset, nav_map.shape[1] - celloffset)

    if start is not None:
        if not left_plot.lines:
            left_plot.plot(start.x, start.y, 'bx', mew=3, label="Goal Position")
        if not right_plot.lines:
            right_plot.plot(start.x, start.y, 'bx', mew=3, label="Goal Position")
    if goal is not None:
        if len(left_plot.lines) == 1:
            left_plot.plot(goal.x, goal.y, 'b.', mew=3, label="Robot Position")
        if len(right_plot.lines) == 1:
            right_plot.plot(goal.x, goal.y, 'b.', mew=3, label="Robot Position")
    if path is not None:
        left_plot.plot([el.x for el in path], [el.y for el in path], 'b', mew=3)
        right_plot.plot([el.x for el in path], [el.y for el in path], 'b', mew=3)

    left_plot.legend()
    right_plot.legend()
    # plt.tight_layout()
    plt.pause(0.01)
