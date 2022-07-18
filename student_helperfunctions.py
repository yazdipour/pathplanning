# -*- coding: utf-8 -*-
""" DIJKSTRA/A* PSEUDOCODE WITH PRIORITY QUEUE
Line
 0  INITIALIZE:
 1  for all edges (i, j) in E (Edges) where i, j in V (Vertices):
 2     if direct neighbor:
 3        weight(i, j) = 1
 4     elif diagonal:
 5        weight(i, j) = sqrt(2)
 6  for all v in V:
 7     d(v) = infinity  # path_distance
 8     b(v) = NULL  # predecessor
 9  d(v0) = 0  # where v0 in V is start vertex
10  Q = v0  # v0 is in the queue
11
12
13  while not Q.empty() or not found:
14      u = Q.get()  # Q = Q without {u}
15      for all v in neighborhood(u):
16          if d(u) + w(u, v) < d(v):
17            d(v) = d(u) + w(u, v)
18            b(v) = u
19            Q = Q.put(v)
20
"""
from __future__ import print_function

from resources.pathplanning_utils import heuristic, Vertex


def dequeue_next(queue, goal):
    """
    Matches Line 14 of the Algorithm.

    As we use a priority queue with cost as primary key, we know that the first
    element in the queue is the one we are looking for. The extraction is also
    the deletion of the first element from the queue.

    Parameters
    ----------
    queue : PriorityQueue of (cost, number, vertex)
        The queue that holds all vertices that still need processing.
    goal : Vertex
        The goal vertex we want to reach.

    Returns
    -------
    next_vertex : Vertex (None if queue is empty)
        The next vertex in the priority queue.
    resign : bool {True, False}
        True if the queue is empty. Goal can't be reached.
    found : bool {True, False}
        True if the new vertex is equal to the goal vertex.

    Examples
    --------
    Queue was not empty and goal has not been found.

    >>> from resources.pathplanning_utils import HeapQueue
    >>> q = HeapQueue()
    >>> goal = Vertex(1, 1)
    >>> start = Vertex(0, 0)
    >>> q.put((0, 0, start))
    >>> a, b, c = dequeue_next(q, goal)
    >>> print("next: {}, resign = {}, found = {}".format(a, b, c))
    next: Vertex(0, 0), resign = False, found = False

    Queue was not empty and goal has been found

    >>> from resources.pathplanning_utils import HeapQueue
    >>> q = HeapQueue()
    >>> goal = Vertex(1, 1)
    >>> q.put((1, 9, goal))
    >>> a, b, c = dequeue_next(q, goal)
    >>> print("next: {}, resign = {}, found = {}".format(a, b, c))
    next: Vertex(1, 1), resign = False, found = True

    Queue is empty and we have to resign (goal can't be found):

    >>> from resources.pathplanning_utils import HeapQueue
    >>> q = HeapQueue()
    >>> goal = Vertex(1, 1)
    >>> a, b, c = dequeue_next(q, goal)
    >>> print("next: {}, resign = {}, found = {}".format(a, b, c))
    next: None, resign = True, found = False
    """
    #   HeapQueue (see utils for more information):
    #   Hint:   Tuple tup in the queue is (cost, n, Vertex)
    #   Hint:   Provides methods empty(), put(tup), tup = get()
    #
    #   Vertex (see utils for more information):
    #   Hint:   Provides variables x and y
    #   Hint:   You can check equality of vertices with "=="
    #
    #
    next_vertex = None
    resign = False
    found = False
    #   (i)     If queue is empty, we have to resign.
    empty_queue = queue.empty()  # TODO: add your code here

    if empty_queue:
        resign = True  # TODO: add your code here
    else:
        #   (ii)    Get the next vertex from the priority queue.
        #           Hint:   Returns tuple (see hints above).

        _, _, next_vertex = queue.get()  # TODO: add your code here

        #   (iii)   Check if the new vertex is equal to the goal vertex, set
        #           found accordingly.
        found = (next_vertex == goal)  # TODO: add your code here

    return next_vertex, resign, found


def next_successor(current, direction, cost_map, goal, mode):
    """This functions adds the direction to the current node and computes the
    path cost and combined (path + heuristic) cost.

    Parameters
    ----------
    current : Vertex
        The current vertex from the queue from which we create the next
        successor.
    direction : WeightedDirection
        The current direction from the N-neighborhood which creates the
        successor when applied.
    cost_map : numpy.array
        The current state of the map that hold the path cost from the cell
        to the start vertex.
    goal : Vertex
        The Vertex we want to reach.
    mode : string
        A string determining if A* or dijkstra is used when calling the
        heuristic function.

    Returns
    -------
    successor : Vertex
        A new vertex (potential successor).
    path_cost : float
        The path cost from the successor vertex to the start vertex.
    combined_cost : float
        The sum of path cost and heuristic value.

    Examples
    --------

    >>> from resources.pathplanning_utils import WeightedDirection
    >>> import numpy as np
    >>> import sys
    >>> mx = sys.float_info.max
    >>> pred = Vertex(0, 0)
    >>> tr = WeightedDirection(1, 1, 1.4, "upper_right")
    >>> c_map = np.array([[0, 1, mx], [1, mx, mx], [mx, mx, mx]])
    >>> print(c_map)  # top left corner is (0, 0)
    [[  0.00000000e+000   1.00000000e+000   1.79769313e+308]
     [  1.00000000e+000   1.79769313e+308   1.79769313e+308]
     [  1.79769313e+308   1.79769313e+308   1.79769313e+308]]
    >>> goal = Vertex(2, 2)
    >>> a, b, c = next_successor(pred, tr, c_map, goal, "astar")
    >>> print("successor: {}, path_cost: {}, combined_cost: {}".format(a, b, c))
    successor: Vertex(1, 1), path_cost: 1.4, combined_cost: 4.22842712475
    >>> a, b, c = next_successor(pred, tr, c_map, goal, "dijkstra")
    >>> print("successor: {}, path_cost: {}, combined_cost: {}".format(a, b, c))
    successor: Vertex(1, 1), path_cost: 1.4, combined_cost: 1.4
    """
    #   Vertex (see utils for more information):
    #   Hint:   Provides variables x and y
    #   Hint:   You can check equality of vertices with "=="
    #
    #   WeightedDirection (see utils for more information):
    #   Hint:   Provides variables dx, dy and cost. dx and dy are the delta
    #           values in their respective axis and cost of that direction.
    #           dx, dy element of {-1, 0, 1}
    #           cost element of {1, sqrt(2)}
    #
    #   heuristic(start, goal, mode) (see resources.pathplanning_utils
    #   for more information)

    #   (v)     Apply the movement and add the cost from the direction
    #           element to the current vertex from step. You
    #           will get a new vertex (successor) and its path cost to
    #           to the start position. Then compute the combined cost
    #           (combined_cost) by adding the path cost and the
    #           heuristic cost (from this position to goal).
    #           Hint:   The heuristic function is provided! See just above.
    # TODO: add your code here

    successor = Vertex(current.x + direction.dx, current.y + direction.dy)
    path_cost = cost_map[current.x][current.y] + direction.w
    combined_cost = path_cost + heuristic(current, goal, mode)
    return successor, path_cost, combined_cost


def validate_successor(successor, nav_map):
    """
    This function checks whether the successor is valid (e.g. not out of range
    or anything like that).

    Parameters
    ----------
    successor : Vertex
        The vertex that needs a validation check.
    nav_map : numpy.array
        The map in which the successor must be valid.

    Returns
    -------
    is_valid : bool
        True if the successor is valid within the map.

    Examples
    --------
    >>> import numpy as np
    >>> nav_map = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]])
    >>> print(nav_map)  # top left corner is (0, 0)
    [[0 0 1]
     [0 0 1]
     [1 1 1]]
    >>> print("valid: {}".format(validate_successor(Vertex(0, 0), nav_map)))
    valid: True
    >>> print("valid: {}".format(validate_successor(Vertex(2, 2), nav_map)))
    valid: False
    >>> print("valid: {}".format(validate_successor(Vertex(-1, 0), nav_map)))
    valid: False
    >>> print("valid: {}".format(validate_successor(Vertex(3, 0), nav_map)))
    valid: False
    >>> print("valid: {}".format(validate_successor(Vertex(0, -1), nav_map)))
    valid: False
    >>> print("valid: {}".format(validate_successor(Vertex(0, 3), nav_map)))
    valid: False
    """
    #   Vertex (see utils for more information):
    #   Hint:   Provides variables x and y
    #   Hint:   You can check equality of vertices with "=="
    #
    #   (vi)    Verify that the successor vertex is valid by:
    #           (vi-i)  Checking its coordinates with the dimensions of
    #                   the nav_map.
    #           (vi-ii) Then verify that the nav_map at the coordinates
    #                   of the successor is 0 (=free)
    #           Hint:   use the "shape" attribute of the map. Returns
    #                   (x_size, y_size)
    # TODO: add your code here

    is_valid = (successor.x >= 0 and successor.x < nav_map.shape[0]) and \
        (successor.y >= 0 and successor.y < nav_map.shape[1])
    return is_valid and (nav_map[successor.x][successor.y] == 0)


def cheaper(successor, path_cost, cost_map):
    """
    Matches Line 16 of the algorithm.

    Parameters
    ----------
    successor : Vertex
        The current vertex.
    path_cost : float
        The current path cost of successor.
    cost_map : numpy.array
        The map in which the path cost of all vertices is saved.

    Returns
    -------
    is_cheaper : bool
        True if the current cost is smaller than the cost from the cost_map

    Examples
    --------
    >>> import numpy as np
    >>> import sys
    >>> mx = sys.float_info.max
    >>> c_map = np.array([[0, 1, mx], [1, mx, mx], [mx, mx, mx]])
    >>> print(c_map)  # top left corner is (0, 0)
    [[  0.00000000e+000   1.00000000e+000   1.79769313e+308]
     [  1.00000000e+000   1.79769313e+308   1.79769313e+308]
     [  1.79769313e+308   1.79769313e+308   1.79769313e+308]]
    >>> successor = Vertex(1, 1)
    >>> path_cost = 1.4
    >>> print("cheaper: {}".format(cheaper(successor, path_cost, c_map)))
    cheaper: True
    >>> successor2 = Vertex(1, 0)
    >>> path_cost2 = 2.4
    >>> print("cheaper: {}".format(cheaper(successor2, path_cost2, c_map)))
    cheaper: False

    """
    #   (vii)   Check whether the computed path cost is cheaper
    #           than the value of the cost_map at the coordinates
    #           of the current vertex. If false then skip this
    #           neighbor. (If skipped then the vertex is reachable
    #           from a different vertex and the cost is cheaper.
    # TODO: add your code here

    is_cheaper = path_cost < cost_map[successor.x][successor.y]
    return is_cheaper
