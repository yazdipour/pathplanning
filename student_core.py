# -*- coding: utf-8 -*-
"""
Implementation of the pathplanning core algorithm for Dijkstra and A*

DIJKSTRA/A* PSEUDOCODE WITH PRIORITY QUEUE
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

import sys
import math

import numpy as np

from resources.pathplanning_utils import plot_map
from resources.pathplanning_utils import (
    PredecessorMatrix, HeapQueue, heuristic)
from student_helperfunctions import (dequeue_next, next_successor,
                                     validate_successor, cheaper)


def plan_path(start, goal, path_matrix):
    """Backtracks through the path_matrix and builds a path along the track.

    Parameters
    ----------
    goal : Vertex
        The vertex from which the traversal through the path_matrix starts.
    start : Vertex
        The vertex at which the traversal terminates.
    path_matrix : PredecessorMatrix
        The path_matrix which was been computed in the A*/Dijkstra algorithm.

    Returns
    -------
    path : list of Vertex
        The path as a list of vertices.

    Notes
    -----
    Expects that start is reachable from goal otherwise will fail with
    Exception!

    Examples
    --------
    >>> from resources.pathplanning_utils import PredecessorMatrix, Vertex
    >>> import numpy as np
    >>> nav_map = np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [1, 1, 1, 1]])
    >>> print(nav_map)
    [[0 0 0 1]
     [0 0 0 1]
     [0 0 0 1]
     [1 1 1 1]]
    >>> pm = PredecessorMatrix(nav_map)
    >>> pm.p_map[1, 0] = Vertex(0, 0)
    >>> pm.p_map[0, 1] = Vertex(0, 0)
    >>> pm.p_map[1, 1] = Vertex(0, 0)
    >>> pm.p_map[2, 0] = Vertex(1, 0)
    >>> pm.p_map[2, 1] = Vertex(1, 0)
    >>> pm.p_map[0, 2] = Vertex(0, 1)
    >>> pm.p_map[1, 2] = Vertex(0, 1)
    >>> pm.p_map[2, 2] = Vertex(1, 1)
    >>> print(np.array([[str(x) for x in y] for y in pm.p_map]))  # (0, 0) top left corner
    [['None' 'Vertex(0, 0)' 'Vertex(0, 1)' 'None']
     ['Vertex(0, 0)' 'Vertex(0, 0)' 'Vertex(0, 1)' 'None']
     ['Vertex(1, 0)' 'Vertex(1, 0)' 'Vertex(1, 1)' 'None']
     ['None' 'None' 'None' 'None']]
    >>> start = Vertex(0, 0)
    >>> goal = Vertex(2, 2)
    >>> path = plan_path(start, goal, pm)
    >>> print([str(x) for x in path])
    ['Vertex(2, 2)', 'Vertex(1, 1)', 'Vertex(0, 0)']
    """
    #   (i)     Build the path by following the chain of predecessors in the
    #           path_matrix from goal to start.
    #
    #   Hint:   path_matrix is a PredecessorMatrix (see
    #           resources.pathplanning_utils for more information) and
    #           provides the following methods:
    #               insert(successor, predecessor),
    #               get_predecessor(vertex)
    # TODO: add your code here

    path = []
    current = goal
    while current != start:
        path.append(current)
        current = path_matrix.get_predecessor(current)

    path.append(start)
    return path


def core_algorithm(nav_map, start, goal, neighbors, mode='astar', silent=False):
    """
    Parameters
    ----------
    nav_map : numpy.array (2d)
        The map which is used to determine valid neighbors. Holds the occupancy
        of each cell with the binary value 0 (=free) or 1 (=occupied).
    start : Vertex
        The vertex from which the process originates (matches the :math:`v_{0}`
        parameter as introduced in the KSR lecture). Note that in the
        visualization this will be the goal position!
    goal : Vertex
        The vertex which is the goal of the process (matches the :math:`v`
        parameter as introduced in the KSR lecture). Note that in the
        visualization this will be the current robot position!
    neighbors : list of WeightedDirection
        A list of possible directions for neighbors for each grid cell. (e.g.
        8N neighborhood)
    mode : str, optional
        Defines whether the A* (astar) or dijkstra algorithm is executed by
        determining whether heuristic returns 0 or the actual value
        (Default: astar).

    Returns
    -------
    PredecessorMatrix
        An object containing a collection of TableEntry.
    print_cost_map : numpy array
        Contains the path cost information from each cell to the start
        position.
    print_combined_cost_map : numpy array
        Contains the summarized costs of path (from cell to start) and
        heuristic (from cell to goal)!


    Notes
    -----
    Dijkstra/A*(astar) by nature are graph based approaches. By using our
    a priori knowledge about the map and its size, we can improve the
    performance.
    This is possible due to the use of an additional map for saving the
    cost from the start to the current coordinates. This also provides
    an O(1) complexity for checking, whether a certain cell has been visited.
    This comes at the cost of additional memory usage, which is still
    manageable at this point.

    We use the cost map  for saving our path cost for each coordinate to the
    start position and also use the array to check, whether coordinates have
    been visited (much faster than accessing the PathTable!).

    We will build the graph on the fly using a neighborhood function which
    describes all possible edges between two grid cells.

    cost_map : numpy.array 2d
        Saves the path costs from the cell to the start position. Can be used
        to check whether a cell has been visited before (by checking for
        equality with the initial value).
    n : int
        This variable is an additional value which is used to force a more
        deterministic behavior when using the builtin PriorityQueue. This
        arises from the fact that we will have duplicate keys / priorities and
        we need a further value to make them distinguishable. This means, a
        tuple in the queue has the following form:
        (combined_cost, n, Vertex).

    References
    ----------
    See KSR Chapter 4.3: Dijkstra and A*

    Examples
    --------
    >>> import numpy as np
    >>> from resources.pathplanning_utils import get_neighborhood
    >>> nav_map = np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [1, 1, 1, 1]])
    >>> print(nav_map)
    [[0 0 0 1]
     [0 0 0 1]
     [0 0 0 1]
     [1 1 1 1]]
    >>> start = Vertex(0, 0)
    >>> goal = Vertex(2, 2)
    >>> n = get_neighborhood(8)
    >>> a, b, c = core_algorithm(nav_map, start, goal, n, "astar", True)
    >>> print(np.array([[str(x) for x in y] for y in a.p_map]))  # (0, 0) top left corner
    [['None' 'Vertex(0, 0)' 'Vertex(0, 1)' 'None']
     ['Vertex(0, 0)' 'Vertex(0, 0)' 'Vertex(0, 1)' 'None']
     ['Vertex(1, 0)' 'Vertex(1, 0)' 'Vertex(1, 1)' 'None']
     ['None' 'None' 'None' 'None']]
    >>> print(b)
    [[ 0.          1.          2.          5.65685425]
     [ 1.          1.41421356  2.41421356  5.65685425]
     [ 2.          2.41421356  2.82842712  5.65685425]
     [ 5.65685425  5.65685425  5.65685425  5.65685425]]
    >>> print(c)
    [[ 2.82842712  3.82842712  4.23606798  2.82842712]
     [ 3.82842712  4.24264069  4.65028154  2.82842712]
     [ 4.23606798  4.65028154  4.24264069  2.82842712]
     [ 2.82842712  2.82842712  2.82842712  2.82842712]]
    """
    # #########################################################################
    # ############# TASK HERE #################################################
    # #########################################################################
    #   TASK:   Use the helper functions which you've implemented and place
    #           them at the right position! There will be three marks for one
    #           function each!
    #           Hint:   Not all helper functions will be needed here!
    n = 0
    x_size, y_size = nav_map.shape
    max_dist = math.sqrt(x_size**2.0 + y_size**2.0)
    h = heuristic(start, goal, mode)

    # initialize cost and path cost maps
    cost_map = np.full_like(nav_map, sys.float_info.max, dtype=np.double)
    pc_map = np.full_like(nav_map, max_dist, dtype=np.double)
    pcc_map = np.full_like(nav_map, h, dtype=np.double)

    cost_map[start.x, start.y] = 0
    pc_map[start.x, start.y] = 0
    pcc_map[start.x, start.y] = h

    # create the predecessor matrix
    pred_matrix = PredecessorMatrix(nav_map)

    # add the first vertex to the priority queue
    queue = HeapQueue()
    queue.put((h, n, start))
    n += 1

    # initialize loop constraints
    resign = False
    found = False
    draw_threshold = 0.0
    draw_value = 0.0
    while not resign and not found:
        next_vertex, resign, found = dequeue_next(queue, goal)
        if resign or found:
            break
        for direction in neighbors:
            #   (i)     Acquire the successor with its path cost and combined
            #           cost. Use the following variable names to store values:
            #           "succ"  (succ)essor Vertex
            #           "sc"    (s)uccessor path (c)ost
            #           "scc"   (s)uccessor (c)ombinded (c)ost
            # TODO: add your code here
            succ, sc, scc = next_successor(
                current=next_vertex, direction=direction, cost_map=cost_map, goal=goal, mode=mode)

            # The draw value is the path cost from the previous task!
            draw_value = sc  # TODO: uncomment after implementing (i)

            #   (ii)    Check whether the successor is in map range.
            is_valid = validate_successor(succ, nav_map)  # TODO: add your code here

            if is_valid:
                #   (iii)   Check whether the cost of the successor is less
                #           than cost currently in the cost_map
                #           Hint:   Use the path cost and not the combined
                #                   cost!
                # TODO: add your code here
                is_cheaper = cheaper(
                    successor=succ, cost_map=cost_map, path_cost=draw_value)
                if is_cheaper:
                    queue.put((scc, n, succ))
                    pred_matrix.insert(succ, next_vertex)
                    n += 1
                    cost_map[succ.x, succ.y] = sc  # path cost
                    pc_map[succ.x, succ.y] = sc  # path cost
                    pcc_map[succ.x, succ.y] = scc  # combined cost
        # PLOT #
        if draw_value > draw_threshold:
            draw_threshold += 15  # magic number (draw every n-th step)
            if not silent:
                plot_map(nav_map, pc_map, pcc_map, start, goal, mode=mode)
    return pred_matrix, pc_map, pcc_map
