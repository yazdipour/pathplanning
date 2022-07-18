# -*- coding: utf-8 -*-
"""
Automatic tests for pathplanning helperfunctions

###############################################################################
############################ NO IMPLEMENTATION HERE ###########################
###############################################################################
"""
# pylint: disable=invalid-name
from __future__ import print_function

import sys

import numpy as np

from ksr_common.string_suite import TestSuite
from ksr_common.hash_function import js_hash

# pylint: disable=wrong-import-order
from resources.test_data import NAV_MAP, START, END, NEIGHBORS
from resources.pathplanning_utils import (Vertex, HeapQueue, heuristic)
try:
    from dev_helperfunctions import (next_successor, validate_successor,
                                     cheaper, dequeue_next)
except ImportError:
    from student_helperfunctions import (next_successor, validate_successor,
                                         cheaper, dequeue_next)

DEV_MODE = False


def _test_core_empty_queue(ts, queue, goal):
    CORE_DEQUEUE_EMPTY_VERTEX = None
    CORE_DEQUEUE_EMPTY_RESIGN = 2261231011
    CORE_DEQUEUE_EMPTY_FOUND = 4097823119

    next_vertex, resign, found = dequeue_next(queue, goal)
    vertex_hash = next_vertex
    resign_hash = js_hash([int(resign)])
    found_hash = js_hash([int(found)])
    vertex_cmp = vertex_hash == CORE_DEQUEUE_EMPTY_VERTEX
    resign_cmp = resign_hash == CORE_DEQUEUE_EMPTY_RESIGN
    found_cmp = found_hash == CORE_DEQUEUE_EMPTY_FOUND
    test_passed = vertex_cmp and resign_cmp and found_cmp

    if DEV_MODE:
        print(vertex_hash, resign_hash, found_hash)

    test_name = "empty queue"
    if not test_passed:
        ts.testcase_open(test_name)
        ts.test_oneline(vertex_cmp, "{} vertex".format(test_name),
                        "An empty queue should return: " + str(None))
        ts.test_oneline(resign_cmp, "{} resign".format(test_name),
                        "Your 'resign' variable has an unexpected value")
        ts.test_oneline(found_cmp, "{} found".format(test_name),
                        "Your 'found' variable has an unexpected value")
        ts.testcase_close(test_passed)
    else:
        ts.test_oneline(test_passed, test_name)
    return test_passed


def _test_core_regular_queue(ts, queue, goal):
    CORE_DEQUEUE_REGULAR_VERTEX = 2416741215
    CORE_DEQUEUE_REGULAR_RESIGN = 4097823119
    CORE_DEQUEUE_REGULAR_FOUND = 4097823119

    next_vertex, resign, found = dequeue_next(queue, goal)
    vertex_hash = js_hash(np.around([next_vertex.x, next_vertex.y], 3))
    vertex_cmp = vertex_hash == CORE_DEQUEUE_REGULAR_VERTEX
    resign_hash = js_hash([int(resign)])
    resign_cmp = resign_hash == CORE_DEQUEUE_REGULAR_RESIGN
    found_hash = js_hash([int(found)])
    found_cmp = found_hash == CORE_DEQUEUE_REGULAR_FOUND
    test_passed = vertex_cmp and resign_cmp and found_cmp

    if DEV_MODE:
        print(vertex_hash, resign_hash, found_hash)

    test_name = "regular queue"
    if not test_passed:
        ts.testcase_open(test_name)
        ts.test_oneline(vertex_cmp, "{} vertex".format(test_name),
                        "Your 'vertex' variable has an unexpected value")
        ts.test_oneline(resign_cmp, "{} resign".format(test_name),
                        "Your 'resign' variable has an unexpected value")
        ts.test_oneline(found_cmp, "{} found".format(test_name),
                        "Your 'found' variable has an unexpected value")
        ts.testcase_close(test_passed)
    else:
        ts.test_oneline(test_passed, test_name)
    return test_passed


def _test_core_goal_queue(ts, queue, goal):
    CORE_DEQUEUE_GOAL_VERTEX = 2416741215
    CORE_DEQUEUE_GOAL_RESIGN = 4097823119
    CORE_DEQUEUE_GOAL_FOUND = 2261231011

    next_vertex, resign, found = dequeue_next(queue, goal)
    vertex_hash = js_hash(np.around([next_vertex.x, next_vertex.y], 3))
    vertex_cmp = vertex_hash == CORE_DEQUEUE_GOAL_VERTEX
    resign_hash = js_hash([int(resign)])
    resign_cmp = resign_hash == CORE_DEQUEUE_GOAL_RESIGN
    found_hash = js_hash([int(found)])
    found_cmp = found_hash == CORE_DEQUEUE_GOAL_FOUND
    test_passed = vertex_cmp and resign_cmp and found_cmp

    if DEV_MODE:
        print(vertex_hash, resign_hash, found_hash)

    test_name = "goal queue"
    if not test_passed:
        ts.testcase_open(test_name)
        ts.test_oneline(vertex_cmp, "{} vertex".format(test_name),
                        "Your 'vertex' variable has an unexpected value")
        ts.test_oneline(resign_cmp, "{} resign".format(test_name),
                        "Your 'resign' variable has an unexpected value")
        ts.test_oneline(found_cmp, "{} found".format(test_name),
                        "Your 'found' variable has an unexpected value")
        ts.testcase_close(test_passed)
    else:
        ts.test_oneline(test_passed, test_name)
    return test_passed


def _test_core_successor(ts, initial_vertex, neighbors, cost_map, goal, mode):
    """Test next_successor implementation"""
    CORE_SUCCESSOR = [(2556740088, 4104108722, 4103677972),
                      (1810289481, 4102250744, 4120942885),
                      (127419105, 4104108722, 4103677972),
                      (2416741215, 4102250744, 4120942885),
                      (1811175549, 4102250744, 4120942885),
                      (632238215, 4104108722, 4103677972),
                      (2418508700, 4102250744, 4120942885),
                      (2580982074, 4104108722, 4103677972)]
    ts.testcase_open("successor")
    passed = True

    try:
        xy_test_vertex = Vertex(cost_map.shape[0]-1, cost_map.shape[1]-1)
        next_successor(xy_test_vertex, NEIGHBORS[0], cost_map, goal, mode)
    except IndexError:
        passed = False
        ts.test_oneline(passed, "x-y-order",
                        "You have likely interchanged x and y in next_successor")
    for idx, direction in enumerate(neighbors):
        loc_succ = True
        succ, cost, combined_cost = next_successor(initial_vertex, direction,
                                                   cost_map, goal, mode)

        vertex_hash = js_hash(np.around([succ.x, succ.y], 3))
        succ_cmp = vertex_hash == CORE_SUCCESSOR[idx][0]
        loc_succ = loc_succ and succ_cmp

        try:
            cost_hash = js_hash(np.around([cost], 3))
            cost_cmp = cost_hash == CORE_SUCCESSOR[idx][1]
            loc_succ = loc_succ and cost_cmp
        except OverflowError:
            cost_cmp = False

        try:
            combined_cost_hash = js_hash(np.around([combined_cost], 3))
            combined_cost_cmp = combined_cost_hash == CORE_SUCCESSOR[idx][2]
            loc_succ = loc_succ and combined_cost_cmp
        except OverflowError:
            combined_cost_cmp = False

        if DEV_MODE:
            print("({}, {}, {})".format(vertex_hash, cost_hash, combined_cost_hash))

        test_name = "successor " + direction.n
        if not loc_succ:
            ts.test_open(test_name)
            ts.test_oneline(succ_cmp, "{} vertex".format(test_name),
                            "Your 'successor' variable has an unexpected value")
            ts.test_oneline(cost_cmp, "{} path cost".format(test_name),
                            "Your path cost is wrong")
            ts.test_oneline(combined_cost_cmp,
                            "{} combined cost".format(test_name),
                            "Your combined cost (path + heuristic) is wrong")
            ts.test_close(loc_succ)
        else:
            ts.test_oneline(loc_succ, test_name)

        passed = passed and loc_succ
    ts.testcase_close(passed)
    return passed


def _test_core_bounds(ts, nav_map):
    X_BOUND_MIN_HASH = 4097823119
    X_BOUND_MAX_HASH = 4097823119
    Y_BOUND_MIN_HASH = 4097823119
    Y_BOUND_MAX_HASH = 4097823119
    XY_VALID_MIN_HASH = 2261231011
    XY_VALID_MAX_HASH = 2261231011
    OCC_HASH = 4097823119
    FREE_HASH = 2261231011

    x_min_v = Vertex(-2, 2)
    x_max_v = Vertex(nav_map.shape[0], 1)
    y_min_v = Vertex(2, -2)
    y_max_v = Vertex(1, nav_map.shape[1])
    xy_min_v = Vertex(0, 0)
    xy_max_v = Vertex(nav_map.shape[0]-1, nav_map.shape[1]-1)
    free_v = Vertex(4, 4)
    occ_v = Vertex(4, 3)
    passed = True

    assert js_hash(nav_map[x_min_v.x, x_min_v.y]) == 4097823119, "DEV: Update test cases!"
    x_bound_min = js_hash([int(validate_successor(x_min_v, nav_map))])
    cmp_x_bound_min = x_bound_min == X_BOUND_MIN_HASH
    passed = passed and cmp_x_bound_min

    ex_x_bound_max_error = ""
    try:
        x_bound_max = js_hash([int(validate_successor(x_max_v, nav_map))])
        cmp_x_bound_max = x_bound_max == X_BOUND_MAX_HASH
    except IndexError as ex:
        cmp_x_bound_max = False
        ex_x_bound_max_error = str(ex)
    passed = passed and cmp_x_bound_max

    assert js_hash(nav_map[y_min_v.x, y_min_v.y]) == 4097823119, "DEV: Update test cases!"
    y_bound_min = js_hash([int(validate_successor(y_min_v, nav_map))])
    cmp_y_bound_min = y_bound_min == Y_BOUND_MIN_HASH
    passed = passed and cmp_y_bound_min

    ex_y_bound_max_error = ""
    try:
        y_bound_max = js_hash([int(validate_successor(y_max_v, nav_map))])
        cmp_y_bound_max = y_bound_max == Y_BOUND_MAX_HASH
    except IndexError as ex:
        cmp_y_bound_max = False
        ex_y_bound_max_error = str(ex)
    passed = passed and cmp_y_bound_max

    assert js_hash(nav_map[xy_min_v.x, xy_min_v.y]) == 4097823119, "DEV: Update test cases!"
    xy_valid_min = js_hash([int(validate_successor(xy_min_v, nav_map))])
    cmp_xy_valid_min = xy_valid_min == XY_VALID_MIN_HASH
    passed = passed and cmp_xy_valid_min

    assert js_hash(nav_map[xy_max_v.x, xy_max_v.y]) == 4097823119, "DEV: Update test cases!"
    try:
        xy_valid_max = js_hash([int(validate_successor(xy_max_v, nav_map))])
        cmp_xy_valid_max = xy_valid_max == XY_VALID_MAX_HASH
    except IndexError:
        cmp_xy_valid_max = False
    passed = passed and cmp_xy_valid_max

    occ_hash = js_hash([int(validate_successor(occ_v, nav_map))])
    cmp_occ = occ_hash == OCC_HASH
    passed = passed and cmp_occ

    free_hash = js_hash([int(validate_successor(free_v, nav_map))])
    cmp_free = free_hash == FREE_HASH
    passed = passed and cmp_free

    test_name = "validate successor"
    if not passed:
        ts.testcase_open(test_name)
        ts.test_oneline(cmp_x_bound_min, "{} x bound min".format(test_name),
                        "Your x value is below the maps minimum")
        ts.test_oneline(cmp_x_bound_max, "{} x bound max".format(test_name),
                        "Your x value is beyond the maximum map length\n"
                        "and threw the following exception:\n"
                        + ex_x_bound_max_error)
        ts.test_oneline(cmp_y_bound_min, "{} y bound min".format(test_name),
                        "Your y value is below the maps minimum")
        ts.test_oneline(cmp_y_bound_max, "{} y bound max".format(test_name),
                        "Your y value is beyond the maximum map length\n"
                        "and threw the following exception:\n"
                        + ex_y_bound_max_error)
        ts.test_oneline(cmp_xy_valid_min, "{} x-y valid min".format(test_name),
                        "Your handling of the lower limit is not correct")
        ts.test_oneline(cmp_xy_valid_max, "{} x-y-order".format(test_name),
                        "You have likely interchanged x and y in validate_successor")
        ts.test_oneline(cmp_occ,
                        "{} on occupied cell".format(test_name),
                        "A successor on an occupied cell was not labeled False")
        ts.test_oneline(cmp_free, "{} on free cell".format(test_name),
                        "A successor on an free cell was not labeled True")
        ts.testcase_close(passed)
    else:
        ts.test_oneline(passed, test_name)
    return passed


def _test_core_cheaper(ts, vertex, cost, cost_map):
    CHEAPER = 2261231011
    NOT_CHEAPER = 4097823119

    passed = False

    cheaper_hash = js_hash([int(cheaper(vertex, cost, cost_map))])
    cmp_cheaper = cheaper_hash == CHEAPER

    cost_map[vertex.x, vertex.y] = cost
    is_not_cheaper_hash = js_hash([int(cheaper(vertex, cost, cost_map))])
    cmp_not_cheaper = is_not_cheaper_hash == NOT_CHEAPER

    passed = cmp_cheaper and cmp_not_cheaper

    test_name = "check cheaper"
    if not passed:
        ts.testcase_open(test_name)
        ts.test_oneline(cmp_cheaper, "is cheaper")
        ts.test_oneline(cmp_not_cheaper, "is not cheaper")
        ts.testcase_close(passed)
    else:
        ts.test_oneline(passed, test_name)
    return passed


def _test_core(nav_map, neighbors, start, goal):
    ts = TestSuite("Core")

    x_size, y_size = nav_map.shape
    assert x_size != y_size, "DEV: Non-square test map highly desirable!"

    h = heuristic(start, goal, 'astar')
    cost_map = np.full_like(nav_map, sys.float_info.max, dtype=np.double)
    # D(V0) = 0 ###############################################################
    cost_map[start.x, start.y] = 0
    cost_map[start.x+1, start.y] = 1

    queue = HeapQueue()

    passed = True

    empty_bool = _test_core_empty_queue(ts, queue, goal)
    passed = passed and empty_bool

    n = 0
    queue.put((h, n, start))  # add the first vertex to the priority queue
    n += 1
    passed = passed and _test_core_regular_queue(ts, queue, goal)

    queue.put((h, n, start))  # add the first vertex to the priority queue
    n += 1
    passed = passed and _test_core_goal_queue(ts, queue, start)

    # test successor
    initial_vertex = Vertex(start.x+1, start.y)
    succ_bool = _test_core_successor(
        ts, initial_vertex, neighbors, cost_map, goal, 'astar'
    )
    passed = passed and succ_bool

    # test how validate_successor handles the edge cases
    passed = passed and _test_core_bounds(ts, nav_map)

    test_cost = 4
    # the test vertex is chosen right at a cost map corner, so a nice
    # IndexError is thrown if x and y are interchanged
    test_vertex = Vertex(x_size-2, y_size-2)
    passed = passed and _test_core_cheaper(ts, test_vertex, test_cost, cost_map)

    ts.suite_close(passed)

    return passed


def test_helperfunctions():
    """pytest-friendly wrapper to call _test_core with preselected arguments"""
    passed = _test_core(NAV_MAP, NEIGHBORS, START, END)
    if __name__ == "test_helperfunctions":
        assert passed


if __name__ == "__main__":
    test_helperfunctions()
