# -*- coding: utf-8 -*-
"""
Test functions of the pathplanning core algorithm

###############################################################################
############################ NO IMPLEMENTATION HERE ###########################
###############################################################################
"""
from __future__ import print_function

import os

import matplotlib.pyplot as plt
import numpy as np

HAS_IMREAD = True
try:
    from imageio import imread
except ImportError:
    HAS_IMREAD = False

# pylint: disable=wrong-import-order, wrong-import-position
from ksr_common.hash_function import js_hash, numpy_hash_normalized
from ksr_common.string_suite import TestSuite, ValidationSuite

from resources.test_data import END, NAV_MAP, NEIGHBORS, START
from resources.pathplanning_utils import (
    Vertex, get_neighborhood, plot_map, plot_map_debug, prepare_map
)

try:
    from dev_core import core_algorithm
    try:
        from dev_core import plan_path
    except ImportError:
        from dev_helperfunctions import plan_path
except ImportError:
    from student_core import core_algorithm
    try:
        from student_core import plan_path
    except ImportError:
        from student_helperfunctions import plan_path

DEV_MODE = False


def astar_test(nav_map, neighbors, source, target):
    """Simple one-step pretest of the implemented A* algorithm"""
    # pylint: disable=invalid-name
    TEST_ASTAR_DISTANCE = 1174933390
    TEST_ASTAR_HEURISTIC = 3150456522
    TEST_ASTAR_PATH = 2477493473
    # pylint: enable=invalid-name
    print("")
    test_name = "A*"
    ts = TestSuite(test_name)
    astar_passed = True
    pred_matrix, d_map, h_map = core_algorithm(
        nav_map, source, target, neighbors, 'astar', silent=True
    )

    distance_hash = numpy_hash_normalized(d_map)
    distance_cmp = distance_hash == TEST_ASTAR_DISTANCE
    astar_passed = astar_passed and distance_cmp
    heuristic_hash = numpy_hash_normalized(h_map)
    heuristic_cmp = heuristic_hash == TEST_ASTAR_HEURISTIC
    astar_passed = astar_passed and heuristic_cmp

    if DEV_MODE:
        print("TEST_ASTAR_DISTANCE = {}".format(distance_hash))
        print("TEST_ASTAR_HEURISTIC = {}".format(heuristic_hash))

    if not astar_passed:
        ts.test_open("{} Distance Grid".format(test_name),
                     ["Tests whether your computed distances for each",
                      "coordinate are correct!"])
        ts.test_close(distance_cmp, "Your distance grid is not correct!")
        ts.test_open("{} Heuristic Grid".format(test_name),
                     ["Tests whether your computed heuristics for each",
                      "coordinate are correct!"])
        ts.test_close(heuristic_cmp, "Your heuristic grid is not correct!")
        ts.suite_close(astar_passed)
        plot_map_debug(
            nav_map, d_map, h_map, source, target, path=None,
            pathmatrix=pred_matrix
        )
        return False

    ts.suite_close(astar_passed)
    return path_test(nav_map, d_map, h_map, pred_matrix, source, target,
                     TEST_ASTAR_PATH)


def dijkstra_test(nav_map, neighbors, source, target):
    """Simple one-step pretest of the implemented Dijkstra algorithm"""
    # pylint: disable=invalid-name
    TEST_DIJKSTRA_DISTANCE = 3275782271
    TEST_DIJKSTRA_HEURISTIC = 2555610717
    TEST_DIJKSTRA_PATH = 2477493473
    # pylint: enable=invalid-name
    print("")
    test_name = "Dijkstra"
    ts = TestSuite(test_name)

    dijkstra_passed = True
    pred_matrix, d_map, h_map = core_algorithm(
        nav_map, source, target, neighbors, 'dijkstra', silent=True
    )
    distance_hash = numpy_hash_normalized(d_map)
    distance_cmp = distance_hash == TEST_DIJKSTRA_DISTANCE
    dijkstra_passed = dijkstra_passed and distance_cmp

    heuristic_hash = numpy_hash_normalized(h_map)
    heuristic_cmp = heuristic_hash == TEST_DIJKSTRA_HEURISTIC
    dijkstra_passed = dijkstra_passed and heuristic_cmp

    if DEV_MODE:
        print("TEST_DIJKSTRA_DISTANCE = {}".format(distance_hash))
        print("TEST_DIJKSTRA_HEURISTIC = {}".format(heuristic_hash))

    if not dijkstra_passed:
        ts.test_open("{} Distance Grid".format(test_name),
                     ["Tests whether your computed distances for each",
                      "coordinate are correct!"])
        ts.test_close(distance_cmp, "Your distance grid is not correct!")
        ts.test_open("{} Heuristic Grid".format(test_name),
                     ["Tests whether your computed heuristics for each",
                      "coordinate are correct!"])
        ts.test_close(heuristic_cmp, "Your heuristic grid is not correct!")
        ts.suite_close(dijkstra_passed)
        plot_map_debug(
            nav_map, d_map, h_map, source, target, path=None,
            pathmatrix=pred_matrix
        )
        return False

    ts.suite_close(dijkstra_passed)
    return path_test(nav_map, d_map, h_map, pred_matrix, source, target,
                     TEST_DIJKSTRA_PATH)


def path_test(nav_map, d_map, h_map, pred_matrix, source, target, test_hash):
    """Shared functionality used by both pretest functions"""
    print("")
    test_name = "Path"
    ts = TestSuite(test_name, ["Tests the path your path planner returns"])
    path_passed = True

    my_path = plan_path(source, target, pred_matrix)
    my_path_number = [[el.x, el.y] for el in my_path]
    path_hash = js_hash(my_path_number)

    path_cmp = path_hash == test_hash
    path_passed = path_passed and path_cmp

    plot_map(nav_map, d_map, h_map, source, target, path=my_path)
    if not path_passed:
        ts.test_oneline(path_cmp, "{} Computation".format(test_name))
        ts.suite_close(path_passed)
        plot_map(nav_map, d_map, h_map, source, target, path=None)
    else:
        ts.suite_close(path_passed)
    return path_passed


def pathplanning_pretest():
    """A simple one-step pretest of both implemented algorithms

    There is actually nothing of real interest here, since most of the work is
    delegated to dijkstra_test and astar_test.

    pytest is configured to also find this test during discovery although it
    does not match the usual "test_*" pattern.
    """
    passed = True
    # Test Dijkstra
    plt.clf()
    dijkstra_result = dijkstra_test(NAV_MAP, NEIGHBORS, START, END)
    passed = passed and dijkstra_result
    # Test Astar
    plt.clf()
    a_star_result = astar_test(NAV_MAP, NEIGHBORS, START, END)
    passed = passed and a_star_result
    if __name__ == "test_core":
        assert passed
    return passed


def run_validation(vs, mode, nav_map, start, goal, hash_tuple, silent=False):
    """Compute and validate one step in the multi-step test

    Input and expected output data is passed as arguments to this function.

    The return value is either True or False, depending on whether all the
    internal checks have succeeded or not.
    """
    # pylint: disable=invalid-name
    DISTANCE_HASH, HEURISTIC_HASH, PATH_HASH = hash_tuple
    # pylint: enable=invalid-name

    plt.clf()

    neighbors = get_neighborhood(8)
    if mode == 'astar':
        validation_name = 'A*'
        vs.suite_info("{} Start".format(validation_name))
        pred_matrix, d_map, h_map = core_algorithm(
            nav_map, start, goal, neighbors, 'astar', silent
        )
    else:
        validation_name = 'Dijkstra'
        vs.suite_info("{} Start".format(validation_name))
        pred_matrix, d_map, h_map = core_algorithm(
            nav_map, start, goal, neighbors, 'dijkstra', silent
        )

    planned_path = plan_path(start, goal, pred_matrix)
    plot_map(nav_map, d_map, h_map, start, goal, planned_path, mode=mode)
    vs.suite_info("{} Validation Start".format(validation_name))

    distance_hash = numpy_hash_normalized(d_map)
    distance_cmp = distance_hash == DISTANCE_HASH

    heuristic_hash = numpy_hash_normalized(h_map)
    heuristic_cmp = heuristic_hash == HEURISTIC_HASH

    path_hash = js_hash([[el.x, el.y] for el in planned_path])
    path_cmp = path_hash == PATH_HASH

    if DEV_MODE:
        print("[{}, {}, {}]".format(distance_hash, heuristic_hash, path_hash))

    if not (distance_cmp and heuristic_cmp and path_cmp):
        vs.validationcase_open(validation_name)
        vs.validation_oneline(distance_cmp,
                              "{} Distance Hash".format(validation_name))
        vs.validation_oneline(heuristic_cmp,
                              "{} Heuristic Hash".format(validation_name))
        vs.validation_oneline(path_cmp, "{} Path Hash".format(validation_name))
        vs.validationcase_close(False)
        return False

    vs.validation_oneline(True, validation_name)
    return True


def test_pathplanning(silent=False):
    """
    Parameters
    ----------
    silent : bool, optional
        Starts without visualization
    """
    # pylint: disable=invalid-name
    DIJKSTRA_HASH = [
        [596486860, 566685125, 877816163],
        [4211979207, 3322734315, 3745999849],
        [3910974588, 1645813182, 2087608585]
    ]

    ASTAR_HASH = [
        [2831870872, 3764695826, 877816163],
        [482017054, 2658979732, 1634291403],
        [3165017498, 4165661301, 2646616493]
    ]
    # pylint: enable=invalid-name

    # load map as image
    basepath = os.path.dirname(__file__)
    try:
        my_nav_map = np.load(os.path.join(basepath, "resources/nav-map.npy"))
    except IOError as ex:
        if HAS_IMREAD:
            my_nav_map = prepare_map(
                imread(os.path.join(basepath, "resources/map.pgm"))
            )
            np.save(os.path.join(basepath, "resources/nav-map.npy"), my_nav_map)
        else:
            raise ex

    # goal parameter
    z_zero = [Vertex(327, 401), Vertex(327, 401), Vertex(389, 56)]
    z_goal = [Vertex(497, 383), Vertex(389, 56), Vertex(562, 285)]
    tpl_list = zip(z_zero, z_goal, ASTAR_HASH, DIJKSTRA_HASH)
    for idx, (start, goal, a_hash, d_hash) in enumerate(tpl_list):
        vs = ValidationSuite("Pathplanning {}".format(idx+1))

        passed = True
        passed_dijkstra = run_validation(vs, 'dijkstra', my_nav_map, start,
                                         goal, d_hash, silent=silent)
        passed = passed and passed_dijkstra
        if not silent:
            plt.pause(5)

        passed_astar = run_validation(vs, 'astar', my_nav_map, start, goal,
                                      a_hash, silent=silent)
        passed = passed and passed_astar
        if not silent:
            plt.pause(1 if __name__ == "test_core" else 5)

        vs.suite_close(passed)
        if __name__ == "test_core":
            assert passed


if __name__ == "__main__":
    if pathplanning_pretest() or DEV_MODE:
        test_pathplanning(silent=False)
