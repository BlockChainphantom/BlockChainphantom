"""End-to-end on simulated homes with known ground truth."""
import numpy as np
import pytest

from rfhome.evaluate import compare
from rfhome.reconstruct import reconstruct, rotate_plan
from rfhome.simulate import example_home, simulate_survey


@pytest.fixture(scope="module")
def home():
    return example_home()


@pytest.mark.parametrize("seed", [1, 4, 5])
def test_reconstruction_accuracy(home, seed):
    plan, rooms, links, nodes = home
    r = reconstruct(simulate_survey(plan, rooms, links, nodes, seed=seed))
    m = compare(r.plan, plan)
    assert m["all_f1"] > 0.9
    assert m["interior_f1"] > 0.8
    assert m["footprint_iou"] > 0.85
    assert m["interior_median_err_m"] < 0.25
    assert 1.5 < r.rti.n_exp < 3.0


def test_radio_explains_walls(home):
    plan, rooms, links, nodes = home
    r = reconstruct(simulate_survey(plan, rooms, links, nodes, seed=2))
    # walls must explain a clear part of the signal, otherwise they are guesses
    assert r.rti.rms_residual < 0.8 * r.rti.rms_residual_no_walls


def test_rotated_house(home):
    plan, rooms, links, nodes = home
    s = simulate_survey(plan, rooms, links, nodes, seed=1)
    th = np.deg2rad(30.0)
    r = reconstruct(s.rotated(th))
    assert abs(np.rad2deg(r.angle) - 30.0) < 3.0
    m = compare(r.plan, rotate_plan(plan, th))
    assert m["all_f1"] > 0.85 and m["footprint_iou"] > 0.8
