"""Hand-checked cases for the metrics module."""
import math

from jevmatch.metrics import all_metrics, per_source_rank, recall_at_gt, thresholded_prf

S, T = ["a", "b"], ["x", "y"]


def test_recall_no_ties():
    sc = {("a", "x"): .9, ("a", "y"): .5, ("b", "x"): .4, ("b", "y"): .8}
    assert recall_at_gt(sc, {("a", "x"), ("b", "y")})["expected"] == 1.0


def test_recall_ties():
    # top-1 is a hit; one slot left, filled from 3 tied pairs holding 1 hit
    sc = {("a", "x"): .9, ("a", "y"): .5, ("b", "x"): .5, ("b", "y"): .5}
    r = recall_at_gt(sc, {("a", "x"), ("b", "y")})
    assert math.isclose(r["expected"], (1 + 1 / 3) / 2)
    assert r["pessimistic"] == 0.5 and r["optimistic"] == 1.0


def test_mrr_p1_ties():
    # source a: x (hit) tied with y at top -> E[rank]=1+... = (2+1)/(1+1)=1.5, P@1 = 1/2
    sc = {("a", "x"): .5, ("a", "y"): .5, ("b", "x"): .1, ("b", "y"): .9}
    r = per_source_rank(sc, {("a", "x"), ("b", "y")})
    assert math.isclose(r["mrr"], (1 / 1.5 + 1) / 2) and math.isclose(r["p_at_1"], (0.5 + 1) / 2)


def test_thresholded_and_missing_cells():
    sc = {("a", "x"): .9, ("b", "y"): .6}  # (a,y), (b,x) missing -> -inf
    m = all_metrics(sc, {("a", "x"), ("b", "y")}, S, T, probabilistic=True)
    assert m["recall_gt_expected"] == 1.0 and m["h1_f1"] == 1.0
    assert m["t7_precision"] == 1.0 and m["t7_recall"] == 0.5
    assert thresholded_prf(sc, {("a", "x")}, 0.5)["precision"] == 0.5
