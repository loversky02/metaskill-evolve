import pytest

from mse.dag import DAG


def test_add_and_get():
    d = DAG()
    root = d.add(utility=0.3, snapshot={"skill.md": "a"})
    n = d.get(root)
    assert n.utility == 0.3 and n.snapshot == {"skill.md": "a"}
    assert n.parent_id is None and n.kind == "task"


def test_best_orders_by_utility():
    d = DAG()
    r = d.add(utility=0.3)
    d.add(parent_id=r, utility=0.5)
    d.add(parent_id=r, utility=0.9)
    d.add(parent_id=r, utility=0.7)
    top = d.best(3)
    assert [round(x.utility, 2) for x in top] == [0.9, 0.7, 0.5]


def test_children():
    d = DAG()
    r = d.add(utility=0.1)
    d.add(parent_id=r, utility=0.2)
    d.add(parent_id=r, utility=0.3)
    assert len(d.children(r)) == 2


def test_frontier_pairs_weights():
    d = DAG()
    r = d.add(utility=0.3)
    d.add(parent_id=r, utility=0.9)
    d.add(parent_id=r, utility=0.6)
    fr = d.frontier((1.0, 0.5, 0.25))
    assert fr[0][0].utility == 0.9 and fr[0][1] == 1.0
    assert len(fr) == 3


def test_kind_filter():
    d = DAG()
    d.add(utility=0.5, kind="task")
    d.add(utility=0.8, kind="meta")
    assert d.best(1, kind="task")[0].utility == 0.5
    assert d.best(1, kind="meta")[0].utility == 0.8


def test_get_missing_raises():
    d = DAG()
    with pytest.raises(KeyError):
        d.get(999)
