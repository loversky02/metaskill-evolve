import pytest

from mse.skills import META_COMPONENTS, SEED_TASK, SkillStore


def test_create_seeds_all_files(tmp_path):
    s = SkillStore.create(tmp_path)
    assert s.task_path.exists()
    assert s.read_task() == SEED_TASK
    for sym in META_COMPONENTS:
        assert s.meta_path(sym).exists()
    assert set(s.meta().keys()) == set(META_COMPONENTS)


def test_snapshot_restore_roundtrip(tmp_path):
    s = SkillStore.create(tmp_path)
    snap = s.snapshot()
    s.write_task("MUTATED")
    s.write_meta("pi", "MUTATED PROPOSER")
    assert s.read_task() == "MUTATED"
    s.restore(snap)
    assert s.read_task() == SEED_TASK
    assert "Proposer" in s.read_meta("pi")


def test_snapshot_keys(tmp_path):
    s = SkillStore.create(tmp_path)
    snap = s.snapshot()
    assert "skill.md" in snap
    assert "meta/proposer.md" in snap
    assert len(snap) == 1 + len(META_COMPONENTS)


def test_invalid_meta_symbol(tmp_path):
    s = SkillStore.create(tmp_path)
    with pytest.raises(KeyError):
        s.meta_path("not_a_symbol")
