import pytest

from tools.stoppolicy.data.fetch_online import presence_from_categories, BudgetedSaver


def test_presence_mapping():
  assert presence_from_categories(["person", "traffic light"]) == (True, False)
  assert presence_from_categories(["stop sign"]) == (False, True)
  assert presence_from_categories(["car", "dog"]) == (False, False)
  assert presence_from_categories(["traffic light", "stop sign"]) == (True, True)


def test_budgeted_saver(tmp_path):
  import numpy as np
  s = BudgetedSaver(tmp_path, max_bytes=20_000)
  n = 0
  while s.add(np.full((128, 128, 3), n % 255, dtype=np.uint8), has_light=bool(n % 2), has_sign=False):
    n += 1
  s.close()
  assert 0 < n < 1000
  meta = __import__("json").loads((tmp_path / "meta.json").read_text())
  assert meta["n_images"] == n


@pytest.mark.slow
def test_fetch_small_sample(tmp_path):
  from tools.stoppolicy.data.fetch_online import fetch
  n = fetch(out_dir=tmp_path, n_target=20, max_bytes=30_000_000)
  assert n >= 10
  labels = [__import__("json").loads(l) for l in (tmp_path / "labels.jsonl").read_text().splitlines()]
  assert any(r["has_light"] or r["has_sign"] for r in labels)
  assert any(not (r["has_light"] or r["has_sign"]) for r in labels)  # negatives kept too
