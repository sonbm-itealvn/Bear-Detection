from det.labels.build_eval_set import select_eval, QUOTA

def _row(i, dom, nbox, split, cls="bear"):
    return {"sha1": f"{split}{dom}{i}", "path_hint": f"/d/{split}/{cls}/{dom}_{i}.jpg", "domain": dom,
            "boxes": [{"xyxy": [0, 0, 1, 1]}] * nbox, "cls": cls, "status": "auto"}

def _rows():
    rows = {"val": [], "test": []}
    for sp in rows:
        for dom in ["inat", "rare", "ir", "youtube", "small", "screen", "paste"]:
            rows[sp] += [_row(i, dom, 1, sp) for i in range(300)]
        rows[sp] += [_row(1000 + i, "ir", 2, sp) for i in range(200)]
        rows[sp] += [_row(5000 + i, "other", 0, sp, "non_bear") for i in range(500)]
    return rows

def test_quota_and_no_paste():
    sel = select_eval(_rows(), novel=set(), seed=0)
    for sp in ("val", "test"):
        assert len(sel[sp]["bear"]) == sum(QUOTA.values()) == 600
        assert len(set(sel[sp]["bear"])) == 600
        assert not any("paste" in s for s in sel[sp]["bear"])
        assert len(sel[sp]["non_bear"]) == 200

def test_novel_first_and_hard_negatives():
    rows = _rows()
    novel = {r["path_hint"] for r in rows["val"] if r["domain"] == "inat" and int(r["sha1"].split("inat")[1]) < 150}
    hard = {r["path_hint"] for r in rows["val"] if r["cls"] == "non_bear"}
    sel = select_eval(rows, novel=novel, seed=0, nb_hard=hard)
    inat = [s for s in sel["val"]["bear"] if "inat" in s]
    assert len(inat) == 100 and all(int(s.split("inat")[1]) < 150 for s in inat)

def test_shortfall_reported(capsys):
    rows = {"val": [_row(i, "small", 1, "val") for i in range(10)]}
    sel = select_eval(rows, novel=set(), seed=0)
    assert len(sel["val"]["bear"]) == 10 and "short" in capsys.readouterr().out
