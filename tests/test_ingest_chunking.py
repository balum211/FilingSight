import numpy as np

from filingsight.chunking import group_units, semantic_breakpoints, split_units
from filingsight.ingest import clean_page, detect_item, rebuild_tables


def test_table_cells_are_glued_into_rows():
    lines = ["Revenue", "$", "11,139", "$", "17,455", "57 %", "Some prose line that is not a number."]
    assert rebuild_tables(lines)[0] == "Revenue | $11,139 | $17,455 | 57 %"
    assert rebuild_tables(lines)[1].startswith("Some prose")


def test_clean_page_drops_page_number_and_reflows_prose():
    raw = "The COVID-19 pandemic has rapidly\nchanged market conditions.\nTotal revenue\n17,455\n54\n"
    text = clean_page(raw)
    assert "rapidly changed market conditions." in text
    assert "Total revenue | 17,455" in text
    assert not text.rstrip().endswith("54")


def test_detect_item_uses_line_start_headings_only():
    page = "see Item 7 of Part II for details\nItem 7. Management's Discussion and Analysis\nbody"
    assert detect_item(page) == "7"
    assert detect_item("Item 1A. Risk Factors") == "1A"
    assert detect_item("nothing here") is None


def test_split_units_adds_table_caption_to_rows():
    text = "Active Riders | 2021 | 2020\nThree Months Ended December 31 | 18,728 | 12,552"
    units = split_units(text, with_captions=True)
    assert units[1] == "Active Riders | 2021 | 2020 › Three Months Ended December 31 | 18,728 | 12,552"


def test_semantic_breakpoints_find_topic_shift():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    vecs = np.stack([a, a, a, b, b, b])
    assert semantic_breakpoints(vecs, percentile=80) == [3]


def test_group_units_respects_max_chars():
    units = ["x" * 400] * 6
    groups = group_units(units, breaks=[], max_chars=1000, min_chars=200)
    assert all(len(g) <= 1500 for g in groups)
    assert len(groups) >= 3
