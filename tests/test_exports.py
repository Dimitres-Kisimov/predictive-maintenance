from __future__ import annotations

from pdm.exports import build_deliverables


def test_deliverables_written_and_non_trivial(pipeline_result, tmp_path):
    sizes = build_deliverables(pipeline_result, tmp_path)
    assert len(sizes) == 2
    names = sorted(p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in sizes)
    assert names == ["pdm_report.pdf", "pdm_workbook.xlsx"]
    for path, size in sizes.items():
        if path.endswith(".pdf"):
            assert size > 10 * 1024, f"{path} suspiciously small ({size} B)"
        else:
            assert size > 5 * 1024, f"{path} suspiciously small ({size} B)"
