from __future__ import annotations

from pathlib import Path

from scripts.cleanup_oneoff_artifacts import cleanup_oneoff_artifacts


def test_cleanup_oneoff_artifacts_removes_target_files_and_dirs(tmp_path: Path) -> None:
    target_file = tmp_path / "ragv2-atomic-abc.db"
    target_dir = tmp_path / "ragv2-search-tempdir"
    keep_file = tmp_path / "ec_bot.db"
    keep_dir = tmp_path / "normal_dir"

    target_file.write_text("x", encoding="utf-8")
    target_dir.mkdir()
    (target_dir / "inner.txt").write_text("x", encoding="utf-8")
    keep_file.write_text("x", encoding="utf-8")
    keep_dir.mkdir()

    report = cleanup_oneoff_artifacts(tmp_path)

    assert "ragv2-atomic-abc.db" in report.removed_files
    assert "ragv2-search-tempdir" in report.removed_dirs
    assert not report.skipped
    assert not target_file.exists()
    assert not target_dir.exists()
    assert keep_file.exists()
    assert keep_dir.exists()


def test_cleanup_oneoff_artifacts_dry_run_does_not_delete(tmp_path: Path) -> None:
    target_file = tmp_path / "ragv2-search-xyz.db"
    target_dir = tmp_path / "ragv2-atomic-tempdir"
    target_file.write_text("x", encoding="utf-8")
    target_dir.mkdir()

    report = cleanup_oneoff_artifacts(tmp_path, dry_run=True)

    assert "ragv2-search-xyz.db" in report.removed_files
    assert "ragv2-atomic-tempdir" in report.removed_dirs
    assert target_file.exists()
    assert target_dir.exists()
