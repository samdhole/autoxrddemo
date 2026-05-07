from pathlib import Path

import pytest

import autoxrd.__main__ as cli


def test_main_raises_clear_error_for_empty_data_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cli, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(cli, "OUTPUT_FILE", tmp_path / "out" / "xrd_memo.html")

    with pytest.raises(RuntimeError, match="No .txt files found"):
        cli.main()
