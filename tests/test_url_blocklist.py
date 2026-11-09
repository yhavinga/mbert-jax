"""The URL blocklist of the final run: hebban.nl (the source of DBRD) and its subdomains, and nothing else."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmbjax.training.url_blocklist import host_is_blocked, parse_blocklist, url_host, url_is_blocked  # noqa: E402

BLOCKED = ("hebban.nl",)


@pytest.mark.parametrize("url", [
    "https://www.hebban.nl/boek/de-avond-is-ongemak",
    "http://hebban.nl",
    "https://HEBBAN.NL:443/recensie?id=1",
    "https://user:pw@www.Hebban.nl/x",
    "https://hebban.nl./boek",
    "hebban.nl/boek",
    "//m.hebban.nl/",
    "https://forum.blog.hebban.nl/topic#1",
])
def test_hebban_and_its_subdomains_are_blocked(url):
    assert url_is_blocked(url, BLOCKED)


@pytest.mark.parametrize("url", [
    "https://nothebban.nl/boek",
    "https://hebban.nl.example.com/",
    "https://example.com/?ref=hebban.nl",
    "https://example.com/hebban.nl/page",
    "https://www.hebban.be/",
    "https://hebbannl.com/",
    "",
    None,
    "https://[2001:db8::1/broken",
])
def test_other_hosts_are_not(url):
    assert not url_is_blocked(url, BLOCKED)


def test_host_of_odd_urls():
    assert url_host("https://[2001:db8::1]:8080/x") == "[2001:db8::1]"
    assert url_host("HTTPS://WWW.Example.COM:80") == "www.example.com"
    assert url_host("https://example.com\\path") == "example.com"
    assert url_host("   ") is None
    assert not host_is_blocked(None, BLOCKED)


def test_blocklist_takes_bare_hosts_only():
    assert parse_blocklist(" Hebban.nl, ,hebban.nl ") == ("hebban.nl",)
    assert parse_blocklist("b.example,a.example") == ("a.example", "b.example")
    for bad in ("https://hebban.nl", "hebban.nl/boek", "hebban.nl:443", "", " , "):
        with pytest.raises(ValueError):
            parse_blocklist(bad)


def test_scan_counts_the_rows_the_loader_leaves_out(tmp_path):
    urls = ["https://www.hebban.nl/a", "https://example.com/", None, "http://HEBBAN.nl/b", "https://nothebban.nl/c"]
    first = tmp_path / "nl" / "nl_part_00007.parquet"
    first.parent.mkdir()
    pq.write_table(pa.table({"text": [f"t{i}" for i in range(5)], "url": urls}), first, row_group_size=2)
    second = tmp_path / "nl" / "nl_part_00011.parquet"
    pq.write_table(pa.table({"text": ["a", "b"], "url": ["https://example.org", "https://x.hebban.nl/"]}), second)
    files_list = tmp_path / "files.txt"
    files_list.write_text(f"{first}\n{second}\n")
    out = tmp_path / "scan.json"
    subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "experiments" / "final" / "blocklist_scan.py"),
                    "--files-list", str(files_list), "--blocklist", "hebban.nl", "--out", str(out)], check=True)
    scan = json.loads(out.read_text())
    assert scan["documents"] == 7 and scan["blocked"] == 3
    assert [(f["culturax_file"], f["rows"], f["blocked_rows"]) for f in scan["files"]] == [(7, 5, [0, 3]), (11, 2, [1])]
    assert scan["hosts"] == {"www.hebban.nl": 1, "hebban.nl": 1, "x.hebban.nl": 1}


def test_scan_refuses_files_without_urls(tmp_path):
    path = tmp_path / "share.parquet"
    pq.write_table(pa.table({"text": ["a"]}), path)
    (tmp_path / "files.txt").write_text(f"{path}\n")
    result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "experiments" / "final" / "blocklist_scan.py"),
                             "--files-list", str(tmp_path / "files.txt"), "--blocklist", "hebban.nl",
                             "--out", str(tmp_path / "scan.json")], capture_output=True, text=True)
    assert result.returncode != 0 and "has no column url" in result.stderr
