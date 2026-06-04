"""Tests for python.zrt.report.chrome_trace — template-based trace HTML export."""

import base64
import gzip
import json

import pytest

from python.zrt.report.chrome_trace import (
    _get_template_path,
    _inject_trace_data_into_template,
    export_trace_html,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

_MINIMAL_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Default Title</title>
</head>
<body>
<script id="viewer-data" type="text/plain">
__TRACE_DATA__
</script>
</body>
</html>"""


@pytest.fixture
def tmpl_path(tmp_path):
    """Create a minimal template file and return its path."""
    p = tmp_path / "trace_viewer_shell.html"
    p.write_text(_MINIMAL_TEMPLATE, encoding="utf-8")
    return p


@pytest.fixture
def trace_dict():
    """Return a small but realistic Chrome Trace Event Format dict."""
    return {
        "traceEvents": [
            {
                "name": "mm",
                "cat": "compute",
                "ph": "X",
                "pid": 0,
                "tid": "compute_0",
                "ts": 1000,
                "dur": 500,
                "args": {"op_type": "aten.mm.default"},
            },
        ],
        "displayTimeUnit": "ms",
        "metadata": {"name": "test_trace"},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# _get_template_path
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetTemplatePath:
    def test_returns_path_when_template_exists(self, tmpl_path, monkeypatch):
        """_get_template_path returns the path when the template file exists."""
        monkeypatch.setattr(
            "python.zrt.report.chrome_trace._TEMPLATE_PATH", tmpl_path,
        )
        result = _get_template_path()
        assert result == tmpl_path

    def test_returns_none_when_template_missing(self, tmp_path, monkeypatch):
        """_get_template_path returns None when the template does not exist."""
        missing = tmp_path / "nonexistent.html"
        monkeypatch.setattr(
            "python.zrt.report.chrome_trace._TEMPLATE_PATH", missing,
        )
        result = _get_template_path()
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# _inject_trace_data_into_template
# ═══════════════════════════════════════════════════════════════════════════════

class TestInjectTraceDataIntoTemplate:
    def test_produces_valid_html(self, tmpl_path, trace_dict, monkeypatch):
        """Output is valid HTML with viewer-data script present."""
        monkeypatch.setattr(
            "python.zrt.report.chrome_trace._TEMPLATE_PATH", tmpl_path,
        )
        out = tmpl_path.parent / "out.html"
        result = _inject_trace_data_into_template(trace_dict, out, title="My Trace")
        assert result == out
        assert out.is_file()

        html = out.read_text(encoding="utf-8")
        assert '<script id="viewer-data" type="text/plain">' in html
        assert "__TRACE_DATA__" not in html

    def test_trace_data_is_gzip_base64_embedded(self, tmpl_path, trace_dict, monkeypatch):
        """The embedded viewer-data can be decoded back to the original JSON."""
        monkeypatch.setattr(
            "python.zrt.report.chrome_trace._TEMPLATE_PATH", tmpl_path,
        )
        out = tmpl_path.parent / "out.html"
        _inject_trace_data_into_template(trace_dict, out)

        html = out.read_text(encoding="utf-8")
        # Extract the base64 content between the script tags
        start = html.find('<script id="viewer-data" type="text/plain">')
        end = html.find('</script>', start)
        b64 = html[start + len('<script id="viewer-data" type="text/plain">\n'):end].strip()

        # Decode and verify
        compressed = base64.b64decode(b64)
        decompressed = gzip.decompress(compressed).decode("utf-8")
        restored = json.loads(decompressed)
        assert restored == trace_dict

    def test_sets_title(self, tmpl_path, trace_dict, monkeypatch):
        """Title is replaced in the output HTML."""
        monkeypatch.setattr(
            "python.zrt.report.chrome_trace._TEMPLATE_PATH", tmpl_path,
        )
        out = tmpl_path.parent / "out.html"
        _inject_trace_data_into_template(trace_dict, out, title="Custom Trace Title")

        html = out.read_text(encoding="utf-8")
        assert "<title>Custom Trace Title</title>" in html
        assert "<title>Default Title</title>" not in html

    def test_keeps_default_title_when_not_provided(self, tmpl_path, trace_dict, monkeypatch):
        """When title is empty, the default title from the template is kept."""
        monkeypatch.setattr(
            "python.zrt.report.chrome_trace._TEMPLATE_PATH", tmpl_path,
        )
        out = tmpl_path.parent / "out.html"
        _inject_trace_data_into_template(trace_dict, out)

        html = out.read_text(encoding="utf-8")
        assert "<title>Default Title</title>" in html

    def test_raises_when_template_missing(self, tmp_path, trace_dict):
        """FileNotFoundError when template does not exist at _TEMPLATE_PATH."""
        missing = tmp_path / "nonexistent_template.html"
        # We must monkeypatch so the function uses our missing path
        import python.zrt.report.chrome_trace as mod
        old = mod._TEMPLATE_PATH
        mod._TEMPLATE_PATH = missing
        try:
            with pytest.raises(FileNotFoundError):
                _inject_trace_data_into_template(trace_dict, tmp_path / "out.html")
        finally:
            mod._TEMPLATE_PATH = old

    def test_accepts_json_string_input(self, tmpl_path, trace_dict, monkeypatch):
        """trace_data can be a pre-serialized JSON string."""
        monkeypatch.setattr(
            "python.zrt.report.chrome_trace._TEMPLATE_PATH", tmpl_path,
        )
        json_str = json.dumps(trace_dict)
        out = tmpl_path.parent / "out.html"
        _inject_trace_data_into_template(json_str, out)

        # Should not double-serialize
        html = out.read_text(encoding="utf-8")
        assert "__TRACE_DATA__" not in html

    def test_creates_parent_directories(self, tmpl_path, trace_dict, monkeypatch):
        """Output parent directories are created automatically."""
        monkeypatch.setattr(
            "python.zrt.report.chrome_trace._TEMPLATE_PATH", tmpl_path,
        )
        out = tmpl_path.parent / "sub" / "deep" / "out.html"
        _inject_trace_data_into_template(trace_dict, out)
        assert out.is_file()


# ═══════════════════════════════════════════════════════════════════════════════
# export_trace_html
# ═══════════════════════════════════════════════════════════════════════════════

class TestExportTraceHtml:
    def test_reads_json_and_produces_html(self, tmpl_path, trace_dict, monkeypatch):
        """export_trace_html reads a JSON file and produces a valid HTML."""
        monkeypatch.setattr(
            "python.zrt.report.chrome_trace._TEMPLATE_PATH", tmpl_path,
        )
        json_path = tmpl_path.parent / "trace.json"
        json_path.write_text(json.dumps(trace_dict), encoding="utf-8")

        out = export_trace_html(json_path, title="From JSON")
        assert out.is_file()
        html = out.read_text(encoding="utf-8")
        assert "__TRACE_DATA__" not in html
        assert "<title>From JSON</title>" in html

    def test_default_output_path(self, tmpl_path, trace_dict, monkeypatch):
        """When output_path is not given, defaults to .json → .html."""
        monkeypatch.setattr(
            "python.zrt.report.chrome_trace._TEMPLATE_PATH", tmpl_path,
        )
        json_path = tmpl_path.parent / "my_trace.json"
        json_path.write_text(json.dumps(trace_dict), encoding="utf-8")

        out = export_trace_html(json_path)
        assert out.name == "my_trace.html"
        assert out.is_file()

    def test_raises_when_template_missing(self, tmp_path, trace_dict):
        """FileNotFoundError when the viewer template is not available."""
        json_path = tmp_path / "trace.json"
        json_path.write_text(json.dumps(trace_dict), encoding="utf-8")

        import python.zrt.report.chrome_trace as mod
        old = mod._TEMPLATE_PATH
        mod._TEMPLATE_PATH = tmp_path / "nonexistent.html"
        try:
            with pytest.raises(FileNotFoundError):
                export_trace_html(json_path)
        finally:
            mod._TEMPLATE_PATH = old
