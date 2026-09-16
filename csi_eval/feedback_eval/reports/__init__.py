"""Report generators: JSON, HTML, Markdown.

Each module exposes a ``save(report, out_dir)`` returning a Path.
"""

from .json_report import save as save_json  # noqa: F401
from .html_report import save as save_html  # noqa: F401
from .markdown_report import save as save_markdown  # noqa: F401

__all__ = ["save_json", "save_html", "save_markdown"]
