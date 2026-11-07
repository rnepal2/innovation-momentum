"""Render the research draft locally; publication files are never touched."""

from __future__ import annotations

import hashlib
import re
from html import escape
from pathlib import Path

from latex2mathml.converter import convert
from markdown_it import MarkdownIt
from markdown_it.token import Token

HERE = Path(__file__).resolve().parent

STYLE = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin: 0; background: Canvas; color: CanvasText;
       font: 17px/1.7 Georgia, "Times New Roman", serif; }
main { max-width: 1200px; margin: 48px auto 80px; padding: 0 32px; }
h1, h2, h3 { line-height: 1.3; font-weight: 600; }
h1 { max-width: 28ch; margin: 0 0 20px; font-size: 34px; }
h2 { margin: 36px 0 16px; font-size: 25px; }
h3 { margin: 28px 0 12px; font-size: 20px; }
p, li { max-width: 88ch; }
p { margin: 0 0 18px; }
a { color: LinkText; text-underline-offset: 3px; }
img { display: block; max-width: 100%; height: auto; }
figure { margin: 32px 0; }
.figure-table { margin: 32px 0; font: 15px/1.6 system-ui, sans-serif; color: inherit; }
.figure-caption p { margin: 0 0 16px; }
.figure-caption p:last-child { margin-bottom: 0; }
.equation, .table-scroll { max-width: 100%; overflow-x: auto; margin: 24px 0; }
.equation { padding: 8px 0; }
table { width: 100%; border-collapse: collapse; font: 14px/1.5 system-ui, sans-serif; }
th, td { padding: 9px 12px; text-align: left; border-bottom: 1px solid #dce0e3; }
th { border-top: 1px solid #b6bec4; }
code { font: 0.88em ui-monospace, monospace; overflow-wrap: anywhere; }
@media (max-width: 760px) {
  main { margin-top: 28px; padding: 0 18px; }
  h1 { font-size: 28px; }
  .figure-table, .figure-table tbody, .figure-table tr, .figure-table td {
    display: block; width: 100% !important;
  }
  .figure-table td { padding: 0 !important; }
  .figure-caption { margin-top: 18px; }
}
@media print { :root { color-scheme: light; } main { margin: 0; } }
"""


def figure_layout(tokens, markdown):
    """Apply the side caption only to HTML, keeping the Markdown source portable."""
    result, index = [], 0
    while index < len(tokens):
        if tokens[index].type != "table_open":
            result.append(tokens[index])
            index += 1
            continue
        stop = next(i for i in range(index + 1, len(tokens)) if tokens[i].type == "table_close")
        block = tokens[index : stop + 1]
        cells = [token for token in block if token.type == "inline"]
        image = next(
            (
                child
                for cell in cells
                for child in cell.children or []
                if child.type == "image"
                and child.attrGet("src") == "figures/observation_limits_preview.png"
            ),
            None,
        )
        if image is None:
            result.extend(block)
            index = stop + 1
            continue
        if len(cells) != 4 or any(cell.content for cell in cells[:2]):
            raise ValueError("Figure 2 requires an empty header and one plot/caption row")
        image.attrSet("src", "figures/observation_limits.png")
        for key, value in {
            "width": "528",
            "height": "360",
            "aria-describedby": "figure-2-caption",
            "style": "display: block; width: 100%; max-width: 528px; height: auto !important; aspect-ratio: 22 / 15; object-fit: contain;",
        }.items():
            image.attrSet(key, value)
        picture = markdown.renderer.renderInline(cells[2].children, markdown.options, {})
        caption = markdown.renderer.renderInline(cells[3].children, markdown.options, {})
        table = Token("html_block", "", 0)
        table.content = (
            '<table id="figure-2" class="figure-table" role="presentation" width="100%" '
            'style="width: 100%; table-layout: fixed; border: 0; background: transparent;">'
            '<tbody><tr><td width="58%" style="width: 58%; padding: 0 24px 0 0; vertical-align: middle; border: 0; background: transparent;">'
            + picture
            + '</td><td width="42%" style="width: 42%; padding: 0; vertical-align: middle; border: 0; background: transparent;">'
            '<div id="figure-2-caption" class="figure-caption"><p>'
            + caption
            + "</p></div></td></tr></tbody></table>\n"
        )
        result.append(table)
        index = stop + 1
    return result


def render(out=None):
    source = (HERE / "manuscript.md").read_text()
    title = source.splitlines()[0].removeprefix("# ")

    def equation(match):
        mathml = convert(match.group(1).strip(), display="block")
        return '\n<div class="equation">' + mathml + "</div>\n"

    prepared = re.sub(r"(?m)^\\\[\s*\n(.*?)\n\\\]\s*$", equation, source, flags=re.DOTALL)
    markdown = MarkdownIt("commonmark", {"html": True, "xhtmlOut": True}).enable("table")
    markdown.renderer.rules["table_open"] = lambda *_: '<div class="table-scroll"><table>\n'
    markdown.renderer.rules["table_close"] = lambda *_: "</table></div>\n"
    tokens = figure_layout(markdown.parse(prepared), markdown)
    body = markdown.renderer.render(tokens, markdown.options, {})
    digest = hashlib.sha256(source.encode()).hexdigest()
    document = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<meta name="source-sha256" content="{digest}">\n'
        f"<title>{escape(title)}</title>\n<style>{STYLE}</style>\n</head>\n"
        f"<body>\n<main>\n{body}</main>\n</body>\n</html>\n"
    )
    target = Path(out) if out is not None else HERE / "manuscript.html"
    target.write_text(document)
    return target


if __name__ == "__main__":
    print(render())
