# reporting.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd


def safe_slug(text: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "item"


def df_to_html_table(df: pd.DataFrame) -> str:
    fmt_df = df.copy()

    def format_value(x):
        if isinstance(x, str):
            return x
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return ""
        except Exception:
            pass

        if isinstance(x, (int, np.integer)):
            return str(int(x))

        if isinstance(x, (float, np.floating)):
            x = float(x)
            if np.isinf(x):
                return "∞" if x > 0 else "-∞"
            ax = abs(x)
            if ax == 0:
                return "0"
            if ax >= 10:
                return f"{x:.2f}"
            if ax >= 1:
                return f"{x:.3f}"
            if ax >= 0.1:
                return f"{x:.4f}"
            if ax >= 0.01:
                return f"{x:.5f}"
            if ax >= 0.001:
                return f"{x:.6f}"
            if ax >= 1e-4:
                return f"{x:.7f}"
            return f"{x:.2e}"

        return str(x)

    table_html = fmt_df.to_html(
        index=False,
        escape=True,
        classes="tbl",
        border=0,
        formatters={col: format_value for col in fmt_df.columns},
    )
    return f'<div class="table-wrapper">{table_html}</div>'


class HtmlReport:
    def __init__(self, report_path: Path, title: str):
        self.report_path = report_path
        self._f = open(report_path, "w", encoding="utf-8")
        self._write_header(title)

    def _write_header(self, title: str) -> None:
        self._f.write(
            f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="preconnect" href="https://rsms.me/">
<link rel="stylesheet" href="https://rsms.me/inter/inter.css">
<style>
:root{{
  --bg0:#F8F9FA;
  --bg1:#FFFFFF;
  --card:#FFFFFF;
  --text:#1A1A1A;
  --muted:#5A5A5A;
  --border:#E0E0E0;
  --accent:#0066CC;
  --accent-light:#E6F2FF;
  --good:#059669;
  --warn:#D97706;
  --bad:#DC2626;
  --shadow:rgba(0,0,0,.08);
  font-family: Inter, sans-serif;
  font-feature-settings: 'liga' 1, 'calt' 1;
}}

@supports (font-variation-settings: normal) {{
  :root {{ font-family: InterVariable, sans-serif; }}
}}

*{{box-sizing:border-box}}
body{{
  margin:0;
  color:var(--text);
  background:var(--bg0);
  line-height:1.6;
}}

.container{{
  max-width:1280px;
  margin:0 auto;
  padding:24px 20px 80px;
}}

h1{{
  font-size:28px;
  margin:0 0 12px;
  font-weight:700;
  color:var(--text);
}}
h2{{
  font-size:22px;
  margin:24px 0 12px;
  font-weight:600;
  color:var(--text);
}}
h3{{
  font-size:17px;
  margin:20px 0 10px;
  font-weight:600;
  color:var(--text);
}}
p{{
  line-height:1.6;
  color:var(--muted);
  margin:10px 0;
}}
a{{
  color:var(--accent);
  text-decoration:none;
}}
a:hover{{
  text-decoration:underline;
}}

.topbar{{
  display:flex;
  flex-wrap:wrap;
  gap:10px;
  align-items:center;
  margin-bottom:16px;
  padding-bottom:16px;
  border-bottom:1px solid var(--border);
}}
.badge{{
  display:inline-block;
  padding:4px 12px;
  border-radius:6px;
  background:var(--bg0);
  border:1px solid var(--border);
  color:var(--muted);
  font-size:13px;
  font-weight:500;
}}

.tabbar{{
  display:flex;
  flex-wrap:wrap;
  gap:8px;
  margin:16px 0;
  padding:6px;
  background:var(--bg0);
  border-radius:10px;
  border:1px solid var(--border);
}}
.tabbtn{{
  background:transparent;
  border:none;
  color:var(--muted);
  padding:8px 16px;
  border-radius:6px;
  cursor:pointer;
  font-size:14px;
  font-weight:500;
  transition:all .15s ease;
}}
.tabbtn:hover{{
  background:var(--card);
  color:var(--text);
}}
.tabbtn.active{{
  background:var(--accent);
  color:#FFFFFF;
  box-shadow:0 2px 8px rgba(0,102,204,.3);
}}

.tabcontent{{display:none}}
.tabcontent.active{{display:block}}

.card{{
  background:var(--card);
  border:1px solid var(--border);
  border-radius:12px;
  padding:20px;
  box-shadow:0 2px 8px var(--shadow);
  margin:16px 0;
}}

.grid{{display:grid;grid-template-columns:1fr;gap:16px}}
@media(min-width:980px){{
  .grid-2{{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:16px;
    align-items:start;
  }}
}}

img.figure{{
  width:100%;
  height:auto;
  border-radius:8px;
  border:1px solid var(--border);
  background:var(--bg0);
}}

.table-wrapper{{
  overflow-x:auto;
  margin:12px 0;
  border-radius:8px;
  border:1px solid var(--border);
}}

table.tbl{{
  width:100%;
  min-width:600px;
  border-collapse:collapse;
  background:var(--card);
  font-size:14px;
}}
.tbl th,.tbl td{{
  padding:12px 14px;
  text-align:left;
  border-bottom:1px solid var(--border);
}}
.tbl th{{
  font-weight:600;
  color:var(--text);
  background:var(--bg0);
  white-space:nowrap;
}}
.tbl td{{
  color:var(--text);
}}
.tbl tr:last-child td{{
  border-bottom:none;
}}
.tbl tr:hover td{{
  background:var(--bg0);
}}

details.footnote{{
  margin-top:12px;
  padding:12px 16px;
  border-radius:8px;
  border:1px solid var(--border);
  background:var(--bg0);
}}
details.footnote summary{{
  cursor:pointer;
  color:var(--text);
  font-size:14px;
  font-weight:500;
}}
details.footnote[open] summary{{
  margin-bottom:8px;
}}
.small{{
  font-size:13px;
  color:var(--muted);
}}
.footer{{
  margin-top:24px;
  padding-top:16px;
  border-top:1px solid var(--border);
  font-size:13px;
  color:var(--muted);
}}
</style>
</head>
<body>
<div class="container">
"""
        )
        self._f.write(f"<h1>{title}</h1>\n")

    def add_html(self, html: str) -> None:
        self._f.write(html + "\n")

    def add_paragraph(self, text: str) -> None:
        self._f.write(f"<p>{text}</p>\n")

    def add_h2(self, text: str) -> None:
        self._f.write(f"<h2>{text}</h2>\n")

    def add_h3(self, text: str) -> None:
        self._f.write(f"<h3>{text}</h3>\n")

    def add_card_start(self) -> None:
        self._f.write('<div class="card">\n')

    def add_card_end(self) -> None:
        self._f.write("</div>\n")

    def add_image(self, rel_path: str, alt: str) -> None:
        self._f.write(f'<img class="figure" src="{rel_path}" alt="{alt}">\n')

    def start_tabs(self, tabs: Sequence[Tuple[str, str]], default_tab_id: str) -> None:
        btns = []
        for tab_id, label in tabs:
            btns.append(f'<button class="tabbtn" data-tab="{tab_id}">{label}</button>')
        self._f.write('<div class="tabbar">\n' + "\n".join(btns) + "\n</div>\n")
        self._f.write(
            f"""
<script>
function openTab(tabId) {{
  const tabs = document.querySelectorAll('.tabcontent');
  const btns = document.querySelectorAll('.tabbtn');
  tabs.forEach(t => t.classList.remove('active'));
  btns.forEach(b => b.classList.remove('active'));
  const t = document.getElementById(tabId);
  if (t) t.classList.add('active');
  const b = document.querySelector(`.tabbtn[data-tab="${{tabId}}"]`);
  if (b) b.classList.add('active');
}}
document.addEventListener('click', (e) => {{
  const btn = e.target.closest('.tabbtn');
  if (!btn) return;
  openTab(btn.getAttribute('data-tab'));
}});
document.addEventListener('DOMContentLoaded', () => {{
    // Adjust container width based on widest table
    const tables = document.querySelectorAll('table.tbl');
    let maxWidth = 800;
    tables.forEach(tbl => {{
        const rect = tbl.getBoundingClientRect();
        if (rect.width > maxWidth) {{
            maxWidth = rect.width;
        }}
    }});
    const container = document.querySelector('.container');
    if (container && maxWidth > 800) {{
        container.style.maxWidth = Math.min(maxWidth + 64, window.innerWidth - 32) + 'px';
    }}
    // Open default tab
  let tab = "{default_tab_id}";
  openTab(tab);
}});
</script>
"""
        )

    def open_tab_content(self, tab_id: str) -> None:
        self._f.write(f'<div class="tabcontent" id="{tab_id}">\n')

    def close_tab_content(self) -> None:
        self._f.write("</div>\n")

    def add_footnote(self, title: str, html_body: str) -> None:
        self._f.write(
            f"""
<details class="footnote">
  <summary>{title}</summary>
  <div class="small">{html_body}</div>
</details>
"""
        )

    def close(self) -> None:
        self._f.write(
            f"""
<div class="footer">
Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
</div>
</div>
</body>
</html>
"""
        )
        self._f.close()


def add_table_with_column_guide(
    report: HtmlReport,
    df: pd.DataFrame,
    col_guide: Dict[str, str],
    title: str = "Column guide",
) -> None:
    """
    Writes an HTML table and then a small column-name explanation beneath it.
    Only columns that exist in df are shown.
    """
    report.add_html(df_to_html_table(df))

    items = []
    for col, desc in col_guide.items():
        if col in df.columns:
            items.append(f"<li><b>{col}</b>: {desc}</li>")

    if not items:
        return

    report.add_html(
        f"""
<div class="small" style="margin-top:8px">
  <b>{title}:</b>
  <ul style="margin:6px 0 0 18px">
    {''.join(items)}
  </ul>
</div>
"""
    )
