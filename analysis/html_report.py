"""
HTML report generation utilities with modern styling.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd


def _safe_slug(text: str) -> str:
    """Convert text to URL-safe slug."""
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "item"


def _df_to_html_table(df: pd.DataFrame) -> str:
    """Convert DataFrame to HTML table with proper formatting."""
    fmt_df = df.copy()
    
    def format_value(x):
        if isinstance(x, float):
            if np.isinf(x):
                return "∞" if x > 0 else "-∞"
            return f"{x:.4f}"
        return str(x)
    
    return fmt_df.to_html(
        index=False,
        escape=True,
        classes="tbl",
        border=0,
        formatters={col: format_value for col in fmt_df.columns},
    )


class HtmlReport:
    """HTML report builder with modern styling."""
    
    def __init__(self, report_path: Path, title: str):
        self.report_path = report_path
        self._f = open(report_path, "w", encoding="utf-8")
        self._write_header(title)

    def _write_header(self, title: str) -> None:
        """Write HTML header with modern CSS styling."""
        self._f.write(
            f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  --bg-primary: #f5f7fa;
  --bg-secondary: #ffffff;
  --bg-tertiary: #e8ecf1;
  --text-primary: #1a202c;
  --text-secondary: #4a5568;
  --text-muted: #718096;
  --border: #cbd5e0;
  --border-light: #e2e8f0;
  --accent-primary: #667eea;
  --accent-secondary: #764ba2;
  --success: #48bb78;
  --warning: #ed8936;
  --error: #f56565;
  --shadow-sm: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.08);
  --shadow-md: 0 4px 6px rgba(0,0,0,0.1), 0 2px 4px rgba(0,0,0,0.06);
  --shadow-lg: 0 10px 15px rgba(0,0,0,0.1), 0 4px 6px rgba(0,0,0,0.05);
}}

* {{ 
  box-sizing: border-box; 
  margin: 0;
  padding: 0;
}}

body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: var(--text-primary);
  line-height: 1.6;
  min-height: 100vh;
  padding: 20px;
}}

.container {{
  max-width: 1400px;
  margin: 0 auto;
  background: var(--bg-primary);
  border-radius: 16px;
  box-shadow: var(--shadow-lg);
  overflow: hidden;
}}

.header {{
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 40px;
  text-align: center;
}}

h1 {{
  font-size: 32px;
  font-weight: 700;
  margin-bottom: 16px;
  text-shadow: 0 2px 4px rgba(0,0,0,0.2);
}}

.header-subtitle {{
  font-size: 16px;
  opacity: 0.9;
  margin-top: 8px;
}}

.content {{
  padding: 32px;
}}

h2 {{ 
  font-size: 24px; 
  font-weight: 700;
  margin: 32px 0 16px;
  color: var(--accent-primary);
  border-bottom: 3px solid var(--accent-primary);
  padding-bottom: 8px;
}}

h3 {{ 
  font-size: 18px; 
  font-weight: 600;
  margin: 24px 0 12px;
  color: var(--text-primary);
}}

p {{ 
  margin: 12px 0;
  color: var(--text-secondary);
}}

a {{ 
  color: var(--accent-primary);
  text-decoration: none;
  transition: color 0.2s;
}}

a:hover {{ 
  color: var(--accent-secondary);
  text-decoration: underline;
}}

.topbar {{
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: center;
  margin-bottom: 24px;
  padding: 16px;
  background: var(--bg-secondary);
  border-radius: 12px;
  box-shadow: var(--shadow-sm);
}}

.badge {{
  display: inline-block;
  padding: 6px 14px;
  border-radius: 20px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  font-size: 12px;
  font-weight: 600;
  box-shadow: var(--shadow-sm);
}}

.badge.warning {{
  background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
}}

.tabbar {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 24px 0;
  padding: 16px;
  background: var(--bg-secondary);
  border-radius: 12px;
  box-shadow: var(--shadow-sm);
}}

.tabbtn {{
  background: var(--bg-tertiary);
  border: 2px solid transparent;
  color: var(--text-primary);
  padding: 10px 20px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 600;
  transition: all 0.3s;
}}

.tabbtn:hover {{
  background: var(--bg-primary);
  border-color: var(--accent-primary);
  transform: translateY(-2px);
  box-shadow: var(--shadow-sm);
}}

.tabbtn.active {{
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  box-shadow: var(--shadow-md);
}}

.tabcontent {{
  display: none;
  animation: fadeIn 0.3s;
}}

.tabcontent.active {{
  display: block;
}}

@keyframes fadeIn {{
  from {{ opacity: 0; transform: translateY(10px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}

.card {{
  background: var(--bg-secondary);
  border-radius: 12px;
  padding: 24px;
  box-shadow: var(--shadow-md);
  margin: 20px 0;
  transition: box-shadow 0.3s;
}}

.card:hover {{
  box-shadow: var(--shadow-lg);
}}

.grid {{
  display: grid;
  grid-template-columns: 1fr;
  gap: 20px;
}}

@media(min-width: 980px) {{
  .grid-2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    align-items: start;
  }}
}}

img.figure {{
  width: 100%;
  height: auto;
  border-radius: 12px;
  border: 2px solid var(--border-light);
  box-shadow: var(--shadow-sm);
  transition: transform 0.3s;
}}

img.figure:hover {{
  transform: scale(1.02);
  box-shadow: var(--shadow-md);
}}

table.tbl {{
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  overflow: hidden;
  border-radius: 12px;
  box-shadow: var(--shadow-sm);
  border: 1px solid var(--border-light);
}}

.tbl th, .tbl td {{
  padding: 12px 16px;
  text-align: left;
  border-bottom: 1px solid var(--border-light);
}}

.tbl th {{
  font-weight: 700;
  color: white;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  text-transform: uppercase;
  font-size: 11px;
  letter-spacing: 0.5px;
}}

.tbl tr:nth-child(even) {{
  background: var(--bg-tertiary);
}}

.tbl tr:hover {{
  background: #edf2f7;
}}

.tbl td {{
  color: var(--text-secondary);
  font-size: 13px;
}}

details.footnote {{
  margin-top: 16px;
  padding: 16px;
  border-radius: 12px;
  border: 2px solid var(--border-light);
  background: var(--bg-tertiary);
  transition: all 0.3s;
}}

details.footnote:hover {{
  border-color: var(--accent-primary);
  box-shadow: var(--shadow-sm);
}}

details.footnote summary {{
  cursor: pointer;
  color: var(--accent-primary);
  font-size: 14px;
  font-weight: 600;
  padding: 8px;
  border-radius: 6px;
  transition: background 0.2s;
}}

details.footnote summary:hover {{
  background: var(--bg-secondary);
}}

.small {{
  font-size: 13px;
  color: var(--text-muted);
  line-height: 1.6;
}}

.footer {{
  margin-top: 40px;
  padding: 24px;
  text-align: center;
  font-size: 13px;
  color: var(--text-muted);
  border-top: 2px solid var(--border-light);
  background: var(--bg-tertiary);
}}

.alert {{
  padding: 16px 20px;
  border-radius: 8px;
  margin: 16px 0;
  border-left: 4px solid;
}}

.alert-info {{
  background: #ebf8ff;
  border-color: #667eea;
  color: #2c5282;
}}

.alert-success {{
  background: #f0fff4;
  border-color: var(--success);
  color: #276749;
}}

.alert-warning {{
  background: #fffaf0;
  border-color: var(--warning);
  color: #7c2d12;
}}
</style>
</head>
<body>
<div class="container">
<div class="header">
<h1>{title}</h1>
<div class="header-subtitle">Comprehensive Bayesian Statistical Analysis</div>
</div>
<div class="content">
"""
        )

    def add_html(self, html: str) -> None:
        """Add raw HTML content."""
        self._f.write(html + "\n")

    def add_paragraph(self, text: str) -> None:
        """Add a paragraph."""
        self._f.write(f"<p>{text}</p>\n")

    def add_h2(self, text: str) -> None:
        """Add an H2 heading."""
        self._f.write(f"<h2>{text}</h2>\n")

    def add_h3(self, text: str) -> None:
        """Add an H3 heading."""
        self._f.write(f"<h3>{text}</h3>\n")

    def add_card_start(self) -> None:
        """Start a card container."""
        self._f.write('<div class="card">\n')

    def add_card_end(self) -> None:
        """End a card container."""
        self._f.write("</div>\n")

    def add_image(self, rel_path: str, alt: str) -> None:
        """Add an image."""
        self._f.write(f'<img class="figure" src="{rel_path}" alt="{alt}">\n')

    def start_tabs(self, tabs: Sequence[Tuple[str, str]], default_tab_id: str) -> None:
        """Initialize tabbed interface."""
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
  let tab = "{default_tab_id}";
  openTab(tab);
}});
</script>
"""
        )

    def open_tab_content(self, tab_id: str) -> None:
        """Open a tab content section."""
        self._f.write(f'<div class="tabcontent" id="{tab_id}">\n')

    def close_tab_content(self) -> None:
        """Close a tab content section."""
        self._f.write("</div>\n")

    def add_footnote(self, title: str, html_body: str) -> None:
        """Add an expandable footnote."""
        self._f.write(
            f"""
<details class="footnote">
  <summary>{title}</summary>
  <div class="small">{html_body}</div>
</details>
"""
        )

    def close(self) -> None:
        """Close the HTML document."""
        self._f.write(
            f"""
<div class="footer">
  Generated on {datetime.now().strftime("%B %d, %Y at %H:%M:%S")}
  <br>
  Powered by PyMC Bayesian Analysis
</div>
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
    Add a table with column documentation.
    
    Parameters
    ----------
    report : HtmlReport
        Report instance to add to
    df : pd.DataFrame
        Data to display
    col_guide : Dict[str, str]
        Column name -> description mapping
    title : str
        Title for the column guide section
    """
    report.add_html(_df_to_html_table(df))

    items = []
    for col, desc in col_guide.items():
        if col in df.columns:
            items.append(f"<li><b>{col}</b>: {desc}</li>")

    if not items:
        return

    report.add_html(
        f"""
<div class="small" style="margin-top:12px; padding:12px; background: var(--bg-tertiary); border-radius:8px;">
  <b style="color: var(--accent-primary);">{title}:</b>
  <ul style="margin:8px 0 0 24px; line-height: 1.8;">
    {''.join(items)}
  </ul>
</div>
"""
    )