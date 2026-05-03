"""Model structure SVG renderer (Section 4).

Generates an interactive SVG showing the model's high-level architecture
with module flow, repeat counts, and sub-structure annotations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.report.report_types import ReportContext, BlockDetail


# ── Color palette for module types ────────────────────────────────────────────

_MODULE_COLORS = {
    "input":        ("#f8fafc", "#cbd5e1"),
    "embedding":    ("#eff6ff", "#93c5fd"),
    "block":        ("#fefcff", "#d8b4fe"),
    "output":       ("#eef2ff", "#a5b4fc"),
}

_SUB_COLORS = {
    "norm":      ("#f8fafc", "#cbd5e1"),
    "attention": ("#ecfeff", "#67e8f9"),
    "residual":  ("#f1f5f9", "#cbd5e1"),
    "router":    ("#fff7ed", "#fdba74"),
    "dispatch":  ("#fef2f2", "#fca5a5"),
    "experts":   ("#eef2ff", "#a5b4fc"),
    "combine":   ("#fef2f2", "#fca5a5"),
    "shared":    ("#fdf2f8", "#f9a8d4"),
}


def _block_type(name: str) -> str:
    n = (name or "").lower()
    if "embed" in n or "tok_embeddings" in n:
        return "embedding"
    if "input" in n:
        return "input"
    if "output" in n or "lm_head" in n or "norm" in n:
        return "output"
    if "block" in n or "layer" in n or "moe" in n:
        return "block"
    return "block"


def render_structure_svg(
    blocks: "list[BlockDetail]",
    *,
    phase: str = "decode",
    width: int = 1200,
    height: int = 360,
) -> str:
    """Generate a model structure SVG with Forward/Backward tab switching.

    Parameters
    ----------
    blocks : list[BlockDetail]
        Top-level model blocks from ReportContext.
    phase : str
        Phase label ("prefill" | "decode" | "train").
    width, height : int
        SVG canvas dimensions.

    Returns
    -------
    str
        SVG markup ready to embed in HTML.
    """
    lines: list[str] = []

    def add(s: str) -> None:
        lines.append(s)

    viewBox = f"0 0 {width} {height}"

    # ── Arrow marker def ──
    add('<defs>')
    add('<marker id="arrowHead" markerHeight="10" markerWidth="10" '
        'orient="auto" refX="7" refY="3">')
    add('<path d="M0,0 L0,6 L8,3 z" fill="#94a3b8"/>')
    add('</marker>')
    add('</defs>')

    # ── Background ──
    add(f'<rect fill="#fbfdff" height="{height - 16}" rx="28" stroke="#dbe4ee" '
        f'width="{width - 16}" x="8" y="8"/>')

    # ── Title ──
    phase_label = phase.upper() if phase else ""
    add(f'<text fill="#0f172a" font-size="20" font-weight="800" x="26" y="34">'
        f'Forward / Main Path — {phase_label}</text>')
    add(f'<text fill="#64748b" font-size="12" font-weight="600" x="26" y="56">'
        f'SVG 连线化模型结构图：展示主干流、重复层和子结构关系</text>')

    # ── Classify blocks ──
    input_blocks = [b for b in blocks if _block_type(b.name) == "input" or _block_type(b.name) == "embedding"]
    main_blocks = [b for b in blocks if _block_type(b.name) == "block"]
    output_blocks = [b for b in blocks if _block_type(b.name) == "output"]

    # If no explicit classification, use first/last blocks
    if not input_blocks and blocks:
        input_blocks = [blocks[0]]
    if not output_blocks and len(blocks) > 1:
        output_blocks = [blocks[-1]]
    if not main_blocks and len(blocks) > 2:
        main_blocks = blocks[1:-1]

    # ── Layout constants ──
    box_y = 100
    box_h = 140
    arrow_y = box_y + box_h // 2
    gap = 24

    # ── Calculate X positions dynamically ──
    input_w = 140
    embed_w = 180 if input_blocks else 0
    output_w = 210

    # Main block area
    main_count = len(main_blocks)
    if main_count > 0:
        main_w = 360
        main_x_start = 26 + input_w + gap + embed_w + gap
        sub_h = 24
        sub_gap = 4
        sub_pad_y = 10
        # Adjust block height for substructures
        subs_total = sum(
            len([ss for ss in b.sub_structures if ss.name])
            for b in main_blocks
        )
        # Use first main block's sub-structures
        sub_count = max(
            len([ss for ss in main_blocks[0].sub_structures if ss.name])
            if main_blocks else 0, 1
        )
        main_h = box_h + sub_pad_y * 2 + (sub_h + sub_gap) * min(sub_count, 6) + 30
    else:
        main_w = 0
        main_x_start = 26 + input_w + gap + embed_w + gap
        main_h = box_h

    output_x = main_x_start + main_w + gap if main_count > 0 else main_x_start

    # ── Render blocks ──
    prev_right = 26

    # --- Input block ---
    if input_blocks:
        b = input_blocks[0]
        fill, stroke = _MODULE_COLORS["input"]
        x = prev_right
        add(f'<rect fill="{fill}" height="{box_h}" rx="22" stroke="{stroke}" '
            f'stroke-width="2" width="{input_w}" x="{x}" y="{box_y}"/>')
        add(f'<text fill="#0f172a" font-size="20" font-weight="800" '
            f'x="{x + 16}" y="{box_y + 30}">Input</text>')
        add(f'<text fill="#475569" font-size="12" font-weight="600" '
            f'x="{x + 16}" y="{box_y + 54}">B×S</text>')
        add(f'<text fill="#0f172a" font-size="22" font-weight="850" '
            f'x="{x + 16}" y="{box_y + 120}">{b.total_ms:.3f} ms</text>')
        prev_right = x + input_w
        # Arrow
        next_x = prev_right + gap
        add(f'<line marker-end="url(#arrowHead)" stroke="#94a3b8" stroke-width="3" '
            f'x1="{prev_right}" x2="{next_x - gap}" y1="{arrow_y}" y2="{arrow_y}"/>')

    # --- Embedding block ---
    if input_blocks and len(input_blocks) > 1 and _block_type(input_blocks[1].name) == "embedding":
        b = input_blocks[1]
        fill, stroke = _MODULE_COLORS["embedding"]
        x = prev_right + gap
        add(f'<rect fill="{fill}" height="{box_h}" rx="22" stroke="{stroke}" '
            f'stroke-width="2" width="{embed_w}" x="{x}" y="{box_y}"/>')
        add(f'<text fill="#0f172a" font-size="20" font-weight="800" '
            f'x="{x + 16}" y="{box_y + 30}">Embedding</text>')
        add(f'<text fill="#475569" font-size="12" font-weight="600" '
            f'x="{x + 16}" y="{box_y + 54}">hidden=7168</text>')
        add(f'<text fill="#0f172a" font-size="22" font-weight="850" '
            f'x="{x + 16}" y="{box_y + 120}">{b.total_ms:.3f} ms</text>')
        prev_right = x + embed_w
        # Arrow
        add(f'<line marker-end="url(#arrowHead)" stroke="#94a3b8" stroke-width="3" '
            f'x1="{prev_right}" x2="{main_x_start}" y1="{arrow_y}" y2="{arrow_y}"/>')

    # --- Main block(s) ---
    for mi, b in enumerate(main_blocks):
        fill, stroke = _MODULE_COLORS["block"]
        x = main_x_start

        repeat_label = f" × {b.repeat}" if b.repeat > 1 else ""
        add(f'<rect fill="{fill}" height="{main_h}" rx="22" stroke="{stroke}" '
            f'stroke-width="2" width="{main_w}" x="{x}" y="{box_y - 42}"/>')
        add(f'<text fill="#0f172a" font-size="20" font-weight="800" '
            f'x="{x + 16}" y="{box_y - 12}">{b.name}{repeat_label}</text>')
        # Model params
        params_text = ""
        if b.model_params:
            pm = b.model_params
            parts = []
            if pm.get("num_experts"):
                parts.append(f"routed={pm['num_experts']}")
            if pm.get("active_per_token"):
                parts.append(f"active/token={pm['active_per_token']}")
            params_text = " · ".join(parts)
        add(f'<text fill="#475569" font-size="12" font-weight="600" '
            f'x="{x + 16}" y="{box_y + 12}">{params_text}</text>')
        add(f'<text fill="#0f172a" font-size="22" font-weight="850" '
            f'x="{x + 16}" y="{box_y + main_h - 20}">{b.total_ms:.3f} ms</text>')

        # Sub-structures as pill boxes
        sub_y = box_y + 36
        col = 0
        row = 0
        cols_per_row = 3
        sub_x_base = x + 16

        for ss in b.sub_structures:
            name_lower = (ss.name or "").lower()
            # Determine sub category
            sub_cat = "attention"
            if "norm" in name_lower:
                sub_cat = "norm"
            elif "attn" in name_lower or "attention" in name_lower:
                sub_cat = "attention"
            elif "residual" in name_lower or "add" in name_lower:
                sub_cat = "residual"
            elif "router" in name_lower or "gate" in name_lower:
                sub_cat = "router"
            elif "dispatch" in name_lower:
                sub_cat = "dispatch"
            elif "expert" in name_lower or "ffn" in name_lower or "mlp" in name_lower:
                sub_cat = "experts"
            elif "combine" in name_lower:
                sub_cat = "combine"
            elif "shared" in name_lower:
                sub_cat = "shared"

            sub_fill, sub_stroke = _SUB_COLORS.get(sub_cat, ("#f8fafc", "#cbd5e1"))
            pill_w = 80
            pill_h = 24
            pill_x = sub_x_base + col * (pill_w + 8)
            pill_y = sub_y + row * (pill_h + 6)

            if pill_y + pill_h > box_y + main_h - 30:
                row = 0
                col += 1
                if col >= cols_per_row:
                    break
                pill_x = sub_x_base + col * (pill_w + 8)
                pill_y = sub_y + row * (pill_h + 6)

            add(f'<rect fill="{sub_fill}" height="{pill_h}" rx="12" '
                f'stroke="{sub_stroke}" width="{pill_w}" x="{pill_x}" y="{pill_y}"/>')
            add(f'<text fill="#334155" font-size="11" font-weight="700" '
                f'x="{pill_x + 10}" y="{pill_y + 16}">{ss.name or "?"}</text>')
            col += 1
            if col >= cols_per_row:
                col = 0
                row += 1

        prev_right = x + main_w

    # Arrow from main block to output
    if main_count > 0 and output_blocks:
        add(f'<line marker-end="url(#arrowHead)" stroke="#94a3b8" stroke-width="3" '
            f'x1="{prev_right}" x2="{output_x}" y1="{arrow_y}" y2="{arrow_y}"/>')
        mid = (prev_right + output_x) // 2
        add(f'<text fill="#64748b" font-size="11" text-anchor="middle" '
            f'x="{mid}" y="{arrow_y - 10}">residual stream</text>')

    # --- Output block ---
    if output_blocks:
        b = output_blocks[0]
        fill, stroke = _MODULE_COLORS["output"]
        add(f'<rect fill="{fill}" height="{box_h}" rx="22" stroke="{stroke}" '
            f'stroke-width="2" width="{output_w}" x="{output_x}" y="{box_y}"/>')
        add(f'<text fill="#0f172a" font-size="20" font-weight="800" '
            f'x="{output_x + 16}" y="{box_y + 30}">Output</text>')
        add(f'<text fill="#475569" font-size="12" font-weight="600" '
            f'x="{output_x + 16}" y="{box_y + 54}">Final Norm + LM Head</text>')
        add(f'<text fill="#0f172a" font-size="22" font-weight="850" '
            f'x="{output_x + 16}" y="{box_y + 120}">{b.total_ms:.3f} ms</text>')

    return f'<svg class="arch-svg" viewBox="{viewBox}" xmlns="http://www.w3.org/2000/svg">\n' + \
           "\n".join(lines) + "\n</svg>"


def render_structure_html(blocks, phase="decode") -> str:
    """Render the full structure HTML section with SVG and tab UI.

    Parameters
    ----------
    blocks : list[BlockDetail]
        Top-level model blocks.
    phase : str
        Phase label.

    Returns
    -------
    str
        Complete HTML for the structure section.
    """
    svg_fwd = render_structure_svg(blocks, phase=phase)

    return f"""
<div class="section">
    <h2>模型结构（交互式 SVG 视图）</h2>
    <div class="viz-tabs" style="display:flex;gap:0;margin-bottom:0">
        <button class="viz-tab active" onclick="switchViz('viz-forward',this)"
         style="padding:8px 20px;border:1px solid #3b82f6;background:#3b82f6;color:#fff;
         border-radius:8px 8px 0 0;cursor:pointer;font-size:13px;font-weight:600">
         Forward 结构视图</button>
        <button class="viz-tab" onclick="switchViz('viz-backward',this)"
         style="padding:8px 20px;border:1px solid #e2e8f0;background:#f8fafc;color:#64748b;
         border-radius:8px 8px 0 0;cursor:pointer;font-size:13px;font-weight:600;
         border-bottom:none">
         Backward 结构视图</button>
    </div>
    <div class="viz-panel active" id="viz-forward" style="display:block">
        {svg_fwd}
    </div>
    <div class="viz-panel" id="viz-backward" style="display:none;padding:40px;text-align:center;
         background:#f8fafc;border:1px solid #e2e8f0;border-radius:0 8px 8px 8px;
         color:#64748b;font-size:14px">
        Backward 结构视图将在支持逆向图捕获后可用。<br>
        <small style="color:#94a3b8">需要 train_backward 阶段的 GraphHierarchy 数据</small>
    </div>
    <div style="color:#64748b;font-size:12px;margin-top:8px">
        点击上方标签即可切换结构视图。SVG 图中使用连线显式表示主路径，适合培训、评审和报告演示。
    </div>
</div>"""
