import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
RAW_SHOOTING = ROOT / "data" / "raw" / "shooting" / "2025-26.csv"

NAVY = "#0B1F3A"
INK = "#14213D"
BLUE = "#2563EB"
CYAN = "#06B6D4"
GOLD = "#D4A017"
GREEN = "#16A34A"
RED = "#DC2626"
VIOLET = "#7C3AED"
PINK = "#DB2777"
MUTED = "#64748B"
GRID = "rgba(148, 163, 184, 0.18)"
TEAM_COL = "Team (2025-26)"

# House accent: basketball orange, used everywhere Streamlit's default
# red would otherwise leak through (buttons, sliders, active tab, etc.)
ORANGE = "#EA580C"
EMBER = "#F97316"
COURT = "#F4EEE2"
ICE = "#0284C7"
HEAT = "#DC2626"

# Real primary team colors, so team badges read as team colors, not a
# single reused theme accent.
TEAM_COLORS = {
    "ATL": "#E03A3E", "BKN": "#3B3B3B", "BOS": "#007A33", "CHA": "#1D1160",
    "CHI": "#CE1141", "CLE": "#860038", "DAL": "#0064B1", "DEN": "#4D90CD",
    "DET": "#C8102E", "GSW": "#1D428A", "HOU": "#CE1141", "IND": "#002D62",
    "LAC": "#C8102E", "LAL": "#552583", "MEM": "#5D76A9", "MIA": "#98002E",
    "MIL": "#00471B", "MIN": "#236192", "NOP": "#B4975A", "NYK": "#006BB6",
    "OKC": "#007AC1", "ORL": "#0077C0", "PHI": "#ED174C", "PHX": "#E56020",
    "POR": "#E03A3E", "SAC": "#5A2D81", "SAS": "#8A8D8F", "TOR": "#753BBD",
    "UTA": "#002B5C", "WAS": "#E31837",
}


def team_color(abbr):
    return TEAM_COLORS.get(abbr, MUTED)


def team_pill_html(abbr, suffix="2025-26", extra_style=""):
    color = team_color(abbr)
    return (
        f'<span class="team-pill" style="color:{color}; border-color:{color}66; '
        f'background:{color}1A; {extra_style}">{abbr} · {suffix}</span>'
    )


st.set_page_config(
    page_title="NBA 2026-27 Player Predictions",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# NBA.com-style shot zones (from leaguedashplayershotlocations, "By Zone").
# Used to paint the half-court hot/cold shot chart.
ZONES = [
    "restricted_area",
    "in_the_paint_non_ra",
    "mid_range",
    "left_corner_3",
    "right_corner_3",
    "above_the_break_3",
]

ZONE_LABELS = {
    "restricted_area": "Restricted Area",
    "in_the_paint_non_ra": "Paint (Non-RA)",
    "mid_range": "Mid-Range",
    "left_corner_3": "Left Corner 3",
    "right_corner_3": "Right Corner 3",
    "above_the_break_3": "Above the Break 3",
}


@st.cache_data
def load_data():
    scores = pd.read_csv(OUTPUTS / "player_scores_2025_26.csv")
    predictions = pd.read_csv(OUTPUTS / "predictions_2026_27.csv")
    midrange = pd.read_csv(OUTPUTS / "midrange_leaders_2025_26.csv")
    metrics = pd.read_csv(OUTPUTS / "model_metrics.csv")
    selected = pd.read_csv(OUTPUTS / "selected_models.csv")

    player = scores.merge(
        predictions,
        on=["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"],
        how="inner",
    )

    player = player.merge(
        midrange[
            [
                "PLAYER_ID",
                "mid_range_fga",
                "mid_range_fg_pct",
                "MIDRANGE_SCORE",
            ]
        ],
        on="PLAYER_ID",
        how="left",
    )

    zone_avg = {}
    if RAW_SHOOTING.exists():
        shooting = pd.read_csv(RAW_SHOOTING)
        zone_cols = ["PLAYER_ID"] + [f"{z}_{s}" for z in ZONES for s in ("fgm", "fga", "fg_pct")]
        shooting = shooting[[c for c in zone_cols if c in shooting.columns]]

        # mid_range_fga / mid_range_fg_pct already came in from the midrange
        # leaderboard merge above — drop the duplicates so the merge below
        # doesn't silently rename both to *_x / *_y.
        to_merge = shooting.drop(
            columns=[c for c in shooting.columns if c != "PLAYER_ID" and c in player.columns]
        )
        player = player.merge(to_merge, on="PLAYER_ID", how="left")

        for zone in ZONES:
            fgm_sum = shooting.get(f"{zone}_fgm", pd.Series(dtype=float)).sum()
            fga_sum = shooting.get(f"{zone}_fga", pd.Series(dtype=float)).sum()
            zone_avg[zone] = (fgm_sum / fga_sum) if fga_sum else 0.0

    for stat in ["PTS", "REB", "AST", "FG3M", "STL", "BLK"]:
        player[f"CHANGE_{stat}"] = player[f"PROJECTED_{stat}"] - player[stat]

    selected_map = dict(zip(selected["target"], selected["selected_model"]))
    model_rows = []
    for target in ["PTS", "REB", "AST", "FG3M", "STL", "BLK"]:
        target_metrics = metrics[metrics["target"] == target]
        baseline = target_metrics[target_metrics["model"] == "Previous Season Baseline"].iloc[0]
        selected_name = selected_map[target]
        chosen = target_metrics[target_metrics["model"] == selected_name].iloc[0]
        improvement = (baseline["mae"] - chosen["mae"]) / baseline["mae"]
        model_rows.append(
            {
                "Target": target,
                "Selected Model": selected_name,
                "Baseline MAE": baseline["mae"],
                "Selected MAE": chosen["mae"],
                "RMSE": chosen["rmse"],
                "R²": chosen["r2"],
                "MAE Improvement": improvement,
            }
        )

    return player, pd.DataFrame(model_rows), metrics, zone_avg


players, model_summary, model_metrics, zone_league_avg = load_data()


st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Manrope:wght@400;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Manrope', 'Arial', sans-serif;
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(148, 163, 184, 0.18);
        }

        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            font-family: 'Oswald', sans-serif;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            font-size: 1rem;
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(14px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes gradientFlow {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        @keyframes pulseGlow {
            0%, 100% { box-shadow: 0 0 0 0 rgba(22, 163, 74, 0.45); opacity: 1; }
            50% { box-shadow: 0 0 0 6px rgba(22, 163, 74, 0); opacity: 0.75; }
        }

        @keyframes bounceBall {
            0%, 100% { transform: translateY(0) rotate(0deg); }
            50% { transform: translateY(-7px) rotate(14deg); }
        }

        @keyframes growBar {
            from { transform: scaleX(0); }
            to { transform: scaleX(1); }
        }

        @keyframes popIn {
            from { opacity: 0; transform: scale(0.94) translateY(10px); }
            to { opacity: 1; transform: scale(1) translateY(0); }
        }

        .hero {
            padding: 1.9rem 2.1rem;
            border-radius: 20px;
            position: relative;
            overflow: hidden;
            background:
                repeating-linear-gradient(135deg, rgba(255,255,255,0.035) 0px, rgba(255,255,255,0.035) 2px, transparent 2px, transparent 26px),
                radial-gradient(circle at 88% -20%, rgba(249,115,22,0.4), transparent 55%),
                linear-gradient(135deg, #0A0F1E 0%, #14213D 55%, #0A0F1E 100%);
            background-size: auto, auto, 260% 260%;
            animation: gradientFlow 18s ease infinite, fadeInUp 0.6s ease both;
            margin-bottom: 1.2rem;
            box-shadow: 0 18px 40px rgba(10, 15, 30, 0.35);
        }

        .hero::after {
            content: "";
            position: absolute;
            left: 0; right: 0; bottom: 0;
            height: 4px;
            background: linear-gradient(90deg, #EA580C, #F97316 45%, #FDBA74 75%, transparent);
        }

        .hero h1 {
            margin: 0;
            font-family: 'Oswald', sans-serif;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.015em;
            font-size: 2.3rem;
            color: #FFFFFF;
        }

        .hero h1 .accent {
            color: #F97316;
        }

        .hero p {
            margin: 0.5rem 0 0 0;
            color: #AEB9D4;
            font-size: 1rem;
            max-width: 640px;
        }

        .hero-ball {
            display: inline-block;
            animation: bounceBall 2.4s ease-in-out infinite;
            margin-right: 0.15rem;
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            margin-top: 0.9rem;
            margin-right: 0.55rem;
            padding: 0.3rem 0.75rem;
            border-radius: 999px;
            border: 1px solid rgba(249, 115, 22, 0.4);
            background: rgba(249, 115, 22, 0.12);
            color: #FDBA74;
            font-size: 0.78rem;
            font-weight: 600;
        }

        .info-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.3rem 0.75rem;
            border-radius: 999px;
            border: 1px solid rgba(234, 88, 12, 0.3);
            background: rgba(234, 88, 12, 0.08);
            color: #C2410C;
            font-size: 0.82rem;
            font-weight: 600;
        }

        .live-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #22C55E;
            display: inline-block;
            animation: pulseGlow 1.8s ease-in-out infinite;
        }

        .kpi-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 0.9rem;
            margin: 0.2rem 0 1.3rem;
        }

        .kpi-card {
            position: relative;
            padding: 0.95rem 1.1rem 0.95rem 1.35rem;
            border-radius: 14px;
            background: #FFFFFF;
            border: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            overflow: hidden;
            transition: transform 0.22s ease, box-shadow 0.22s ease;
            animation: fadeInUp 0.5s ease both;
        }

        .kpi-card::before {
            content: "";
            position: absolute;
            left: 0; top: 0; bottom: 0;
            width: 5px;
            background: var(--kpi-color, #EA580C);
        }

        .kpi-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 14px 26px rgba(15, 23, 42, 0.12);
        }

        .kpi-label {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: #64748B;
        }

        .kpi-value {
            font-family: 'Oswald', sans-serif;
            font-size: 2.15rem;
            font-weight: 600;
            color: #14213D;
            line-height: 1.15;
            font-variant-numeric: tabular-nums;
            margin-top: 0.1rem;
        }

        .kpi-sub {
            font-size: 0.8rem;
            color: #64748B;
            margin-top: 0.2rem;
        }

        @keyframes heatPulse {
            0%, 100% { filter: drop-shadow(0 0 0px rgba(234, 88, 12, 0)); }
            50% { filter: drop-shadow(0 0 9px rgba(234, 88, 12, 0.9)); }
        }

        @keyframes coldPulse {
            0%, 100% { filter: drop-shadow(0 0 0px rgba(2, 132, 199, 0)); }
            50% { filter: drop-shadow(0 0 7px rgba(2, 132, 199, 0.75)); }
        }

        .zone-hot { animation: heatPulse 1.7s ease-in-out infinite; }
        .zone-cold { animation: coldPulse 2.6s ease-in-out infinite; }

        .team-pill {
            display: inline-block;
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            border: 1px solid;
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            vertical-align: middle;
            margin-left: 0.5rem;
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }

        .team-pill:hover {
            transform: translateY(-1px) scale(1.05);
            box-shadow: 0 4px 10px rgba(20, 33, 61, 0.2);
        }

        .fire-badge {
            display: inline-block;
            animation: bounceBall 1.6s ease-in-out infinite;
        }

        .score-bars {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            margin-top: 0.35rem;
            animation: fadeInUp 0.5s ease both;
        }

        .score-bar-row {
            display: grid;
            grid-template-columns: 92px 1fr 38px;
            align-items: center;
            gap: 0.6rem;
        }

        .score-bar-name {
            font-size: 0.78rem;
            color: #64748B;
            font-weight: 600;
        }

        .score-bar-track {
            position: relative;
            height: 9px;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.16);
            overflow: hidden;
        }

        .score-bar-fill {
            position: absolute;
            inset: 0;
            border-radius: 999px;
            transform-origin: left center;
            animation: growBar 0.9s cubic-bezier(0.22, 1, 0.36, 1) both;
        }

        .score-bar-value {
            font-size: 0.76rem;
            font-weight: 700;
            text-align: right;
        }

        .spotlight-card {
            border: 1px solid rgba(212, 160, 23, 0.35);
            border-radius: 16px;
            padding: 1rem 1.2rem;
            background: linear-gradient(135deg, rgba(212, 160, 23, 0.12), rgba(37, 99, 235, 0.06));
            animation: popIn 0.5s cubic-bezier(0.22, 1, 0.36, 1) both;
            margin-bottom: 1.1rem;
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 26px rgba(37, 99, 235, 0.18);
            border-color: rgba(37, 99, 235, 0.45);
        }

        [data-testid="stPlotlyChart"] {
            transition: transform 0.25s ease, box-shadow 0.25s ease;
            border-radius: 12px;
        }

        [data-testid="stPlotlyChart"]:hover {
            transform: translateY(-3px);
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.10);
        }

        [data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
            transition: box-shadow 0.25s ease;
        }

        [data-testid="stDataFrame"]:hover {
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.12);
        }

        [data-baseweb="tab-list"] button[data-baseweb="tab"] {
            transition: color 0.2s ease, transform 0.2s ease;
        }

        [data-baseweb="tab-list"] button[data-baseweb="tab"]:hover {
            color: #2563EB;
            transform: translateY(-1px);
        }

        [data-baseweb="tab-highlight"] {
            transition: left 0.25s ease, width 0.25s ease, background-color 0.25s ease !important;
        }

        .stButton > button {
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px) scale(1.02);
            box-shadow: 0 8px 18px rgba(37, 99, 235, 0.22);
        }

        .stButton > button:active {
            transform: scale(0.97);
        }

        .stMultiSelect [data-baseweb="tag"] {
            transition: transform 0.15s ease;
        }

        .stMultiSelect [data-baseweb="tag"]:hover {
            transform: translateY(-1px) scale(1.05);
        }

        .section-note {
            color: #64748B;
            font-size: 0.92rem;
            margin-top: -0.45rem;
            margin-bottom: 1rem;
        }

        .small-label {
            color: #64748B;
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            background: rgba(255, 255, 255, 0.015);
            transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
        }

        div[data-testid="stMetricLabel"] {
            color: #64748B;
        }

        .footer-note {
            border-top: 1px solid rgba(148, 163, 184, 0.18);
            color: #64748B;
            font-size: 0.82rem;
            margin-top: 2.2rem;
            padding-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def polish(fig, height=430, legend=True):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=65, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=legend,
        hoverlabel=dict(font_size=13),
        transition=dict(duration=450, easing="cubic-in-out"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def score_bars_html(row):
    """Animated score-bar mini panel for a single player row."""
    items = [
        ("Scoring", row.get("SCORING_SCORE"), BLUE),
        ("Playmaking", row.get("PLAYMAKING_SCORE"), CYAN),
        ("Offense", row.get("OFFENSE_SCORE"), GOLD),
        ("Defense", row.get("DEFENSE_SCORE"), GREEN),
        ("All-round", row.get("ALL_ROUND_SCORE"), VIOLET),
        ("3-point", row.get("THREE_POINT_SCORE"), PINK),
    ]
    rows_html = []
    i = 0
    for label, value, color in items:
        if pd.isna(value):
            continue
        pct = max(0.0, min(100.0, float(value)))
        delay = i * 0.07
        rows_html.append(
            f'<div class="score-bar-row">'
            f'<div class="score-bar-name">{label}</div>'
            f'<div class="score-bar-track"><div class="score-bar-fill" '
            f'style="width:{pct:.1f}%; background:{color}; animation-delay:{delay:.2f}s;"></div></div>'
            f'<div class="score-bar-value" style="color:{color};">{value:.0f}</div>'
            f"</div>"
        )
        i += 1
    return f'<div class="score-bars">{"".join(rows_html)}</div>'


def kpi_card_html(label, value, sub="", color=ORANGE):
    """One PowerBI-style KPI tile: colored accent bar, big tabular number."""
    return (
        f'<div class="kpi-card" style="--kpi-color:{color};">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f"</div>"
    )


def kpi_row_html(cards):
    return f'<div class="kpi-row">{"".join(cards)}</div>'


# --- Shot chart: half-court zone geometry -----------------------------
# Coordinates follow the NBA convention (tenths of a foot, hoop at the
# origin); court-space is converted to SVG pixel-space at draw time.
_R3 = 237.5
_CORNER_X = 220.0
_CORNER_Y = math.sqrt(_R3**2 - _CORNER_X**2)
_THETA_CORNER = math.degrees(math.acos(_CORNER_X / _R3))


def _arc(cx, cy, r, a1, a2, n=50):
    pts = []
    for i in range(n + 1):
        t = math.radians(a1 + (a2 - a1) * i / n)
        pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))
    return pts


def _to_svg(x, y):
    return x + 250, 312.5 - y


def _pts_str(pts):
    return " ".join(f"{sx:.1f},{sy:.1f}" for sx, sy in (_to_svg(x, y) for x, y in pts))


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def zone_fill_color(diff):
    """Diverging cold -> neutral -> hot color for a shot-zone FG% delta."""
    t = max(-1.0, min(1.0, diff / 0.12))
    cold = _hex_to_rgb(ICE)
    mid = (203, 213, 225)
    hot = _hex_to_rgb(HEAT)
    anchor = cold if t < 0 else hot
    f = abs(t)
    rgb = tuple(mid[i] + (anchor[i] - mid[i]) * f for i in range(3))
    return "#%02x%02x%02x" % tuple(int(round(v)) for v in rgb)


def zone_status(diff, fga):
    if fga is None or pd.isna(fga) or fga < 0.3 or diff is None or pd.isna(diff):
        return "⬜ No data"
    if diff >= 0.06:
        return "🔥 Hot"
    if diff <= -0.06:
        return "❄️ Cold"
    return "• Neutral"


def _zone_info(row, zone_avgs, zone):
    fga = row.get(f"{zone}_fga")
    pct = row.get(f"{zone}_fg_pct")
    fgm = row.get(f"{zone}_fgm")
    if pd.isna(fga) or fga < 0.3:
        return {"color": "#CBD5E1", "opacity": 0.22, "anim": "", "pct": None, "fga": fga, "fgm": fgm, "nodata": True}
    diff = pct - zone_avgs.get(zone, pct)
    color = zone_fill_color(diff)
    opacity = min(1.0, 0.5 + 0.5 * min(fga / 5.0, 1.0))
    anim = "zone-hot" if diff >= 0.06 else ("zone-cold" if diff <= -0.06 else "")
    return {"color": color, "opacity": opacity, "anim": anim, "pct": pct, "fga": fga, "fgm": fgm, "nodata": False}


def _zone_title(zone, info):
    label = ZONE_LABELS[zone]
    if info["nodata"]:
        return f"{label}: no attempts this season"
    return f"{label}: {info['pct'] * 100:.1f}% on {info['fga']:.1f} FGA/gm"


def build_shot_chart_svg(row, zone_avgs):
    """Half-court zone map colored hot/cold vs league average, with a
    live CSS pulse on zones that run meaningfully hot or cold."""
    shapes = []
    labels = []

    # Above-the-break 3 forms the background: everything beyond the arc.
    info = _zone_info(row, zone_avgs, "above_the_break_3")
    bx1, by1 = _to_svg(-250, 300)
    bx2, by2 = _to_svg(250, -47.5)
    shapes.append(
        f'<rect x="{min(bx1, bx2):.1f}" y="{min(by1, by2):.1f}" width="{abs(bx2 - bx1):.1f}" '
        f'height="{abs(by2 - by1):.1f}" class="{info["anim"]}" fill="{info["color"]}" '
        f'fill-opacity="{info["opacity"]:.2f}"><title>{_zone_title("above_the_break_3", info)}</title></rect>'
    )
    labels.append((0, 268, info))

    # Mid-range wraps the paint on three sides, bounded outside by the arc.
    info = _zone_info(row, zone_avgs, "mid_range")
    arc_top = _arc(0, 0, _R3, _THETA_CORNER, 180 - _THETA_CORNER, 60)
    mid_pts = (
        [(220, -47.5), (220, _CORNER_Y)]
        + arc_top
        + [(-220, _CORNER_Y), (-220, -47.5), (-80, -47.5), (-80, 142.5), (80, 142.5), (80, -47.5), (220, -47.5)]
    )
    shapes.append(
        f'<polygon points="{_pts_str(mid_pts)}" class="{info["anim"]}" fill="{info["color"]}" '
        f'fill-opacity="{info["opacity"]:.2f}"><title>{_zone_title("mid_range", info)}</title></polygon>'
    )
    labels.append((-152, 46, info))
    labels.append((152, 46, info))

    # Paint (non-restricted-area).
    info = _zone_info(row, zone_avgs, "in_the_paint_non_ra")
    px1, py1 = _to_svg(-80, -47.5)
    px2, py2 = _to_svg(80, 142.5)
    shapes.append(
        f'<rect x="{min(px1, px2):.1f}" y="{min(py1, py2):.1f}" width="{abs(px2 - px1):.1f}" '
        f'height="{abs(py2 - py1):.1f}" class="{info["anim"]}" fill="{info["color"]}" '
        f'fill-opacity="{info["opacity"]:.2f}"><title>{_zone_title("in_the_paint_non_ra", info)}</title></rect>'
    )
    labels.append((0, 108, info))

    # Restricted area.
    info = _zone_info(row, zone_avgs, "restricted_area")
    ra_pts = _arc(0, 0, 40, 0, 180, 30)
    shapes.append(
        f'<polygon points="{_pts_str(ra_pts)}" class="{info["anim"]}" fill="{info["color"]}" '
        f'fill-opacity="{info["opacity"]:.2f}"><title>{_zone_title("restricted_area", info)}</title></polygon>'
    )
    labels.append((0, 16, info))

    # Corners.
    for zone, x1, x2, label_x in [("left_corner_3", -250, -220, -236), ("right_corner_3", 220, 250, 236)]:
        info = _zone_info(row, zone_avgs, zone)
        cx1, cy1 = _to_svg(x1, -47.5)
        cx2, cy2 = _to_svg(x2, _CORNER_Y)
        shapes.append(
            f'<rect x="{min(cx1, cx2):.1f}" y="{min(cy1, cy2):.1f}" width="{abs(cx2 - cx1):.1f}" '
            f'height="{abs(cy2 - cy1):.1f}" class="{info["anim"]}" fill="{info["color"]}" '
            f'fill-opacity="{info["opacity"]:.2f}"><title>{_zone_title(zone, info)}</title></rect>'
        )
        labels.append((label_x, 22, info))

    # Court line art, drawn on top of the colored zone fills.
    lines = [
        f'<polygon points="{_pts_str([(-80, -47.5), (-80, 142.5), (80, 142.5), (80, -47.5)])}" '
        f'fill="none" stroke="{INK}" stroke-opacity="0.55" stroke-width="2"/>',
        f'<polygon points="{_pts_str(_arc(0, 142.5, 60, 0, 360, 60))}" fill="none" stroke="{INK}" '
        f'stroke-opacity="0.4" stroke-width="1.6"/>',
        f'<polyline points="{_pts_str(ra_pts)}" fill="none" stroke="{INK}" stroke-opacity="0.55" stroke-width="2"/>',
        f'<polyline points="{_pts_str([(-220, -47.5), (-220, _CORNER_Y)] + arc_top + [(220, _CORNER_Y), (220, -47.5)])}" '
        f'fill="none" stroke="{INK}" stroke-opacity="0.65" stroke-width="2.4"/>',
    ]
    bbx1, bby1 = _to_svg(-30, -7.5)
    bbx2, bby2 = _to_svg(30, -7.5)
    lines.append(f'<line x1="{bbx1:.1f}" y1="{bby1:.1f}" x2="{bbx2:.1f}" y2="{bby2:.1f}" stroke="{INK}" stroke-opacity="0.7" stroke-width="3"/>')
    hx, hy = _to_svg(0, 0)
    lines.append(f'<circle cx="{hx:.1f}" cy="{hy:.1f}" r="7" fill="none" stroke="{ORANGE}" stroke-width="2.4"/>')
    ox1, oy1 = _to_svg(-250, 300)
    ox2, oy2 = _to_svg(250, -47.5)
    lines.append(
        f'<rect x="{min(ox1, ox2):.1f}" y="{min(oy1, oy2):.1f}" width="{abs(ox2 - ox1):.1f}" '
        f'height="{abs(oy2 - oy1):.1f}" fill="none" stroke="{INK}" stroke-opacity="0.22" stroke-width="2"/>'
    )

    label_svgs = []
    for x, y, info in labels:
        sx, sy = _to_svg(x, y)
        text = "—" if info["nodata"] else f"{info['pct'] * 100:.0f}%"
        label_svgs.append(
            f'<text x="{sx:.1f}" y="{sy:.1f}" text-anchor="middle" font-size="15" font-weight="700" '
            f'font-family="Manrope, sans-serif" fill="{INK}" stroke="{COURT}" stroke-width="4" '
            f'stroke-linejoin="round" paint-order="stroke">{text}</text>'
        )

    return (
        '<svg viewBox="0 0 500 360" style="width:100%; max-width:460px; display:block; margin:0 auto;">'
        f'<rect x="0" y="0" width="500" height="360" fill="{COURT}" rx="14"/>'
        + "".join(shapes)
        + "".join(lines)
        + "".join(label_svgs)
        + "</svg>"
    )


def top_player(df, column):
    return df.loc[df[column].idxmax()]


def metric_label(metric):
    names = {
        "PROJECTED_PTS": "Points per game",
        "PROJECTED_REB": "Rebounds per game",
        "PROJECTED_AST": "Assists per game",
        "PROJECTED_FG3M": "3-pointers made per game",
        "PROJECTED_STL": "Steals per game",
        "PROJECTED_BLK": "Blocks per game",
    }
    return names[metric]


def score_label(column):
    return {
        "OFFENSE_SCORE": "Offense",
        "DEFENSE_SCORE": "Defense",
        "ALL_ROUND_SCORE": "All-round",
        "SCORING_SCORE": "Scoring",
        "PLAYMAKING_SCORE": "Playmaking",
        "THREE_POINT_SCORE": "3-point",
        "MIDRANGE_SCORE": "Mid-range",
    }[column]


def player_radar(selected_rows):
    categories = ["Scoring", "Playmaking", "Offense", "Defense", "All-round", "3-point"]
    columns = [
        "SCORING_SCORE",
        "PLAYMAKING_SCORE",
        "OFFENSE_SCORE",
        "DEFENSE_SCORE",
        "ALL_ROUND_SCORE",
        "THREE_POINT_SCORE",
    ]

    fig = go.Figure()
    for _, row in selected_rows.iterrows():
        values = [row[c] if pd.notna(row[c]) else 0 for c in columns]
        fig.add_trace(
            go.Scatterpolar(
                r=values + values[:1],
                theta=categories + categories[:1],
                fill="toself",
                name=row["PLAYER_NAME"],
                opacity=0.58,
            )
        )

    fig.update_layout(
        height=430,
        margin=dict(l=35, r=35, t=50, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 100], gridcolor=GRID, showticklabels=False),
            angularaxis=dict(gridcolor=GRID),
        ),
        legend=dict(orientation="h", y=-0.1),
    )
    return fig


st.markdown(
    f"""
    <div class="hero">
        <h1><span class="hero-ball">🏀</span> NBA <span class="accent">2026-27</span> Player Predictions</h1>
        <p>Season projections, shot charts, breakout signals, and model validation for the league's returning players —
        built to poke around in, not just read.</p>
        <div class="hero-badge"><span class="live-dot"></span> Model-generated projections</div>
        <div class="hero-badge">🗓️ Teams shown reflect 2025-26 rosters</div>
        <div class="hero-badge">📈 {len(players):,} players tracked</div>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.subheader("Filters")

    all_teams = sorted(players["TEAM_ABBREVIATION"].dropna().unique().tolist())
    selected_teams = st.multiselect("Team (2025-26 season)", all_teams, placeholder="All teams")
    st.caption("Team rosters reflect 2025-26; 2026-27 trades and free agency are not modeled.")

    min_age = int(players["AGE_2026_27"].min())
    max_age = int(players["AGE_2026_27"].max())
    age_range = st.slider("2026-27 age", min_age, max_age, (min_age, max_age))

    min_games = st.slider("Minimum 2025-26 games", 20, int(players["GP"].max()), 20)

    st.divider()
    st.caption("Filters apply to prediction and league-leader views. Player Explorer always lets you search the full player pool.")

    st.divider()
    if st.button("🎲 Surprise me", use_container_width=True):
        st.session_state["spotlight_id"] = players.sample(1)["PLAYER_ID"].iloc[0]
        st.balloons()
    st.caption("Spotlight a random player from the full league.")


filtered = players[
    players["AGE_2026_27"].between(age_range[0], age_range[1])
    & (players["GP"] >= min_games)
].copy()

if selected_teams:
    filtered = filtered[filtered["TEAM_ABBREVIATION"].isin(selected_teams)]

if filtered.empty:
    st.warning("No players match the current filters. Adjust the sidebar filters.")
    st.stop()


if "spotlight_id" in st.session_state:
    pick = players[players["PLAYER_ID"] == st.session_state["spotlight_id"]]
    if not pick.empty:
        p = pick.iloc[0]
        st.markdown(
            f"""
            <div class="spotlight-card">
                <div class="small-label">🎲 Random spotlight</div>
                <h3 style="margin:0.25rem 0 0.7rem 0;">{p['PLAYER_NAME']}
                    {team_pill_html(p['TEAM_ABBREVIATION'])}
                </h3>
                <p style="margin:0 0 0.7rem 0; color:#64748B;">
                    Projected 2026-27: <b>{p['PROJECTED_PTS']:.1f}</b> PTS ·
                    <b>{p['PROJECTED_REB']:.1f}</b> REB ·
                    <b>{p['PROJECTED_AST']:.1f}</b> AST ·
                    <b>{p['PROJECTED_FG3M']:.1f}</b> 3PM
                </p>
                {score_bars_html(p)}
            </div>
            """,
            unsafe_allow_html=True,
        )


overview_tab, explorer_tab, leaders_tab, shotchart_tab, shooting_tab, breakout_tab, model_tab = st.tabs(
    [
        "📊 Overview",
        "🔍 Player Explorer",
        "🏆 League Leaders",
        "🎯 Shot Chart",
        "🧪 Shooting Lab",
        "🚀 Breakout Watch",
        "🧠 Model Performance",
    ]
)


with overview_tab:
    st.subheader("2026-27 projection leaders")
    st.markdown('<div class="section-note">Returning players meeting the current filters.</div>', unsafe_allow_html=True)

    ppg = top_player(filtered, "PROJECTED_PTS")
    rpg = top_player(filtered, "PROJECTED_REB")
    apg = top_player(filtered, "PROJECTED_AST")
    threes = top_player(filtered, "PROJECTED_FG3M")

    st.markdown(
        kpi_row_html(
            [
                kpi_card_html("Points leader", f"{ppg['PROJECTED_PTS']:.1f}", f"{ppg['PLAYER_NAME']} {team_pill_html(ppg['TEAM_ABBREVIATION'])}", ORANGE),
                kpi_card_html("Rebounds leader", f"{rpg['PROJECTED_REB']:.1f}", f"{rpg['PLAYER_NAME']} {team_pill_html(rpg['TEAM_ABBREVIATION'])}", BLUE),
                kpi_card_html("Assists leader", f"{apg['PROJECTED_AST']:.1f}", f"{apg['PLAYER_NAME']} {team_pill_html(apg['TEAM_ABBREVIATION'])}", VIOLET),
                kpi_card_html("3PM leader", f"{threes['PROJECTED_FG3M']:.1f}", f"{threes['PLAYER_NAME']} {team_pill_html(threes['TEAM_ABBREVIATION'])}", CYAN),
            ]
        ),
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    with left:
        top10 = filtered.nlargest(10, "PROJECTED_PTS").sort_values("PROJECTED_PTS")
        fig = px.bar(
            top10,
            x="PROJECTED_PTS",
            y="PLAYER_NAME",
            orientation="h",
            title="Projected scoring leaders",
            labels={"PROJECTED_PTS": "Projected PPG", "PLAYER_NAME": "", "TEAM_ABBREVIATION": TEAM_COL},
            hover_data={"TEAM_ABBREVIATION": True, "PROJECTED_PTS": ":.2f"},
        )
        fig.update_traces(marker_color=BLUE)
        st.plotly_chart(polish(fig, legend=False), use_container_width=True)

    with right:
        two_way = filtered.nlargest(20, "ALL_ROUND_SCORE")
        fig = px.scatter(
            two_way,
            x="OFFENSE_SCORE",
            y="DEFENSE_SCORE",
            size="ALL_ROUND_SCORE",
            color="ALL_ROUND_SCORE",
            hover_name="PLAYER_NAME",
            hover_data={"TEAM_ABBREVIATION": True, "ALL_ROUND_SCORE": ":.1f"},
            title="Top all-round profiles",
            labels={"OFFENSE_SCORE": "Offense score", "DEFENSE_SCORE": "Defense score", "TEAM_ABBREVIATION": TEAM_COL},
            color_continuous_scale="Blues",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(polish(fig, legend=False), use_container_width=True)

    st.subheader("Highest-rated 2025-26 profiles")
    offense = top_player(filtered, "OFFENSE_SCORE")
    defense = top_player(filtered, "DEFENSE_SCORE")
    all_round = top_player(filtered, "ALL_ROUND_SCORE")
    st.markdown(
        kpi_row_html(
            [
                kpi_card_html("Offense score", f'{offense["OFFENSE_SCORE"]:.0f}<span style="font-size:1.1rem;color:#94A3B8;"> /100</span>', f"{offense['PLAYER_NAME']} {team_pill_html(offense['TEAM_ABBREVIATION'])}", GOLD),
                kpi_card_html("Defense score", f'{defense["DEFENSE_SCORE"]:.0f}<span style="font-size:1.1rem;color:#94A3B8;"> /100</span>', f"{defense['PLAYER_NAME']} {team_pill_html(defense['TEAM_ABBREVIATION'])}", GREEN),
                kpi_card_html("All-round score", f'{all_round["ALL_ROUND_SCORE"]:.0f}<span style="font-size:1.1rem;color:#94A3B8;"> /100</span>', f"{all_round['PLAYER_NAME']} {team_pill_html(all_round['TEAM_ABBREVIATION'])}", RED),
            ]
        ),
        unsafe_allow_html=True,
    )


with explorer_tab:
    st.subheader("Player Explorer")
    st.markdown('<div class="section-note">Compare up to three players across current production, projections, and custom analytics scores.</div>', unsafe_allow_html=True)

    names = sorted(players["PLAYER_NAME"].tolist())
    defaults = [name for name in ["Shai Gilgeous-Alexander", "Nikola Jokić"] if name in names]
    chosen_players = st.multiselect(
        "Players",
        names,
        default=defaults,
        max_selections=3,
        placeholder="Choose up to three players",
    )

    if not chosen_players:
        st.info("Choose at least one player to begin.")
    else:
        selected_rows = players[players["PLAYER_NAME"].isin(chosen_players)].copy()

        for _, row in selected_rows.iterrows():
            st.markdown(
                f'<h3 style="margin:0.6rem 0 0.2rem 0;">{row["PLAYER_NAME"]}'
                f'{team_pill_html(row["TEAM_ABBREVIATION"])}</h3>',
                unsafe_allow_html=True,
            )
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Projected PPG", f'{row["PROJECTED_PTS"]:.2f}', f'{row["CHANGE_PTS"]:+.2f}')
            c2.metric("Projected RPG", f'{row["PROJECTED_REB"]:.2f}', f'{row["CHANGE_REB"]:+.2f}')
            c3.metric("Projected APG", f'{row["PROJECTED_AST"]:.2f}', f'{row["CHANGE_AST"]:+.2f}')
            c4.metric("Projected 3PM", f'{row["PROJECTED_FG3M"]:.2f}', f'{row["CHANGE_FG3M"]:+.2f}')
            c5.metric("Projected STL", f'{row["PROJECTED_STL"]:.2f}', f'{row["CHANGE_STL"]:+.2f}')
            c6.metric("Projected BLK", f'{row["PROJECTED_BLK"]:.2f}', f'{row["CHANGE_BLK"]:+.2f}')
            st.markdown(score_bars_html(row), unsafe_allow_html=True)

        left, right = st.columns([1.05, 1])
        with left:
            compare = []
            for _, row in selected_rows.iterrows():
                for label, actual, projected in [
                    ("PTS", row["PTS"], row["PROJECTED_PTS"]),
                    ("REB", row["REB"], row["PROJECTED_REB"]),
                    ("AST", row["AST"], row["PROJECTED_AST"]),
                    ("3PM", row["FG3M"], row["PROJECTED_FG3M"]),
                    ("STL", row["STL"], row["PROJECTED_STL"]),
                    ("BLK", row["BLK"], row["PROJECTED_BLK"]),
                ]:
                    compare.append({"Player": row["PLAYER_NAME"], "Stat": label, "Season": "2025-26", "Value": actual})
                    compare.append({"Player": row["PLAYER_NAME"], "Stat": label, "Season": "2026-27 projection", "Value": projected})

            compare_df = pd.DataFrame(compare)
            fig = px.bar(
                compare_df,
                x="Stat",
                y="Value",
                color="Season",
                facet_col="Player" if len(chosen_players) > 1 else None,
                barmode="group",
                title="Current vs projected production",
                color_discrete_map={"2025-26": MUTED, "2026-27 projection": BLUE},
            )
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            st.plotly_chart(polish(fig, 440), use_container_width=True)

        with right:
            st.plotly_chart(player_radar(selected_rows), use_container_width=True)

        table_cols = [
            "PLAYER_NAME",
            "TEAM_ABBREVIATION",
            "AGE_2026_27",
            "MIN",
            "TS_PCT",
            "USG_PCT",
            "OFF_RATING",
            "DEF_RATING",
            "ALL_ROUND_SCORE",
        ]
        display = selected_rows[table_cols].copy()
        display["TS_PCT"] = display["TS_PCT"] * 100
        display["USG_PCT"] = display["USG_PCT"] * 100
        display.columns = ["Player", TEAM_COL, "Age", "MIN", "TS%", "USG%", "Off Rating", "Def Rating", "All-Round Score"]
        st.dataframe(
            display,
            hide_index=True,
            use_container_width=True,
            column_config={
                "TS%": st.column_config.NumberColumn(format="%.1f%%"),
                "USG%": st.column_config.NumberColumn(format="%.1f%%"),
                "All-Round Score": st.column_config.NumberColumn(format="%.1f"),
            },
        )


with leaders_tab:
    st.subheader("League Leaders")
    st.markdown('<div class="section-note">Switch between projected counting stats and 2025-26 analytical scores.</div>', unsafe_allow_html=True)

    leader_mode = st.radio("View", ["2026-27 projections", "2025-26 analytics"], horizontal=True)
    top_n = st.slider("Players shown", 5, 25, 12)

    if leader_mode == "2026-27 projections":
        options = {
            "Points": "PROJECTED_PTS",
            "Rebounds": "PROJECTED_REB",
            "Assists": "PROJECTED_AST",
            "3-pointers": "PROJECTED_FG3M",
            "Steals": "PROJECTED_STL",
            "Blocks": "PROJECTED_BLK",
        }
        choice = st.selectbox("Metric", list(options))
        column = options[choice]
        leaders = filtered.nlargest(top_n, column).sort_values(column)
        fig = px.bar(
            leaders,
            x=column,
            y="PLAYER_NAME",
            orientation="h",
            title=f"Projected {choice.lower()} leaders",
            labels={column: metric_label(column), "PLAYER_NAME": "", "TEAM_ABBREVIATION": TEAM_COL},
            hover_data={"TEAM_ABBREVIATION": True, column: ":.2f"},
        )
        fig.update_traces(marker_color=BLUE)
        st.plotly_chart(polish(fig, max(430, top_n * 31), legend=False), use_container_width=True)

        out = leaders[["PLAYER_NAME", "TEAM_ABBREVIATION", column]].sort_values(column, ascending=False).copy()
        out.columns = ["Player", TEAM_COL, choice]
        out.insert(0, "Rank", range(1, len(out) + 1))
        st.dataframe(out, hide_index=True, use_container_width=True)

    else:
        options = {
            "Offense": "OFFENSE_SCORE",
            "Defense": "DEFENSE_SCORE",
            "All-round": "ALL_ROUND_SCORE",
            "Scoring": "SCORING_SCORE",
            "Playmaking": "PLAYMAKING_SCORE",
            "3-point shooting": "THREE_POINT_SCORE",
            "Mid-range": "MIDRANGE_SCORE",
        }
        choice = st.selectbox("Score", list(options))
        column = options[choice]
        score_df = filtered.dropna(subset=[column]).nlargest(top_n, column).sort_values(column)
        fig = px.bar(
            score_df,
            x=column,
            y="PLAYER_NAME",
            orientation="h",
            title=f"Top {choice.lower()} profiles",
            labels={column: f"{score_label(column)} score", "PLAYER_NAME": "", "TEAM_ABBREVIATION": TEAM_COL},
            hover_data={"TEAM_ABBREVIATION": True, column: ":.1f"},
        )
        fig.update_traces(marker_color=CYAN)
        st.plotly_chart(polish(fig, max(430, top_n * 31), legend=False), use_container_width=True)

        out = score_df[["PLAYER_NAME", "TEAM_ABBREVIATION", column]].sort_values(column, ascending=False).copy()
        out.columns = ["Player", TEAM_COL, "Score"]
        out.insert(0, "Rank", range(1, len(out) + 1))
        st.dataframe(out, hide_index=True, use_container_width=True)


with shotchart_tab:
    st.subheader("Shot Chart")
    st.markdown(
        '<div class="section-note">A zone-by-zone hot/cold read on 2025-26 shooting, colored against league-average '
        'efficiency from that same spot on the floor. Zones pulse live when they run meaningfully hot or cold.</div>',
        unsafe_allow_html=True,
    )

    has_shot_data = bool(zone_league_avg) and f"{ZONES[0]}_fga" in players.columns
    if not has_shot_data:
        st.info("Zone shooting data isn't available in this build — run `src/collect_data.py` to fetch it.")
    else:
        sc_names = sorted(players["PLAYER_NAME"].tolist())
        default_sc = "Stephen Curry" if "Stephen Curry" in sc_names else sc_names[0]
        sc_player = st.selectbox("Player", sc_names, index=sc_names.index(default_sc), key="shotchart_player")
        sc_row = players[players["PLAYER_NAME"] == sc_player].iloc[0]

        chart_col, side_col = st.columns([1.25, 1])

        with chart_col:
            st.markdown(
                f'<div style="text-align:center; margin-bottom:0.5rem;">'
                f'<span style="font-family:\'Oswald\',sans-serif; font-weight:600; font-size:1.2rem; letter-spacing:0.02em;">{sc_row["PLAYER_NAME"].upper()}</span> '
                f"{team_pill_html(sc_row['TEAM_ABBREVIATION'])}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(build_shot_chart_svg(sc_row, zone_league_avg), unsafe_allow_html=True)
            st.markdown(
                '<div style="display:flex; justify-content:center; gap:1.1rem; margin-top:0.6rem; font-size:0.78rem; color:#64748B;">'
                f'<span>🔥 <b style="color:{HEAT};">Hot</b> vs league avg</span>'
                f'<span>❄️ <b style="color:{ICE};">Cold</b> vs league avg</span>'
                '<span>⬜ Not enough volume</span>'
                "</div>",
                unsafe_allow_html=True,
            )

        with side_col:
            st.markdown("##### Reading the chart")
            st.markdown(
                "- **Color** is FG% in that zone minus the league average from the same zone — orange runs hot, blue runs cold.\n"
                "- **Pulse** flags zones at least 6 points above or below league average; the glow is a live CSS animation, not a static badge.\n"
                "- **Saturation** tracks volume (FGA/gm) — a washed-out zone means the color can't be trusted yet.\n"
                "- **Gray** zones had fewer than 0.3 attempts per game: no real signal either way."
            )

            zone_rows = []
            for zone in ZONES:
                fga = sc_row.get(f"{zone}_fga")
                pct = sc_row.get(f"{zone}_fg_pct")
                avg = zone_league_avg.get(zone)
                has_volume = pd.notna(fga) and fga >= 0.3 and pd.notna(pct)
                diff = (pct - avg) if has_volume and avg is not None else None
                zone_rows.append(
                    {
                        "Zone": ZONE_LABELS[zone],
                        "FGA/gm": f"{fga:.1f}" if pd.notna(fga) else "—",
                        "FG%": f"{pct * 100:.1f}%" if pd.notna(pct) else "—",
                        "Lg avg": f"{avg * 100:.1f}%" if avg is not None else "—",
                        "Status": zone_status(diff, fga),
                    }
                )
            zone_df = pd.DataFrame(zone_rows)
            st.dataframe(zone_df, hide_index=True, use_container_width=True)


with shooting_tab:
    st.subheader("Shooting Lab")
    st.markdown('<div class="section-note">Volume and efficiency are shown together so low-volume percentage leaders do not dominate the rankings.</div>', unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        min_3pa = st.slider("Minimum 3PA per game", 0.0, 8.0, 2.0, 0.5)
        threes = players[players["FG3A"] >= min_3pa].copy()
        fig = px.scatter(
            threes,
            x="FG3A",
            y="FG3_PCT",
            size="FG3M",
            color="THREE_POINT_SCORE",
            hover_name="PLAYER_NAME",
            hover_data={"TEAM_ABBREVIATION": True, "FG3A": ":.1f", "FG3M": ":.1f", "FG3_PCT": ":.3f"},
            title="3-point volume vs efficiency",
            labels={"FG3A": "3PA per game", "FG3_PCT": "3P%", "THREE_POINT_SCORE": "3PT score", "TEAM_ABBREVIATION": TEAM_COL},
            color_continuous_scale="Blues",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(polish(fig, 480), use_container_width=True)

    with right:
        mid = players.dropna(subset=["MIDRANGE_SCORE"]).copy()
        fig = px.scatter(
            mid,
            x="mid_range_fga",
            y="mid_range_fg_pct",
            size="MIDRANGE_SCORE",
            color="MIDRANGE_SCORE",
            hover_name="PLAYER_NAME",
            hover_data={"TEAM_ABBREVIATION": True, "mid_range_fga": ":.1f", "mid_range_fg_pct": ":.3f"},
            title="Mid-range volume vs efficiency",
            labels={"mid_range_fga": "Mid-range FGA per game", "mid_range_fg_pct": "Mid-range FG%", "MIDRANGE_SCORE": "Mid-range score", "TEAM_ABBREVIATION": TEAM_COL},
            color_continuous_scale="Teal",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(polish(fig, 480), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Top 3-point profiles")
        top_three = players.dropna(subset=["THREE_POINT_SCORE"]).nlargest(10, "THREE_POINT_SCORE")
        view = top_three[["PLAYER_NAME", "TEAM_ABBREVIATION", "FG3M", "FG3A", "FG3_PCT", "THREE_POINT_SCORE"]].copy()
        view.columns = ["Player", TEAM_COL, "3PM", "3PA", "3P%", "3PT Score"]
        st.dataframe(view, hide_index=True, use_container_width=True)

    with c2:
        st.markdown("#### Top mid-range profiles")
        top_mid = players.dropna(subset=["MIDRANGE_SCORE"]).nlargest(10, "MIDRANGE_SCORE")
        view = top_mid[["PLAYER_NAME", "TEAM_ABBREVIATION", "mid_range_fga", "mid_range_fg_pct", "MIDRANGE_SCORE"]].copy()
        view.columns = ["Player", TEAM_COL, "Mid FGA", "Mid FG%", "Mid-range Score"]
        st.dataframe(view, hide_index=True, use_container_width=True)


with breakout_tab:
    st.subheader("Breakout Watch")
    st.markdown('<div class="section-note">Projected change is measured against each player\'s 2025-26 per-game production. It is a model signal, not a guarantee.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    max_breakout_age = c1.slider("Maximum age", 20, 30, 25)
    min_current_ppg = c2.slider("Minimum current PPG", 0.0, 20.0, 8.0, 0.5)
    breakout_count = c3.slider("Players shown", 5, 20, 10)

    breakout_pool = players[
        (players["AGE_2025_26"] <= max_breakout_age)
        & (players["PTS"] >= min_current_ppg)
    ].copy()

    top_gain = breakout_pool.nlargest(1, "CHANGE_PTS")
    if not top_gain.empty:
        g = top_gain.iloc[0]
        st.markdown(
            f'<div class="info-badge"><span class="fire-badge">🔥</span> '
            f'Biggest projected jump: <b>{g["PLAYER_NAME"]}</b> '
            f'{team_pill_html(g["TEAM_ABBREVIATION"])} '
            f'&nbsp;+{g["CHANGE_PTS"]:.1f} PPG</div>',
            unsafe_allow_html=True,
        )

    left, right = st.columns(2)

    with left:
        gains = breakout_pool.nlargest(breakout_count, "CHANGE_PTS").sort_values("CHANGE_PTS")
        fig = px.bar(
            gains,
            x="CHANGE_PTS",
            y="PLAYER_NAME",
            orientation="h",
            title="Largest projected scoring gains",
            labels={"CHANGE_PTS": "Projected PPG change", "PLAYER_NAME": "", "TEAM_ABBREVIATION": TEAM_COL},
            hover_data={"TEAM_ABBREVIATION": True, "PTS": ":.2f", "PROJECTED_PTS": ":.2f", "AGE_2025_26": True},
        )
        fig.update_traces(marker_color=GREEN)
        st.plotly_chart(polish(fig, 450, legend=False), use_container_width=True)

    with right:
        established = players[players["PTS"] >= 15].nsmallest(breakout_count, "CHANGE_PTS").sort_values("CHANGE_PTS", ascending=False)
        fig = px.bar(
            established,
            x="CHANGE_PTS",
            y="PLAYER_NAME",
            orientation="h",
            title="Largest projected scoring declines",
            labels={"CHANGE_PTS": "Projected PPG change", "PLAYER_NAME": "", "TEAM_ABBREVIATION": TEAM_COL},
            hover_data={"TEAM_ABBREVIATION": True, "PTS": ":.2f", "PROJECTED_PTS": ":.2f", "AGE_2025_26": True},
        )
        fig.update_traces(marker_color=RED)
        st.plotly_chart(polish(fig, 450, legend=False), use_container_width=True)

    st.markdown("#### Breakout candidate table")
    table = breakout_pool.nlargest(20, "CHANGE_PTS")[[
        "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE_2025_26", "PTS", "PROJECTED_PTS", "CHANGE_PTS", "ALL_ROUND_SCORE"
    ]].copy()
    table.columns = ["Player", TEAM_COL, "Age", "2025-26 PPG", "Projected PPG", "Change", "All-Round Score"]
    st.dataframe(table, hide_index=True, use_container_width=True)


with model_tab:
    st.subheader("Model Performance")
    st.markdown('<div class="section-note">Each target was backtested on 2025-26 and compared with a simple previous-season baseline.</div>', unsafe_allow_html=True)

    average_improvement = model_summary["MAE Improvement"].mean()
    strongest = model_summary.loc[model_summary["MAE Improvement"].idxmax()]
    weakest = model_summary.loc[model_summary["MAE Improvement"].idxmin()]

    c1, c2, c3 = st.columns(3)
    c1.metric("Average MAE improvement", f"{average_improvement:.1%}")
    c2.metric("Strongest improvement", strongest["Target"], f'{strongest["MAE Improvement"]:.1%}')
    c3.metric("Smallest improvement", weakest["Target"], f'{weakest["MAE Improvement"]:.1%}')

    long_metrics = model_summary.melt(
        id_vars="Target",
        value_vars=["Baseline MAE", "Selected MAE"],
        var_name="Model",
        value_name="MAE",
    )
    fig = px.bar(
        long_metrics,
        x="Target",
        y="MAE",
        color="Model",
        barmode="group",
        title="Mean absolute error: baseline vs selected model",
        color_discrete_map={"Baseline MAE": MUTED, "Selected MAE": BLUE},
    )
    st.plotly_chart(polish(fig, 440), use_container_width=True)

    display = model_summary.copy()
    display["MAE Improvement"] = display["MAE Improvement"].map(lambda x: f"{x:.1%}")
    display["Baseline MAE"] = display["Baseline MAE"].round(3)
    display["Selected MAE"] = display["Selected MAE"].round(3)
    display["RMSE"] = display["RMSE"].round(3)
    display["R²"] = display["R²"].round(3)
    st.dataframe(display, hide_index=True, use_container_width=True)

    with st.expander("How the modeling works"):
        st.markdown(
            """
            - Each training row represents one player-season and predicts the next consecutive NBA season.
            - The holdout is chronological rather than a random train/test split.
            - Ridge Regression and Random Forest are compared for each target.
            - The selected model is the one with the lower validation MAE.
            - A previous-season baseline is reported so the machine-learning model must beat a simple alternative.
            - Team is kept as 2025-26 display metadata rather than a model feature, which makes the projections less sensitive to trades and free agency moves.
            """
        )


st.markdown(
    """
    <div class="footer-note">
        Projections are statistical estimates for returning players with meaningful 2025-26 playing time. Future injuries,
        retirements, trades, coaching changes, and role changes are not known to the model. Custom player scores are project
        analytics and are not official NBA metrics. Team labels throughout reflect each player's 2025-26 roster.
    </div>
    """,
    unsafe_allow_html=True,
)
