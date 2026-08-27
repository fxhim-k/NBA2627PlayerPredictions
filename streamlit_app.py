from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

NAVY = "#0B1F3A"
BLUE = "#2563EB"
CYAN = "#06B6D4"
GOLD = "#D4A017"
GREEN = "#16A34A"
RED = "#DC2626"
MUTED = "#64748B"
GRID = "rgba(148, 163, 184, 0.18)"


st.set_page_config(
    page_title="NBA 2026-27 Player Predictions",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)


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

    return player, pd.DataFrame(model_rows), metrics


players, model_summary, model_metrics = load_data()


st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(148, 163, 184, 0.18);
        }

        .hero {
            padding: 1.5rem 1.7rem;
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 18px;
            background: linear-gradient(120deg, rgba(37, 99, 235, 0.10), rgba(6, 182, 212, 0.04));
            margin-bottom: 1.1rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.15rem;
            letter-spacing: -0.04em;
        }

        .hero p {
            margin: 0.45rem 0 0 0;
            color: #64748B;
            font-size: 1rem;
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
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


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
    """
    <div class="hero">
        <h1>NBA 2026-27 Player Predictions</h1>
        <p>Player analytics, season projections, shooting profiles, breakout signals, and model validation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.subheader("Filters")

    all_teams = sorted(players["TEAM_ABBREVIATION"].dropna().unique().tolist())
    selected_teams = st.multiselect("Team", all_teams, placeholder="All teams")

    min_age = int(players["AGE_2026_27"].min())
    max_age = int(players["AGE_2026_27"].max())
    age_range = st.slider("2026-27 age", min_age, max_age, (min_age, max_age))

    min_games = st.slider("Minimum 2025-26 games", 20, int(players["GP"].max()), 20)

    st.divider()
    st.caption("Filters apply to prediction and league-leader views. Player Explorer always lets you search the full player pool.")


filtered = players[
    players["AGE_2026_27"].between(age_range[0], age_range[1])
    & (players["GP"] >= min_games)
].copy()

if selected_teams:
    filtered = filtered[filtered["TEAM_ABBREVIATION"].isin(selected_teams)]

if filtered.empty:
    st.warning("No players match the current filters. Adjust the sidebar filters.")
    st.stop()


overview_tab, explorer_tab, leaders_tab, shooting_tab, breakout_tab, model_tab = st.tabs(
    [
        "Overview",
        "Player Explorer",
        "League Leaders",
        "Shooting Lab",
        "Breakout Watch",
        "Model Performance",
    ]
)


with overview_tab:
    st.subheader("2026-27 projection leaders")
    st.markdown('<div class="section-note">Returning players meeting the current filters.</div>', unsafe_allow_html=True)

    ppg = top_player(filtered, "PROJECTED_PTS")
    rpg = top_player(filtered, "PROJECTED_REB")
    apg = top_player(filtered, "PROJECTED_AST")
    threes = top_player(filtered, "PROJECTED_FG3M")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"PPG · {ppg['PLAYER_NAME']}", f"{ppg['PROJECTED_PTS']:.2f}")
    c2.metric(f"RPG · {rpg['PLAYER_NAME']}", f"{rpg['PROJECTED_REB']:.2f}")
    c3.metric(f"APG · {apg['PLAYER_NAME']}", f"{apg['PROJECTED_AST']:.2f}")
    c4.metric(f"3PM · {threes['PLAYER_NAME']}", f"{threes['PROJECTED_FG3M']:.2f}")

    left, right = st.columns(2)

    with left:
        top10 = filtered.nlargest(10, "PROJECTED_PTS").sort_values("PROJECTED_PTS")
        fig = px.bar(
            top10,
            x="PROJECTED_PTS",
            y="PLAYER_NAME",
            orientation="h",
            title="Projected scoring leaders",
            labels={"PROJECTED_PTS": "Projected PPG", "PLAYER_NAME": ""},
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
            labels={"OFFENSE_SCORE": "Offense score", "DEFENSE_SCORE": "Defense score"},
            color_continuous_scale="Blues",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(polish(fig, legend=False), use_container_width=True)

    st.subheader("Highest-rated 2025-26 profiles")
    c1, c2, c3 = st.columns(3)
    offense = top_player(filtered, "OFFENSE_SCORE")
    defense = top_player(filtered, "DEFENSE_SCORE")
    all_round = top_player(filtered, "ALL_ROUND_SCORE")
    c1.metric("Offense score", offense["PLAYER_NAME"], f'{offense["OFFENSE_SCORE"]:.1f} / 100')
    c2.metric("Defense score", defense["PLAYER_NAME"], f'{defense["DEFENSE_SCORE"]:.1f} / 100')
    c3.metric("All-round score", all_round["PLAYER_NAME"], f'{all_round["ALL_ROUND_SCORE"]:.1f} / 100')


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
            st.markdown(f"### {row['PLAYER_NAME']} · {row['TEAM_ABBREVIATION']}")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Projected PPG", f'{row["PROJECTED_PTS"]:.2f}', f'{row["CHANGE_PTS"]:+.2f}')
            c2.metric("Projected RPG", f'{row["PROJECTED_REB"]:.2f}', f'{row["CHANGE_REB"]:+.2f}')
            c3.metric("Projected APG", f'{row["PROJECTED_AST"]:.2f}', f'{row["CHANGE_AST"]:+.2f}')
            c4.metric("Projected 3PM", f'{row["PROJECTED_FG3M"]:.2f}', f'{row["CHANGE_FG3M"]:+.2f}')
            c5.metric("Projected STL", f'{row["PROJECTED_STL"]:.2f}', f'{row["CHANGE_STL"]:+.2f}')
            c6.metric("Projected BLK", f'{row["PROJECTED_BLK"]:.2f}', f'{row["CHANGE_BLK"]:+.2f}')

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
        display.columns = ["Player", "Team", "Age", "MIN", "TS%", "USG%", "Off Rating", "Def Rating", "All-Round Score"]
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
            labels={column: metric_label(column), "PLAYER_NAME": ""},
            hover_data={"TEAM_ABBREVIATION": True, column: ":.2f"},
        )
        fig.update_traces(marker_color=BLUE)
        st.plotly_chart(polish(fig, max(430, top_n * 31), legend=False), use_container_width=True)

        out = leaders[["PLAYER_NAME", "TEAM_ABBREVIATION", column]].sort_values(column, ascending=False).copy()
        out.columns = ["Player", "Team", choice]
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
            labels={column: f"{score_label(column)} score", "PLAYER_NAME": ""},
            hover_data={"TEAM_ABBREVIATION": True, column: ":.1f"},
        )
        fig.update_traces(marker_color=CYAN)
        st.plotly_chart(polish(fig, max(430, top_n * 31), legend=False), use_container_width=True)

        out = score_df[["PLAYER_NAME", "TEAM_ABBREVIATION", column]].sort_values(column, ascending=False).copy()
        out.columns = ["Player", "Team", "Score"]
        out.insert(0, "Rank", range(1, len(out) + 1))
        st.dataframe(out, hide_index=True, use_container_width=True)


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
            labels={"FG3A": "3PA per game", "FG3_PCT": "3P%", "THREE_POINT_SCORE": "3PT score"},
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
            labels={"mid_range_fga": "Mid-range FGA per game", "mid_range_fg_pct": "Mid-range FG%", "MIDRANGE_SCORE": "Mid-range score"},
            color_continuous_scale="Teal",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(polish(fig, 480), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Top 3-point profiles")
        top_three = players.dropna(subset=["THREE_POINT_SCORE"]).nlargest(10, "THREE_POINT_SCORE")
        view = top_three[["PLAYER_NAME", "TEAM_ABBREVIATION", "FG3M", "FG3A", "FG3_PCT", "THREE_POINT_SCORE"]].copy()
        view.columns = ["Player", "Team", "3PM", "3PA", "3P%", "3PT Score"]
        st.dataframe(view, hide_index=True, use_container_width=True)

    with c2:
        st.markdown("#### Top mid-range profiles")
        top_mid = players.dropna(subset=["MIDRANGE_SCORE"]).nlargest(10, "MIDRANGE_SCORE")
        view = top_mid[["PLAYER_NAME", "TEAM_ABBREVIATION", "mid_range_fga", "mid_range_fg_pct", "MIDRANGE_SCORE"]].copy()
        view.columns = ["Player", "Team", "Mid FGA", "Mid FG%", "Mid-range Score"]
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

    left, right = st.columns(2)

    with left:
        gains = breakout_pool.nlargest(breakout_count, "CHANGE_PTS").sort_values("CHANGE_PTS")
        fig = px.bar(
            gains,
            x="CHANGE_PTS",
            y="PLAYER_NAME",
            orientation="h",
            title="Largest projected scoring gains",
            labels={"CHANGE_PTS": "Projected PPG change", "PLAYER_NAME": ""},
            hover_data={"PTS": ":.2f", "PROJECTED_PTS": ":.2f", "AGE_2025_26": True},
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
            labels={"CHANGE_PTS": "Projected PPG change", "PLAYER_NAME": ""},
            hover_data={"PTS": ":.2f", "PROJECTED_PTS": ":.2f", "AGE_2025_26": True},
        )
        fig.update_traces(marker_color=RED)
        st.plotly_chart(polish(fig, 450, legend=False), use_container_width=True)

    st.markdown("#### Breakout candidate table")
    table = breakout_pool.nlargest(20, "CHANGE_PTS")[[
        "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE_2025_26", "PTS", "PROJECTED_PTS", "CHANGE_PTS", "ALL_ROUND_SCORE"
    ]].copy()
    table.columns = ["Player", "Team", "Age", "2025-26 PPG", "Projected PPG", "Change", "All-Round Score"]
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
            - Team is kept as display metadata rather than a model feature, which makes the projections less sensitive to trades.
            """
        )


st.markdown(
    """
    <div class="footer-note">
        Projections are statistical estimates for returning players with meaningful 2025-26 playing time. Future injuries,
        retirements, trades, coaching changes, and role changes are not known to the model. Custom player scores are project
        analytics and are not official NBA metrics.
    </div>
    """,
    unsafe_allow_html=True,
)
