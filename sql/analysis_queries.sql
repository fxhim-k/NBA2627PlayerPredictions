-- NBA2627PlayerPredictions
-- Example SQL questions for the finished SQLite database.

-- 1. Top projected scorers
SELECT
    PLAYER_NAME,
    TEAM_ABBREVIATION,
    PROJECTED_PTS
FROM predictions_2026_27
ORDER BY PROJECTED_PTS DESC
LIMIT 15;


-- 2. Top projected rebounders
SELECT
    PLAYER_NAME,
    TEAM_ABBREVIATION,
    PROJECTED_REB
FROM predictions_2026_27
ORDER BY PROJECTED_REB DESC
LIMIT 15;


-- 3. Best all-round players from the 2025-26 analytics score
SELECT
    PLAYER_NAME,
    TEAM_ABBREVIATION,
    OFFENSE_SCORE,
    DEFENSE_SCORE,
    ALL_ROUND_SCORE
FROM player_scores_2025_26
ORDER BY ALL_ROUND_SCORE DESC
LIMIT 15;


-- 4. Best high-volume 3-point profiles
SELECT
    PLAYER_NAME,
    TEAM_ABBREVIATION,
    FG3M,
    FG3A,
    FG3_PCT,
    THREE_POINT_SCORE
FROM player_scores_2025_26
WHERE FG3A >= 5
ORDER BY THREE_POINT_SCORE DESC
LIMIT 15;


-- 5. Biggest projected scoring increases
SELECT
    p.PLAYER_NAME,
    p.TEAM_ABBREVIATION,
    h.PTS AS PTS_2025_26,
    p.PROJECTED_PTS,
    ROUND(p.PROJECTED_PTS - h.PTS, 2) AS PROJECTED_CHANGE
FROM predictions_2026_27 p
JOIN player_history h
    ON p.PLAYER_ID = h.PLAYER_ID
WHERE h.SEASON = '2025-26'
ORDER BY PROJECTED_CHANGE DESC
LIMIT 15;


-- 6. Compare model errors
SELECT
    target,
    model,
    ROUND(mae, 3) AS MAE,
    ROUND(rmse, 3) AS RMSE,
    ROUND(r2, 3) AS R2
FROM model_metrics
ORDER BY target, mae;
