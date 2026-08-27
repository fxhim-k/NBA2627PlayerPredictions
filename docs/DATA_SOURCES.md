# Data sources

## Main source

The project uses the open-source `nba_api` Python client to access NBA Stats endpoints.

Historical player statistics:
- `LeagueDashPlayerStats`
- Base measure
- Advanced measure
- Per-game regular-season data

Current shot-zone statistics:
- `LeagueDashPlayerShotLocations`
- `By Zone`
- Per-game regular-season data

The project uses `PLAYER_ID` as the player key.

## Why the project saves raw CSV files

The API is used only during the collection step.

After a season is downloaded, it is stored under:

```text
data/raw/
```

The modeling and analytics scripts then work from those local files.

This has two benefits:

1. the project is reproducible after the initial download
2. repeated model runs do not keep calling NBA Stats

## API reliability

NBA Stats endpoints can occasionally time out or return no data.

The collection script:
- retries failed requests
- skips files already downloaded
- does not stop the main project if the optional shot-location request fails

The mid-range analysis depends on shot-location data. The six main 2026-27 projections do not.
