# Validation runs

`RUN_INDEX.tsv` lists 33 validation runs from October 17 to 28. Every run plays the 25 public games and the 30 custom games; the 9-hour and 18-hour runs play further games as well. All of them run at about 4.9 minutes per game: 4.5 hours for 55 games, 9 hours for 110, 18 hours for 222.

| Column | Meaning |
|---|---|
| `run`, `start_utc` | the run's name and its start (month-day, UTC) |
| `games`, `hours`, `min_per_game` | games played, run length, and their ratio |
| `pub25_levels_of_183`, `copycat_levels_of_214` | levels cleared on the public 25 and on the 30 custom games |
| `pub25_LW_pct`, `copycat_LW_pct` | level-weighted completion: L(L+1) / N(N+1) for a game with L of its N levels cleared, averaged over the games |
| `pub25_official` | the official score on the public 25 |
| `pub25_eff` | the run's action efficiency: `pub25_official` / `pub25_LW_pct` |
| `copycat_est` | the estimated score on the custom games: their level-weighted completion times the run's `pub25_eff`, since those games have no action baselines |

The estimate on all 55 games is (25 × `pub25_official` + 30 × `copycat_est`) / 55. This is the score that section 2.3 of the [write-up](../docs/writeup.md) reports.

`make_fig_score_over_time.py` computes the estimate per run, groups 14 of the runs into the six stages of the write-up and draws Figure 1 (`../docs/figures/arc3_validation_score_over_time.png`). The other runs are exploratory: the context-length search, precision tests, arms at other context lengths, a rejected variant, and one pair in which most games were cancelled (`kag71`). The script's header lists them.

```bash
python3 make_fig_score_over_time.py
```

The script needs matplotlib. It prints the mean and range of each stage and rewrites the figure.
