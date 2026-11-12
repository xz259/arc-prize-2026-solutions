# Validation runs

`RUN_INDEX.tsv` lists the validation runs from October 17 to 28. Each run plays the 25 public games and 30 custom games (longer runs also play more games) at about 4.9 minutes per game: 4.5 hours for the 55 games, 9 hours for 110, 18 hours for 222.

| Column | Meaning |
|---|---|
| `pub25_levels_of_183`, `copycat_levels_of_214` | levels cleared on the public 25 and on the 30 custom games |
| `pub25_LW_pct`, `copycat_LW_pct` | level-weighted completion: L(L+1) / N(N+1) for a game with L of its N levels cleared |
| `pub25_official` | the official score on the public 25 |
| `pub25_eff` | the run's action efficiency: `pub25_official` / `pub25_LW_pct` |
| `copycat_est` | the estimated score on the custom games: their level-weighted completion times the run's `pub25_eff`, since those games have no action baselines |

The estimate on all 55 games is (25 × `pub25_official` + 30 × `copycat_est`) / 55. `make_fig_score_over_time.py` computes it per run, groups the runs into the stages of the write-up and draws Figure 1 (`../docs/figures/arc3_validation_score_over_time.png`).
