# Technical document — <team name>

> Draft. Fill each section as we go, not on the last day. The session log has the history.
> Required sections come from `docs/oficial/SUBMISSION.md`. A few pages is plenty.

## 1. Approach

What we built, and why this rather than something else.

## 2. Assets used

Which provided assets we used and how. Also which ones we tried and **dropped**, and why.

| Asset | Used? | How / why not |
|---|---|---|
| Tactical ATD (tracking) | | |
| Smart Tagging | | |
| Homography | | |
| Tactical video | | |
| Broadcast video + ATD | | |

## 3. Training data

Every labelled dataset used: source, who labelled it, how much.
None of it may come from the provided game.

## 4. Possession rules implemented

Where we drew the line on loose balls, challenges, deflections and restarts.
What counts as "control" in our system, and how we detect dead ball.

## 5. Handling missing and noisy tracking

The ball is missing on 41% of graded frames, tracks fragment and swap identity.
What we did about it.

## 6. Limitations

Where it fails, and what we would fix with more time.

## 7. Reproduction

```bash
pip install -r requirements.txt
# put the data bundle in ./data (or set MTAH_DATA_DIR)
python scripts/make_submission.py --name <team_name>
```

This regenerates `evaluation/results/<team_name>.csv` byte for byte.
