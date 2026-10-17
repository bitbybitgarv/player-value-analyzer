# Premier League Player Value Analyzer

A small machine learning project that estimates a Premier League player's market value
from his age, position and per-90 stats, then shows where the estimate and the real price
disagree.

**Live app:** https://epl-player-value.streamlit.app

## What it does
- Ranks 292 outfield players (2024/25, 900+ minutes) by how far the model's value
  is from the Transfermarkt market value
- Shows each player's market value, model value, and a caution flag where the gap
  is not informative (very young players, superstars, very old players, price floor)
- Compares each player with his own squad-mates to reduce club effects

## How it works
1. Cleaned an FBref 2024/25 player-stats dataset: fixed a duplicate club name, dropped
   columns that failed consistency checks, found that half the players were missing a
   block of stats, corrected mislabeled positions, converted counts to per-90 rates
2. Added Transfermarkt market values (the last valuation before the end of the season) and
   age at season end, matching players by name with manual fixes
3. Trained a ridge regression on the log of value, with age hinge features that capture the
   fall in value after about 28 and 31
4. Evaluated with repeated 5-fold cross-validation: RMSE about 0.48 in log units, R2 about
   0.75. Typical predictions are off by a factor of about 1.6.

## What I learned
I was surprised to see how some factors were clearly dominating over the other factors in deciding
the player's market value.Trying to predict the value with different models made me understand the
approach. I am very eager to work on something which is complex and problem solving
## Limitations
- One season and 292 players, so the model is rough and only large gaps mean anything
- It cannot see potential, injuries, contracts, reputation or commercial appeal, so young
  players and superstars often look "overvalued"
- The club is not a model input, but minutes and clean sheets still carry some team
  information
- Goalkeepers are excluded, because the stats do not support them
- The model was chosen by comparing variants on the same players, so the accuracy is
  somewhat optimistic

## Data sources
- FBref 2024/25 Premier League player stats (Kaggle)
- Transfermarkt data (Kaggle: "Football Data From Transfermarkt")
Raw data is not included in this repository.

## Run it locally
Install the packages in requirements.txt, then run `streamlit run app.py`
