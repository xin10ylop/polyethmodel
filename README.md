# polyethmodel — Polymarket "ETH Up or Down 5m" trading system

The ETH replica of [polybtcmodel](https://github.com/xin10ylop/polybtcmodel), built 2026-11-09 with the same protocol: tune on
Aug 20 – Sep 30 2026, test on Oct 1 – Nov 3, and three never-touched periods (June, July, Nov 4–7). Same simulator, same
strategies (A pre-open directional, B open mispricing, C, D settlement capture), the ETH siblings of the three BTC models
(`xin10ylop/eth1` = btc-predict, `eth2` = btc-movement, `eth3` = btcmovement2), the same exhaustive exit grids and the same
book-depth vote. Everything below is out of sample for the period it is reported on.

**Verdict in one paragraph.** ETH's 5-minute market is tradable with strategy A and the edge per share is of the same order as
BTC's (test: +2.8 ¢ per share, 66 % win rate at $100 per trade with the ETH exit), but it is **not a more profitable bot**: (1) the ETH
models' *direction* at T−45 is weaker than BTC's (AUC 0.51–0.53 vs 0.52–0.55), exactly as the three ETH repos' own research says —
what is more predictable about ETH is its volatility, which A does not trade; (2) B, C and D, which earn most of BTC's money, **lose money
on ETH's test period**; (3) the ETH markets are about **fifteen times thinner** than BTC's: the median window prints $10–27 at 0.49–0.51
before the open (BTC: $400–600), so a $100 pre-open bid would fill in roughly one window in ten, and in the windows deep enough to fill,
A's edge is close to zero. Capacity-weighted, A on ETH is worth tens of dollars a day, not hundreds. The honest use of this repo is a
small add-on to the BTC bot at a $20–50 stake, run in shadow first.

## 1. Data (`scripts/`)

| what | source | result |
|---|---|---|
| ETH 5-minute markets | gamma series `eth-up-or-down-5m` (id 10683), slugs `eth-updown-5m-<T>` | 46,338 windows 2026-06-01 → 11-09, 98 % with `priceToBeat`, 99.8 % resolved; trade prints 162 days (405 MB) |
| resolution rule | market descriptions | **point price until 2026-08-06, 60-s Chainlink ETH/USD TWAP from 2026-08-07 00:00 UTC** (BTC switched in early September). Reference lag used in scoring: L = 0 before, L = 30 after (`research/ref_timing_eth.py`: the price-to-beat sits within the TWAP window; the Chainlink-vs-Binance basis is too noisy to pin it closer) |
| Binance ETHUSDT spot 1 s klines + USD-M perp aggTrades | data.binance.vision, 2026-05-15 → 11-08 | eth3 day files `/home/user/data_eth/days`, 1-s timeline `/home/user/data_eth/timeline` (eth3's own builder) |
| truth table | `scripts/truth.py` | 46,368 windows; Polymarket outcome agrees with the Binance label 95.7 % (point months, nominal window) / 94.1 % (TWAP months, effective window [T−30, T+270]) |
| ETHUSDT perp book depth (30-s snapshots) and metrics | data.binance.vision bookDepth / metrics | 160 days, `/home/user/data_eth/extra` |
| master table | `scripts/build_master.py` | `/home/user/data_eth/master.parquet`: 46,050 windows × 1,519 columns (outcome, truth, regimes, eth3 at 29 decision offsets, eth1 at T−120/−60/0) |

## 2. The models on ETH

* **eth3** (`btcmodel`, ETH version of btcmovement2): the frozen `models/wf5_20260430` (trained through 2026-04-30), scored at every
  offset exactly like BTC (`scripts/score_m3.py` = score_m2.py with the ETH paths; batch-vs-Predictor consistency 0.0). eth3 adds eight
  ETH-specific reversal features; no BTC input.
* **eth1** (`ethpredict`, ETH version of btc-predict): the shipped boosters were trained through 2026-11-02, so they were **not** used.
  `scripts/train_ep_folds.py` retrains the spot-variant core model (ETH spot + BTC spot as lead, 136 / 169 features, no perp, premium,
  Coinbase or alt data) in honest 2-month walk-forward blocks, as `train_bp_*.py` did for BTC. Out of sample: s5 IC 0.017, hit 52.2 %,
  AUC 0.531; s1 IC 0.039, hit 53.0 % (the production model's own holdout numbers are 0.016 / 52.0 %). Live path `polyeth/ep_live.py`
  reproduces the training features exactly (parity test in the module).
* **eth2** (`ethmove`): not scored — it needs its own multi-month ETH + BTC data build (monthly npz, seasonal and regime files) and the
  BTC sibling was not part of the live rule. Low priority; noted for completeness.

Monthly AUC of the heads against the Polymarket outcome (BTC's frozen heads in brackets):

| month | eth3 close head @T−45 | eth3 @T+90 | eth1 z_s5 @T−60 | confident windows (\|p−0.5\| ≥ 0.05): n / accuracy |
|---|---|---|---|---|
| Jun | 0.507 | 0.771 | 0.536 | 693 / 51.5 % |
| Jul | 0.520 | 0.769 | 0.532 | 661 / 54.5 % |
| Aug | 0.526 (BTC 0.525) | 0.778 | 0.522 (0.509) | 1,176 / 52.3 % (BTC 1,977 / 53.2 %) |
| Sep | 0.519 (0.521) | 0.790 | 0.517 (0.516) | 775 / 53.2 % (1,489 / 53.8 %) |
| Oct | 0.520 (0.527) | 0.794 | 0.527 (0.529) | 985 / 52.6 % (1,850 / 54.8 %) |
| Nov 1–7 | 0.522 (0.546) | 0.805 | 0.536 (0.518) | 317 / 52.7 % |

## 3. Market depth: the binding constraint (`research/capacity_eth.py`)

Per window, median over the month; "mid" = $ printed at 0.49–0.51 (UP-equivalent) before the nominal open, i.e. the most a resting
0.50 bid could have taken; "hi" = $ printed at ≥ 0.85 after the open (A's sell limit).

| | ETH Jun | ETH Aug | ETH Oct | BTC Jun | BTC Aug | BTC Oct |
|---|---|---|---|---|---|---|
| volume per window | $3,608 | $2,223 | $1,480 | $39,496 | $25,103 | $20,596 |
| pre-open prints | 9 | 10 | 7 | 74 | 70 | 72 |
| mid $ | $20 | $25 | $10 | $606 | $487 | $392 |
| windows with mid ≥ $100 | 9 % | 8 % | 3 % | 96 % | 96 % | 94 % |
| windows with mid ≥ $50 | 24 % | 27 % | 10 % | 99 % | 98 % | 98 % |
| hi $ | $1,480 | $890 | $705 | $14,651 | $9,597 | $8,987 |

ETH volume per window has also halved from June to October while BTC's fell by a third.

## 4. The BTC rules applied unchanged to ETH (`polyeth/final_report_data.py`, `reports/summary.json`)

| strategy (BTC's frozen rule) | Aug 1–19 | tune | test | verdict on ETH |
|---|---|---|---|---|
| A: eth3 close head @T−45, \|p−0.5\| ≥ 0.05, eth1 z_s5 agrees, buy 0.50, sell 0.85 else settle | +0.1 ¢, 59 % | +2.4 ¢, 61.5 %, 785 trades | **+2.5 ¢, 61.8 %, 770 trades, t 1.7** | works, weaker than BTC (test +4.7 ¢ / 64.3 %) |
| A with the previous exits (0.80 / SL 0.05 / T+240) | +1.0 ¢ | +1.6 ¢ | +2.8 ¢, 64.0 % | — |
| A + book vote (ETHUSDT 1 % depth imbalance agrees) | +1.2 ¢ | +2.4 ¢ | +4.3 ¢, 63.9 %, 535 trades | helps on test, see §6 |
| B: open mispricing, delta 0.12, sell 0.85 else settle | +3.9 ¢ | +2.4 ¢ | **−5.7 ¢, 52 %, t −2.4** | **rejected** (every delta 0.06–0.15 and every exit in the grid is negative on test) |
| C: "B five seconds early" | +3.4 ¢ | +3.0 ¢ | −1.7 ¢ | rejected |
| D: T+90, model ≥ 0.90, market in [0.80, 0.90] | +0.2 ¢ | −2.4 ¢ | −1.5 ¢ (86 % win, loses on the misses) | rejected; none of the eleven D variants in `reports/D_winrate.csv` is positive on tune and test |
| A+B+D combined, test, $100 per trade | | | $309 over 34 days | B and D cancel A |

## 5. Every exit combination for A (`research/exit_grid_time.py`, `exit_grid_full.py`, `settle_exit_A.py`)

* **141,120 cells** (TP 0.52–0.99 × SL 0.01–0.49 × hold T+5…T+300 in 5-s steps, 1-cent grid; `reports/A_exit_grid_time.csv`), frozen signal
  (thr 0.05, eth1 agreement). Selection on tune: the ridge is **no stop (SL 0.01) with a high sell limit and a long hold**: best cell per hold
  TP 0.74 @T+180 (+2.4 ¢ tune / +1.8 test), TP 0.79 @T+240 (+2.4 / +3.1, win 64–66 %), TP 0.93 @T+270 (+2.9 / +3.1, win 56 %). Any
  stop ≥ 0.13 is negative on tune at every TP; stops between 0.04 and 0.10 cost 0.5–1 ¢ versus no stop. Same shape as BTC.
* **Settlement variants** (`reports/A_settle_exit_main.csv`, paired day bootstrap vs the 0.80/0.05/T+240 exits): on tune "**sell 0.80 then
  settle**" is the best exit (+2.8 ¢, 65.9 % win, p = 0.005), ahead of 0.85-then-settle (+2.4 ¢, 61.5 %); on test it is +2.8 ¢ / 66.0 % vs
  +2.5 ¢ / 61.8 %. "Hold to settlement, no exits" has the highest mean on test (+4.4 ¢) but 54 % win rate and a −$1,000 worst day.
  **Adopted for ETH: sell 0.80 then hold to settlement** (`CFG_A`), i.e. the only ETH-specific change to A's exits.

## 6. A's own parameters on ETH (`research/eth_A_grid.py`, `research/eth_A_confirm.py`, `reports/eth_A_grid.csv`, `reports/eth_A_confirm.csv`)

Decision at T−45, sell 0.80 then settle, $100 per trade. Win rate / ¢ per share; periods Jun · Jul · Aug 1–19 · tune · test · Nov 4–7.

| vote | thr | win rate | edge (¢) | trades/day (test) |
|---|---|---|---|---|
| eth1 agrees | 0.03 | 63.6 · 67.6 · 64.6 · 66.6 · 64.7 · 70.6 | +1.0 · +4.3 · +1.7 · +3.4 · +1.7 · +6.5 | 65 |
| **eth1 agrees (adopted)** | **0.05** | **65.9 · 71.7 · 63.3 · 65.9 · 66.0 · 74.7** | **+2.9 · +7.5 · +0.7 · +2.8 · +2.8 · +9.8** | 23 |
| eth1 agrees | 0.06 | 67.2 · 72.5 · 65.0 · 67.7 · 64.9 · 74.2 | +3.9 · +8.1 · +2.0 · +4.3 · +1.9 · +9.4 | 13 |
| eth1 + book agree | 0.03 | 64.7 · 67.9 · 64.7 · 67.7 · 65.9 · 70.9 | +1.8 · +4.5 · +1.8 · +4.2 · +2.7 · +6.7 | 45 |
| eth1 + book agree | 0.05 | 67.4 · 71.1 · 62.8 · 66.2 · 68.4 · 72.3 | +4.0 · +7.0 · +0.2 · +3.0 · +4.7 · +7.8 | 16 |
| eth1 + book agree | 0.06 | 68.2 · 71.7 · 65.3 · 68.7 · 67.7 · 72.1 | +4.5 · +7.3 · +2.3 · +5.0 · +4.2 · +7.7 | 9 |
| no vote | 0.05 | 59.2 · 64.4 · 59.0 · 60.5 · 60.2 · 65.3 (0.85 exit) | +0.4 · +4.9 · +0.2 · +1.5 · +1.2 · +5.5 | 32 |

Every row with the eth1 vote is positive in all six periods; the eth1 vote itself adds 2–3 points of win rate everywhere (BTC: +1–2).
The **book vote on ETH is mixed**: it lifts test (+4.7 vs +2.8 ¢) and June but not July, tune or Nov 4–7, so it is logged in shadow
(`BOOK_VOTE=0`), not adopted. Thresholds 0.06 raise the win rate further at a third of the trades; thr 0.03 with both votes keeps
45 trades/day at a lower edge. Decision at T−60 or T−30 is not better than T−45 (`reports/eth_A_grid.csv`).

**Capacity-weighted** (`research/eth_A_confirm.py`; fill = min(stake, $ printed at 0.49–0.51 before the open), sell 0.85 then settle):

| rule | $ per day at $25 stake: Jun · Jul · tune · test | at $100 | at $100 ignoring capacity |
|---|---|---|---|
| eth1, thr 0.05 | 1.6 · 40 · 3.6 · 4.2 | −2.7 · 58 · 17 · −2.5 | 96 · 243 · 88 · 115 |
| eth1 + book, thr 0.03 | 13 · 63 · 40 · 17 | 10 · 94 · 97 · 12 | 142 · 388 · 325 · 273 |
| eth1, thr 0.03 | 2.4 · 88 · 44 · −1.9 | −18 · 161 · 119 · −21 | 112 · 491 · 376 · 222 |

The windows deep enough to fill a $100 bid are the busy ones, and in those A's edge is about zero (test −$2.5/day). At a $25 stake the
rule fills most of its trades and makes $2–60 a day. This is the number that matters for sizing ETH.

## 7. Live path (`polyeth/paper_trader.py`, `polyeth/ep_live.py`, `polyeth/bookdepth.py`, `scripts/setup_box.sh`)

`polyeth/paper_trader.py` is the BTC trader with the asset swapped: ETHUSDT spot 1-s klines (websocket mirror + REST backfill), BTCUSDT
1-s klines as the **lead feed** (the eth1 spot variant's only other input), OKX ETH-USDT-SWAP trades standing in for the Binance perp on
US hosts, eth3's deployment model `prod_20261102` (the backtest used the frozen `wf5_20260430`; set `M3_MODEL` to switch), the eth1
production spot boosters through `ep_live.py`, markets looked up by `eth-updown-5m-<T>`, logs in `/home/user/paper_eth`. Only **A** is
enabled (`ENABLED = {"A"}`, sell 0.80 then settle); B, C, D stay in the code switched off. The ETHUSDT book-depth vote is logged with every
decision (`book_imb`, `book_agree`) and gates only with `BOOK_VOTE=1`. Offline check on 2026-11-06 from the trader's own timeline: the
T−45 close head matches the master table on 3 of 5 windows exactly and within 0.002 on the other two (regime-cache chunking, same as BTC).
Stake: start at **$20–25 per trade** (section 6), `MAX_OPEN` 2; the executor and risk limits are the BTC ones.

## 8. Layout

```
polyeth/     polysim.py · strategy_eval.py · experiments.py · final_strategy.py (CFG_A: thr 0.05, eth1 agree, sell 0.80 then settle)
             final_report_data.py (A/A0/AB/B/C/D on old/tune/test + B grid) · bookdepth.py (ETHUSDT) · ep_live.py (eth1 live path)
             paper_trader.py · live_executor.py / paper_audit.py are shared with polybtcmodel
scripts/     fetch_polymarket{,_range}.py (series 10683) · fetch_binance.py (ETHUSDT -> eth3 day files) · fetch_extra_feeds.py (ETHUSDT depth)
             truth.py · score_m3.py · train_ep_folds.py · build_master.py · run_eth3_pipeline.sh / run_eth3_gap.sh / run_eth_research.sh · setup_box.sh
research/    ref_timing_eth.py · capacity_eth.py · eth_A_grid.py · eth_A_confirm.py · exit_grid_time.py · exit_grid_full.py · settle_exit_A.py
             d_winrate.py · ideas10.py · book_vote_A.py · book_vote_monthly.py · sl_bootstrap.py
reports/     summary.json · trades_*.csv · B_exit_grid.csv · A_exit_grid_{time,full}.csv · A_settle_exit_main.csv · D_winrate.csv
             eth_A_grid.csv · eth_A_confirm.csv · A_book_vote.csv
```
Data lives outside git in `/home/user/data_eth` (markets, trades, days, timeline, truth, forecasts, forecasts_L0, extra).
