# Going live with strategy A (BTC and ETH), $5 stakes, new model

Audit of 2026-11-09; the new rule was wired in and audited a second time on 2026-11-10 (findings 14-20 below).

**What runs: only strategy A on both coins, with the NEW model** (README section 10; `A_RULE=new` is the default, `A_RULE=old` brings the
previous rule back):
* at about T-43.8, as soon as the spot price of second T-45 has arrived, the purpose-built model scores the window from the feature engine's row
  (`polybtc/newa_live.py`, models `models/ourcase_<COIN>_d-45_live.*`, trained on every window Jan 2025 - Nov 7 2026);
* it trades only when |score| is in its top 15 % (threshold stored with the model): score > 0 buys UP, < 0 buys DOWN;
* BUY limit 0.50 (post-only), sent right after the score, cancelled at T-30 if unfilled (`A_CANCEL_S`); on a fill a SELL limit rests at 0.85 on BTC / 0.80 on ETH; whatever is not
  sold is held to settlement and redeemed.
The old rule (close head + second model) is computed AFTER the order has gone out and written to `forecasts.jsonl` as `old_action`, for
comparison only: an error or a delay in the old models cannot stop or slow the new rule. `BOOK_VOTE` has no effect on the new rule.
B, C, D and E are off and cannot place orders (`STRATEGIES=A` is the default).

The live model is the research model WITHOUT the book-depth inputs: the exchange's live depth feed does not reach 1 % from the price, so
those inputs cannot be rebuilt live. Walk-forward accuracy is nearly the same (AUC vs the Polymarket outcome, Jun-Nov: BTC 0.528 vs 0.531
with the book, ETH 0.535 vs 0.536). Backtest of exactly this live rule, every order assumed filled at 0.50, BTC $100 + ETH $25 per trade,
Jun 1 - Nov 7 (`research/deep/live_rule_table.py`): account $81,190 (old rule $40,015), win 66.1 % (64.1 %), max drawdown $1,780 ($3,010),
about 43 signals a day per coin. Live path = research path: `python tests/test_newa_parity.py polybtc|polyeth` puts the trader in the
exact state it has at the decision (bars through T-46 final, spot price of T-45 arrived, slow features computed at T-52) and compares with
the research score: 60 windows per coin, largest difference 0.0015 (BTC) / 0.0019 (ETH) against thresholds of 0.129 / 0.140, no trade
decision different.

## Settings added on 2026-11-10 after the first real trades (all optional; defaults unchanged)
The first real BTC orders (bid placed, cancelled, filled, sold at 0.85 with no fee) showed what the backtest table cannot: most signals never
fill at 0.50. Research on the real trade tape (`research/deep/exec_opt.py`, `cancel_late.py`, `tp_entry.py`, `early_decision.py`,
`trades_per_day.py`, `max_stake.py`), BTC, Jun 1 - Nov 7, $5 stake; "every signal filled at 0.50" = $21 a day:

| setup | trades/day | c/share | $/day at $5 |
|---|---|---|---|
| decision T-45, bid 0.50 to T-30, never a fee (the default) | 8.5 | +3.7 | 3.2 |
| + `ALLOW_TAKER=1` (take the offer when our side is already offered at <= 0.50) and `A_CANCEL_S=-15` | 21 | +2.9 | 6.0 |
| **`A_DEC=-120`** (decide 120 s before the open, model `ourcase_BTC_d-120_live`), `ALLOW_TAKER=1`, bid kept to T-15 | 29 | +3.0 | 8.6 |

* `A_DEC=-120`: decision time. The earlier bid rests while fewer traders have seen the move; positive in all three periods checked
  (Jun 1 - Aug 19, Aug 20 - Sep 30, Oct 1 - Nov 7: +3.4 / +3.0 / +2.5 c a share). Live path = research path:
  `A_DEC=-120 python tests/test_newa_parity.py polybtc` (40 windows, largest score difference 0.0005). BTC only: no ETH model at T-120.
* `A_CANCEL_S`: when the unfilled bid is cancelled (default T-30 at T-45, T-15 at earlier decisions). Keeping the bid past T-15, to the
  open or after it, adds trades with about no edge.
* `ALLOW_TAKER=1`: a bid refused as "crosses the book" is sent again as a plain limit at 0.50: it fills at 0.50 or better and pays the
  taker fee (about 1.75 c a share). Never more than 0.50.
* `A_TP=0.90` or `A_TP=settle`: sell price. 0.90 adds 0.1-0.2 c a share; holding to settlement about 0.7 c with larger swings.
* Stake: the volume sold at 0.50 while the bid rests caps the useful size at about $200 a trade on BTC (`max_stake.py`).

## What the audit found and fixed
| # | defect | effect if left | status |
|---|---|---|---|
| 1 | B's decision block ignored the "enabled" switch and sent real orders | B would have traded live on both bots although only A is wanted | fixed (guards + `LIVE_STRATEGIES = {"A"}`) |
| 2 | ETH bot: OKX contract size 0.01 (BTC's) instead of 0.1 ETH | perp volumes 10x too small in the ETH model inputs | fixed |
| 3 | ETH bot: one-day hole in the BTC lead feed between its archive and the live backfill | second model computed on missing data for up to two days after a start | fixed (backfill starts where each archive ends, both bots) |
| 4 | no "too late" guard on A | after a start or a stall the bot could bid at 0.50 seconds before or after the open | fixed (decision skipped after T-40.5; executor refuses a bid within 1 s of its cancel time) |
| 5 | starting one coin's bot killed the other (`setup_box.sh` process pattern) | BTC bot silently stopped when ETH was set up | fixed |
| 6 | each bot's start / stop cancelled EVERY order of the wallet | the other coin's take-profit orders vanished and were never re-posted | fixed (a bot cancels only its own orders; a missing take-profit is posted again every 3 s) |
| 7 | take-profit posted once, never retried | a rejected SELL turned the trade into hold-to-settlement | fixed |
| 8 | SIGTERM could deadlock on the executor lock; executor was created after the 3-minute backfill | a crashed run's bid stayed on the book for minutes; unclean stops | fixed (re-entrant lock, executor first, fills made while down are recovered) |
| 9 | slow web lookups (settlement) could delay the cancel | bid resting into the open | fixed (5 s timeout, none between T-60 and T-28, none while a bid rests) |
| 10 | no data-quality gate | trades on a stale spot feed, synthetic perp bars or a dead second feed | fixed (no trade, reason logged) |
| 11 | ETH restart re-posted the take-profit at 0.85 instead of 0.80; ETH executor default log dir | wrong exit after a restart | fixed |
| 12 | defaults were $10 stake, 4 open, -$100 daily stop | too loose for a $5 test | now $5, 3 open, -$25 per bot per day |
| 13 | trader stops working after 15 days of uptime (timeline capacity) | silent stop | now exits and restarts; refresh the archive at least every two weeks by re-running `setup_box.sh` |
| 14 | (2nd audit) after every start the last day or two of perp bars are rebuilt from spot, and the gate only looked at the last 60 s | first trades after a restart on inputs the model never saw (score off by up to 0.025 six minutes after a start) | fixed: no trade while more than 30 rebuilt perp seconds lie in the last hour, or any in the last minute (`PERP_CLEAN_S=3600`, `PERP_CLEAN_MAX=30`). **The bot does not trade in its first hour.** |
| 15 | the decision waited for a perp trade in second T-45 that the model never uses | order sent 1-2 s later than needed, or the window missed on a slow perp feed | fixed: decides as soon as the spot price of T-45 is in (about T-43.8); from T-42 it decides without it and logs `open_known: false` |
| 16 | the new rule ran inside the old model's code path | an error or an implausible value in the old heads cancelled the new rule's trade | fixed: row -> score -> gate -> order first; old heads afterwards, in their own error handler |
| 17 | `paper_audit.py` still checked the old rule and crashed on A | daily audit useless for the live rule | fixed: checks `new_score` against `new_thr`, replays the new score from the archive |
| 18 | settlement lookups resumed at T-38 although the bid now rests until T-30 | a slow lookup could delay the cancel | fixed: none between T-60 and T-28 |
| 19 | real orders possible on the OKX fallback feed | live trades on inputs far from the backtest | fixed: with `DRY_RUN=0` orders are refused unless `PERP_FEED` is `binance` or `binance_rest` (`ALLOW_OKX_LIVE=1` overrides) |
| 20 | a mistyped `A_RULE` silently ran the old rule; the model file was only loaded after the 3-minute backfill; executor defaults were $10 / 4 open / -$100 when started without `setup_box.sh` | wrong rule or wrong size without notice | fixed: start refuses any `A_RULE` other than `new` / `old`, loads the model first; defaults are $5 / 3 open / -$25 everywhere |

Verified: SDK calls match `polymarket-client` 0.18.0 (pinned); 15 fake-exchange checks of the real-order path pass on both bots
(`python tests/test_executor_fake.py polybtc|polyeth`); both bots ran here against live feeds in dry-run; ETH live-path forecasts match
the archive (30 windows: no trade-decision difference).

## What could NOT be tested here (do these on the box, in this order)
1. **A real order has never been sent by this code.** Start with `LIVE=1 DRY_RUN=1` for an hour and read `live.log`. Then `DRY_RUN=0` and
   watch the first three trades next to the Polymarket page: the BUY appears at 0.50 about 43 s before the open and disappears 30 s before it if unfilled; after a fill a SELL
   appears at 0.85 / 0.80; after the close the position settles and, if it won, is redeemed (needs `POLYMARKET_RELAYER_API_KEY`).
2. **Order status after a full fill.** The bot learns about fills by asking for the order's status. If the first filled BUY is not followed
   within two seconds by a "FILLED" line in `live.log`, stop (touch `$PAPER_DIR/STOP`) and send me the log.
3. **Fill rate.** The bid rests from about T-43.5 to T-30. In the trade history (a resting bid fills only when someone sells into it), a
   0.50 bid over that stretch would have filled in 17-33 % of BTC signals (7-14 fills a day; filled trades won 63-64 %, about +4 cents a
   share) and in 3-6 % of ETH signals (2-3 fills a day; edge about +1 cent, not distinguishable from zero). It is refused outright when
   someone already offers our side at 0.50 (post-only, never a fee). The backtest table above assumes every order fills, so it is an upper
   bound; the real fill rate and the result of the filled trades in `live_orders.jsonl` / `live_trades.jsonl` are what this $5 test measures.
   The T-30 cancel was chosen on BTC evidence; for ETH there are too few fills in the history to say whether T-30 or the open is better.
4. **Wait one hour after each start** before expecting trades (finding 14). `paper.log` shows "NO TRADE (data quality): ... rebuilt perp
   seconds" for signals in that hour. If that line keeps appearing after the first hour, the perp feed is dropping seconds: send me
   `forecasts.jsonl` (fields `synth_last60`, `synth_clean`, `perp_lag`).

## Commands
```bash
# BTC bot
export POLYMARKET_PRIVATE_KEY=... POLYMARKET_WALLET_ADDRESS=... POLYMARKET_RELAYER_API_KEY=... POLYMARKET_RELAYER_API_KEY_ADDRESS=...
PERP_FEED=binance_rest LIVE=1 DRY_RUN=1 STAKE_USD=5 bash /home/user/polybtcmodel/scripts/setup_box.sh      # dry run first; DRY_RUN=0 for real orders
```
```bash
# ETH bot (same wallet is fine: each bot only touches its own orders)
PERP_FEED=binance_rest LIVE=1 DRY_RUN=1 STAKE_USD=5 bash /home/user/polyethmodel/scripts/setup_box.sh
```
```bash
# stop new orders of one bot at once (cancels its resting bid): BTC /home/user/paper, ETH /home/user/paper_eth
touch /home/user/paper/STOP
```
Logs per bot in its PAPER_DIR: `paper.log` (decisions), `live.log` (orders), `live_orders.jsonl`, `live_trades.jsonl`, `forecasts.jsonl`
(every decision with its data-quality fields), `book.jsonl`.

## Limits to keep in mind
* Each bot has its own daily stop (-$25): the account can lose $50 in a day before both halt. Max 3 open positions per bot.
* After every start the perp bars between the archive's last day and "now" are rebuilt from spot; the bot trades only once a full hour
  of real perp bars has been received (after one hour its scores are within 0.004 of the archive's). The same applies after a perp-feed outage.
* The model was trained on archives in which one day (2026-10-28) has no perp data; not corrected.
* Positions filled between the last status check and a crash are picked up at the next start only if the state file survived.
* **Perp feed: use `PERP_FEED=binance_rest` (or `binance`) on the non-US box**, as in the 2026-11-05 audit (Frankfurt: A forecasts within 0.006 of the
  archive, 1 decision of 127 different, 0 synthetic seconds). The default `okx` is only a fallback for US hosts: with it the forecasts are far
  from the backtest (README: direction flips on 10 of 30 windows) and the new data-quality gate will block many trades. The ETH bot's Binance
  perp handlers are the BTC ones with the symbol changed; they could not be run from this container (Binance futures refuses US hosts), so check
  in the ETH bot's first dry-run hours that `forecasts.jsonl` shows `synth_clean: 0` from the second hour on and `perp_lag` under 2 s.
