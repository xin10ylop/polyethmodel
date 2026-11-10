# Going live with strategy A (BTC and ETH), $5 stakes, new model

Audit of 2026-11-09, updated the same day for the new rule.

**What runs: only strategy A on both coins, with the NEW model** (README section 10; `A_RULE=new` is the default, `A_RULE=old` brings the
previous rule back):
* at about T-43, once the bar of second T-45 is final, the purpose-built model scores the window from the feature engine's row
  (`polybtc/newa_live.py`, models `models/ourcase_<COIN>_d-45_live.*`, trained on every window Jan 2025 - Nov 7 2026);
* it trades only when |score| is in its top 15 % (threshold stored with the model): score > 0 buys UP, < 0 buys DOWN;
* BUY limit 0.50 (post-only), cancelled at T-40 if unfilled; on a fill a SELL limit rests at 0.85 on BTC / 0.80 on ETH; whatever is not
  sold is held to settlement and redeemed.
The old rule (close head + second model) is still computed and written to `forecasts.jsonl` as `old_action`, for comparison only.
B, C, D and E are off and cannot place orders (`STRATEGIES=A` is the default).

The live model is the research model WITHOUT the book-depth inputs: the exchange's live depth feed does not reach 1 % from the price, so
those inputs cannot be rebuilt live. Walk-forward accuracy is nearly the same (AUC vs the Polymarket outcome, Jun-Nov: BTC 0.528 vs 0.531
with the book, ETH 0.535 vs 0.536). Backtest of exactly this live rule, every order assumed filled at 0.50, BTC $100 + ETH $25 per trade,
Jun 1 - Nov 7 (`research/deep/live_rule_table.py`): account $81,190 (old rule $40,015), win 66.1 % (64.1 %), max drawdown $1,780 ($3,010),
about 43 signals a day per coin. Live path = research path: `python tests/test_newa_parity.py polybtc|polyeth` reproduces the research
score exactly on both coins (BTC 40 windows, ETH 60 windows, max difference 0.00000; with the regime features pre-computed at T-70 as live
does: no trade decision changes in 120 windows).

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
| 9 | slow web lookups (settlement) could delay the T-40 cancel | bid resting into the open | fixed (5 s timeout, none between T-60 and T-38, none while a bid rests) |
| 10 | no data-quality gate | trades on a stale spot feed, synthetic perp bars or a dead second feed | fixed (no trade, reason logged) |
| 11 | ETH restart re-posted the take-profit at 0.85 instead of 0.80; ETH executor default log dir | wrong exit after a restart | fixed |
| 12 | defaults were $10 stake, 4 open, -$100 daily stop | too loose for a $5 test | now $5, 3 open, -$25 per bot per day |
| 13 | trader stops working after 15 days of uptime (timeline capacity) | silent stop | now exits and restarts; refresh the archive at least every two weeks by re-running `setup_box.sh` |

Verified: SDK calls match `polymarket-client` 0.18.0 (pinned); 15 fake-exchange checks of the real-order path pass on both bots
(`python tests/test_executor_fake.py polybtc|polyeth`); both bots ran here against live feeds in dry-run; ETH live-path forecasts match
the archive (30 windows: no trade-decision difference).

## What could NOT be tested here (do these on the box, in this order)
1. **A real order has never been sent by this code.** Start with `LIVE=1 DRY_RUN=1` for an hour and read `live.log`. Then `DRY_RUN=0` and
   watch the first three trades next to the Polymarket page: the BUY appears at 0.50 and disappears at T-40 if unfilled; after a fill a SELL
   appears at 0.85 / 0.80; after the close the position settles and, if it won, is redeemed (needs `POLYMARKET_RELAYER_API_KEY`).
2. **Order status after a full fill.** The bot learns about fills by asking for the order's status. If the first filled BUY is not followed
   within two seconds by a "FILLED" line in `live.log`, stop (touch `$PAPER_DIR/STOP`) and send me the log.
3. **Fill rate.** The bid now goes out at about T-42.7 (the model waits for the T-45 bar) and is cancelled at T-40: under three seconds of resting. In the trade history a 0.50 bid placed at T-45 would have filled by T-40 in about
   28 % of BTC signals and 4 % of ETH signals, and it is refused outright when someone already offers our side at 0.50 (post-only, never a fee).
   Expect few fills, especially on ETH. The real numbers from `live_orders.jsonl` are exactly what this test is for.

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
* After every start the perp bars between the archive's last day and "now" are rebuilt from spot until live bars arrive; the gate only
  checks the last 60 seconds. Start the bots right after `setup_box.sh` has fetched yesterday's archive, as the script does.
* Positions filled between the last status check and a crash are picked up at the next start only if the state file survived.
* **Perp feed: use `PERP_FEED=binance_rest` (or `binance`) on the non-US box**, as in the 2026-11-05 audit (Frankfurt: A forecasts within 0.006 of the
  archive, 1 decision of 127 different, 0 synthetic seconds). The default `okx` is only a fallback for US hosts: with it the forecasts are far
  from the backtest (README: direction flips on 10 of 30 windows) and the new data-quality gate will block many trades. The ETH bot's Binance
  perp handlers are the BTC ones with the symbol changed; they could not be run from this container (Binance futures refuses US hosts), so check
  in the ETH bot's first dry-run hour that `forecasts.jsonl` shows `synth_last60: 0` and `perp_lag` under 2 s.
