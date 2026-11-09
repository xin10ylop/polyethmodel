#!/bin/bash
# One-shot setup of the ETH paper trader (polyethmodel) on a fresh Ubuntu box (run as a user with sudo).
# The research scripts hard-code /home/user/... paths, so the repos are placed there.
set -e
SUDO=$(command -v sudo || true); [ "$(id -u)" = 0 ] && SUDO=""        # works as a sudo user or as root
ME=$(id -un)
GH=${GH:-https://github.com/xin10ylop}      # use git@github.com:xin10ylop if you clone over SSH
BR=claude/polymarket-btc-trading-system-ewbqwd
echo "== feed reachability (US IPs are blocked by Binance's main API; the mirrors below must answer)"
curl -sS -m 10 "https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1s&limit=1" | head -c 80; echo
curl -sS -m 10 "https://www.okx.com/api/v5/market/trades?instId=BTC-USDT-SWAP&limit=1" | head -c 80; echo
echo "== Binance perp (the archive's own feed; HTTP 451 = restricted location, then keep PERP_FEED=okx):"
curl -sS -m 10 -o /dev/null -w "fapi.binance.com -> HTTP %{http_code}\n" "https://fapi.binance.com/fapi/v1/aggTrades?symbol=BTCUSDT&limit=1" || true
curl -sS -m 10 -o /dev/null -w "clob.polymarket.com -> HTTP %{http_code}\n" "https://clob.polymarket.com/markets?limit=1" || true
$SUDO mkdir -p /home/user && $SUDO chown "$ME" /home/user
cd /home/user
mkdir -p /home/user/xin10ylop; for r in eth1 eth3; do [ -d /home/user/xin10ylop/$r ] || git clone $GH/$r.git /home/user/xin10ylop/$r; done   # ETH models (eth1 = btc-predict sibling, eth3 = btcmovement2 sibling)
[ -d btc-predict ] || git clone $GH/btc-predict.git                 # its per-second archive format serves the BTC-spot lead feed
[ -d polyethmodel ] || git clone -b $BR $GH/polyethmodel.git
[ -d polybtcmodel ] || git clone -b $BR $GH/polybtcmodel.git     # its fetch_binance.py fills btc-predict/data/sec/spot (BTC lead)
git -C polyethmodel pull -q; git -C polybtcmodel pull -q           # polyethmodel is the copy that runs
$SUDO apt-get install -y -qq git curl python3-venv python3-full libgomp1 > /dev/null   # libgomp1: OpenMP runtime for LightGBM
[ -d /home/user/venv ] || python3 -m venv /home/user/venv
PY=/home/user/venv/bin/python3
$PY -m pip install -q --upgrade pip
$PY -m pip install -q -r polybtcmodel/requirements.txt websocket-client polymarket-client   # polymarket-client: official SDK, used only when LIVE=1
echo "== 10 days of Binance archive history (spot, ETH, perp) in every format the models need"
START=$(date -u -d "11 days ago" +%F); END=$(date -u -d "yesterday" +%F)
$PY polyethmodel/scripts/fetch_binance.py $START $END 4              # ETHUSDT spot 1 s + perp aggTrades -> /home/user/data_eth/days
$PY polybtcmodel/scripts/fetch_binance.py $START $END 4              # BTC archive (btc-predict/data/sec/spot = the lead feed)
BTC_DATA_DIR=/home/user/data_eth BTC_ASSET=ETH $PY -c "import sys; sys.path.insert(0,'/home/user/xin10ylop/eth3'); from btcmodel.timeline import build_timeline; build_timeline()"
echo "== start the paper trader (auto-restart loop, logs in /home/user/paper_eth/, outside git)"
export PAPER_DIR=/home/user/paper_eth; mkdir -p $PAPER_DIR
cd /home/user/polyethmodel
for pid in $(ps -eo pid,cmd | awk '/[p]aper_trader.py/ {print $1}'); do kill $pid; done; sleep 2    # one trader only
export BOOK_FEED=${BOOK_FEED:-binance_perp} BOOK_VOTE=${BOOK_VOTE:-0}   # book-depth vote for A (polybtc/bookdepth.py): binance_perp needs a non-US host; BOOK_VOTE=1 gates trades, 0 = shadow log only
curl -sS -m 10 -o /dev/null -w "fapi.binance.com depth -> HTTP %{http_code} (451 = US host: use BOOK_FEED=binance_spot and keep BOOK_VOTE=0 until validated)\n" "https://fapi.binance.com/fapi/v1/depth?symbol=BTCUSDT&limit=5" || true
export PERP_FEED=${PERP_FEED:-okx}     # PERP_FEED=binance (stream) or binance_rest (polling) on a non-US host; okx otherwise
export LIVE=${LIVE:-0} DRY_RUN=${DRY_RUN:-1} STAKE_USD=${STAKE_USD:-10} MAX_OPEN=${MAX_OPEN:-4} DAILY_LOSS_LIMIT=${DAILY_LOSS_LIMIT:--100}   # LIVE=1 DRY_RUN=0 + POLYMARKET_* keys = real orders
if [ "$LIVE" = "1" ] && [ "$DRY_RUN" = "0" ]; then
  for v in POLYMARKET_PRIVATE_KEY POLYMARKET_WALLET_ADDRESS; do [ -n "${!v:-}" ] || { echo "LIVE=1 DRY_RUN=0 but $v is not set in this shell: not starting"; exit 1; }; done
  [ -n "${POLYMARKET_RELAYER_API_KEY:-}" ] || echo "   WARNING: POLYMARKET_RELAYER_API_KEY not set: winners held to settlement will not be redeemed automatically"
  echo "   *** REAL ORDERS: stake \$$STAKE_USD, max open $MAX_OPEN, daily loss limit \$$DAILY_LOSS_LIMIT ***"
fi
echo "   perp feed $PERP_FEED | live $LIVE (dry_run $DRY_RUN, stake \$$STAKE_USD)"
nohup bash -c 'while true; do /home/user/venv/bin/python3 -W ignore polyeth/paper_trader.py >> $PAPER_DIR/stdout.log 2>&1; echo "trader exited $(date -u)" >> $PAPER_DIR/stdout.log; sleep 5; done' > /dev/null 2>&1 &
sleep 5; tail -3 $PAPER_DIR/paper.log 2>/dev/null || true
echo "running.  log:   tail -f /home/user/paper_eth/paper.log"
echo "          tally: PAPER_DIR=/home/user/paper_eth /home/user/venv/bin/python3 /home/user/polyethmodel/polyeth/paper_trader.py --summary"
echo "          audit: PAPER_DIR=/home/user/paper_eth /home/user/venv/bin/python3 /home/user/polyethmodel/polyeth/paper_audit.py --day $(date -u -d yesterday +%F)"
