#!/bin/bash
# ETH markets switched from point price to the 60-s Chainlink TWAP at 2026-08-07 00:00 UTC (gamma descriptions): L=0 through Aug 6, L=30 from Aug 7.
set -e; export BTC_DATA_DIR=/home/user/data_eth BTC_ASSET=ETH; LOG=/home/user/data_eth/logs
until grep -q "ALL DONE" $LOG/pipeline.log; do sleep 60; done
cd /home/user/polyethmodel
M2_REF_LAG=0 M2_OUT=/home/user/data_eth/forecasts_L0 python3 scripts/score_m3.py 2026-08-03 2026-08-06 4 > $LOG/score_m3_L0_gap.log 2>&1 && echo "L0 gap done"
M2_REF_LAG=30 M2_OUT=/home/user/data_eth/forecasts python3 scripts/score_m3.py 2026-08-07 2026-08-19 4 > $LOG/score_m3_L30_gap.log 2>&1 && echo "L30 gap done"
echo GAP DONE
