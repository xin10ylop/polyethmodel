#!/bin/bash
# Chain: wait for the Binance ETH day files -> eth3 timeline -> truth table -> eth3 consistency test -> eth3 forecasts
# (L=0 point-price months Jun 1 - Aug 2; L=30 TWAP months Aug 20 - Nov 8; the Aug 3-19 gap is run once the switch date is known)
set -e
export BTC_DATA_DIR=/home/user/data_eth BTC_ASSET=ETH
LOG=/home/user/data_eth/logs
until grep -q "^done" $LOG/binance.log; do sleep 20; done
echo "days: $(ls /home/user/data_eth/days | wc -l)"
cd /home/user/xin10ylop/eth3 && python3 -c "
from btcmodel.timeline import build_timeline; import datetime as dt
build_timeline(dt.date(2026,5,15), dt.date(2026,11,8))" > $LOG/timeline.log 2>&1 && echo "timeline built: $(cat /home/user/data_eth/timeline/meta.txt | head -3 | tr '\n' ' ')"
cd /home/user/polyethmodel && python3 scripts/truth.py 2026-06-01 2026-11-08 > $LOG/truth.log 2>&1 && echo "truth: $(tail -n 1 $LOG/truth.log)"
python3 scripts/score_m3.py test > $LOG/score_m3_test.log 2>&1; echo "consistency: $(tail -n 1 $LOG/score_m3_test.log)"
M2_REF_LAG=0 M2_OUT=/home/user/data_eth/forecasts_L0 python3 scripts/score_m3.py 2026-06-01 2026-08-02 4 > $LOG/score_m3_L0.log 2>&1 && echo "L0 done"
M2_REF_LAG=30 M2_OUT=/home/user/data_eth/forecasts python3 scripts/score_m3.py 2026-08-20 2026-11-08 4 > $LOG/score_m3_L30.log 2>&1 && echo "L30 done"
echo ALL DONE
