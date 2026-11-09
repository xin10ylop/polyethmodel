#!/bin/bash
# sequential (memory): A parameter grid, book vote per period, exhaustive exit grids, settle variants, D variants
LOG=/home/user/data_eth/logs; cd /home/user/polyethmodel
MAIN_DAYS=$(python3 -c "import pandas as pd; print(','.join(str(d.date()) for d in pd.date_range('2026-08-01','2026-11-03')))")
python3 research/eth_A_grid.py > $LOG/eth_A_grid.log 2>&1; echo "eth_A_grid $?"
python3 research/book_vote_A.py > $LOG/book_vote_A.log 2>&1; echo "book_vote_A $?"
python3 research/exit_grid_time.py > $LOG/exit_grid_time.log 2>&1; echo "exit_grid_time $?"
python3 research/exit_grid_full.py > $LOG/exit_grid_full.log 2>&1; echo "exit_grid_full $?"
POLY_DAYS=$MAIN_DAYS LABEL=main python3 research/settle_exit_A.py > $LOG/settle_exit_A.log 2>&1; echo "settle_exit_A $?"
POLY_DAYS=$MAIN_DAYS python3 research/d_winrate.py > $LOG/d_winrate.log 2>&1; echo "d_winrate $?"
echo RESEARCH DONE
