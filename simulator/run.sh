#!/bin/bash
set -x
python3 simulator.py --interval 60 --policy fifo workloads/test.csv 2>&1 | tee trace_log/log_fifo
python3 simulator.py --interval 60 --policy easyscale workloads/test.csv 2>&1 | tee trace_log/log_easyscale
python3 simulator.py --interval 60 --policy easyscale_homo workloads/test.csv 2>&1 | tee trace_log/log_easyscale_homo
python3 plot_summary.py
