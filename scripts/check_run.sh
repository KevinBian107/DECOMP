#!/bin/bash
# Compact monitoring snapshot for a detached run_all.py run.
# Usage:  bash scripts/check_run.sh [pid_file]
# Default pid_file: logs/run_expansion.pid

set -u
PID_FILE="${1:-logs/run_expansion.pid}"
LOG_FILE="${PID_FILE%.pid}.log"

echo "=== process ==="
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    ps -p "$PID" -o pid,etime,pcpu,rss,command 2>&1 | head -n 3 || echo "  pid $PID not alive"
else
    echo "  no pid file at $PID_FILE"
fi

echo ""
echo "=== sessions cached ==="
echo "  GLM csvs : $(ls data/cache/*_glm_results.csv 2>/dev/null | wc -l | tr -d ' ')"
echo "  CCA csvs : $(ls data/cache/*_cca_results.csv 2>/dev/null | wc -l | tr -d ' ')"
echo "  SVCA summaries : $(ls data/cache/*_svca_summary.csv 2>/dev/null | wc -l | tr -d ' ')"

echo ""
echo "=== last log lines (filtered) ==="
if [ -f "$LOG_FILE" ]; then
    grep -E "(INFO|WARN|Error|Traceback|Done)" "$LOG_FILE" 2>/dev/null | tail -n 6
else
    echo "  no log file at $LOG_FILE"
fi

echo ""
echo "=== ONE cache size ==="
du -sh ~/Downloads/ONE/ 2>&1 | head -n 1

echo ""
echo "=== outputs ==="
ls -la outputs/ 2>/dev/null | grep -E "\.(png|json)$" | awk '{print "  " $9 " (" $5 " bytes, " $6 " " $7 " " $8 ")"}'
