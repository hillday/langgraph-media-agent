#!/bin/bash

# 获取脚本所在目录
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

PID_FILE="app.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    
    # 检查进程是否存在
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Stopping LangGraph Media Agent (PID $PID)..."
        kill "$PID"
        
        # 等待进程完全退出
        while ps -p "$PID" > /dev/null 2>&1; do
            echo "Waiting for process to exit..."
            sleep 1
        done
        
        echo "Service stopped."
        rm "$PID_FILE"
    else
        echo "Service is not running (PID $PID not found)."
        rm "$PID_FILE"
    fi
else
    echo "No PID file found. Is the service running?"
    exit 1
fi
