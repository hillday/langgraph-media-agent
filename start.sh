#!/bin/bash

# 获取脚本所在目录，确保不管从哪里执行都能正确进入项目根目录
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# 定义日志文件路径
LOG_FILE="app.log"
PID_FILE="app.pid"

# 检查是否已经有进程在运行
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Service is already running with PID $PID"
        exit 1
    else
        # 进程已死，但 pid 文件还在，清理掉
        echo "Found stale PID file. Cleaning up..."
        rm "$PID_FILE"
    fi
fi

echo "Starting LangGraph Media Agent..."

# 如果使用了虚拟环境，可以在这里激活
# source venv/bin/activate

# 使用 nohup 后台启动应用，并将 stdout 和 stderr 重定向到 app.log
nohup python app.py > "$LOG_FILE" 2>&1 &

# 获取后台进程的 PID 并保存
PID=$!
echo $PID > "$PID_FILE"

echo "Service started in background with PID $PID"
echo "Logs are being written to $LOG_FILE"
echo "You can check logs using: tail -f $LOG_FILE"
