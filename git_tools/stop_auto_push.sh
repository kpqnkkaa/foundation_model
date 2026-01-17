#!/bin/bash

# 停止自动备份服务

REPO_DIR="/mnt/nvme1n1/lululemon/fm_shijing"
PID_FILE="$REPO_DIR/.auto_push.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "⚠️  未找到 PID 文件，服务可能未运行"
    exit 1
fi

PID=$(cat "$PID_FILE")

if ps -p "$PID" > /dev/null 2>&1; then
    # 终止进程及其子进程
    pkill -P "$PID" 2>/dev/null
    kill "$PID" 2>/dev/null
    sleep 1
    
    # 确认是否已停止
    if ps -p "$PID" > /dev/null 2>&1; then
        kill -9 "$PID" 2>/dev/null
    fi
    
    rm -f "$PID_FILE"
    echo "✅ 自动备份服务已停止"
else
    echo "⚠️  进程不存在 (PID: $PID)，清理 PID 文件"
    rm -f "$PID_FILE"
fi

