#!/bin/bash

# 启动自动备份服务

REPO_DIR="/mnt/nvme1n1/lululemon/fm_shijing"
SCRIPT_PATH="$REPO_DIR/git_tools/scheduled_push_with_env.sh"
PID_FILE="$REPO_DIR/.auto_push.pid"
LOG_FILE="$REPO_DIR/.auto_push.log"

cd "$REPO_DIR" || exit 1

# 检查是否已经在运行
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠️  自动备份服务已在运行 (PID: $OLD_PID)"
        echo "   如需重启，请先运行: ./git_tools/stop_auto_push.sh"
        exit 1
    else
        # PID 文件存在但进程不存在，删除旧文件
        rm -f "$PID_FILE"
    fi
fi

# 确保脚本可执行
chmod +x "$SCRIPT_PATH"

# 启动服务（后台运行）
nohup bash "$SCRIPT_PATH" >> "$LOG_FILE" 2>&1 &
NEW_PID=$!

# 保存 PID
echo "$NEW_PID" > "$PID_FILE"

echo "✅ 自动备份服务已启动"
echo "   PID: $NEW_PID"
echo "   日志: $LOG_FILE"
echo "   查看日志: tail -f $LOG_FILE"
echo "   停止服务: ./git_tools/stop_auto_push.sh"

