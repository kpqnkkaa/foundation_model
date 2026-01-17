#!/bin/bash

# 定时自动推送脚本（包含环境导出）
# 每10分钟检查一次，如果有变化则推送
# 自动排除 .pt 文件

REPO_DIR="/mnt/nvme1n1/lululemon/fm_shijing"
ENV_NAME="fm_shijing"  # 根据实际情况修改
CONDA_PATH="/mnt/nvme1n1/lululemon/xjj/miniconda3/bin/conda"
LOG_FILE="${REPO_DIR}/.auto_push.log"
INTERVAL=600  # 10分钟 = 600秒

# 确保在正确的目录
cd "$REPO_DIR" || exit 1

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 检查是否有变化（排除 .pt 文件）
check_changes() {
    # 检查代码变化（排除 .pt 文件）
    git diff --quiet -- ':!*.pt' && git diff --cached --quiet -- ':!*.pt'
    local code_changed=$?
    
    # 检查环境文件变化
    if [ -d "environment" ]; then
        git diff --quiet environment/
        local env_changed=$?
    else
        local env_changed=1  # 如果目录不存在，认为有变化
    fi
    
    # 如果有任何变化，返回1（有变化）
    if [ $code_changed -ne 0 ] || [ $env_changed -ne 0 ]; then
        return 1
    fi
    return 0
}

# 导出环境
export_env() {
    log "开始导出 Conda 环境: $ENV_NAME"
    mkdir -p "$REPO_DIR/environment"
    
    # 使用 conda env export
    if [ -f "$CONDA_PATH" ]; then
        source "$(dirname $CONDA_PATH)/../etc/profile.d/conda.sh"
        conda activate "$ENV_NAME" 2>/dev/null
        conda env export --no-builds > "$REPO_DIR/environment/environment.yml" 2>>"$LOG_FILE"
        conda env export --from-history > "$REPO_DIR/environment/environment.yml.history" 2>>"$LOG_FILE"
        log "环境导出完成"
    else
        log "警告: Conda 路径不存在: $CONDA_PATH"
    fi
}

# 推送更改（排除 .pt 文件）
push_changes() {
    log "检测到变化，开始推送..."
    
    # 导出环境
    export_env
    
    # 添加所有更改（但排除 .pt 文件）
    git add -A
    # 从暂存区移除 .pt 文件
    git reset HEAD -- '*.pt' 2>/dev/null || true
    
    # 提交
    local commit_msg="Auto backup: $(date '+%Y-%m-%d %H:%M:%S')"
    git commit -m "$commit_msg" >> "$LOG_FILE" 2>&1
    
    # 推送
    if git push origin main >> "$LOG_FILE" 2>&1; then
        log "✅ 推送成功"
    else
        log "❌ 推送失败，请检查日志"
    fi
}

# 主循环
log "自动备份服务启动 (间隔: ${INTERVAL}秒)"
while true; do
    # 导出环境（每次检查都导出）
    export_env
    
    # 检查变化
    if check_changes; then
        log "无变化，跳过推送"
    else
        push_changes
    fi
    
    # 等待指定时间
    sleep "$INTERVAL"
done

