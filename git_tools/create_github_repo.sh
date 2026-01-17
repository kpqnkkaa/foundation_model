#!/bin/bash

# 使用 GitHub API 创建仓库
# 需要先设置 GITHUB_TOKEN 环境变量

REPO_NAME="foundation_model"
GITHUB_USER="kpqnkkaa"
# 从环境变量读取 token，不要硬编码在脚本中
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

echo "正在创建 GitHub 仓库: $REPO_NAME..."

# 使用 GitHub API 创建仓库
curl -X POST \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/user/repos \
  -d "{
    \"name\": \"$REPO_NAME\",
    \"description\": \"Foundation Model for Gaze Estimation and Following\",
    \"private\": false,
    \"auto_init\": false
  }" 2>&1

if [ $? -eq 0 ]; then
    echo "✅ 仓库创建成功！"
    echo "现在可以运行: git push -u origin main"
else
    echo "❌ 仓库创建失败，可能已经存在"
    echo "如果仓库已存在，可以直接推送"
fi

