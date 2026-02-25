#!/bin/bash

# 设置代理
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897

# 安装 Node.js (如果没有)
if ! command -v node &> /dev/null; then
    echo "Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
fi

# 验证 Node.js
echo "Node.js version: $(node -v)"
echo "npm version: $(npm -v)"

# 安装 Claude Code
echo "Installing Claude Code..."
npm install -g @anthropic-ai/claude-code

# 验证安装
echo "Claude Code installed:"
claude --version
