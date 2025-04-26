#!/bin/bash

echo "🔵 开始安装依赖包..."

# 更新pip
python3 -m pip install --upgrade pip

# 安装必要依赖
pip install -r requirements.txt

echo "✅ 依赖安装完成！"