#!/bin/bash
# Gram-_Database 一键部署脚本
# 适用于 Ubuntu 20.04/22.04

set -e

echo "🚀 开始部署 Gram-_Database 完整版本..."

# 检查是否为 root 用户
if [[ $EUID -ne 0 ]]; then
   echo "请使用 root 用户运行此脚本"
   exit 1
fi

# 更新系统
echo "📦 更新系统包..."
apt update && apt upgrade -y

# 安装必要工具
echo "🔧 安装必要工具..."
apt install -y git curl docker.io docker-compose python3 python3-pip

# 启动 Docker
echo "🐳 配置 Docker..."
systemctl start docker
systemctl enable docker
usermod -aG docker $USER

# 克隆项目
echo "📥 下载项目代码..."
if [ -d "Gram-_Database" ]; then
    cd Gram-_Database
    git pull origin feature/separate-views
else
    git clone -b feature/separate-views https://github.com/yiyabo/Gram-_Database.git
    cd Gram-_Database
fi

# 配置环境变量
echo "⚙️  配置环境变量..."
cat > .env << EOF
FLASK_ENV=production
SECRET_KEY=$(openssl rand -hex 32)
PORT=5000
EOF

# 构建并启动服务
echo "🏗️  构建并启动服务..."
cd deployment
docker-compose down 2>/dev/null || true
docker-compose build --no-cache
docker-compose up -d

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 30

# 检查服务状态
echo "🔍 检查服务状态..."
docker-compose ps

# 测试服务
echo "🧪 测试服务..."
if curl -f http://localhost/health > /dev/null 2>&1; then
    echo "✅ 服务启动成功！"
    echo "🌐 访问地址: http://$(curl -s ifconfig.me)"
    echo "🔧 管理命令:"
    echo "   查看日志: docker-compose logs -f"
    echo "   重启服务: docker-compose restart"
    echo "   停止服务: docker-compose down"
else
    echo "❌ 服务启动失败，请检查日志:"
    docker-compose logs
fi

echo "🎉 部署完成！"
