#!/bin/bash

# 🚀 Gram-Negative Bacteria Prediction System 部署脚本
# 适用于 Ubuntu/Debian 系统

set -e  # 遇到错误时退出

echo "🚀 开始部署 Gram-Negative Bacteria Prediction System"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 函数：打印彩色消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否为 root 用户
if [[ $EUID -eq 0 ]]; then
   print_error "请不要使用 root 用户运行此脚本"
   exit 1
fi

# 配置变量
DOMAIN_NAME=${1:-"your-domain.com"}
EMAIL=${2:-"your-email@example.com"}
PROJECT_DIR=$(pwd)

print_info "域名: $DOMAIN_NAME"
print_info "邮箱: $EMAIL"
print_info "项目目录: $PROJECT_DIR"

# 1. 更新系统
print_info "更新系统包..."
sudo apt update && sudo apt upgrade -y

# 2. 安装 Docker 和 Docker Compose
print_info "安装 Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    print_info "Docker 安装完成"
else
    print_info "Docker 已安装"
fi

# 安装 Docker Compose
print_info "安装 Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    print_info "Docker Compose 安装完成"
else
    print_info "Docker Compose 已安装"
fi

# 3. 安装 Certbot (Let's Encrypt)
print_info "安装 Certbot..."
sudo apt install -y certbot python3-certbot-nginx

# 4. 创建必要的目录
print_info "创建部署目录..."
sudo mkdir -p /var/log/gram_predictor
sudo mkdir -p /var/www/certbot
sudo chown -R $USER:$USER /var/log/gram_predictor

# 5. 配置环境变量
print_info "配置环境变量..."
cat > .env << EOF
SECRET_KEY=$(openssl rand -hex 32)
DOMAIN_NAME=$DOMAIN_NAME
EMAIL=$EMAIL
EOF

# 6. 更新 Nginx 配置中的域名
print_info "更新 Nginx 配置..."
sed -i "s/your-domain.com/$DOMAIN_NAME/g" deployment/nginx.conf

# 7. 构建 Docker 镜像
print_info "构建 Docker 镜像..."
cd $PROJECT_DIR
docker-compose -f deployment/docker-compose.yml build

# 8. 启动服务 (临时不使用 SSL)
print_info "启动服务..."
# 临时修改 nginx 配置以不使用 SSL
cp deployment/nginx.conf deployment/nginx.conf.backup
cat > deployment/nginx_temp.conf << 'EOF'
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    upstream gram_predictor_app {
        server gram-predictor:8080;
    }
    
    server {
        listen 80;
        server_name your-domain.com;
        
        location /.well-known/acme-challenge/ {
            root /var/www/certbot;
        }
        
        location /static/ {
            alias /var/www/static/;
        }
        
        location / {
            proxy_pass http://gram_predictor_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }
    }
}
EOF

sed -i "s/your-domain.com/$DOMAIN_NAME/g" deployment/nginx_temp.conf
cp deployment/nginx_temp.conf deployment/nginx.conf

# 启动服务
docker-compose -f deployment/docker-compose.yml up -d

# 等待服务启动
print_info "等待服务启动..."
sleep 30

# 9. 获取 SSL 证书
print_info "获取 SSL 证书..."
sudo certbot certonly --webroot \
    --webroot-path=/var/www/certbot \
    --email $EMAIL \
    --agree-tos \
    --no-eff-email \
    -d $DOMAIN_NAME \
    -d www.$DOMAIN_NAME

# 10. 恢复完整的 Nginx 配置
print_info "恢复 SSL Nginx 配置..."
cp deployment/nginx.conf.backup deployment/nginx.conf

# 复制 SSL 证书到 Docker volume
sudo docker cp /etc/letsencrypt/live/$DOMAIN_NAME/fullchain.pem $(docker-compose -f deployment/docker-compose.yml ps -q nginx):/etc/nginx/ssl/
sudo docker cp /etc/letsencrypt/live/$DOMAIN_NAME/privkey.pem $(docker-compose -f deployment/docker-compose.yml ps -q nginx):/etc/nginx/ssl/

# 重启 Nginx
docker-compose -f deployment/docker-compose.yml restart nginx

# 11. 设置自动续期
print_info "设置 SSL 证书自动续期..."
(crontab -l 2>/dev/null; echo "0 12 * * * /usr/bin/certbot renew --quiet && docker-compose -f $PROJECT_DIR/deployment/docker-compose.yml restart nginx") | crontab -

# 12. 配置防火墙
print_info "配置防火墙..."
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable

# 13. 健康检查
print_info "检查服务状态..."
sleep 10

if curl -f http://localhost/health > /dev/null 2>&1; then
    print_info "✅ 服务部署成功！"
    print_info "🌐 访问地址: https://$DOMAIN_NAME"
else
    print_warning "⚠️  服务可能未完全启动，请检查日志"
fi

# 清理临时文件
rm -f deployment/nginx_temp.conf

print_info "🎉 部署完成！"
print_info ""
print_info "📋 部署信息:"
print_info "   域名: https://$DOMAIN_NAME"
print_info "   日志目录: /var/log/gram_predictor"
print_info "   SSL 证书: /etc/letsencrypt/live/$DOMAIN_NAME"
print_info ""
print_info "🔧 常用命令:"
print_info "   查看日志: docker-compose -f deployment/docker-compose.yml logs -f"
print_info "   重启服务: docker-compose -f deployment/docker-compose.yml restart"
print_info "   停止服务: docker-compose -f deployment/docker-compose.yml down"
print_info "   更新服务: docker-compose -f deployment/docker-compose.yml pull && docker-compose -f deployment/docker-compose.yml up -d"
