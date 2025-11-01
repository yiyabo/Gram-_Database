# 📖 网站部署指南

本指南将帮助您将 Gram-Negative Bacteria Prediction System 部署到生产环境，让其他人可以通过网页访问。

## 🎯 部署方式选择

### 方式一：云服务器部署（推荐）
适合：希望完全控制服务器环境的用户
成本：中等（服务器费用 + 域名费用）
技术要求：中等

### 方式二：云平台部署
适合：希望简化运维的用户
成本：低到中等
技术要求：低

## 🚀 完整部署流程

### 步骤 1：准备云服务器

**推荐云服务商：**
- **国外用户**：AWS EC2、Google Cloud、DigitalOcean
- **国内用户**：阿里云、腾讯云、华为云

**服务器配置建议：**
- CPU: 2核心以上
- 内存: 4GB 以上（推荐8GB，因为有机器学习模型）
- 存储: 40GB 以上 SSD
- 操作系统: Ubuntu 20.04 或 22.04

### 步骤 2：购买域名

**域名注册商：**
- **国外**：Namecheap、GoDaddy、Cloudflare
- **国内**：阿里云、腾讯云、华为云

**重要提醒：**
- 如果服务器在中国大陆，域名需要备案
- 建议选择 `.com`、`.org` 等国际域名

### 步骤 3：DNS 配置

将域名指向您的服务器：

```bash
# A 记录配置示例
your-domain.com     A     YOUR_SERVER_IP
www.your-domain.com A     YOUR_SERVER_IP
```

### 步骤 4：上传代码到服务器

```bash
# 在本地打包代码
git clone your-repository
cd Gram-_Database

# 上传到服务器（替换 YOUR_SERVER_IP 和 YOUR_DOMAIN）
scp -r . root@YOUR_SERVER_IP:/opt/gram-predictor/
```

### 步骤 5：运行部署脚本

```bash
# 登录服务器
ssh root@YOUR_SERVER_IP

# 进入项目目录
cd /opt/gram-predictor

# 运行部署脚本（替换为您的实际域名和邮箱）
./deployment/deploy.sh your-domain.com your-email@example.com
```

## 🔧 手动部署（如果自动脚本失败）

### 1. 安装 Docker

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

### 2. 安装 Docker Compose

```bash
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 3. 构建和启动服务

```bash
cd deployment
docker-compose up -d --build
```

### 4. 配置 SSL 证书

```bash
# 安装 Certbot
sudo apt install certbot python3-certbot-nginx

# 获取证书
sudo certbot --nginx -d your-domain.com -d www.your-domain.com
```

## 🔍 部署后检查

### 1. 检查服务状态

```bash
# 查看容器状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

### 2. 测试网站功能

访问以下地址进行测试：
- https://your-domain.com - 主页
- https://your-domain.com/predict - 预测功能
- https://your-domain.com/generate - 生成功能
- https://your-domain.com/health - 健康检查

## 📊 监控和维护

### 查看日志

```bash
# 应用日志
docker-compose logs gram-predictor

# Nginx 日志
docker-compose logs nginx

# 系统日志
tail -f /var/log/gram_predictor/app.log
```

### 更新部署

```bash
# 拉取最新代码
git pull origin main

# 重新构建并部署
docker-compose up -d --build
```

### 备份数据

```bash
# 备份重要文件
tar -czf backup-$(date +%Y%m%d).tar.gz data/ model/ 
```

## 🛡️ 安全建议

1. **定期更新系统**：
   ```bash
   sudo apt update && sudo apt upgrade
   ```

2. **配置防火墙**：
   ```bash
   sudo ufw allow 22,80,443/tcp
   sudo ufw enable
   ```

3. **监控资源使用**：
   ```bash
   htop
   df -h
   ```

4. **设置自动备份**：
   ```bash
   # 添加到 crontab
   0 2 * * * /path/to/backup-script.sh
   ```

## 🔧 故障排除

### 常见问题：

1. **容器无法启动**：
   - 检查端口是否被占用：`netstat -tlnp`
   - 查看日志：`docker-compose logs`

2. **SSL 证书问题**：
   - 检查域名解析：`nslookup your-domain.com`
   - 重新申请证书：`sudo certbot renew`

3. **模型加载失败**：
   - 确保模型文件存在：`ls -la model/`
   - 检查文件权限：`chmod -R 644 model/`

4. **内存不足**：
   - 监控内存使用：`free -h`
   - 考虑增加服务器配置或优化模型

## 💰 成本估算

**月度成本预估：**
- 域名：$10-15/年
- 云服务器（4GB内存）：$20-50/月
- SSL证书：免费（Let's Encrypt）
- 总计：约 $25-55/月

## 📞 获取帮助

如果部署过程中遇到问题：

1. 检查本文档的故障排除部分
2. 查看项目 GitHub Issues
3. 检查云服务商的文档和支持

---

🎉 **恭喜！您的网站现在已经可以供全世界的用户访问了！**
