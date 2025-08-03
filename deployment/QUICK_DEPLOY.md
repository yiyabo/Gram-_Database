# 🚀 快速部署指南

## 📋 部署前准备清单

✅ **必需项：**
- [ ] 云服务器（推荐 4GB+ 内存）
- [ ] 域名（如：my-gram-predictor.com）
- [ ] 邮箱地址（用于 SSL 证书）

✅ **推荐服务商：**
- **国外**：DigitalOcean ($20/月)、AWS、Google Cloud
- **国内**：阿里云、腾讯云 (需备案)

## 🎯 三种部署方式

### 方式 1️⃣：一键自动部署（推荐新手）

```bash
# 1. 上传代码到服务器
git clone https://github.com/your-username/Gram-_Database.git
cd Gram-_Database

# 2. 运行一键部署脚本
chmod +x deployment/deploy.sh
./deployment/deploy.sh your-domain.com your-email@example.com

# 3. 等待 10-15 分钟完成部署
```

### 方式 2️⃣：Docker 快速部署

```bash
# 1. 安装 Docker
curl -fsSL https://get.docker.com | sh

# 2. 启动服务
cd deployment
docker-compose up -d

# 3. 配置域名和 SSL（手动）
```

### 方式 3️⃣：云平台一键部署

#### Railway (最简单)
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template?template=https://github.com/your-username/Gram-_Database)

#### Heroku
[![Deploy to Heroku](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy?template=https://github.com/your-username/Gram-_Database)

#### Vercel (适合演示)
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/your-username/Gram-_Database)

## 🔧 部署后必做的 3 件事

### 1. 测试网站功能
```bash
# 访问这些地址确认工作正常
curl https://your-domain.com/health
curl https://your-domain.com/predict
```

### 2. 设置监控
```bash
# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

### 3. 配置安全
```bash
# 开启防火墙
sudo ufw enable
sudo ufw allow 80,443/tcp

# 设置自动更新
sudo apt install unattended-upgrades
```

## ⚡ 常见问题快速解决

| 问题 | 解决方案 |
|------|----------|
| 🔴 网站无法访问 | 检查 DNS 解析：`nslookup your-domain.com` |
| 🔴 SSL 证书错误 | 重新申请：`sudo certbot renew` |
| 🔴 内存不足 | 增加 swap：`sudo fallocate -l 2G /swapfile` |
| 🔴 模型加载失败 | 检查文件权限：`chmod -R 644 model/` |
| 🔴 磁盘空间不足 | 使用轻量级版本：修改 `requirements.txt` 为轻量级依赖 |
| 🔴 TensorFlow 太大 | 部署演示版本：使用 `lightweight_app.py` |

### 🚨 磁盘空间不足解决方案

如果遇到 `no space left on device` 错误：

#### 方案 1：轻量级部署（推荐）

```bash
# 1. 切换到轻量级版本
cp requirements_lightweight.txt requirements.txt

# 2. 修改 Procfile
echo "web: gunicorn lightweight_app:app --bind 0.0.0.0:\$PORT --workers 2" > Procfile

# 3. 重新部署
git add . && git commit -m "Switch to lightweight deployment"
git push
```

#### 方案 2：升级云平台套餐

- Railway: 升级到 Pro 计划 ($5/月)
- Heroku: 使用 Standard-1X ($25/月)
- DigitalOcean App Platform: Basic ($12/月)

## 💰 成本预估

| 方案 | 月费用 | 适合人群 |
|------|--------|----------|
| 🌟 DigitalOcean | $25/月 | 个人/小团队 |
| 🏢 AWS EC2 | $30-50/月 | 企业用户 |
| 🆓 Railway | 免费额度 | 学习/演示 |
| 🎯 阿里云 | ¥150/月 | 国内用户 |

## 📞 获取帮助

🆘 **遇到问题？**

1. 检查 [部署日志](./DEPLOYMENT_GUIDE.md#故障排除)
2. 搜索 [GitHub Issues](https://github.com/your-username/Gram-_Database/issues)
3. 联系技术支持

---

🎉 **5 分钟后，您的网站就可以全球访问了！**
