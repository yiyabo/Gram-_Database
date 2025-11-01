# 🚀 服务器升级指南 - 部署完整ML功能

## 🎯 为什么需要升级？

您的项目包含：
- 🧠 **复杂ML模型**：LSTM+MLP混合分类器
- 🔬 **扩散模型**：序列生成功能  
- 🤖 **ESM-2模型**：蛋白质编码器
- 📊 **完整预测管道**：特征提取 + 深度学习

**资源需求**：
- 💾 **内存**：4GB+ (ML模型加载)
- 💿 **存储**：8GB+ (PyTorch + TensorFlow)
- ⚡ **CPU**：2核+ (推理计算)

## 🔥 推荐升级方案

### 方案 1：🌟 DigitalOcean Droplet (推荐)

**配置**：Basic 4GB / 2 vCPUs / 80GB SSD
**价格**：$24/月
**优势**：
- ✅ 完全控制服务器
- ✅ Docker 友好
- ✅ 简单易用
- ✅ 优秀的文档

**快速部署**：
```bash
# 1. 创建 Droplet (Ubuntu 22.04)
# 2. 连接服务器
ssh root@your-droplet-ip

# 3. 安装 Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# 4. 克隆代码
git clone https://github.com/yiyabo/Gram-_Database.git
cd Gram-_Database

# 5. 一键部署
chmod +x deployment/deploy.sh
./deployment/deploy.sh your-domain.com your-email@example.com
```

### 方案 2：🏢 AWS EC2 (企业级)

**配置**：t3.medium (2 vCPU, 4GB RAM)
**价格**：~$30/月
**优势**：
- ✅ 企业级可靠性
- ✅ 丰富的服务生态
- ✅ 免费层可试用

### 方案 3：🇨🇳 阿里云 (国内用户)

**配置**：2核4GB / 40GB高效云盘
**价格**：¥150-200/月
**优势**：
- ✅ 国内访问速度快
- ✅ 中文技术支持
- ✅ 备案支持

### 方案 4：💰 升级现有云平台

#### Railway Pro
- **价格**：$5/月
- **资源**：更多 CPU/内存/存储
- **操作**：在 Railway Dashboard 点击 "Upgrade"

#### Heroku Standard-1X  
- **价格**：$25/月
- **资源**：512MB → 1GB 内存
- **操作**：在 Heroku Dashboard 升级 Dyno 类型

## 🛠️ 详细部署步骤

### DigitalOcean 完整部署教程

#### 步骤 1：创建服务器
1. 访问 [DigitalOcean](https://digitalocean.com)
2. 创建账户（新用户有 $200 免费额度）
3. 点击 "Create" → "Droplets"
4. 选择配置：
   - **镜像**：Ubuntu 22.04 LTS
   - **套餐**：Basic $24/月 (4GB / 2 vCPUs)
   - **数据中心**：选择离您最近的
   - **认证**：SSH密钥或密码

#### 步骤 2：配置域名 (可选)
```bash
# 在 DigitalOcean 或域名商处设置 DNS
# A 记录：your-domain.com → 服务器IP
```

#### 步骤 3：连接并部署
```bash
# 连接服务器
ssh root@YOUR_SERVER_IP

# 安装必要工具
apt update && apt upgrade -y
apt install -y git curl

# 安装 Docker
curl -fsSL https://get.docker.com | sh
systemctl start docker
systemctl enable docker

# 克隆项目
git clone https://github.com/yiyabo/Gram-_Database.git
cd Gram-_Database

# 运行部署脚本
chmod +x deployment/deploy.sh
./deployment/deploy.sh your-domain.com your-email@example.com

# 或手动 Docker 部署
cd deployment
docker-compose up -d
```

#### 步骤 4：验证部署
```bash
# 检查服务状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 测试 API
curl http://your-server-ip/health
```

## 🔧 性能优化建议

### 1. 启用 GPU (如需要)
```bash
# 选择支持 GPU 的实例
# DigitalOcean: GPU Droplets
# AWS: p3.2xlarge
# 阿里云: GPU 实例
```

### 2. 增加 Swap
```bash
# 增加虚拟内存
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

### 3. 启用缓存
```bash
# Redis 缓存模型预测结果
# 在 docker-compose.yml 中已配置
```

## 💰 成本对比

| 服务商 | 配置 | 月费用 | 年费用 | 性价比 |
|--------|------|--------|--------|--------|
| 🌟 DigitalOcean | 4GB/2核 | $24 | $288 | ⭐⭐⭐⭐⭐ |
| 🏢 AWS EC2 | t3.medium | $30 | $360 | ⭐⭐⭐⭐ |
| 🇨🇳 阿里云 | 2核4GB | ¥150 | ¥1800 | ⭐⭐⭐⭐ |
| Railway Pro | 增强资源 | $5 | $60 | ⭐⭐⭐ |
| Heroku Standard | 1GB内存 | $25 | $300 | ⭐⭐ |

## 🚀 推荐操作

**立即行动方案**：
1. **注册 DigitalOcean** (有 $200 免费额度)
2. **创建 4GB Droplet** 
3. **运行一键部署脚本**
4. **5分钟内完成完整部署**

**命令汇总**：
```bash
# 一条命令完成部署
curl -sSL https://raw.githubusercontent.com/yiyabo/Gram-_Database/feature/separate-views/deployment/quick-install.sh | bash
```

## 📞 技术支持

遇到问题？
- 📧 DigitalOcean 24/7 技术支持
- 📚 完整的部署文档在 `deployment/` 目录
- 🔧 自动故障诊断和修复脚本

---

🎉 **升级后您将获得**：
- ✅ 完整的 ML 预测功能
- ✅ 实时序列生成
- ✅ 高性能推理服务
- ✅ 企业级稳定性
