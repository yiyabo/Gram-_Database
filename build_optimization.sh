# 云平台构建优化命令
# 在你的云平台构建脚本中使用这些命令：

# 1. 清理pip缓存
pip install --no-cache-dir -r requirements_cloud_minimal.txt
pip cache purge

# 2. 清理apt缓存（如果使用Ubuntu/Debian）
apt-get clean
rm -rf /var/lib/apt/lists/*

# 3. 清理Python编译缓存
find . -type d -name __pycache__ -delete
find . -name "*.pyc" -delete

# 4. 使用更小的基础镜像
# python:3.9-slim 替代 python:3.9
# 或者使用 python:3.9-alpine（最小）
