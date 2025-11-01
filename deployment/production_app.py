#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生产环境启动文件
配置为生产环境使用，关闭调试模式，优化性能设置
"""

import os
import logging
from gram_predictor.app import app

# 生产环境配置
class ProductionConfig:
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-change-this-in-production')
    
    # 文件上传配置
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = '/tmp/uploads'
    
    # 日志配置
    LOG_LEVEL = logging.INFO
    LOG_FILE = '/var/log/gram_predictor/app.log'

# 应用配置
app.config.from_object(ProductionConfig)

# 配置日志
if not app.debug:
    # 创建日志目录
    log_dir = os.path.dirname(ProductionConfig.LOG_FILE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # 配置文件日志
    file_handler = logging.FileHandler(ProductionConfig.LOG_FILE)
    file_handler.setLevel(ProductionConfig.LOG_LEVEL)
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    )
    file_handler.setFormatter(formatter)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(ProductionConfig.LOG_LEVEL)

# 创建上传目录
os.makedirs(ProductionConfig.UPLOAD_FOLDER, exist_ok=True)

if __name__ == '__main__':
    # 生产环境不应该使用内置服务器
    # 应该使用 Gunicorn 等 WSGI 服务器
    print("警告：生产环境请使用 Gunicorn 启动应用")
    print("运行命令：gunicorn production_app:app")
    app.run(host='0.0.0.0', port=8080)
