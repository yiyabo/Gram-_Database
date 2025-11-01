#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
云平台部署启动文件
简化版本，适用于 Railway、Heroku 等云平台
"""

import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'gram_predictor'))

# 设置环境变量
os.environ.setdefault('FLASK_ENV', 'production')
os.environ.setdefault('SECRET_KEY', 'fallback-secret-key-change-this')

# 导入并配置应用
try:
    from gram_predictor.app import app
    
    # 云平台配置
    app.config.update(
        DEBUG=False,
        SECRET_KEY=os.environ.get('SECRET_KEY', 'your-secret-key'),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB
    )
    
    if __name__ == '__main__':
        port = int(os.environ.get('PORT', 8080))
        app.run(host='0.0.0.0', port=port, debug=False)

except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保所有依赖都已正确安装")
    sys.exit(1)
except Exception as e:
    print(f"启动错误: {e}")
    sys.exit(1)
