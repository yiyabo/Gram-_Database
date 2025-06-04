#!/usr/bin/env python3
"""
测试Web应用的启动脚本
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("🔍 测试导入...")
    
    # 测试基础导入
    from generation_service import get_generation_service
    print("✅ 生成服务导入成功")
    
    # 测试Flask应用导入
    from app import app
    print("✅ Flask应用导入成功")
    
    # 测试生成服务初始化
    gen_service = get_generation_service()
    print("✅ 生成服务初始化成功")
    
    # 获取模型信息
    model_info = gen_service.get_model_info()
    print(f"📊 模型状态: {model_info}")
    
    print("\n🎉 所有测试通过！Web应用已准备就绪")
    print("\n🚀 启动命令:")
    print("cd gram_predictor && python app.py")
    print("\n🌐 访问地址:")
    print("http://localhost:8080")
    print("http://localhost:8080/generate (生成页面)")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()