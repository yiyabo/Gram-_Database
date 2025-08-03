# Gunicorn 生产环境配置文件

# 服务器套接字
bind = "0.0.0.0:8080"
backlog = 2048

# 工作进程
workers = 4  # 根据 CPU 核心数调整 (通常为 2 * CPU核心数 + 1)
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2

# 安全
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# 日志
accesslog = "/var/log/gram_predictor/access.log"
errorlog = "/var/log/gram_predictor/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# 进程管理
max_requests = 1000
max_requests_jitter = 50
preload_app = True
daemon = False
pidfile = "/var/run/gram_predictor.pid"
user = "www-data"  # 根据实际用户调整
group = "www-data"  # 根据实际组调整

# 性能调优
worker_tmp_dir = "/dev/shm"  # 使用内存文件系统提高性能

# SSL (如果需要在应用层处理 SSL)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"
