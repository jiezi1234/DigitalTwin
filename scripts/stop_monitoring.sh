#!/bin/bash

echo "正在停止监控容器 (dt-prometheus, dt-grafana, dt-loki, dt-tempo)..."
docker stop -t 30 dt-loki 2>/dev/null
docker stop dt-prometheus dt-grafana dt-tempo 2>/dev/null
docker rm dt-prometheus dt-grafana dt-loki dt-tempo 2>/dev/null

# 删除专用网络
NETWORK_NAME="dt-monitoring-network"
docker network rm $NETWORK_NAME 2>/dev/null || true

echo "监控系统已停止并清理。"
