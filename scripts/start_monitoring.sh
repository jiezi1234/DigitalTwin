#!/bin/bash

# 获取当前目录
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 创建专用网络来支持 DNS 解析 (容器名互访)
NETWORK_NAME="dt-monitoring-network"
docker network create $NETWORK_NAME 2>/dev/null || true

echo "正在停止旧容器..."
docker stop -t 30 dt-loki 2>/dev/null
docker stop dt-prometheus dt-grafana dt-tempo 2>/dev/null
docker rm dt-prometheus dt-grafana dt-loki dt-tempo 2>/dev/null

echo "正在启动 Prometheus (dt-prometheus)..."
docker run -d \
  --name dt-prometheus \
  --network $NETWORK_NAME \
  --network-alias prometheus \
  -p 9090:9090 \
  -v "$DIR/../monitoring/prometheus.yml:/etc/prometheus/prometheus.yml" \
  -v "$DIR/../monitoring/prometheus_data:/prometheus" \
  --add-host=host.docker.internal:host-gateway \
  prom/prometheus:latest

echo "正在启动 Grafana (dt-grafana)..."
docker run -d \
  --name dt-grafana \
  --network $NETWORK_NAME \
  -p 3000:3000 \
  -v "$DIR/../monitoring/grafana_data:/var/lib/grafana" \
  -e "GF_SECURITY_ADMIN_PASSWORD=admin" \
  -e "GF_AUTH_ANONYMOUS_ENABLED=true" \
  -e "GF_AUTH_ANONYMOUS_ORG_ROLE=Admin" \
  -e "GF_USERS_DEFAULT_LANGUAGE=zh-Hans" \
  grafana/grafana:latest

echo "正在启动 Loki (dt-loki)..."
docker run -d \
  --name dt-loki \
  --network $NETWORK_NAME \
  --network-alias loki \
  -p 3100:3100 \
  -v "$DIR/../monitoring/loki-config.yml:/etc/loki/local-config.yaml" \
  -v "$DIR/../monitoring/loki_data:/loki" \
  grafana/loki:latest

echo "正在启动 Tempo (dt-tempo)..."
docker run -d \
  --name dt-tempo \
  --network $NETWORK_NAME \
  --network-alias tempo \
  -p 3200:3200 \
  -p 4317:4317 \
  -p 4318:4318 \
  -v "$DIR/../monitoring/tempo-config.yml:/etc/tempo/config.yaml" \
  grafana/tempo:2.6.1 \
  -config.file=/etc/tempo/config.yaml

echo "启动完成！"
echo "Prometheus: http://localhost:9090"
echo "Grafana: http://localhost:3000"
echo "Loki (OTLP HTTP): http://localhost:3100/otlp"
echo "Tempo (OTLP HTTP): http://localhost:4318"
