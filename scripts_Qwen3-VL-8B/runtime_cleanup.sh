#!/bin/bash

runtime_cleanup_all() {
  echo "Stopping Ray and vLLM processes..."

  command -v ray >/dev/null 2>&1 && ray stop --force >/dev/null 2>&1 || true

  pkill -f "vllm_service_init/start_vllm_server.py" 2>/dev/null || true

  sleep 3
}
