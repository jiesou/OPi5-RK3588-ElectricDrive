#!/bin/bash
sudo service nvargus-daemon restart
sudo bash -c "echo 3 > /proc/sys/vm/drop_caches"
sudo jetson_clocks
echo "==优化完成，启动中=="

cd /home/jetson/services/ElectricDrive
source .venv/bin/activate
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=4096:glibc.rtld.optional_static_tls=4096
export ONNXRUNTIME_EXECUTION_PROVIDES="['CUDAExecutionProvider, CPUExecutionProvider']"
python3 -m pyqt
