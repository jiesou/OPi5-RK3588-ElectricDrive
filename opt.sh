#!/bin/bash
sudo service nvargus-daemon restart
sudo bash -c "echo 3 > /proc/sys/vm/drop_caches"
sudo jetson_clocks
echo "==优化完成=="
