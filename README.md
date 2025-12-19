过程:
uv venv --system-site-packages
uv pip install ultralytics==8.3.233 --overrides overrides.txt
uv pip install PySide6 --overrides overrides.txt 
uv pip install aiohttp
uv pip install numpy==1.23.5 --overrides overrides.txt

DISPLAY=:0 python -m pyqt



https://elinux.org/Jetson_Zoo#ONNX_Runtime
ONNX Runtime 1.11.0 支持 opset 16 算子

不用 docker，裸 inference
sudo docker run -d \
    --name inference-server \
    --runtime=nvidia \
    --gpus all \
    -p 9001:9001 \
    --volume ~/.inference/cache:/tmp:rw \
    --security-opt="no-new-privileges" \
    --cap-drop="ALL" \
    --cap-add="NET_BIND_SERVICE" \
    -e NOTEBOOK_ENABLED=true -p 9002:9002 \
    -e ONNXRUNTIME_EXECUTION_PROVIDERS="[TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider]" \
    roboflow/roboflow-inference-server-jetson-4.6.1:latest

sudo docker run -d \
    --name inference-server \
    --runtime=nvidia \
    --privileged \
    -p 9001:9001 \
    --volume ~/.inference/cache:/tmp:rw \
    -e NOTEBOOK_ENABLED=true \
    -p 9002:9002 \
    -e ONNXRUNTIME_EXECUTION_PROVIDERS="[CUDAExecutionProvider,CPUExecutionProvider]" \
    roboflow/roboflow-inference-server-jetson-4.6.1:latest
    
sudo docker run -it --name electric \
  --device=/dev/video0 \
  --device=/dev/video1 \
  --network host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /tmp/argus_socket:/tmp/argus_socket \
  -v ~/services/ElectricDrive/:/app/ElectricDrive \
  ubuntu
 
pip install opencv-python
pip install inference_sdk
pip uninstall opencv-python opencv-contrib-python   

可能需要安装 numpy==1.21.6 numpy==1.23.5
需要跳过：
onnxruntime-gpu ; sys_platform == 'never'
zxing-cpp ; sys_platform == 'never'


export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=4096:glibc.rtld.optional_static_tls=4096
export ONNXRUNTIME_EXECUTION_PROVIDES="['CUDAExecutionProvider, CPUExecutionProvider']"
python3 inference-gst.py
  
curl -X POST "http://localhost:9001/infer/workflows/roboflow-docs/model-comparison" \
-H "Content-Type: application/json" \
-d '{
    "inputs": {
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/workflows/examples/bleachers.jpg"
        },
        "model1": "yolov8n-640",
        "model2": "yolov11n-640"
    }
}'
