/usr/src/tensorrt/bin/trtexec \
--onnx=electricdrivev2.0.onnx \
--saveEngine=electricdrivev2.0.engine \
--fp16 \
--workspace=1024 \
--verbose
