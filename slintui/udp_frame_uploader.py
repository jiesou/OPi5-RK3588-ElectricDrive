import threading
import socket
import struct
import time
import cv2
import threading
import time
import socket
import struct
from urllib.parse import urlparse
from camera_viewport import camera_viewport
from settings import stored_settings


CHUNK_LENGTH = 1472
FRAMERATE = 10


class UdpFrameUploader:
    """
    独立线程模块，负责：
    1. 从 stateBus 获取最新帧
    2. 编码为 JPEG
    3. 按照 UDP 协议分片发送到服务器
    
    发送格式：
    - 包头 8 字节: frame_index(4字节) + chunk_index(2字节) + chunk_total(2字节)
    - 包体: JPEG 数据分片
    """

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = None
        self._sock = None
        self._server_addr = None
        self._frame_index = 0

    def start(self):
        """启动上传线程"""
        if self._thread is not None and self._thread.is_alive():
            print("[UdpFrameUploader] 线程已在运行")
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._upload_loop, daemon=True)
        self._thread.start()
        print("[UdpFrameUploader] 启动成功")

    def stop(self):
        """停止上传线程"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        print("[UdpFrameUploader] 已停止")

    def _setup_socket(self, url: str) -> bool:
        """解析 URL 并创建 UDP socket"""
        try:
            parsed = urlparse(url)
            if parsed.scheme != "udp":
                print(f"[UdpFrameUploader] 错误: URL scheme 必须是 udp, 当前是 {parsed.scheme}")
                return False
            
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or 8099
            
            if self._sock is not None:
                self._sock.close()
            
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._server_addr = (host, port)
            
            print(f"[UdpFrameUploader] Socket 创建成功，目标: {host}:{port}")
            return True
        except Exception as e:
            print(f"[UdpFrameUploader] Socket 创建失败: {e}")
            return False

    def _upload_loop(self):
        """主循环：获取帧 -> 编码 -> 分片发送"""
        interval = 1.0 / FRAMERATE
        
        while not self._stop_event.is_set():
            loop_start = time.time()
            
            # 获取配置的 URL
            feed_url = stored_settings.get("image_feed_udp_url", "")
            if not feed_url:
                time.sleep(1)
                continue
            
            # 创建或重建 socket
            if self._sock is None:
                if not self._setup_socket(feed_url):
                    time.sleep(1)
                    continue
            
            # 从 camera_viewport 获取最新帧
            frame = camera_viewport.get_latest_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            
            # 编码为 JPEG
            try:
                ret, jpeg_data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ret or jpeg_data is None:
                    print("[UdpFrameUploader] JPEG 编码失败")
                    time.sleep(0.05)
                    continue
                
                jpeg_bytes = jpeg_data.tobytes()
                data_len = len(jpeg_bytes)
                
                # 发送分片
                self._send_frame(jpeg_bytes, data_len)
                
            except Exception as e:
                print(f"[UdpFrameUploader] 编码或发送错误: {e}")
            
            # 控制帧率
            elapsed = time.time() - loop_start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _send_frame(self, data: bytes, data_len: int):
        """
        将一帧 JPEG 数据按协议分片发送
        
        包头结构 (8字节，小端):
        - frame_index: uint32 (4字节)
        - chunk_index: uint16 (2字节)
        - chunk_total: uint16 (2字节)
        """
        if self._sock is None or self._server_addr is None:
            return
        
        # 计算分片
        header_size = 8
        chunk_payload_len = CHUNK_LENGTH - header_size
        chunk_total = (data_len + chunk_payload_len - 1) // chunk_payload_len
        
        for chunk_index in range(chunk_total):
            offset = chunk_index * chunk_payload_len
            this_len = min(chunk_payload_len, data_len - offset)
            
            # 组包: 包头 + 数据
            header = struct.pack('<IHH', self._frame_index, chunk_index, chunk_total)
            payload = data[offset:offset + this_len]
            packet = header + payload
            
            try:
                self._sock.sendto(packet, self._server_addr)
            except Exception as e:
                print(f"[UdpFrameUploader] 发送分片失败 frame={self._frame_index} chunk={chunk_index}: {e}")
                return
        
        self._frame_index += 1


# 全局单例
uploader = UdpFrameUploader()
