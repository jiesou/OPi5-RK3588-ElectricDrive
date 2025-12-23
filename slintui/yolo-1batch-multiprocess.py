from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Tuple
import time
import numpy as np
import cv2
from rknnlite.api import RKNNLite as RKNN

@dataclass
class Detection:
    """检测结果的数量统计"""
    terminal: int = 0
    cross: int = 0
    excopper: int = 0
    exterminal: int = 0

@dataclass
class Box:
    """单个检测框"""
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    conf: float
    source: int

@dataclass
class YoloResult:
    """完整的YOLO检测结果"""
    detection: Detection
    boxes: List[Box]

# -----------------------------------------------------------------------------
# Helper Functions (Worker Logic)
# -----------------------------------------------------------------------------

def dfl(position):
    # Distribution Focal Loss (DFL)
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    
    # Softmax
    y = np.exp(y)
    y = y / np.sum(y, axis=2, keepdims=True)
    
    acc_metrix = np.arange(mc).reshape(1, 1, mc, 1, 1).astype(np.float32)
    y = (y * acc_metrix).sum(2)
    return y

def box_process(position, img_size):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([img_size[1]//grid_h, img_size[0]//grid_w]).reshape(1,2,1,1)

    position = dfl(position)
    box_xy  = grid + 0.5 - position[:,0:2,:,:]
    box_xy2 = grid + 0.5 + position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

    return xyxy

def post_process_single(input_data, img_size, obj_thresh):
    """对单个 batch 的输出做解码"""
    boxes, classes_conf, scores = [], [], []
    default_branch = 3
    pair_per_branch = len(input_data) // default_branch

    for i in range(default_branch):
        boxes.append(box_process(input_data[pair_per_branch * i], img_size))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # Filter by confidence
    box_confidences = scores.reshape(-1)
    class_max_score = np.max(classes_conf, axis=-1)
    classes = np.argmax(classes_conf, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= obj_thresh)
    
    final_scores = (class_max_score * box_confidences)[_class_pos]
    final_boxes = boxes[_class_pos]
    final_classes = classes[_class_pos]

    return final_boxes, final_classes, final_scores

def worker_func(input_queue: mp.Queue, output_queue: mp.Queue, model_path: str, img_size: Tuple[int, int], obj_thresh: float):
    """独立进程的工作函数"""
    try:
        rknn = RKNN()
        print(f"Worker loading model: {model_path}")
        if rknn.load_rknn(model_path) != 0:
            print("Load RKNN model failed")
            return
        
        # 自动分配 NPU 核心
        if rknn.init_runtime(core_mask=RKNN.NPU_CORE_AUTO) != 0:
            print("Init runtime environment failed")
            return
            
        while True:
            task = input_queue.get()
            if task is None:
                break
            
            idx, img = task
            
            # Inference (add batch dim)
            # img is (640, 640, 3), need (1, 640, 640, 3)
            outputs = rknn.inference(inputs=[img[None, ...]])
            
            # Post process
            boxes, classes, scores = post_process_single(outputs, img_size, obj_thresh)
            
            output_queue.put((idx, boxes, classes, scores))
            
    except Exception as e:
        print(f"Worker exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'rknn' in locals() and rknn is not None:
            rknn.release()

# -----------------------------------------------------------------------------
# Main Class
# -----------------------------------------------------------------------------

class Yolo:
    """YOLO检测器，专为电拖装接评估场景设计 (RKNN 多进程并行版)"""
    def __init__(self):
        self.latest_result: YoloResult = YoloResult(
            detection=Detection(),
            boxes=[]
        )
        self.CLASSES = ("cross", "excopper", "exterminal", "terminal")
        self.IMG_SIZE = (640, 640)
        self.OBJ_THRESH = 0.25
        self.NMS_THRESH = 0.45
        
        self.model_path = "./batch1-rkfork-electricdrivev20.2.17.1.rknn"
        
        # 初始化进程池
        self.input_queues = [mp.Queue() for _ in range(3)]
        self.output_queue = mp.Queue()
        self.processes = []
        
        print("Starting Yolo workers...")
        for i in range(3):
            p = mp.Process(
                target=worker_func,
                args=(self.input_queues[i], self.output_queue, self.model_path, self.IMG_SIZE, self.OBJ_THRESH),
                daemon=True
            )
            p.start()
            self.processes.append(p)
        print("Yolo workers started.")

    def pre_process(self, bgr: np.ndarray) -> Tuple[List[np.ndarray], List[dict]]:
        """预处理：生成3个切片"""
        h, w = bgr.shape[:2]

        # 1. Left Crop
        left_crop = cv2.resize(bgr[0:720, 0:720], self.IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        
        # 2. Right Crop
        right_crop = cv2.resize(bgr[0:720, w - 720 : w], self.IMG_SIZE, interpolation=cv2.INTER_LINEAR)

        # 3. Letterbox
        scaled = cv2.resize(bgr, (int(w * 0.5), int(h * 0.5)), interpolation=cv2.INTER_LINEAR)
        letter = cv2.copyMakeBorder(scaled, 140, 140, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # Convert to RGB
        imgs = [
            cv2.cvtColor(left_crop, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(right_crop, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(letter, cv2.COLOR_BGR2RGB),
        ]

        ratio_crop = self.IMG_SIZE[0] / 720.0
        metas = [
            {"ratio": ratio_crop, "pad": (0.0, 0.0), "offset": (0.0, 0.0), "source": 0},
            {"ratio": ratio_crop, "pad": (0.0, 0.0), "offset": (w - 720.0, 0.0), "source": 1},
            {"ratio": 0.5, "pad": (0.0, 140.0), "offset": (0.0, 0.0), "source": 2},
        ]

        return imgs, metas

    def detect(self, bgr: np.ndarray) -> YoloResult:
        """执行检测"""
        h, w = bgr.shape[:2]
        t_start = time.perf_counter()

        # 1. Preprocess
        imgs, metas = self.pre_process(bgr)
        
        # 2. Distribute tasks
        for i in range(3):
            self.input_queues[i].put((i, imgs[i]))
            
        # 3. Collect results
        results_dict = {}
        for _ in range(3):
            idx, boxes, classes, scores = self.output_queue.get()
            results_dict[idx] = (boxes, classes, scores)
            
        # 4. Aggregate and Map coordinates
        all_boxes = []
        all_classes = []
        all_scores = []
        all_sources = []
        
        for i in range(3):
            boxes, classes, scores = results_dict[i]
            if len(boxes) == 0:
                continue
                
            meta = metas[i]
            ratio = meta["ratio"]
            pad_x, pad_y = meta["pad"]
            offset_x, offset_y = meta["offset"]
            
            # Vectorized coordinate mapping
            # x1, x2
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio + offset_x
            # y1, y2
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio + offset_y
            
            # Clip to image bounds
            np.clip(boxes[:, 0], 0, w, out=boxes[:, 0])
            np.clip(boxes[:, 1], 0, h, out=boxes[:, 1])
            np.clip(boxes[:, 2], 0, w, out=boxes[:, 2])
            np.clip(boxes[:, 3], 0, h, out=boxes[:, 3])
            
            all_boxes.append(boxes)
            all_classes.append(classes)
            all_scores.append(scores)
            all_sources.append(np.full(len(classes), meta["source"], dtype=int))
            
        if not all_boxes:
            self.latest_result = YoloResult(detection=Detection(), boxes=[])
            return self.latest_result

        all_boxes = np.concatenate(all_boxes)
        all_classes = np.concatenate(all_classes)
        all_scores = np.concatenate(all_scores)
        all_sources = np.concatenate(all_sources)
        
        # 5. Global NMS using cv2.dnn.NMSBoxesBatched
        # Convert xyxy to xywh
        boxes_wh = all_boxes.copy()
        boxes_wh[:, 2] = boxes_wh[:, 2] - boxes_wh[:, 0]  # w
        boxes_wh[:, 3] = boxes_wh[:, 3] - boxes_wh[:, 1]  # h
        
        indices = cv2.dnn.NMSBoxesBatched(
            boxes_wh.tolist(), 
            all_scores.tolist(), 
            all_classes.tolist(), 
            self.OBJ_THRESH, 
            self.NMS_THRESH
        )
        
        final_boxes_list: List[Box] = []
        counts = {"terminal": 0, "cross": 0, "excopper": 0, "exterminal": 0}
        
        if len(indices) > 0:
            indices = indices.flatten()
            f_boxes = all_boxes[indices]
            f_scores = all_scores[indices]
            f_classes = all_classes[indices]
            f_sources = all_sources[indices]
            
            for box, score, cl, src in zip(f_boxes, f_scores, f_classes, f_sources):
                class_name = self.CLASSES[int(cl)]
                final_boxes_list.append(Box(
                    x1=int(box[0]),
                    y1=int(box[1]),
                    x2=int(box[2]),
                    y2=int(box[3]),
                    label=class_name,
                    conf=float(score),
                    source=int(src)
                ))
                if class_name in counts:
                    counts[class_name] += 1

        t_end = time.perf_counter()
        print(f"Total detection time: {(t_end - t_start)*1000:.2f} ms")
        
        if hasattr(self, 'last_push_result_time'):
            print(f"Time since last push: {time.perf_counter() - self.last_push_result_time:.2f} seconds")
        self.last_push_result_time = time.perf_counter()

        detection = Detection(
            terminal=counts["terminal"],
            cross=counts["cross"],
            excopper=counts["excopper"],
            exterminal=counts["exterminal"]
        )
        
        self.latest_result = YoloResult(detection=detection, boxes=final_boxes_list)
        return self.latest_result

# 全局单例
yolo = Yolo()
