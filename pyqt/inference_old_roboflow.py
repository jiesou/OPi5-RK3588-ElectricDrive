import multiprocessing as mp
import threading
import time
from typing import Optional

from .state_bus import stateBus, Detection


def _process_worker(in_q: mp.Queue, out_q: mp.Queue, model_id: str, api_key: str):
    """Runs in a separate process: loads model and performs inference on frames received via in_q.

    Communicates back via out_q with messages of form:
      {'type':'status','warming': True/False}
      {'type':'detections','detections': [...], 'terminal': int, 'cross': int}
    Receiving None on in_q is used as a shutdown sentinel.
    """
    try:
        out_q.put({'type': 'status', 'warming': True})

        # lazy import model inside worker process
        import importlib
        infmod = importlib.import_module('inference')
        get_model = getattr(infmod, 'get_model')
        model = get_model(model_id=model_id, api_key=api_key)

        # optional warmup: consume the latest frame if available
        try:
            frame = None
            while not in_q.empty():
                frame = in_q.get_nowait()
            if frame is not None and hasattr(model, 'infer'):
                _ = model.infer([frame])
        except Exception:
            pass

        out_q.put({'type': 'status', 'warming': False})

        while True:
            try:
                # block for a short time waiting for a frame
                frame = in_q.get(timeout=1.0)
                if frame is None:
                    break

                # run inference
                if not hasattr(model, 'infer'):
                    continue

                results = model.infer([frame])[0]

                detections = []
                terminal_cnt = 0
                cross_cnt = 0

                if results is not None and hasattr(results, 'predictions'):
                    for pred in results.predictions:
                        cls_name = getattr(pred, 'class_name', getattr(pred, 'label', ''))
                        conf = float(getattr(pred, 'confidence', 0.0))
                        x = float(getattr(pred, 'x', 0.0))
                        y = float(getattr(pred, 'y', 0.0))
                        w = float(getattr(pred, 'width', 0.0))
                        h = float(getattr(pred, 'height', 0.0))
                        detections.append({
                            'x': x, 'y': y, 'width': w, 'height': h,
                            'class': cls_name, 'confidence': conf,
                        })

                        if 'cross' in cls_name:
                            cross_cnt += 1
                        elif 'terminal' in cls_name:
                            terminal_cnt += 1

                out_q.put({'type': 'detections', 'detections': detections,
                           'terminal': terminal_cnt, 'cross': cross_cnt})

            except Exception:
                # keep running unless a sentinel is received; ignore intermittent errors
                continue

    except Exception as e:
        try:
            out_q.put({'type': 'status', 'warming': False})
            out_q.put({'type': 'error', 'error': str(e)})
        except Exception:
            pass


class InferenceManager:
    """Manages a separate inference process plus a small monitor thread in the main process.

    - Uses a multiprocessing.Process to perform CPU/GPU-bound model loading and inference.
    - Sends frames via a small multiprocessing.Queue (maxsize=1) to implement drop-old-frame behavior.
    - Receives detections via an output queue and updates `stateBus` from the monitor thread.
    """

    def __init__(self, model_id: str = "electricdrive-yqqfl/8", api_key: str = "sD0Vt3yPtXgXajBC3sfT"):
        self._proc: Optional[mp.Process] = None
        self._in_q: Optional[mp.Queue] = None
        self._out_q: Optional[mp.Queue] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_stop = threading.Event()
        self._running = False
        self._model_id = model_id
        self._api_key = api_key

    def start(self):
        if self._running:
            return
        # create queues
        self._in_q = mp.Queue(maxsize=1)
        self._out_q = mp.Queue()

        # start process
        self._proc = mp.Process(target=_process_worker,
                                args=(self._in_q, self._out_q, self._model_id, self._api_key),
                                daemon=True)
        self._proc.start()

        # start monitor thread to read out_q and update stateBus
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        # connect to frame updates and mark running
        stateBus.frame_updated.connect(self._on_frame_updated)
        self._running = True

    def stop(self):
        if not self._running:
            return

        # disconnect frame updates
        try:
            stateBus.frame_updated.disconnect(self._on_frame_updated)
        except Exception:
            pass

        # send sentinel to process
        try:
            if self._in_q is not None:
                try:
                    # clear queue then send sentinel
                    while not self._in_q.empty():
                        try:
                            self._in_q.get_nowait()
                        except Exception:
                            break
                    self._in_q.put_nowait(None)
                except Exception:
                    pass
        except Exception:
            pass

        # stop monitor
        self._monitor_stop.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

        # terminate process if still alive
        try:
            if self._proc is not None and self._proc.is_alive():
                self._proc.join(timeout=1.0)
                if self._proc.is_alive():
                    self._proc.terminate()
        except Exception:
            pass

        # cleanup queues
        try:
            if self._in_q is not None:
                self._in_q.close()
        except Exception:
            pass
        try:
            if self._out_q is not None:
                self._out_q.close()
        except Exception:
            pass

        self._in_q = None
        self._out_q = None
        self._proc = None
        self._monitor_thread = None
        self._running = False

        # clear visual detections and counts
        try:
            stateBus.set_visual_detections([])
            stateBus.set_detections(Detection())
            stateBus.set_inference_status(False)
        except Exception:
            pass

    def is_running(self) -> bool:
        return self._running

    def _on_frame_updated(self):
        # push latest frame to process input queue (drop oldest, non-blocking)
        if not self._running or self._in_q is None:
            return
        try:
            frame = stateBus.get_last_frame()
            if frame is None:
                return
            # try to keep only the latest frame in the queue
            try:
                if self._in_q.full():
                    try:
                        self._in_q.get_nowait()
                    except Exception:
                        pass
                self._in_q.put_nowait(frame.copy())
            except Exception:
                pass
        except Exception:
            pass

    def _monitor_loop(self):
        # monitor out_q for messages and update stateBus accordingly
        while not self._monitor_stop.is_set():
            try:
                if self._out_q is None:
                    time.sleep(0.1)
                    continue
                msg = None
                try:
                    msg = self._out_q.get(timeout=0.5)
                except Exception:
                    msg = None

                if msg is None:
                    continue

                mtype = msg.get('type')
                if mtype == 'status':
                    warming = bool(msg.get('warming', False))
                    try:
                        stateBus.set_inference_status(warming)
                    except Exception:
                        pass
                elif mtype == 'detections':
                    dets = msg.get('detections', [])
                    terminal = int(msg.get('terminal', 0))
                    cross = int(msg.get('cross', 0))
                    try:
                        stateBus.set_visual_detections(dets)
                        stateBus.set_detections(Detection(terminal=terminal, cross=cross))
                    except Exception:
                        pass
                elif mtype == 'error':
                    # propagate error as non-fatal status change
                    try:
                        stateBus.set_inference_status(False)
                    except Exception:
                        pass
                else:
                    # unknown message
                    pass

            except Exception:
                # transient monitor error; continue
                continue


# singleton instance
inference_manager = InferenceManager()
