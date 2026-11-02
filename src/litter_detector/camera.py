"""Optional local webcam support. No physical camera is opened on import."""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time

import cv2


@dataclass(frozen=True)
class Frame:
    frame_id: int
    received_at: float
    image: object


class LatestFrame:
    """One slot: a slow consumer cannot build an unbounded queue."""
    def __init__(self):
        self._condition = threading.Condition()
        self._frame = None
        self.closed = False

    def publish(self, frame):
        with self._condition:
            self._frame = frame
            self._condition.notify_all()

    def get(self, after_id=-1, timeout=0.2):
        with self._condition:
            self._condition.wait_for(lambda: self.closed or (self._frame is not None and self._frame.frame_id > after_id), timeout)
            if self._frame is not None and self._frame.frame_id > after_id:
                return self._frame
            return None

    def close(self):
        with self._condition:
            self.closed = True
            self._condition.notify_all()


def run_camera(detector, index=0, process_fps=15, stop=None, callback=None, capture_factory=None):
    if detector.profile == "underwater":
        raise ValueError("Профиль underwater предназначен только для файлов датасета")
    if process_fps <= 0:
        raise ValueError("Частота обработки должна быть положительной")
    stop = stop if stop is not None else threading.Event()
    slot = LatestFrame()
    cap = (capture_factory or cv2.VideoCapture)(index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Не удалось открыть обычную веб-камеру")
    def capture():
        frame_id = 0
        try:
            while not stop.is_set():
                ok, image = cap.read()
                if not ok:
                    break
                slot.publish(Frame(frame_id, time.perf_counter(), image))
                frame_id += 1
        finally:
            slot.close()
    worker = threading.Thread(target=capture, daemon=True)
    worker.start()
    frame_id = -1
    try:
        while not stop.is_set():
            frame = slot.get(frame_id)
            if frame is None:
                if slot.closed:
                    break
                continue
            frame_id = frame.frame_id
            start = time.perf_counter()
            annotated, record = detector.predict(frame.image, frame_id=frame_id)
            record["receive_to_result_ms"] = (time.perf_counter()-frame.received_at)*1000
            if callback:
                callback(annotated, record)
            else:
                cv2.imshow("Litter detector | Q: exit", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            stop.wait(max(0, 1/process_fps - (time.perf_counter()-start)))
    finally:
        stop.set()
        cap.release()
        worker.join(timeout=2)
        slot.close()
        if not callback:
            cv2.destroyAllWindows()
