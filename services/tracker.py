"""A minimal IoU-based tracker to assign persistent IDs to detections across frames.

This is a lightweight tracker (greedy matching) — good for simple cases where objects
move smoothly between frames. It doesn't use a Kalman filter or appearance embeddings
but is dependency-free and easy to understand.

API:
  tracker = IoUTracker(max_lost=5, iou_threshold=0.3)
  ids = tracker.update(list_of_bboxes)

`list_of_bboxes` should be a list of (x1, y1, x2, y2) integers in the same order
as the detections you create. The returned `ids` list aligns with input order.
"""
from typing import List, Tuple, Dict


def iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    denom = float(boxAArea + boxBArea - interArea)
    if denom <= 0:
        return 0.0
    return interArea / denom


class IoUTracker:
    def __init__(self, max_lost: int = 5, iou_threshold: float = 0.3):
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self.tracks: Dict[int, Dict] = {}  # id -> {bbox, lost}
        self.next_id = 1

    def update(self, detections: List[Tuple[int, int, int, int]]) -> List[int]:
        """Assign track ids for the given list of bboxes.

        Returns a list of track ids aligned with the input `detections` order.
        """
        assigned_ids: List[int] = [-1] * len(detections)

        if not detections:
            # increment lost counters and cleanup
            to_delete = []
            for tid, t in self.tracks.items():
                t['lost'] += 1
                if t['lost'] > self.max_lost:
                    to_delete.append(tid)
            for tid in to_delete:
                del self.tracks[tid]
            return []

        # Build IoU matrix of detections x tracks
        track_ids = list(self.tracks.keys())
        iou_matrix = []
        for d in detections:
            row = []
            for tid in track_ids:
                row.append(iou(d, self.tracks[tid]['bbox']))
            iou_matrix.append(row)

        matched_tracks = set()
        matched_dets = set()

        # Greedy matching: for each detection, find best track if above threshold
        for det_idx, row in enumerate(iou_matrix):
            best_iou = 0.0
            best_tid = None
            for col_idx, val in enumerate(row):
                tid = track_ids[col_idx]
                if tid in matched_tracks:
                    continue
                if val > best_iou:
                    best_iou = val
                    best_tid = tid

            if best_tid is not None and best_iou >= self.iou_threshold:
                # assign
                assigned_ids[det_idx] = best_tid
                matched_tracks.add(best_tid)
                matched_dets.add(det_idx)
                # update track bbox and reset lost
                self.tracks[best_tid]['bbox'] = detections[det_idx]
                self.tracks[best_tid]['lost'] = 0

        # Create new tracks for unmatched detections
        for idx, det in enumerate(detections):
            if idx in matched_dets:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {'bbox': det, 'lost': 0}
            assigned_ids[idx] = tid

        # Increment lost for unmatched tracks and cleanup
        for tid in list(self.tracks.keys()):
            if tid not in matched_tracks and tid not in assigned_ids:
                self.tracks[tid]['lost'] += 1
                if self.tracks[tid]['lost'] > self.max_lost:
                    del self.tracks[tid]

        return assigned_ids
