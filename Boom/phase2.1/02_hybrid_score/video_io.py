"""OpenCV worker executed only by the project's compatible Python environment."""
from pathlib import Path
import json
import sys
import cv2


def process(request: dict) -> list[dict]:
    """Read real metadata or extract requested real frames into PNG files."""
    rows = []
    for item in request['videos']:
        path = Path(item['path'])
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise RuntimeError(f'Cannot decode {path}')
            if request['operation'] == 'inventory':
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                rows.append(dict(video=path.name, frames=int(frames), fps=fps,
                    duration_s=frames/fps if fps else None, width=int(cap.get(3)),
                    height=int(cap.get(4)), MB=path.stat().st_size/1e6))
            elif request['operation'] == 'frames':
                for frame in item['frames']:
                    cap.set(cv2.CAP_PROP_POS_MSEC, float(frame['seconds'])*1000)
                    ok, image = cap.read()
                    target = Path(frame['output'])
                    if ok:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if not cv2.imwrite(str(target), image):
                            raise RuntimeError(f'Cannot write {target}')
                    rows.append(dict(seconds=frame['seconds'], path=str(target) if ok else None))
            else:
                raise ValueError(request['operation'])
        finally:
            cap.release()
    return rows


if __name__ == '__main__':
    print(json.dumps(process(json.load(sys.stdin)), ensure_ascii=False))
