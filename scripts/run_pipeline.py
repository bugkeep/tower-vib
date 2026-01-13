import os
from pathlib import Path
import cv2 as cv
import numpy as np
import logging
from datetime import datetime
#=======参数设置======#
ROOT=Path(__file__).resolve().parents[1]
VIDEO=ROOT /"data"/"demo_tower.mp4"
OUT_NPZ=ROOT/"results"/"cache"/"day28_frames.npz"
def setup_logging(log_dir:str|Path="results/logs",name:str="tower-vib",level:int=logging.INFO)->logging.Logger:
    """
       Create a logger that logs to both console and a timestamped file.

       Returns:
           logger: configured logger
       """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{name}_{ts}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # prevent double logging via root logger

    # IMPORTANT: avoid adding handlers multiple times if called repeatedly
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)

    # File handler (UTF-8 for Windows)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info(f"Logging to file: {log_path.resolve()}")
    return logger

def read_video_frames(cap):
    grays=[]
    for i in range(200):
        ok,frame=cap.read()
        if not ok:
            raise RuntimeError(f"this frame {i} error")
        gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        grays.append(gray)
    return grays

def save_frame_npz(grays,fps):
    grays_arr=np.stack(grays,axis=0)
    np.savez(OUT_NPZ,grays=grays_arr,fps=float(fps))
    print("save:",OUT_NPZ,"shape:",grays_arr.shape,"fps:",fps)

def roi(grays,roi_xywh):
    x,y,w,h=roi_xywh
    patches=[]
    for i,g in enumerate(grays):
        patch=g[y:y+h,x:x+w]
        if patch.shape[0]!=h or patch.shape[1]!=w:
            raise RuntimeError(f"ROI out of bounds at frame {i}, got {patch.shape}, expect {(h,w)}")
        patches.append(patch.copy())
    return patches

def main():
    logger = setup_logging(ROOT / "results" / "logs", name="day28")
    cap=cv.VideoCapture(VIDEO)
    print("cap opened:", cap.isOpened())
    if not cap.isOpened():
        raise RuntimeError("cannot open video (check path/codec)")
    fps=cap.get(cv.CAP_PROP_FPS)
    grays=read_video_frames(cap)
    roi_xywh = (200,100,128,128)
    grays_roi=roi(grays,roi_xywh)

    logger.info("start pipeline")
    logger.info(f"video_path={VIDEO}")
    logger.info(f"fps={fps}")
    logger.info(f"roi_xywh={roi_xywh}")
    logger.warning("this is a warning example")
    save_frame_npz(grays_roi, fps)
    cap.release()


if __name__=="__main__":
    main()