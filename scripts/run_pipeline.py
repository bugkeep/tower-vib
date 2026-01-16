from pathlib import Path
import cv2 as cv
import logging
import  subprocess
import  sys
from roi_mean_curve_check import run_roi_mean_probe
from  datetime import datetime
#=======参数设置======#
ROOT=Path(__file__).resolve().parents[1]
VIDEO=ROOT /"data"/"demo_tower.mp4"
OUT_NPZ=ROOT/"results"/"cache"/"day28_frames.npz"
def setup_logging(log_dir:str|Path="results/logs",name:str="tower-vib",level:int=logging.INFO)->logging.Logger:
    """
    日志工具
    :param log_dir:
    :param name:
    :param level:
    :return:
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

def step_extract_roi(video,roi,max_frames):
    """
    裁剪工具在extract中，会在chache中生成.npz文件
    :param video:
    :param roi: x，y ，w，h
    :param max_frames: 最大帧
    :return:
    """
    logger = logging.getLogger(__name__)
    cmd = [
        sys.executable, "scripts/extract_roi.py",
        "--video", str(video),
        "--roi", *map(str, roi),
        "--max_frames", str(max_frames),
    ]
    subprocess.run(cmd,check=True,cwd=ROOT)
    out_npz=list((ROOT/"results"/"cache").glob("*_main_roi_frames.npz"))
    logger.info(f"candidates={[p.name for p in out_npz]}")
    if len(out_npz)==0:
        raise RuntimeError("out_npz==0")
    return out_npz

def read_video_frames(cap):
    """
    读取前200帧
    :param cap:
    :return:
    """
    grays=[]
    for i in range(200):
        ok,frame=cap.read()
        if not ok:
            raise RuntimeError(f"this frame {i} error")
        gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        grays.append(gray)
    return grays

# def compute_mean_curve(roi_frames:np.ndarray)->np.ndarray:
#     """
#     均值曲线计算：根据roi_mean_curve_check
#     roi_frames:
#         -灰度：(N,H,W)
#         -彩色：(N,H,W,3)
#     输出 mean_curve:(N,)
#     """
#     roi_frames=np.asarray(roi_frames)
#     if roi_frames.ndim==3:
#         mean_curve=roi_frames.mean(axis=(1,2))
#     elif roi_frames.ndim==4:
#         mean_curve=roi_frames.mean(axis=(1,2,3))
#     else:
#         raise RuntimeError("roi_frames.ndim must be 3 or 4")
#     return mean_curve.reshape(-1)
# def save_frame_npz(grays,fps):
#     grays_arr=np.stack(grays,axis=0)
#     np.savez(OUT_NPZ,grays=grays_arr,fps=float(fps))
#     print("save:",OUT_NPZ,"shape:",grays_arr.shape,"fps:",fps)

# def roi(grays,roi_xywh):
#     x,y,w,h=roi_xywh
#     patches=[]
#     for i,g in enumerate(grays):
#         patch=g[y:y+h,x:x+w]
#         if patch.shape[0]!=h or patch.shape[1]!=w:
#             raise RuntimeError(f"ROI out of bounds at frame {i}, got {patch.shape}, expect {(h,w)}")
#         patches.append(patch.copy())
#     return patche

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
    )
    ts = datetime.now().strftime("%Y-%m-%d")
    logger = setup_logging( ROOT / "results" / "logs", name=f"{ts}")
    cap=cv.VideoCapture(VIDEO)
    print("cap opened:", cap.isOpened())
    if not cap.isOpened():
        raise RuntimeError("cannot open video (check path/codec)")
    fps=cap.get(cv.CAP_PROP_FPS)
    grays=read_video_frames(cap)
    roi_xywh = (200,100,128,128)
    # grays_roi=roi(grays,roi_xywh)
    npz_path=step_extract_roi(VIDEO,roi_xywh,200)
    # save_frame_npz(grays_roi, fps)
    #npz文件全部跑一边
    for i in  npz_path:
        run_roi_mean_probe(i)

    cap.release()


if __name__ == "__main__":
    main()