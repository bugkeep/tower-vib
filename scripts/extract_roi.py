import argparse
import cv2 as cv
from pathlib import Path
from datetime import datetime
import numpy as np
ts=datetime.now().strftime("%Y-%m-%d")
ROOT=Path.cwd()
out_mp4=ROOT/"results"/"roi"/f"{ts}_main_roi.mp4"
out_npz=ROOT/"results"/"cache"/f"{ts}_main_roi_frames.npz"
out_mp4.parent.mkdir(parents=True,exist_ok=True)
out_npz.parent.mkdir(parents=True,exist_ok=True)
def parse_args():
    p=argparse.ArgumentParser(description="2026-1-15:extract ROI frames from a video")
    p.add_argument("--video",required=True,help="Path to input video file")
    p.add_argument("--roi",required=True,nargs=4,type=int, metavar=("X","Y","W","H"),help="ROI rectangle:x y w h")
    p.add_argument("--max_frames",default=200,type=int,help="Max frames to process(default:200)")
    return p.parse_args()

def getframe(cap,dps):
    frams=[]
    for i in range(dps):
        ok,fram=cap.read()
        if not ok:
            break
        frams.append(fram)
    return frams

def roi(frams,x,y,w,h):
    H,W=frams[0].shape[:2]
    if  not ((0<=x<W) and (0<=y<H) and (x+w<=W) and (y+h<=H) and (w>0) and (h>0)) :
        raise RuntimeError(f"ROI out bounds: frame (W,H)={W,H},roi={x,y,w,h}")
    frams_roi=[]
    for frame in frams:
        frame_roi=frame[y:y+h,x:x+w]
        frams_roi.append(frame_roi)
    return frams_roi

def save_ROI_mp4(frames_roi,fps,out_path):
    h2,w2=frames_roi[0].shape[:2]
    fourcc=cv.VideoWriter_fourcc(*"mp4v")
    writer=cv.VideoWriter(str(out_path),fourcc,float(fps),(w2,h2))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed to open.Check codec/fps/path.")
    for f in frames_roi:
        writer.write(f)
    writer.release()

def main():
    args=parse_args()
    video_path=args.video
    x,y,w,h=args.roi
    max_frames=args.max_frames
    cap=cv.VideoCapture(video_path)
    fps=cap.get(cv.CAP_PROP_FPS)
    fps = fps if fps and fps > 1 else 25
    frams=getframe(cap,max_frames)
    if len(frams) ==0:
        raise RuntimeError("No frames read ...")
    print("frame:",frams[0].shape)
    frams_roi=roi(frams,x,y,w,h)
    save_ROI_mp4(frams_roi,fps,out_mp4)
    print("frames_roi:",frams_roi[0].shape)
    print("roi_frame:",frams_roi[0].min(),frams_roi[0].max())
    arr=np.stack(frams_roi,axis=0)
    np.savez_compressed(out_npz, frames=arr, roi=[x, y, w, h], video=str(video_path), fps=float(fps))
    print("video:",video_path)
    print("roi:",(x,y,w,h))
    print("max_frames:",max_frames)
    cap.release()

if __name__=="__main__":
    main()