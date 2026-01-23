import cv2 as cv
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
ts=datetime.now().strftime("%Y-%m-%d")
def save_phasecorr_plots(
        df,
        out_dir,
        resp_th:float=0.05,
        disp_name:str=f"{ts}_phase_corr.png",
        resp_name:str=f"{ts}_phase_corr_response.png",
        mark_low_resp:bool=True,
        dpi:int=150
):
    """
    保存phase correlation结果图：
    1）dx/dy 折线图
    2）response折线图（带阈值，可选标记低response点）

    :param df:
    :param out_dir:
    :param resp_th:
    :param disp_name:
    :param resp_name:
    :param mark_low_resp:
    :param dpi:
    :return: 输出目录（str/path）
    """
    out_dir=Path(out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    #-----1)dx/dy plot-----
    fig,ax=plt.subplots(figsize=(10,4))
    ax.plot(df["frames"],df["dx"],label="dx")
    ax.plot(df["frames"],df["dy"],label="dy")
    ax.legend()
    ax.set_title("Phase correlation displacement(unwrap)")
    ax.set_xlabel("frames")
    ax.set_ylabel("pixels")
    fig.tight_layout()
    fig.savefig(out_dir/disp_name,dpi=dpi)
    plt.close(fig)
    #-----2)response plot-----
    fig,ax=plt.subplots(figsize=(10,3))
    ax.plot(df["frames"],df["response"],label="response")
    ax.axhline(resp_th,linestyle="--")
    ax.set_title("Phase correlation response")
    ax.set_xlabel("frames")
    ax.set_ylabel("response")
    ax.legend()
    if mark_low_resp:
        if "valid" in df.columns:
            bad=df[df["valid"]==0]
        else:
            bad=df[df["response"]<resp_th]
        if len(bad)>0:
            ax.scatter(bad["frames"],bad["response"],s=12)

    fig.tight_layout()
    fig.savefig(out_dir/resp_name,dpi=dpi)
    plt.close(fig)
def unwrap_shift(dx, dy, H, W):
    # 折回到 [-W/2, W/2] 和 [-H/2, H/2]
    if dx >  W/2: dx -= W
    if dx < -W/2: dx += W
    if dy >  H/2: dy -= H
    if dy < -H/2: dy += H
    return dx, dy
def setup_logging():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)
def preprocess_frame(frame:np.ndarray)->np.ndarray:
    """
    输入uint8/灰度->float32，减均值避免直流影响,3通道转换为单通道
    """
    if frame.ndim == 3:
        frame=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    f=frame.astype(np.float32)
    return f-f.mean()

def complete_phase_corr(ref_f,tgt_f,hann=None):
    """
    返回dx，dy，response
    """
    if hann is None:
        hann=cv.createHanningWindow(ref_f.shape[::-1],cv.CV_32F)
    (dx,dy),response=cv.phaseCorrelate(ref_f,tgt_f,hann)
    return dx,dy,response
def compute_displacements_with_refresh(frames,ref_idx=0,max_frames=None,resp_th = 0.05,refresh_on_low=True):
    """
    批处理
    """
    logger = setup_logging()
    n=len(frames)
    end=n if max_frames is None else min(n,max_frames)

    ref=preprocess_frame(frames[ref_idx])
    hann = cv.createHanningWindow(ref.shape[::-1], cv.CV_32F)
    out=[]
    current_ref_frame_idx = ref_idx
    H,W=ref.shape[:2]
    for i in range(ref_idx+1,end):
        tgt=preprocess_frame(frames[i])
        dx,dy,response=complete_phase_corr(ref,tgt,hann)
        dx,dy=unwrap_shift(dx,dy,H,W)
        resp_use = float(np.clip(response, 0.0, 1.0))
        valid_response = 1 if resp_use >= resp_th else 0
        out.append((i, dx, dy, resp_use, valid_response, current_ref_frame_idx))
    #----关键：低response刷新参考帧
        if refresh_on_low and resp_use < resp_th:
            current_ref_frame_idx = i
            logger.info(f"refresh ref at frame {current_ref_frame_idx}, response={response:.3f}")
            ref=tgt
            H, W = ref.shape[:2]
            hann = cv.createHanningWindow(ref.shape[::-1], cv.CV_32F)
    return  out
def compute_multi_roi_displacements(
        roi_frames_list,
        ref_idx=1,
        max_frames=None
):
    """
    多roi批处理
    """
    num_rois=len(roi_frames_list)
    N=len(roi_frames_list[0]) if max_frames is None  else min(len(roi_frames_list[0]),max_frames)
    results={
        "frames":list(range(ref_idx+1,N))
    }
    for roi_idx in range(num_rois):
        results[f"dx_roi{roi_idx}"]=[]
        results[f"dy_roi{roi_idx}"]=[]
        results[f"response_roi{roi_idx}"]=[]

    for roi_idx,roi_frames in enumerate(roi_frames_list):
        ref=preprocess_frame(roi_frames[ref_idx])
        H, W = ref.shape[:2]
        hann = cv.createHanningWindow(ref.shape[::-1], cv.CV_32F)
        for i in range(ref_idx+1,N):
            tgt=preprocess_frame(roi_frames[i])
            dx,dy,response=complete_phase_corr(ref,tgt,hann)
            unwrap_shift(dx,dy,H,W)
            results[f"dx_roi{roi_idx}"].append(dx)
            results[f"dy_roi{roi_idx}"].append(dy)
            results[f"response_roi{roi_idx}"].append(response)
    results["dx_mean"]=np.mean([results[f"dx_roi{i}"] for i in range(num_rois)],axis=0)
    results["dy_mean"]=np.mean([results[f"dy_roi{i}"] for i in range(num_rois)],axis=0)
    results["response_mean"]=np.mean([results[f"response_roi{i}"] for i in range(num_rois)],axis=0)
    return pd.DataFrame(results)
def check_roi_consistency(results,threshold=2.0):
    """
    检查多个roi的位移一致性
    """
    num_rois=len([col for col in results.columns if col.startswith("dx_roi")])

    inconsistent=[]
    for i in range(len(results)):
        dx_values=[results[f"dx_roi{j}"].iloc[i] for j in range(num_rois)]
        dy_values=[results[f"dy_roi{j}"].iloc[i] for j in range(num_rois)]

        dx_std=np.std(dx_values)
        dy_std=np.std(dy_values)

        if dx_std>threshold or dy_std>threshold:
            inconsistent.append({
                "frames":results["frames"].iloc[i],
                "dx_std":dx_std,
                "dy_std":dy_std
            })
    return inconsistent
def compute_displacements_sliding_window(
        frames,
        window_size=10,
        step=1,
        max_frames=None
):
    """
    滑动窗口
    """
    results={
        "frames":[],
        "dx":[],
        "dy":[],
        "response":[]
    }
    N=len(frames) if max_frames is None else min(len(frames),max_frames)
    for i in range(0,N-window_size+1,step):
        window_frames=frames[i:i+window_size]
        ref=preprocess_frame(window_frames[0])
        tgt=preprocess_frame(window_frames[-1])
        dx,dy,responses=complete_phase_corr(ref,tgt)
        results["frames"].append(i+window_size-1)
        results["dx"].append(dx)
        results["dy"].append(dy)
        results["response"].append(responses)
    return pd.DataFrame(results)

def main():
    logger=setup_logging()
    ap=argparse.ArgumentParser()
    ap.add_argument("--roi_cache",type=str,required=True,help="roi文件")
    ap.add_argument("--output_dir",type=str,required=True,help="输出目录(传目录)")
    ap.add_argument("--max_frames",type=int,default=200)
    ap.add_argument("--ref_idx",type=int,default=0)
    ap.add_argument("--response_threshold",type=float,default=0.05)
    ap.add_argument("--refresh_strategy",type=str,default="response",
                    choices=["response","interval","accumulation"])
    ap.add_argument("--refresh_interval",type=int,default=50)
    ap.add_argument("--max_accumulation",type=int,default=10.0)
    args=ap.parse_args()

    data=np.load(args.roi_cache)
    frames=data["frames"]
    logger.info(f"frames: shape={frames.shape}, dtype={frames.dtype}")
    results=compute_displacements_with_refresh(frames,args.ref_idx,args.max_frames,args.response_threshold)
    df=pd.DataFrame(results,columns=["frames","dx","dy","response","valid","ref_frame_idx"])
    low = (df["valid"] == 0).sum()
    logger.info(f"low-response frames (<{args.response_threshold}): {low}/{len(df)}")
    save_phasecorr_plots(df, args.output_dir, args.response_threshold)
    out_dir=Path(args.output_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    out_csv=out_dir/"displacements.csv"
    df.to_csv(out_csv,index=False)

    logger.info(f"Results saved to: {out_csv}")
if __name__ == "__main__":
    main()