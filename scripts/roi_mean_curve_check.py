import logging
import numpy as np
from pathlib import  Path
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
base=Path(__file__).resolve().parents[1]
def compute_mean_curve(npz_path):
    """
        计算每帧ROI的像素均值
    """
    logger = logging.getLogger(__name__)
    logger.info("计算每帧的ROI的像素均值")
    data=np.load(npz_path)
    logger.info(f"npz_path={npz_path}")
    logger.info(f"data.files={data.files}")
    if "frames" not in data.files:
        raise RuntimeError("frames not in npz")
    roi_frames=data["frames"]
    mean_curve=np.mean(roi_frames,axis=(1,2))
    logger.info(f"计算均值曲线: {len(mean_curve)} 个数据点")
    logger.info(f"均值范围: [{mean_curve.min():.2f}, {mean_curve.max():.2f}]")
    return mean_curve

def save_mean_curve_plot(mean_curve,out_path,title="ROI Mean Curve"):
    """
    保存mean_curve为曲线图
    """
    logger = logging.getLogger(__name__)
    output_path=Path(out_path)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    plt.figure(figsize=(12,6))
    plt.plot(mean_curve,linewidth=1.5)
    plt.xlabel('Frame Number',fontsize=12)
    plt.ylabel('Mean Pixel Value',fontsize=12)
    plt.title(title,fontsize=14)
    plt.grid(True,alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path,dpi=150,bbox_inches='tight')
    plt.close()
    logger.info(f"保存均值曲线: {output_path}")

def save_mean_curve_csv(mean_curve,out_path):
    """
    保存mean_curve为csv
    """
    logger = logging.getLogger(__name__)
    output_path=Path(out_path)
    print("mean_curve.shape =", mean_curve.shape)
    if mean_curve.ndim == 1:
        df = pd.DataFrame({"frame": np.arange(len(mean_curve)),
                           "mean_value": mean_curve})
    elif mean_curve.ndim == 2 and mean_curve.shape[1] == 3:
        df = pd.DataFrame({"frame": np.arange(len(mean_curve)),
                           "mean_r": mean_curve[:, 0],
                           "mean_g": mean_curve[:, 1],
                           "mean_b": mean_curve[:, 2],
                           "mean_value": mean_curve.mean(axis=1)},
        )
    else:
        raise ValueError("unexpected mean_curve shape")
    df.to_csv(output_path,index=False)
    logger.info(f"保存均值曲线: {output_path}")
def run_roi_mean_probe(npz_path):
    """
    运行roi_mean_curve_check
    """
    ts = datetime.now().strftime("%Y-%m-%d")
    mean_curve = compute_mean_curve(npz_path)
    save_mean_curve_plot(mean_curve,base/"results"/"plots"/f"{ts}_mean_curve.png")
    save_mean_curve_csv(mean_curve,base/"results"/"data"/f"{ts}_mean_curve.csv")

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
    )
    ts = datetime.now().strftime("%Y-%m-%d")
    npz_path=base/"results"/"cache"/f"{ts}_main_roi_frames.npz"
    run_roi_mean_probe(npz_path)
if __name__=="__main__":
    main()