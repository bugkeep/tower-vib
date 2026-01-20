import numpy as np
import logging
import pandas as pd
from pathlib import Path
import argparse

def setup_logging():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def check_arry(arry,name,expected_shape=None,expected_dtype=None):
    if not isinstance(arry,np.ndarray):
        return {
        "name": name,
        "shape": None,
        "dtype": None,
        "has_nan": False,
        "has_inf": False,
        "is_valid": False,
        "errors": ["不是数组"]
        }
    is_float = np.issubdtype(arry.dtype, np.floating)
    result={
        'name':name,
        "shape":arry.shape,
        "dtype":str(arry.dtype),
        "has_nan":bool(np.isnan(arry).any())if is_float else False,
        "has_inf":bool(np.isinf(arry).any())if is_float else False,
        "is_valid":True,
        "errors":[]
    }
    if  result["has_nan"]:
        result["is_valid"]=False
        result["errors"].append("有NaN")
    if  result["has_inf"]:
        result["is_valid"]=False
        result["errors"].append("有Inf")

    if expected_dtype is not None:
        if arry.dtype != np.dtype(expected_dtype):
            result["is_valid"] = False
            result["errors"].append(
                f"数据类型错误，期望{np.dtype(expected_dtype)}，实际{arry.dtype}"
            )

    if expected_shape is not None:
        if result["shape"] != expected_shape:
            result["is_valid"]=False
            result["errors"].append(f"形状错误，期望{expected_shape}，实际{result['shape']}")

    if np.issubdtype(arry.dtype, np.number) and arry.size > 0:
        result["min"] = float(np.nanmin(arry))
        result["max"] = float(np.nanmax(arry))

    return result

def check_video_frames(cache_path,logger):
    """
    检查的是*frames.npz
    :param cache_path:
    :param logger:
    :return:
    """
    if not Path(cache_path).exists():
        msg=f"npz文件{cache_path}不存在"
        logger.error(msg)
        return {"name":"video_frames","is_valid":False,"errors":[msg],"shape": None, "dtype": None}
    try:
        data=np.load(cache_path)
        frames=data["frames"]
        logger.info(f"检查视频帧:{cache_path}")
        return check_arry(frames,"video_frames",expected_dtype=np.uint8)
    except Exception as e:
        logger.error(f"读取失败:{e}")
        return {"name": "video_frames", "is_valid": False, "errors": [f"读取失败{e}"], "shape": None, "dtype": None}


def check_roi_frames(cache_path,logger):
    """
    检查的是main_roi_frames.npz
    :param cache_path:
    :param logger:
    :return:
    """
    if not Path(cache_path).exists():
        msg=f"缓存目录{cache_path}不存在"
        logger.error(msg)
        return {"name":"roi_frames","is_valid":False,"errors":[msg],"shape": None, "dtype": None}
    try:
        data=np.load(cache_path)
        frames=data["frames"]
        logger.info(f"检查ROI帧:{cache_path}")
        return check_arry(frames,"roi_frames",expected_dtype=np.uint8)
    except Exception as e:
        logger.error(f"读取失败:{e}")
        return {"name": "roi_frames", "is_valid": False, "errors": [f"读取失败{e}"], "shape": None, "dtype": None}


def check_mean_curve(data_path,logger):
    """
    检查的是mean_curve.csv
    :param data_path:
    :param logger:
    :return:
    """
    if not Path(data_path).exists():
        msg=f"csv文件{data_path}不存在"
        logger.error(msg)
        return {"name": "mean_curve", "is_valid": False, "errors": [msg], "shape": None, "dtype": None}
    try:
        df=pd.read_csv(data_path)
        required = {"mean_r", "mean_g", "mean_b", "mean_value"}
        missing = required - set(df.columns)
        if missing:
            msg = f"csv 文件{data_path}缺少列: {sorted(missing)}"
            logger.error(msg)
            return {"name": "mean_curve", "is_valid": False, "errors": [msg], "shape": None, "dtype": None}
        mean_curve=df["mean_value"].to_numpy(dtype=np.float32)
        logger.info(f"检查均值曲线:{data_path}")
        result=check_arry(mean_curve,"mean_curve",expected_dtype=np.float32)
        if len(mean_curve)>1:
            diff=np.diff(mean_curve)
            abs_diff=np.abs(diff)
            mu=abs_diff.mean()
            sigma=abs_diff.std(ddof=0)
            threshold=mu+3.0*sigma
            threshold = max(threshold,1e-6)
            abnormal_indices=np.where(abs_diff>threshold)[0]
            if len(abnormal_indices)>0:
                result["is_valid"]=False
                result["errors"].append(f"发现异常突变(阈值={threshold:.2f}): 帧边界 {abnormal_indices.tolist()}")
        return result
    except Exception as e:
        logger.error(f"读取失败:{e}")
        return {"name": "mean_curve", "is_valid": False, "errors": [f"读取失败{e}"], "shape": None, "dtype": None}


def generate_report(results,output_path,logger):
    output_path=Path(output_path)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with open(output_path,"w",encoding="utf-8") as f:
        f.write(f"#Tower-Vib 防爆检查报告\n")
        valid_count=sum(1 for r in results if r["is_valid"])
        f.write(f"**总项数**: {len(results)} | **通过**: {valid_count} | **失败**: {len(results)-valid_count}\n\n")
        f.write("----\n")
        for res in results:
            icon="✅" if res["is_valid"] else "❌"
            f.write(f"\n### {icon} {res['name']}\n")
            f.write(f"- shape: {res['shape']}\n")
            f.write(f"- type: {res['dtype']}\n")
            if res.get("min") is not None:
                f.write(f"-Range: [{res['min']:.2f},{res['max']:.2f}]\n")
            if res["errors"]:
                f.write("**错误列表：**\n")
                for err in res["errors"]:
                    f.write(f"-{err}\n")


    logger.info(f"报告已生成:{output_path}")

#=========配置区域=====================
CHECK_RULES = [{
    "pattern": "cache/*_main_roi_frames.npz",
    "func": check_roi_frames,
    "desc": "视频/ROI帧缓冲检查"

}, {
    "pattern": "cache/????-??-??_frames.npz",
    "func": check_video_frames,
    "desc": "视频帧缓冲检查"
},
{
    "pattern": "data/*_mean_curve.csv",
    "func": check_mean_curve,
    "desc": "检查csv均值曲线"
},
]


def run_checks_by_rules(results,logger):
    """
    根据规则自动扫描并检查文件
    :param results:
    :param logger:
    :return:
    """
    results_dir = Path(results)
    all_results=[]

    logger.info("开始检查...")
    for rule in CHECK_RULES:
        pattern=rule["pattern"]
        check_func=rule["func"]

        matched_files=list(results_dir.glob(pattern))
        if not matched_files:
            logger.warning(f"未找到匹配文件:{pattern}")
            continue
        for file_path in matched_files:
            try:
                result=check_func(file_path,logger)
                if result:
                    result["file_name"]=file_path.name
                    all_results.append(result)
            except Exception as e:
                logger.error(f"检查失败:{e}")
    return all_results




def main():
    parser=argparse.ArgumentParser(description="Tower-Vib 防爆检查工具")
    parser.add_argument("--results_dir",type=str,default="results")
    parser.add_argument("--output",type=str,default="tower-vib/sanity_check_report.txt")
    args=parser.parse_args()
    logger=setup_logging()
    base_dir=Path(__file__).resolve().parents[1]
    all_results=run_checks_by_rules(base_dir/Path(args.results_dir),logger)
    generate_report(all_results,args.output,logger)

if __name__ == '__main__':
    main()