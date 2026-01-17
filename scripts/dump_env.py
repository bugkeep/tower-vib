from __future__ import annotations
from datetime import datetime
import os
import argparse
import sys
import platform
import subprocess
import json
from os import mkdir
from typing import Iterable,Dict,Any,Optional
from pathlib import Path
def parse_args():
    parser = argparse.ArgumentParser(description="tower-vib:")
    parser.add_argument('--output_dir', type=str, default='tower-vib', help='输出目录')
    parser.add_argument('--format', type=str, choices=['txt', 'json', 'both'], default='txt', help='输出格式')
    parser.add_argument('--generate_requirements', action='store_true', help='是否生成requiremens.txt')
    parser.add_argument('--dump_env_info', action='store_true', help='是否生成json文件')
    return parser.parse_args()
def get_python_info():
    python_info={
        'python_version':sys.version,
        'python_path':sys.executable
    }
    return python_info
def get_system_info():
    system_info={
        'system':platform.system(),
        'release':platform.release(),
        'version':platform.version(),
    }
    return system_info
def get_cpu_info():
    cpu_info={
        'arch':platform.machine(),
        'cpu_count':os.cpu_count(),
        'cpu_model':platform.processor()
    }
    return cpu_info
import subprocess

def get_gpu_info_nvidia_smi():
    """
    返回: (gpus, error)
    - gpus: list[dict]，每张卡一个 dict
    - error: None 或字符串错误原因
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,pci.bus_id,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits"
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError:
        return [], "nvidia-smi not found (可能未安装NVIDIA驱动或不在PATH)"
    except subprocess.CalledProcessError as e:
        return [], f"nvidia-smi failed: {e.output.strip()}"

    gpus = []
    lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        # 与 query 字段一一对应
        keys = [
            "index","name","uuid","driver_version","pci_bus_id",
            "mem_total_mb","mem_used_mb","mem_free_mb","util_gpu_percent","temp_c"
        ]
        d = dict(zip(keys, parts))
        # 尝试把数字字段转成 int（失败就保持字符串）
        for k in ["index","mem_total_mb","mem_used_mb","mem_free_mb","util_gpu_percent","temp_c"]:
            try:
                d[k] = int(float(d[k]))
            except Exception:
                pass
        gpus.append(d)

    return gpus, None

def get_gpu_info_torch():
    """
    返回: (info, error)
    - info: dict
    - error: None 或字符串
    """
    try:
        import torch
    except Exception as e:
        return {}, f"torch not available: {e}"

    info = {
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version_built": getattr(torch.version, "cuda", None),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    gpus = []
    if info["cuda_available"]:
        for i in range(info["device_count"]):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "index": i,
                "name": props.name,
                "total_memory_mb": int(props.total_memory / (1024**2)),
                "major": props.major,
                "minor": props.minor,
                "multi_processor_count": props.multi_processor_count,
            })
    info["gpus"] = gpus
    return info, None
def collect_gpu_info():
    result = {
        "nvidia_smi": {"gpus": [], "error": None},
        "torch": {"info": {}, "error": None},
        "summary": ""
    }

    smi_gpus, smi_err = get_gpu_info_nvidia_smi()
    result["nvidia_smi"]["gpus"] = smi_gpus
    result["nvidia_smi"]["error"] = smi_err

    torch_info, torch_err = get_gpu_info_torch()
    result["torch"]["info"] = torch_info
    result["torch"]["error"] = torch_err

    # 总结逻辑（你也可以按需要调整）
    if smi_gpus:
        result["summary"] = f"检测到 {len(smi_gpus)} 张 NVIDIA GPU（来自 nvidia-smi）。"
    elif torch_info.get("cuda_available"):
        result["summary"] = f"torch 显示 CUDA 可用，检测到 {torch_info.get('device_count', 0)} 张 GPU（来自 torch）。"
    else:
        # 没有检测到
        if smi_err and "not found" in smi_err:
            result["summary"] = "未检测到 NVIDIA GPU（或未安装/未配置 NVIDIA 驱动导致 nvidia-smi 不可用）。"
        elif smi_err:
            result["summary"] = f"nvidia-smi 运行失败，且 torch 未检测到 CUDA：{smi_err}"
        else:
            result["summary"] = "未检测到可用 GPU（nvidia-smi 无输出，torch 也未检测到 CUDA）。"

    return result

def get_package_versions(
    packages: Optional[Iterable[str]] = None,
    include_python: bool = True,
) -> Dict[str, Any]:
    """
    获取依赖包版本信息，未安装的包会标记为 not_installed，不抛异常。

    返回示例:
    {
      "python": "3.11.7",
      "torch": {"version": "2.2.1", "status": "installed"},
      "numpy": {"version": None, "status": "not_installed", "error": "..."}
    }
    """
    if packages is None:
        packages = [
            # 你可以按需要增删
            "torch", "torchvision", "torchaudio",
            "numpy", "scipy", "pandas",
            "opencv-python", "Pillow",
            "matplotlib",
            "transformers", "accelerate",
            "cuda-python",
        ]

    result: Dict[str, Any] = {}

    if include_python:
        import sys
        result["python"] = sys.version.split()[0]

    # 先尝试 importlib.metadata（标准库）
    try:
        from importlib.metadata import version as _version, PackageNotFoundError  # py3.8+
        def get_ver(name: str):
            try:
                return _version(name), None
            except PackageNotFoundError as e:
                return None, str(e)
            except Exception as e:
                return None, repr(e)

    except Exception:
        # 降级到 pkg_resources（setuptools）
        def get_ver(name: str):
            try:
                import pkg_resources
                return pkg_resources.get_distribution(name).version, None
            except Exception as e:
                return None, repr(e)

    for p in packages:
        ver, err = get_ver(p)
        if ver is not None:
            result[p] = {"version": ver, "status": "installed"}
        else:
            result[p] = {"version": None, "status": "not_installed", "error": err}

    return result

def dump_env_info(path,format):
    result = {
        "python": get_python_info(),
        "system":get_system_info() ,
        "packages": get_package_versions(),
        "gpu_info": collect_gpu_info(),
        "cpu": get_cpu_info(),
    }
    if format=="json":
        with open(path/"env_info.json", "w") as f:
            json.dump(result, f, indent=4,ensure_ascii=False)
            print("已保存到 env_info.json")
    elif format=="txt":
        with open(path/"env_info.txt", "w") as f:
            f.write(json.dumps(result, indent=4,ensure_ascii=False))
            print("已保存到 env_info.txt")
    elif format=="both":
        with open(path / "env_info.json", "w") as f:
            json.dump(result, f, indent=4)
        with open(path / "env_info.txt", "w") as f:
            f.write(json.dumps(result, indent=4))
    else:
        raise ValueError("unexpected format")

def generate_requirements_txt(out_dir: Path):
    results = get_package_versions()
    with open(out_dir/"requirements.txt", "w",encoding="UTF-8") as f:
        f.write("#tower-vib依赖包列表\n")
        f.write(f"#生存时间：{datetime.now().isoformat()}")
        for k, v in results.items():
            if k == "python":
                f.write(f"#python=={v}\n")
            elif v.get("status") == "installed":
                f.write(f"{k}=={v['version']}\n")
            else:
                f.write(f"#{k} not installed\n")

def main():
    args = parse_args()
    path = Path(args.output_dir)
    path.mkdir(parents=True, exist_ok=True)
    if args.generate_requirements:
        print("正在生成requirements.txt...")
        generate_requirements_txt(path)
        print("生成完成")
    if args.dump_env_info:
        print("正在保存环境信息...")
        dump_env_info(path,args.format)
        print("保存完成")
if __name__=="__main__":
    main()