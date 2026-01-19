from __future__ import annotations

from datetime import datetime
import platform
import re
import sys
from  pathlib import Path
from typing import Any, Dict, Tuple, Optional, Iterable
import json
import os

def _extra_pkg_version_status(package:Dict[str,Any])->Dict[str,str]:
    """
    统一将env_info.json里面package中的值转化为(pkg_name,version)
    :param v:
    :return:
    """
    flat:Dict[str,str]={}
    for name,v in (package or {}).items():
        if isinstance(v,str):
            flat[name]=v
        elif isinstance(v,dict):
            flat[name]=str(v.get("version","unkonwn"))
        else:
            flat[name]="Not installed"
    return flat

def compare_environments(env1_dir,env2_dir):
    with open(env1_dir/"env_info.json","r",encoding="utf-8") as f:
        env1=json.load(f)
    with open(env2_dir/"env_info.json","r",encoding="utf-8") as f:
        env2=json.load(f)
    diff_report: Dict[str,Dict[str,Any]]={
        "python":{},
        "system":{},
        "packages":{}
    }
    def record_diff(section:str,key:str,v1:Any,v2:Any)->None:
        if v1!=v2:
            diff_report[section][key]={
                "env1":v1,
                "env2":v2
            }
    py1=env1.get("python",{})
    py2=env2.get("python",{})
    record_diff("python","python_version",py1.get("python_version"),py2.get("python_version"))
    record_diff("python","python_path",py1.get("python_path"),py2.get("python_path"))

    sys1=env1.get("system",{})
    sys2=env2.get("system",{})
    record_diff("system","system",sys1.get("system"),sys2.get("system"))
    record_diff("system","release",sys1.get("release"),sys2.get("release"))
    record_diff("system","version",sys1.get("version"),sys2.get("version"))

    package1=_extra_pkg_version_status(env1.get("packages",{}))
    package2=_extra_pkg_version_status(env2.get("packages",{}))
    all_pkgs=set(package1.keys())|set(package2.keys())
    diff={}
    for pkg in sorted(all_pkgs):
        v1=package1.get(pkg,"Not installed")
        v2=package2.get(pkg,"Not installed")
        if v1!=v2:
            record_diff("packages",pkg,v1,v2)
    print("环境对比成功")
    return diff_report

def save_comparison_report(diff_report,out_path)->Path:
    """
    保存环境对比报告
    :param diff_report:
    :param out_path:
    :return:
    """
    out_path=Path(out_path)
    out_path.mkdir(parents=True,exist_ok=True)

    json_path=out_path/"diff_report.json"
    txt_path=out_path/"diff_report.txt"

    with json_path.open("w",encoding="utf-8") as f:
        json.dump(diff_report,f,ensure_ascii=False,indent=2)

    lines=[]
    for section in ["python","system","cpu","gpu","packages"]:
        sec=diff_report.get(section)or{}
        if not sec:
            continue
        lines.append(f"[{section}]")

        for k in sorted(sec.keys()):
            v=sec[k]
            env1=v.get("env1") if isinstance(v,dict) else None
            env2=v.get("env2") if isinstance(v,dict) else None
            lines.append(f"{k}: {env1} -> {env2}")
        lines.append("")

    with txt_path.open("w",encoding="utf-8")as f:
        f.write("\n".join(lines).rstrip()+"\n")

    return json_path

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

def parse_requirements(requirements_path:str)->list[dict]:
    """
    解析requirement.txt文件，返回包含包名，操作符和版本的字典列表
    """
    results=[]
    if not os.path.exists(requirements_path):
        print(f"{requirements_path}不存在")
        return []
    pattern = re.compile(r'^([a-zA-Z0-9_\-.]+(?:\[[^\]]+\])?)\s*([<>=!~]+)\s*(.+)$')
    try:
        with open(requirements_path,'r',encoding='utf-8') as f:
            for line in f:
                line=line.strip()

                if not line or line.startswith('#'):
                    continue
                line =line.split("#")[0].strip()#去掉注释
                match=pattern.match(line)
                if match:
                    package_name,operator,version=match.groups()
                    package_name = package_name.split("[")[0]
                    results.append({
                        "package":package_name,
                        "operator":operator,
                        "version":version
                    })
                else:
                    if line:
                        results.append(
                            {
                                "package":line,
                                "operator":None,
                                "version":None
                            }
                        )
    except Exception as e:
        print(f"解析requirements.txt文件时出错：{e}")
        return []
    return  results

def compare_versions(installed:str|None,required:str|None,operator:str|None)->bool:
    if operator is None:
        return installed is not None
    if installed is None:
        return False
    try:
        from packaging import version as pkg_version
        a=pkg_version.parse(installed)
        b=pkg_version.parse(required)
    except Exception:
        print(f"无法解析版本：{installed}")
        a=installed
        b=required
    if operator==">":
        return a>b
    elif operator==">=":
        return a>=b
    elif operator=="<":
        return a<b
    elif operator=="<=":
        return a<=b
    elif operator=="==":
        return a==b
    elif operator=="!=":
        return a!=b
    else:
        return False

def _normalize_pkg_name(name: str) -> str:
    # requirements / pip 名一般用 -，这里做轻度统一
    return name.strip()

def export_requirements_txt(out_path: str) -> str:
    """
    生成requirement.txt文件
    :param out_path:
    :return:
    """
    pkgs = list_installed_packages()
    lines = [f"{name}=={ver}" for name, ver in sorted(pkgs.items(), key=lambda x: x[0].lower())]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path

def list_installed_packages() -> Dict[str, str]:
    """
    返回当前 Python 环境中所有已安装 pip 包：{package_name: version}
    """
    try:
        from importlib.metadata import distributions  # py3.8+
    except Exception:
        # 极老环境兜底
        import pkg_resources
        return {d.project_name: d.version for d in pkg_resources.working_set}

    pkgs: Dict[str, str] = {}
    for dist in distributions():
        # dist.metadata['Name'] 是规范名字；dist.version 是版本字符串
        name = dist.metadata.get("Name") or dist.name
        if not name:
            continue
        pkgs[name] = dist.version

    return pkgs
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
            "torch", "torchvision", "torchaudio",
            "numpy", "scipy", "pandas",
            "opencv-python", "pillow",
            "matplotlib",
            "transformers", "accelerate",
            "cuda-python",
        ]

    result: Dict[str, Any] = {}

    if include_python:
        result["python"] = sys.version.split()[0]

    # 优先 importlib.metadata
    use_metadata = True
    try:
        from importlib.metadata import version as meta_version, PackageNotFoundError
    except Exception:
        use_metadata = False

    def get_ver(name: str):
        name = _normalize_pkg_name(name)

        # 1) metadata / pkg_resources
        if use_metadata:
            try:
                return meta_version(name), None
            except PackageNotFoundError as e:
                ver, err = None, str(e)
            except Exception as e:
                ver, err = None, repr(e)
        else:
            try:
                import pkg_resources
                return pkg_resources.get_distribution(name).version, None
            except Exception as e:
                ver, err = None, repr(e)

        # 2) 回退：import 模块读 __version__（有些包可能更适配）
        pkg_to_module = {
            "opencv-python": "cv2",
            "pillow": "PIL",
            "scikit-image": "skimage",
        }
        module_name = pkg_to_module.get(name, name.replace("-", "_"))
        try:
            module = __import__(module_name)
            v = getattr(module, "__version__", None)
            if v is not None:
                return str(v), None
        except Exception:
            pass

        return ver, err

    for p in packages:
        ver, err = get_ver(p)
        if ver is not None:
            result[p] = {"version": ver, "status": "installed"}
        else:
            result[p] = {"version": None, "status": "not_installed", "error": err}

    return result

def get_package_version(package_name:str)->Optional[str]:
    """
    获取单个包的已安装的版本号，未安装的返回None
    :param package_name:
    :return:
    """
    versions=get_package_versions([package_name],include_python=False)
    info = versions.get(package_name)
    if isinstance(info,dict) and info.get("status")=="installed":
        return info["version"]
    else:
        return None
def check_package_installed(requirement:dict)->dict:
    package=requirement["package"]
    operator=requirement["operator"]
    required_version=requirement.get("version")
    installed_version=get_package_version(package)
    installed=installed_version is not None

    required=None
    if operator is not None:
        required=f"{operator}{required_version}"
    satisfied=compare_versions(installed_version,required_version,operator)

    return {
        "package":package,
        "operator":operator,
        "required_version":required_version,
        "installed_version":installed_version,
        "installed":installed,
        "required":required,
        "satisfied":satisfied
    }

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

def verify_environment(requirements_path:str)->dict:
    reqs=parse_requirements(requirements_path)
    results=[check_package_installed(r) for r in reqs]

    failed=[r for r in results if not r["satisfied"]]
    success=(len(failed)==0)
    summary={
        "overall": "PASS" if success else "FAIL",
        "total": len(results),
        "failed": len(failed),
        "satisfied": len(results) - len(failed),
    }
    return {
        "success":success,
        "results":results,
        "failed":failed,
        "summary":summary
    }
def generate_check_report(env,output_path:str)->str:
    os.makedirs(output_path,exist_ok=True)
    report_path=os.path.join(output_path,"env_check_report.txt")

    satisfied=[p for p in env["results"]if p["satisfied"]]
    failed=env["failed"]
    def install_hint(item:dict)->str:
        pkg=item["package"]
        req=item["required"]
        if not item["installed"]:
            return f"pip install {pkg}{req or ''}"
        else:
            return f"pip install --upgrade {pkg}{req or ''}"
    with open(report_path,"w",encoding="utf-8") as f:
        f.write("Environment Check Report\n")
        f.write(f"Generated at: {datetime.now().isoformat(timespec='seconds')}\n\n")

        f.write(f"Overall: {'PASS' if env['success'] else 'FAIL'}\n")
        f.write(f"Total packages: {len(env['results'])}\n")
        f.write(f"Satisfied: {len(satisfied)}\n")
        f.write(f"Failed: {len(failed)}\n\n")

        f.write("=== SATISFIED PACKAGES ===\n")
        for item in satisfied:
            f.write(f"- {item['package']} | version: {item['installed_version']} | required: {item['required']}\n")
        f.write("\n")

        f.write("=== FAILED PACKAGES ===\n")
        for item in failed:
            f.write(
                f"- {item['package']} | installed: {item['installed']} | version: {item['installed_version']} | required: {item['required']}\n")
            f.write(f"  -> Hint: {install_hint(item)}\n")
        f.write("\n")
    print("check generate OK")
    return report_path
def generate_env_report(env_info_path,requirements_path,output_path:str)->str:
    os.makedirs(output_path,exist_ok=True)
    report_path=os.path.join(output_path,"env_report.json")
    with open(env_info_path,"r",encoding="utf-8") as f:
        env_info:Dict[str,Any]=json.load(f)
        if "gpu_info" not in env_info or env_info["gpu_info"] in (None,{}):
            env_info["gpu_info"]=collect_gpu_info()
    check_result=verify_environment(requirements_path)
    env_info["check_result"]=check_result

    check_report_path=generate_check_report(check_result,output_path)
    env_info["check_report_path"]=check_report_path
    env_info["check_report_summary"]=(check_result or {}).get("summary")

    env_info["python_version"]=env_info.get("python",{}).get("python_version")
    wanted=["torch","torchvision","numpy","pandas","scipy","scikit-learn","matplotlib","seaborn","tqdm","pillow"]
    versions=get_package_versions(wanted,include_python=False)
    def pick_ver(name:str)->Optional[str]:
        info=versions.get(name)
        if isinstance(info,dict)and info.get("status")=="installed":
            return info["version"]
        return None

    env_info["torch_version"] = pick_ver("torch")
    env_info["torchvision_version"] = pick_ver("torchvision")
    env_info["numpy_version"] = pick_ver("numpy")
    env_info["pandas_version"] = pick_ver("pandas")
    env_info["scipy_version"] = pick_ver("scipy")
    env_info["sklearn_version"] = pick_ver("scikit-learn")  # 注意包名
    env_info["matplotlib_version"] = pick_ver("matplotlib")
    env_info["seaborn_version"] = pick_ver("seaborn")
    env_info["tqdm_version"] = pick_ver("tqdm")
    env_info["pillow_version"] = pick_ver("pillow")
    env_info["cuda_version"] = env_info.get("gpu_info", {}).get("torch", {}).get("info", {}).get("cuda_version_built")

    with open(report_path,"w",encoding="utf-8") as f:
        json.dump(env_info,f,indent=4)
        print("已保存到 env_report.json")
    return report_path