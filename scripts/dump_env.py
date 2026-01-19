from __future__ import annotations
from datetime import datetime
import argparse
import json
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
from env_utils import (
    get_python_info, get_system_info, get_cpu_info,
    collect_gpu_info, get_package_versions
)
def parse_args():
    parser = argparse.ArgumentParser(description="tower-vib:")
    parser.add_argument('--output_dir', type=str, default='tower-vib', help='输出目录')
    parser.add_argument('--format', type=str, choices=['txt', 'json', 'both'], default='txt', help='输出格式')
    parser.add_argument('--generate_requirements', action='store_true', help='是否生成requiremens.txt')
    parser.add_argument('--dump_env_info', action='store_true',default=True, help='是否生成json文件')
    return parser.parse_args()

def dump_env_info(path,format):
    result = {
        "python": get_python_info(),
        "system":get_system_info() ,
        "packages": get_package_versions(),
        "gpu_info": collect_gpu_info(),
        "cpu": get_cpu_info(),
    }
    if format=="json":
        with open(path/"env_info.json", "w",encoding="UTF-8") as f:
            json.dump(result, f, indent=4,ensure_ascii=False)
            print("已保存到 env_info.json")
    elif format=="txt":
        with open(path/"env_info.txt", "w",encoding="UTF-8") as f:
            f.write(json.dumps(result, indent=4,ensure_ascii=False))
            print("已保存到 env_info.txt")
    elif format=="both":
        with open(path / "env_info.json", "w",encoding="UTF-8") as f:
            json.dump(result, f, indent=4,ensure_ascii=False)
        with open(path / "env_info.txt", "w",encoding="UTF-8") as f:
            f.write(json.dumps(result, indent=4,ensure_ascii= False))
        print("已保存env_info.json和env_info.txt")
    else:
        raise ValueError("unexpected format")

def generate_requirements_txt(out_dir: Path):
    results = get_package_versions()
    with open(out_dir/"requirements.txt", "w",encoding="UTF-8") as f:
        f.write("#tower-vib依赖包列表\n")
        f.write(f"#生存时间：{datetime.now().isoformat()}\n")
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