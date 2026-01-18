import argparse
import re
import os
from datetime import datetime
def parse_args():
    parser=argparse.ArgumentParser(description="tower-vib")
    parser.add_argument('--requirements',type=str,default="./requirements.txt",help='requirements.txt文件路径')
    parser.add_argument('--output',type=str,default='tower-vib',help='输出目录')
    parser.add_argument('--exit-on-failure',default=False,action='store_true',help='失败时退出')
    return parser.parse_args()

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

def get_package_versions(packages:str)->str|None:
    package_name = packages.split("[")[0].strip()
    try:
        from importlib.metadata import version as meta_version
    except Exception:
        return None
    try:
        return meta_version(package_name)
    except Exception:
        return None


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

def check_package_installed(requirement:dict)->dict:
    package=requirement["package"]
    operator=requirement["operator"]
    required_version=requirement.get("version")
    installed_version=get_package_versions(package)
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
def verify_environment(requirements_path:str)->dict:
    reqs=parse_requirements(requirements_path)
    results=[check_package_installed(r) for r in reqs]

    failed=[r for r in results if not r["satisfied"]]
    success=(len(failed)==0)
    return {
        "success":success,
        "results":results,
        "failed":failed
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

    return report_path
def main():
    args=parse_args()
    print(f"Requirement路径：{args.requirements}")
    print(f"输出目录：{args.output}")
    print(f"是否退出：{args.exit_on_failure}")

    if args.exit_on_failure:
        print(">>当前模式：严格模式（失败即退出）")
    else:
        print(">>当前模式：非严格模式（失败时继续）")

    fake={"package":"some_pkg","operator":">=","version":"1.0.0"}
    print("Fake test:",check_package_installed(fake))
    requirements=parse_requirements(args.requirements)
    for requirement in requirements:
        result=check_package_installed(requirement)
        print(f"{result['package']}:{result['installed_version']}")
    env = verify_environment(args.requirements)
    report_path=generate_check_report(env,args.output)
    print(f"Report generated at: {report_path}")
    if env["failed"]:
        print("FAILED PACKAGES:")
        for f in env["failed"]:
            print(" -",f["package"],"|installed:",f["installed"],"|required:",f["required"])

    if args.exit_on_failure and not env["success"]:
        raise SystemExit(1)

if __name__=="__main__":
    main()