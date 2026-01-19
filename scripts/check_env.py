import argparse
import os
from datetime import datetime
from env_utils import (
    parse_requirements, compare_versions,
    check_package_installed, get_package_version,verify_environment,generate_check_report
)
def parse_args():
    parser=argparse.ArgumentParser(description="tower-vib")
    parser.add_argument('--requirements',type=str,default="./requirements.txt",help='requirements.txt文件路径')
    parser.add_argument('--output',type=str,default='tower-vib',help='输出目录')
    parser.add_argument('--exit-on-failure',default=False,action='store_true',help='失败时退出')
    return parser.parse_args()

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