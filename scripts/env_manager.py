import argparse
from env_utils import *
from dump_env import dump_env_info, generate_requirements_txt
from pathlib import Path
def main():
    parser=argparse.ArgumentParser(description="tower-vib:统一环境管理脚本")
    subparsers=parser.add_subparsers(dest="command",required= True)

    dummy_parser=subparsers.add_parser("dump",help="记录环境信息")
    dummy_parser.add_argument("--output_dir",type=str,default="tower-vib",help="输出目录")
    dummy_parser.add_argument("--format", type=str, default="both", choices=["json", "txt", "both"])
    dummy_parser.add_argument("--generate_requirements", action="store_true")

    check_parser=subparsers.add_parser("check",help="验证环境")
    check_parser.add_argument("--requirements",type=str,default="tower-vib/requirements.txt")
    check_parser.add_argument("--output_dir",type=str,default="tower-vib",help="输出目录")


    compare_parser=subparsers.add_parser("compare",help="对比环境")
    compare_parser.add_argument("env1",type=str)
    compare_parser.add_argument("env2",type=str)
    compare_parser.add_argument("--output_dir", type=str, default="tower-vib", help="输出目录")

    report_parser=subparsers.add_parser("report",help="生成报告")
    report_parser.add_argument("--output_dir", type=str, default="tower-vib", help="输出目录")
    report_parser.add_argument("--env_info",type=str,dest="env_info",default="tower-vib/env_info.json")
    report_parser.add_argument("--requirements",type=str,default="tower-vib/requirements.txt")
    args=parser.parse_args()
    if args.command=="dump":
        if args.generate_requirements:
            generate_requirements_txt(Path(args.output_dir))
        dump_env_info(Path(args.output_dir), args.format)
    elif args.command=="check":
        check_result=verify_environment(requirements_path=args.requirements)
        generate_check_report(check_result,output_path=args.output_dir)
    elif args.command=="compare":
        diff_report=compare_environments(Path(args.env1),Path(args.env2))
        save_comparison_report(diff_report,args.output_dir)
    elif args.command=="report":
        generate_env_report(args.env_info,args.requirements,args.output_dir)

if __name__ == "__main__":
    main()
