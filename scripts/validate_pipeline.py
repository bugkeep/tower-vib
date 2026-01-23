import argparse
from pathlib import Path
import logging
from sanity_check import run_checks_by_rules
def setup_logging():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)
def validate_pipeline(results_dir,logger):
    all_results=run_checks_by_rules(results_dir,logger)
    return all_results
def generate_validation_report(output_dir,results):
    """
    生成数据流验收报告
    :param output_path:
    :param results:
    :return:
    """
    output_path=Path(output_dir)
    output_path.mkdir(parents=True,exist_ok=True)
    with open(output_path/"validation_report.txt","w",encoding="utf-8") as f:
        f.write(f"#Tower-Vib 数据流验收报告\n")
        valid_count=sum(1 for r in results if r["is_valid"])
        f.write(f"**总项数**: {len(results)} | **通过**: {valid_count} | **失败**: {len(results)-valid_count}\n\n")
        f.write("----\n")
        for items in results:
            k=items["name"]
            v=items["is_valid"]
            icon="✅" if v else "❌"
            f.write(f"\n### {icon} {k}\n")

def parse_args():
    parser=argparse.ArgumentParser(description="Tower-Vib 数据流验收工具")
    parser.add_argument("--results_dir",type=str,default="results")
    parser.add_argument("--output_dir",type=str,default="tower-vib")
    args=parser.parse_args()
    return args
def main():
    args=parse_args()
    logger=setup_logging()
    results=validate_pipeline(args.results_dir,logger)
    generate_validation_report(args.output_dir,results)
    logger.info(f"数据流验收完成")
if __name__ == '__main__':
    main()