import os
import subprocess
import datetime
import time

# ======================= 配置 ========================
VLLM_API_URL = "http://101.37.180.140:8000"   # ✅ 注意这里！只到端口号！
OUTPUT_DIR = "benchmark_output/eval_gsm8k"
TASK_NAME = "gsm8k"
SAVE_NAME = "gsm8k_eval_result"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================= 基本函数 ========================

def run_gsm8k_eval():
    """跑 GSM8K 测试，并计时"""
    print(f"▶️ 开始评测 {TASK_NAME}...")
    start_time = time.time()

    cmd = [
        "python", "-m", "lm_eval",
        "--model", "local-chat-completions",
        "--model_args", f"api_base={VLLM_API_URL},chat_format=openai",
        "--tasks", TASK_NAME,
        "--output_path", os.path.join(OUTPUT_DIR, f"{SAVE_NAME}.json"),
        "--batch_size", "1",
        "--apply_chat_template",
        "--gen_kwargs", "stop_token=\\n\\n"  # <<< ✅ 用这一行，别再用 --eos_text
    ]
    subprocess.run(cmd, shell=True)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"✅ GSM8K 评测完成，用时 {elapsed:.2f} 秒\n")
    return elapsed

def generate_markdown_report(total_time_sec):
    """生成Markdown格式的评测报告"""
    report_path = os.path.join(OUTPUT_DIR, "gsm8k_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# 📊 GSM8K 评测报告\n\n")
        f.write(f"- 测试时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 总耗时: {total_time_sec:.2f} 秒\n\n")

        json_path = os.path.join(OUTPUT_DIR, f"{SAVE_NAME}.json")
        if os.path.exists(json_path):
            f.write(f"## 📄 测试结果 JSON\n\n")
            f.write("```json\n")
            with open(json_path, "r", encoding="utf-8") as jf:
                f.write(jf.read())
            f.write("\n```\n")
        else:
            f.write("❌ 未找到评测结果文件！\n")

    print(f"✅ Markdown报告已生成: {report_path}")

# ======================= 主流程 ========================
if __name__ == "__main__":
    total_time = run_gsm8k_eval()
    generate_markdown_report(total_time)
    print("🎯 GSM8K测试完成，全部输出已保存到 benchmark_output/eval_gsm8k/")
