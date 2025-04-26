# benchmark_utils.py

import time
import requests
import concurrent.futures
import matplotlib.pyplot as plt
import threading
import subprocess
import re
import os
import csv

# ==================== 配置 ====================
server_url = "http://101.37.180.140:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}
payload = {
    "model": "/data/DeepSeek-R1-Distill-Qwen-7B",
    "messages": [{"role": "user", "content": "请简要总结一下Transformer模型的原理。"}],
    "temperature": 0.7
}
concurrency_levels = [1, 2, 5, 10, 15, 20]
vllm_log_file = "/root/vllm.log"  # 日志路径，需要修改成实际环境

# ==================== 全局变量 ====================
throughputs = []
benchmark_results = []
gpu_memory_records = []
prefix_cache_hit_records = []
benchmark_running = True

# === 输出目录 ===
RESULT_DIR = os.path.join(os.getcwd(), "result")
os.makedirs(RESULT_DIR, exist_ok=True)

# ==================== 核心函数 ====================

def _save_plot(fig, filename: str):
    """Convenience helper to save a matplotlib figure in RESULT_DIR."""
    full_path = os.path.join(RESULT_DIR, filename)
    fig.savefig(full_path, dpi=150, bbox_inches="tight")
    print(f"📊 Plot saved to {full_path}")

def get_gpu_memory_usage():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        )
        mem = int(result.decode("utf-8").strip().split("\n")[0])
        return mem
    except Exception:
        return -1


def monitor_gpu_memory():
    while benchmark_running:
        mem = get_gpu_memory_usage()
        timestamp = time.time()
        gpu_memory_records.append((timestamp, mem))
        time.sleep(1)


def monitor_prefix_cache_hit():
    last_position = 0
    pattern = re.compile(r"Prefix cache hit rate: ([0-9.]+)%")
    while benchmark_running:
        try:
            if os.path.exists(vllm_log_file):
                with open(vllm_log_file, "r") as f:
                    f.seek(last_position)
                    lines = f.readlines()
                    last_position = f.tell()

                for line in lines:
                    match = pattern.search(line)
                    if match:
                        hit_rate = float(match.group(1))
                        prefix_cache_hit_records.append((time.time(), hit_rate))
        except Exception:
            pass
        time.sleep(1)


def send_request(i):
    try:
        session = requests.Session()
        start_time = time.time()
        response = session.post(server_url, headers=headers, json=payload, timeout=120)
        end_time = time.time()
        session.close()

        if response.status_code == 200:
            result = response.json()
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            tokens = prompt_tokens + completion_tokens
            request_time = end_time - start_time
            return (tokens, request_time, True)
        else:
            return (0, 0, False)
    except Exception:
        return (0, 0, False)


def run_benchmark():
    global benchmark_running

    monitor_thread_gpu = threading.Thread(target=monitor_gpu_memory)
    monitor_thread_prefix = threading.Thread(target=monitor_prefix_cache_hit)
    monitor_thread_gpu.start()
    monitor_thread_prefix.start()

    for concurrent_requests in concurrency_levels:
        print(f"\n🔵 测试并发数：{concurrent_requests}")
        total_tokens = 0
        request_times = []
        success_count = 0

        start_benchmark = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [executor.submit(send_request, i) for i in range(concurrent_requests)]
            for future in concurrent.futures.as_completed(futures):
                tokens, request_time, success = future.result()
                if success:
                    success_count += 1
                    total_tokens += tokens
                    request_times.append(request_time)
        end_benchmark = time.time()
        wall_clock_time = end_benchmark - start_benchmark

        if success_count > 0:
            avg_latency = sum(request_times) / len(request_times)
            avg_throughput = total_tokens / wall_clock_time
            throughputs.append(avg_throughput)
            benchmark_results.append((concurrent_requests, total_tokens, wall_clock_time, avg_latency, avg_throughput))
            print(f"并发{concurrent_requests}：平均吞吐量 {avg_throughput:.2f} tokens/sec，平均时延 {avg_latency:.2f}秒")
        else:
            throughputs.append(0)
            benchmark_results.append((concurrent_requests, 0, wall_clock_time, 0, 0))
            print(f"并发{concurrent_requests}：请求失败！")

    benchmark_running = False
    monitor_thread_gpu.join()
    monitor_thread_prefix.join()


def analyze_best_concurrency():
    print("\n========== 📈 并发优化分析 ==========\n")

    if not benchmark_results:
        print("没有采集到有效数据")
        return

    best_throughput_entry = max(benchmark_results, key=lambda x: x[4])
    best_latency_entry = min(benchmark_results, key=lambda x: x[3])

    print(f"最佳吞吐量并发数：{best_throughput_entry[0]}，吞吐量={best_throughput_entry[4]:.2f} tokens/sec")
    print(f"最低平均时延并发数：{best_latency_entry[0]}，平均时延={best_latency_entry[3]:.2f} 秒")

    decline_points = []
    for i in range(1, len(benchmark_results)):
        prev_tput = benchmark_results[i-1][4]
        curr_tput = benchmark_results[i][4]
        if curr_tput < prev_tput * 0.95:
            decline_points.append((benchmark_results[i][0], prev_tput, curr_tput))

    if decline_points:
        first_decline = decline_points[0]
        print(f"建议并发不要超过 {first_decline[0]}，因为吞吐量在 {first_decline[0]} 时开始明显下降")
    else:
        print("当前测试范围内并发扩展性良好，无明显吞吐下降")

    print("===================================")


def plot_results():
    # 并发 vs 吞吐量
    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(concurrency_levels, throughputs, marker="o", linestyle="-", linewidth=2)
    plt.xlabel("Concurrent Requests")
    plt.ylabel("Tokens per Second")
    plt.title("Concurrent Requests vs Throughput")
    plt.grid(True)
    _save_plot(fig1, "concurrent_vs_throughput.png")
    plt.close(fig1)

    # GPU 内存
    if gpu_memory_records:
        times, memories = zip(*gpu_memory_records)
        times = [t - times[0] for t in times]

        fig2 = plt.figure(figsize=(10, 6))
        plt.plot(times, memories, linestyle="-", label="GPU Memory (MB)")
        plt.xlabel("Time (s)")
        plt.ylabel("GPU Memory Usage (MB)")
        plt.title("GPU Memory Usage Over Time")
        plt.grid(True)
        _save_plot(fig2, "gpu_memory_over_time.png")
        plt.close(fig2)

    # Prefix cache 命中率
    if prefix_cache_hit_records:
        times, hits = zip(*prefix_cache_hit_records)
        times = [t - times[0] for t in times]

        fig3 = plt.figure(figsize=(10, 6))
        plt.plot(times, hits, linestyle="-", label="Prefix Cache Hit Rate (%)")
        plt.xlabel("Time (s)")
        plt.ylabel("Prefix Cache Hit Rate (%)")
        plt.title("Prefix Cache Hit Rate Over Time")
        plt.grid(True)
        _save_plot(fig3, "prefix_cache_hit_rate.png")
        plt.close(fig3)


def save_to_csv():
    print("正在保存CSV...")

    gpu_time_base = gpu_memory_records[0][0] if gpu_memory_records else 0
    gpu_data = {round(t - gpu_time_base, 2): mem for t, mem in gpu_memory_records}

    prefix_time_base = prefix_cache_hit_records[0][0] if prefix_cache_hit_records else 0
    prefix_data = {round(t - prefix_time_base, 2): hit for t, hit in prefix_cache_hit_records}

    all_times = sorted(set(list(gpu_data.keys()) + list(prefix_data.keys())))

    # 完整指标
    csv_full_path = os.path.join(RESULT_DIR, "benchmark_full_metrics.csv")
    with open(csv_full_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["Time(s)", "GPU Memory (MiB)", "Prefix Cache Hit Rate (%)"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for t in all_times:
            writer.writerow(
                {
                    "Time(s)": t,
                    "GPU Memory (MiB)": gpu_data.get(t, ""),
                    "Prefix Cache Hit Rate (%)": prefix_data.get(t, ""),
                }
            )
    print(f"保存 {csv_full_path}")

    # 并发汇总
    csv_summary_path = os.path.join(RESULT_DIR, "benchmark_concurrency_summary.csv")
    with open(csv_summary_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["Concurrency Level", "Throughput Tokens/sec"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, c in enumerate(concurrency_levels):
            writer.writerow(
                {"Concurrency Level": c, "Throughput Tokens/sec": throughputs[i]}
            )
    print(f"保存 {csv_summary_path}")

    print("正在保存图表...")
    plot_results()
    print("图表保存成功")



if __name__ == "__main__":
    run_benchmark()
    analyze_best_concurrency()
    save_to_csv()
    print("🎯 Benchmark完成！")
    print("===================================")
    print("🎉 感谢使用！")