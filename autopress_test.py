import time
import requests
import concurrent.futures
import matplotlib.pyplot as plt
import threading
import subprocess
import re
import os
import csv

# === 自动创建输出目录 ===
output_dir = "benchmark_output"
os.makedirs(output_dir, exist_ok=True)

# 安全打印，兼容Windows/Linux中文和emoji
def safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', errors='ignore').decode())

# 自动生成超长Prompt
def generate_long_prompt(target_tokens=1000):
    base = "这是关于Transformer原理的总结。"
    repeated = " Transformer是一种基于自注意力机制的模型。" * ((target_tokens * 3) // len(" Transformer是一种基于自注意力机制的模型。"))
    return (base + repeated)[:target_tokens*4]

# 服务器配置
server_url = "http://101.37.180.140:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}
payload = {
    "model": "/data/DeepSeek-R1-Distill-Qwen-7B",
    "messages": [{"role": "user", "content": generate_long_prompt(1000)}],
    "temperature": 0.7
}

concurrency_levels = [1, 2, 5, 10, 15, 20, 30, 50]
benchmark_results = []

vllm_log_file = "/root/vllm.log"

throughputs = []
gpu_memory_records = []
prefix_cache_hit_records = []

benchmark_running = True

def get_gpu_memory_usage():
    try:
        result = subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
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
        except Exception as e:
            print(f"Log monitor error: {e}")
        time.sleep(1)

def send_request(i):
    try:
        session = requests.Session()
        start_time = time.time()
        response = session.post(server_url, headers=headers, json=payload, timeout=180)
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
    except Exception as e:
        print(f"Request {i} failed: {e}")
        return (0, 0, False)

def run_benchmark():
    global benchmark_running

    monitor_thread_gpu = threading.Thread(target=monitor_gpu_memory)
    monitor_thread_prefix = threading.Thread(target=monitor_prefix_cache_hit)
    monitor_thread_gpu.start()
    monitor_thread_prefix.start()

    for concurrent_requests in concurrency_levels:
        safe_print(f"\U0001f535 测试并发数：{concurrent_requests}")

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
        executor.shutdown(wait=True)

        if success_count > 0:
            avg_latency = sum(request_times) / len(request_times)
            avg_throughput = total_tokens / wall_clock_time
            throughputs.append(avg_throughput)
            benchmark_results.append((concurrent_requests, total_tokens, wall_clock_time, avg_latency, avg_throughput))

            safe_print(f"✅ 并发{concurrent_requests}：平均吞吐量 {avg_throughput:.2f} tokens/sec，平均时延 {avg_latency:.2f}秒")
        else:
            throughputs.append(0)
            benchmark_results.append((concurrent_requests, 0, wall_clock_time, 0, 0))
            safe_print(f"❌ 并发{concurrent_requests}：请求失败！")

    benchmark_running = False
    monitor_thread_gpu.join()
    monitor_thread_prefix.join()

def save_to_csv():
    csv_filename = os.path.join(output_dir, "benchmark_full_metrics.csv")
    safe_print(f"📦 保存测试数据到 {csv_filename} ...")

    gpu_time_base = gpu_memory_records[0][0] if gpu_memory_records else 0
    gpu_data = {round(t - gpu_time_base, 2): mem for t, mem in gpu_memory_records}

    prefix_time_base = prefix_cache_hit_records[0][0] if prefix_cache_hit_records else 0
    prefix_data = {round(t - prefix_time_base, 2): hit for t, hit in prefix_cache_hit_records}

    all_times = sorted(set(list(gpu_data.keys()) + list(prefix_data.keys())))

    with open(csv_filename, "w", newline='', encoding='utf-8') as csvfile:
        fieldnames = ["Time(s)", "GPU Memory (MiB)", "Prefix Cache Hit Rate (%)"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for t in all_times:
            writer.writerow({
                "Time(s)": t,
                "GPU Memory (MiB)": gpu_data.get(t, ""),
                "Prefix Cache Hit Rate (%)": prefix_data.get(t, "")
            })

    csv_filename2 = os.path.join(output_dir, "benchmark_concurrency_summary.csv")
    with open(csv_filename2, "w", newline='', encoding='utf-8') as csvfile:
        fieldnames = ["Concurrency Level", "Throughput Tokens/sec"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, c in enumerate(concurrency_levels):
            writer.writerow({
                "Concurrency Level": c,
                "Throughput Tokens/sec": throughputs[i]
            })

    safe_print(f"✅ CSV保存完成！")

def plot_results():
    plt.figure(figsize=(10,6))
    plt.plot(concurrency_levels, throughputs, marker='o', linestyle='-', color='blue')
    plt.xlabel('Concurrent Requests')
    plt.ylabel('Tokens per Second')
    plt.title('Concurrent Requests vs Throughput')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "concurrent_vs_throughput.png"))
    plt.show()

    if gpu_memory_records:
        times, memories = zip(*gpu_memory_records)
        times = [t - times[0] for t in times]
        plt.figure(figsize=(10,6))
        plt.plot(times, memories, linestyle='-', color='green')
        plt.xlabel('Time (s)')
        plt.ylabel('GPU Memory Usage (MB)')
        plt.title('GPU Memory Usage Over Time')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "gpu_memory_over_time.png"))
        plt.show()

    if prefix_cache_hit_records:
        times, hits = zip(*prefix_cache_hit_records)
        times = [t - times[0] for t in times]
        plt.figure(figsize=(10,6))
        plt.plot(times, hits, linestyle='-', color='purple')
        plt.xlabel('Time (s)')
        plt.ylabel('Prefix Cache Hit Rate (%)')
        plt.title('Prefix Cache Hit Rate Over Time')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "prefix_cache_hit_rate.png"))
        plt.show()

def analyze_best_concurrency():
    safe_print("\n========== 📈 并发优化分析 ==========")
    best_entry = max(benchmark_results, key=lambda x: x[4])
    safe_print(f"🚀 最佳吞吐量并发数：{best_entry[0]}，吞吐量={best_entry[4]:.2f} tokens/sec")

if __name__ == "__main__":
    run_benchmark()
    analyze_best_concurrency()
    save_to_csv()
    plot_results()
    safe_print("✅ Benchmark测试完成，结果输出到 benchmark_output/ 目录！")
