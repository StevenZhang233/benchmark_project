import benchmark_utils as bm

if __name__ == "__main__":
    bm.run_benchmark()
    bm.analyze_best_concurrency()
    bm.plot_results()
    bm.save_to_csv()
    print("🎯 Benchmark完成！")
    print("===================================")
    print("🎉 感谢使用！")