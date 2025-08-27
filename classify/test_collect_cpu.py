import json
import subprocess
import threading
import time
import psutil
import sys
import queue
import os

# 创建一个线程安全的队列
print_queue = queue.Queue()

def calculate_average_cpu_usage(file_path="cpu_usage_log.json"):
    with open(file_path, "r") as f:
        data = json.load(f)

    if not data:
        print("No CPU usage data found.")
        return

    total_cpu = sum(entry["cpu_percent"] for entry in data)
    average_cpu = total_cpu / (len(data)-1)

    print(f"Number of samples: {len(data)}")
    print(f"Average CPU usage: {average_cpu:.2f}%")

def printer_worker():
    """专职的打印机线程，从队列获取消息并打印"""
    while True:
        message = print_queue.get()
        if message is None:
            break
        print(message)
        print_queue.task_done()

class CPUMonitor:
    def __init__(self, interval=1, output_file="cpu_usage_log.json"):
        self.sample_interval = interval
        self.output_file = output_file
        self.cpu_usage = []
        self.monitoring = False

    def collect_cpu_usage(self):
        psutil.cpu_percent(interval=None)
        while self.monitoring:
            loop_start_time = time.monotonic()
            usage = psutil.cpu_percent(interval=None)
            timestamp = time.time()
            self.cpu_usage.append({"timestamp": timestamp, "cpu_percent": usage})
            print_queue.put(f"[CPU MONITOR]: {usage}%")
            work_duration = time.monotonic() - loop_start_time
            sleep_time = self.sample_interval - work_duration
            if sleep_time > 0:
                time.sleep(sleep_time)

    def save_to_file(self):
        with open(self.output_file, "w") as f:
            json.dump(self.cpu_usage, f, indent=4)
        print(f"\n[INFO] CPU usage data saved to {self.output_file}")

    def start_monitoring(self):
        self.monitoring = True
        monitor_thread = threading.Thread(target=self.collect_cpu_usage, daemon=True)
        monitor_thread.start()

    def stop_monitoring(self):
        self.monitoring = False

def stream_output_to_queue(stream, prefix):
    """将子进程的输出流逐行放入打印队列"""
    try:
        for line in iter(stream.readline, ''):
            print_queue.put(f"{prefix} {line.strip()}")
    finally:
        # 确保流在线程结束时关闭
        stream.close()

def run_xxx_with_cpu_monitor():
    printer_thread = threading.Thread(target=printer_worker, daemon=True)
    printer_thread.start()

    cpu_monitor = CPUMonitor(interval=1, output_file="cpu_usage_log.json")
    cpu_monitor.start_monitoring()

    process = None
    try:
        command = [sys.executable, r"D:\数据集分析\Flash-IDS-main\process\classify\test.py"]
        working_dir = ".."

        # ==================== 关键修复 ====================
        # 1. 复制当前的环境变量
        env = os.environ.copy()
        # 2. 强制子进程的Python I/O使用UTF-8编码
        env['PYTHONIOENCODING'] = 'utf-8'
        # ==================================================

        print_queue.put(f"[INFO] Starting subprocess: {' '.join(command)}")

        process = subprocess.Popen(
            command, cwd=working_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', # 父进程继续使用UTF-8解码
            bufsize=1,
            env=env # <--- 将修改后的环境变量传递给子进程
        )

        stdout_thread = threading.Thread(target=stream_output_to_queue, args=(process.stdout, "[test.py]"), daemon=True)
        stderr_thread = threading.Thread(target=stream_output_to_queue, args=(process.stderr, "[test.py-ERROR]"), daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        print_queue.put("[INFO] Monitoring CPU and subprocess output...")

        process.wait()

        stdout_thread.join()
        stderr_thread.join()

        print_queue.put(f"\n[INFO] Subprocess 'test.py' finished with exit code {process.returncode}.")

    except Exception as e:
        print_queue.put(f"[ERROR] An unexpected error occurred: {e}")
    finally:
        if process and process.poll() is None:
            process.terminate()

        cpu_monitor.stop_monitoring()
        time.sleep(1)

        print_queue.join()
        print_queue.put(None)
        printer_thread.join(timeout=2)

        cpu_monitor.save_to_file()

if __name__ == "__main__":
    run_xxx_with_cpu_monitor()
    calculate_average_cpu_usage()