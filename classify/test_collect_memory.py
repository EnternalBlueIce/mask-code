import json
import subprocess
import threading
import time
import psutil
import sys
import queue
import os


print_queue = queue.Queue()


def calculate_average_memory_usage(file_path="memory_usage_log.json"):
    with open(file_path, "r") as f:
        data = json.load(f)

    if not data:
        print("No memory usage data found.")
        return

    total_memory = sum(entry["memory_used_MB"] for entry in data)
    average_memory = total_memory / len(data)

    print(f"Number of samples: {len(data)}")
    print(f"Average memory usage: {average_memory:.2f} MB")

def printer_worker():
    while True:
        message = print_queue.get()
        if message is None:
            break
        print(message)
        print_queue.task_done()


class MemoryMonitor:
    def __init__(self, interval=1, output_file="memory_usage_log.json"):
        self.sample_interval = interval
        self.output_file = output_file
        self.memory_usage = []
        self.monitoring = False
        self.target_pid = None
        self.process_handle = None

    def set_target_pid(self, pid):
        """主线程调用此方法来设置目标PID，并正式激活监控。"""
        self.target_pid = pid
        try:
            self.process_handle = psutil.Process(pid)
        except psutil.NoSuchProcess:
            print_queue.put(f"[ERROR] Process with PID {pid} does not exist when setting target.")
            self.monitoring = False # 如果进程在设置时就没了，就停止监控

    def collect_memory_usage(self):
        """在独立线程中运行的核心监控方法。"""
        # 阶段1: 等待PID被设置。
        # self.monitoring 在start_monitoring时已设为True，所以这个循环会运行。
        while self.process_handle is None:
            if not self.monitoring: # 如果在等待期间被外部停止，则优雅退出
                return
            time.sleep(0.1)

        # 阶段2: PID已设置，开始真正的监控循环。
        while self.monitoring:
            try:
                loop_start_time = time.monotonic()

                # 获取特定进程的物理内存占用（RSS）
                memory_used = self.process_handle.memory_info().rss / (1024 * 1024)

                timestamp = time.time()
                self.memory_usage.append({"timestamp": timestamp, "memory_used_MB": memory_used})

                print_queue.put(f"[MEMORY MONITOR (PID:{self.target_pid})]: {memory_used:.2f} MB")

                work_duration = time.monotonic() - loop_start_time
                sleep_time = self.sample_interval - work_duration
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except psutil.NoSuchProcess:
                print_queue.put(f"[INFO] Target process {self.target_pid} has ended. Stopping memory monitor.")
                break # 目标进程已结束，正常退出监控循环
            except Exception as e:
                print_queue.put(f"[ERROR] Memory monitor encountered an error: {e}")
                break

        # 无论如何退出循环，都将监控标志设为False
        self.monitoring = False

    def save_to_file(self):
        with open(self.output_file, "w") as f:
            json.dump(self.memory_usage, f, indent=4)
        print(f"\n[INFO] Memory usage data saved to {self.output_file}")

    def start_monitoring(self):
        """
        启动内存监控线程，并设置监控标志为True。
        这是关键的修复！
        """
        # --- 关键修复：必须在这里设置监控标志！ ---
        self.monitoring = True
        # ----------------------------------------
        monitor_thread = threading.Thread(target=self.collect_memory_usage, daemon=True)
        monitor_thread.start()

    def stop_monitoring(self):
        """停止内存监控。"""
        self.monitoring = False

# --- 3. 子进程输出处理模块 (保持不变) ---
def stream_output_to_queue(stream, prefix):
    try:
        for line in iter(stream.readline, ''):
            print_queue.put(f"{prefix} {line.strip()}")
    finally:
        stream.close()

# --- 4. 主执行函数 (保持不变) ---
def run_xxx_with_memory_monitor():
    printer_thread = threading.Thread(target=printer_worker, daemon=True)
    printer_thread.start()

    memory_monitor = MemoryMonitor(interval=1, output_file="memory_usage_log.json")
    # 现在这个调用会正确地启动监控线程并让它处于“待命”状态
    memory_monitor.start_monitoring()

    process = None
    try:
        command = [sys.executable, r"D:\数据集分析\Flash-IDS-main\process\classify\test.py"]
        working_dir = ".."
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'

        print_queue.put(f"[INFO] Starting subprocess: {' '.join(command)}")

        process = subprocess.Popen(
            command, cwd=working_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', bufsize=1, env=env
        )

        pid = process.pid
        print_queue.put(f"[INFO] Subprocess started with PID: {pid}. Attaching memory monitor.")
        # 主线程将PID告知监控线程，激活其数据采集
        memory_monitor.set_target_pid(pid)

        stdout_thread = threading.Thread(target=stream_output_to_queue, args=(process.stdout, "[test.py]"), daemon=True)
        stderr_thread = threading.Thread(target=stream_output_to_queue, args=(process.stderr, "[test.py-ERROR]"), daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        print_queue.put("[INFO] Monitoring memory of subprocess and its output...")

        process.wait()

        stdout_thread.join()
        stderr_thread.join()

        print_queue.put(f"\n[INFO] Subprocess 'test.py' (PID:{pid}) finished with exit code {process.returncode}.")

    except Exception as e:
        print_queue.put(f"[ERROR] An unexpected error occurred: {e}")
    finally:
        if process and process.poll() is None:
            process.terminate()

        memory_monitor.stop_monitoring()
        time.sleep(1)

        print_queue.join()
        print_queue.put(None)
        printer_thread.join(timeout=2)

        memory_monitor.save_to_file()

if __name__ == "__main__":
    run_xxx_with_memory_monitor()
    calculate_average_memory_usage()