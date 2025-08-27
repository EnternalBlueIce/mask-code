import json
import subprocess
import threading
import time
import psutil

# 定义一个类来收集 CPU 使用率
class CPUMonitor:
    # ... (这部分代码无需修改，保持原样)
    def __init__(self, interval=1, output_file="cpu_usage_log.json"):
        self.interval = interval
        self.output_file = output_file
        self.cpu_usage = []
        self.monitoring = False

    def collect_cpu_usage(self):
        while self.monitoring:
            usage = psutil.cpu_percent(interval=self.interval)
            timestamp = time.time()
            self.cpu_usage.append({"timestamp": timestamp, "cpu_percent": usage})
            # 为了避免和子进程的输出混在一起，可以考虑减少打印频率或完全不打印
            print(f"CPU Usage: {usage}% at {timestamp}")

    def save_to_file(self):
        with open(self.output_file, "w") as f:
            json.dump(self.cpu_usage, f, indent=4)
            print(f"CPU usage data saved to {self.output_file}")

    def start_monitoring(self):
        self.monitoring = True
        monitor_thread = threading.Thread(target=self.collect_cpu_usage)
        monitor_thread.daemon = True
        monitor_thread.start()

    def stop_monitoring(self):
        self.monitoring = False

# 对开启的代码启动CPU监控的函数
def run_xxx_with_cpu_monitor():
    # 创建 CPU 监控实例
    cpu_monitor = CPUMonitor(interval=1, output_file="cpu_usage_log.json")
    cpu_monitor.start_monitoring()  # 启动 CPU 监控

    # ======== 修改的核心部分在这里 ========
    process = None # 先声明变量
    try:
        command = 'python process/test.py'
        working_dir = "../"

        # 使用 Popen 启动子进程，这是非阻塞的
        # 你的主程序会立即继续往下执行
        print("Starting subprocess 'test.py'...")
        process = subprocess.Popen(command, shell=True, cwd=working_dir)

        # 现在主线程是自由的，它会在这里等待子进程执行结束
        process.wait()
        print("Subprocess 'test.py' finished.")

    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running xxx: {e}")
    except FileNotFoundError:
        print(f"Command not found or working directory is incorrect.")
    finally:
        # 无论子进程是成功还是失败，都确保监控能被停止和保存
        # 这是一个更健壮的设计
        if process and process.poll() is None:
            # 如果程序因为某种原因（如用户按Ctrl+C）在这里退出，尝试终止子进程
            process.terminate()
            print("Subprocess terminated.")

        # 在 xxx 执行完后，停止 CPU 使用率监控
        # 稍微等待一下，确保最后一次的CPU数据能被采集到
        time.sleep(cpu_monitor.interval)
        cpu_monitor.stop_monitoring()

        # 保存 CPU 使用率日志
        cpu_monitor.save_to_file()

if __name__ == "__main__":
    run_xxx_with_cpu_monitor()