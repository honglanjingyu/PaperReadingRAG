# stop_api.py
"""强制停止API服务"""

import os
import signal
import sys

def stop_service(port=8000):
    """停止占用端口的服务"""
    try:
        # Windows
        if sys.platform == 'win32':
            os.system(f'netstat -ano | findstr :{port}')
            print(f"\n请手动结束占用端口 {port} 的进程")
        # Linux/Mac
        else:
            result = os.popen(f'lsof -ti:{port}').read().strip()
            if result:
                pids = result.split('\n')
                for pid in pids:
                    os.kill(int(pid), signal.SIGTERM)
                    print(f"已停止进程 {pid}")
            else:
                print(f"端口 {port} 没有运行的进程")
    except Exception as e:
        print(f"停止失败: {e}")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    stop_service(port)