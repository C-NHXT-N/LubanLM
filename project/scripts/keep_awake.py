"""只在指定控制器存活期间阻止自动休眠，不更改永久电源计划。"""
import argparse
import ctypes
import time
import psutil

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pid',type=int,required=True)
    args=p.parse_args()
    process=psutil.Process(args.pid)
    command=process.cmdline()
    controllers=('scripts.train_chat_v2','scripts.improve_chat_v2','scripts.run_chat_corrections','scripts.continue_base_correction','scripts.run_general_2b','scripts.run_general_10b')
    if '-m' not in command or command[command.index('-m')+1] not in controllers:
        raise ValueError('PID is not a recognized training controller')
    started=process.create_time()
    kernel=ctypes.windll.kernel32
    if not kernel.SetThreadExecutionState(0x80000000|0x00000001):
        raise OSError('could not request temporary system-awake state')
    try:
        while psutil.pid_exists(args.pid):
            current=psutil.Process(args.pid)
            if current.create_time()!=started:
                break
            time.sleep(15)
    except psutil.NoSuchProcess:
        pass
    finally:
        kernel.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
    main()
