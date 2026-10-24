"""WSL loopback-to-Windows loopback relay over interop pipes, without firewall changes.

Windows --stdio connects only to a local PhysicalProxy. Linux serves only loopback,
runs the requested installer, and closes its listener afterwards.
"""
import argparse
import os
import socket
import socketserver
import subprocess
import sys
import threading

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stdio',action='store_true');p.add_argument('--port',type=int,required=True)
    p.add_argument('--windows-script');p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args()
    if a.stdio:
        if os.name!='nt':raise RuntimeError('stdio helper requires Windows')
        import msvcrt
        msvcrt.setmode(sys.stdin.fileno(),os.O_BINARY);msvcrt.setmode(sys.stdout.fileno(),os.O_BINARY)
        s=socket.create_connection(('127.0.0.1',a.port),timeout=20);s.settimeout(None)
        def send():
            try:
                while data:=os.read(sys.stdin.fileno(),262144):s.sendall(data)
            except OSError:pass
            finally:
                try:s.shutdown(socket.SHUT_WR)
                except OSError:pass
        threading.Thread(target=send,daemon=True).start()
        try:
            while data:=s.recv(262144):
                sys.stdout.buffer.write(data);sys.stdout.buffer.flush()
        finally:s.close()
        return
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            worker=subprocess.Popen(['/mnt/c/Tools/miniconda3/python.exe','-B',a.windows_script,'--stdio','--port',str(a.port)],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
            def send():
                try:
                    while data:=self.request.recv(262144):
                        worker.stdin.write(data);worker.stdin.flush()
                except (OSError,BrokenPipeError):pass
                finally:
                    try:worker.stdin.close()
                    except OSError:pass
            threading.Thread(target=send,daemon=True).start()
            try:
                while data:=os.read(worker.stdout.fileno(),262144):self.request.sendall(data)
            except OSError:pass
            finally:
                if worker.poll() is None:worker.terminate()
                worker.wait()
    class Server(socketserver.ThreadingTCPServer):
        daemon_threads=True
    with Server(('127.0.0.1',0),Handler) as server:
        threading.Thread(target=server.serve_forever,daemon=True).start()
        env=dict(os.environ)
        for key in list(env):
            if key.lower() in {'http_proxy','https_proxy','all_proxy','no_proxy'}:del env[key]
        url=f'http://127.0.0.1:{server.server_address[1]}'
        env.update(http_proxy=url,https_proxy=url,HTTP_PROXY=url,HTTPS_PROXY=url,NO_PROXY='localhost,127.0.0.1')
        cmd=a.command[1:] if a.command[:1]==['--'] else a.command
        try:status=subprocess.call(cmd,env=env)
        finally:server.shutdown()
    raise SystemExit(status)

if __name__=='__main__':main()
