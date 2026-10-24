"""Temporary loopback CONNECT bridge for installers, using PhysicalMirror sockets."""
from __future__ import annotations
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import select
import socket
import threading

from scripts.direct_mirror import PhysicalMirror

INSTALL_HOSTS = frozenset({
    'dns.alidns.com', 'mirrors.tuna.tsinghua.edu.cn',
    'pypi.tuna.tsinghua.edu.cn', 'mirrors.nju.edu.cn', 'hf-mirror.com',
})


class PhysicalProxy:
    def __init__(self, mirror: PhysicalMirror):
        self.mirror = mirror
        self.stopping = threading.Event()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *args):
                pass

            def do_CONNECT(self):
                upstream = None
                connected = False
                try:
                    host, port = self.path.rsplit(':', 1)
                    if port != '443' or host not in INSTALL_HOSTS:
                        owner.mirror.record(event='proxy_rejected', host=host, port=port)
                        self.send_error(403, 'Only approved HTTPS mirrors are allowed')
                        return
                    upstream = owner.mirror.socket_connection(host, owner.mirror.resolve(host))
                    self.send_response(200, 'Connection established')
                    self.end_headers()
                    self.wfile.flush()
                    connected = True
                    peers = (self.connection, upstream)
                    for peer in peers:
                        peer.settimeout(60)
                    while not owner.stopping.is_set():
                        readable, _, _ = select.select(peers, [], [], 1)
                        for peer in readable:
                            chunk = peer.recv(262144)
                            if not chunk:
                                return
                            (upstream if peer is self.connection else self.connection).sendall(chunk)
                except (OSError, ValueError) as exc:
                    owner.mirror.record(event='proxy_error', error=type(exc).__name__, detail=str(exc))
                    if not connected:
                        self.send_error(502, 'Physical connection failed; no fallback')
                finally:
                    self.close_connection = True
                    if upstream is not None:
                        upstream.close()

            def do_GET(self):
                self.send_error(405, 'Use HTTPS CONNECT')

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'
        self.mirror.record(event='installer_proxy_started', endpoint=self.url)
        return self

    def __exit__(self, *args):
        self.stopping.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.mirror.record(event='installer_proxy_stopped')
