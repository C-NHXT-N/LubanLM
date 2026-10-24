"""Offline checks for physical-only installation and portable native defaults."""
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

from scripts.direct_mirror import PhysicalMirror
from scripts.physical_proxy import PhysicalProxy


class EnvironmentSetupTests(unittest.TestCase):
    def test_no_matching_physical_interface_stops_before_network(self):
        rows=json.dumps([{'index':3,'name':'WLAN','ip':'192.168.0.19'}])
        with patch('scripts.direct_mirror.subprocess.check_output',return_value=rows), \
             patch('scripts.direct_mirror.socket.socket') as socket:
            with self.assertRaises(RuntimeError):
                PhysicalMirror(interface='Meta')
            with self.assertRaises(RuntimeError):
                PhysicalMirror(interface='WLAN',bind_ip='198.18.0.1')
            socket.assert_not_called()

    def test_host_policy_rejects_official_fallback_before_dns(self):
        rows=json.dumps([{'index':3,'name':'WLAN','ip':'192.168.0.19'}])
        with patch('scripts.direct_mirror.subprocess.check_output',return_value=rows):
            mirror=PhysicalMirror(interface='WLAN',allowed_hosts={'hf-mirror.com','dns.alidns.com'})
        with patch.object(mirror,'connection') as connection:
            with self.assertRaises(ValueError):
                mirror.resolve('huggingface.co')
            connection.assert_not_called()

    def test_fake_ip_dns_answer_is_rejected(self):
        rows=json.dumps([{'index':3,'name':'WLAN','ip':'192.168.0.19'}])
        with patch('scripts.direct_mirror.subprocess.check_output',return_value=rows):
            mirror=PhysicalMirror(interface='WLAN')
        connection=Mock()
        response=connection.getresponse.return_value
        response.status=200
        response.read.return_value=b'{"Answer":[{"type":1,"data":"198.18.0.10"}]}'
        with patch.object(mirror,'connection',return_value=connection):
            with self.assertRaises(ConnectionError):
                mirror.resolve('hf-mirror.com')
        connection.close.assert_called_once()

    def test_loopback_proxy_blocks_other_hosts_and_plain_http(self):
        mirror=Mock()
        with PhysicalProxy(mirror) as proxy:
            port=proxy.server.server_port
            connection=http.client.HTTPConnection('127.0.0.1',port,timeout=5)
            connection.request('CONNECT','download.pytorch.org:443')
            response=connection.getresponse()
            self.assertEqual(response.status,403)
            response.read();connection.close()
            connection=http.client.HTTPConnection('127.0.0.1',port,timeout=5)
            connection.request('GET','http://mirrors.tuna.tsinghua.edu.cn/')
            response=connection.getresponse()
            self.assertEqual(response.status,405)
            response.read();connection.close()
        mirror.resolve.assert_not_called()
        mirror.socket_connection.assert_not_called()

    def test_existing_explicit_paths_and_cuda_selection_are_preserved(self):
        env={**os.environ,'LUBAN_DATA':'C:/explicit-data','LUBAN_RUNS':'C:/explicit-runs',
             'CUDA_VISIBLE_DEVICES':'GPU-explicit','PYTHONDONTWRITEBYTECODE':'1'}
        command=[sys.executable,'-c',"from scripts.common import DATA,RUNS;import os,json;print(json.dumps([str(DATA),str(RUNS),os.environ['CUDA_VISIBLE_DEVICES']]))"]
        actual=json.loads(subprocess.check_output(command,env=env,text=True))
        self.assertEqual(actual,['C:\\explicit-data','C:\\explicit-runs','GPU-explicit'])


if __name__=='__main__':
    unittest.main()
