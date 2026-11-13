"""Fail at safe checkpoints; never silently swap a multi-gigabyte Python pool."""
import os
import shutil
from pathlib import Path
import psutil

def check_resources(plan,phase):
    rss=psutil.Process().memory_info().rss
    ceiling=plan.get('sft_rss_bytes_cap',3<<30) if phase=='sft' else plan.get('pretrain_rss_bytes_cap',4<<30)
    if rss>ceiling:raise RuntimeError('RESOURCE_WAIT: worker RSS limit exceeded')
    if psutil.virtual_memory().available<plan.get('memory_free_bytes_floor',1<<30):raise RuntimeError('RESOURCE_WAIT: insufficient available RAM')
    if shutil.disk_usage(Path(plan['data_root'])).free<plan['disk_free_bytes_floor']:raise RuntimeError('RESOURCE_WAIT: disk reserve reached')
    return dict(rss_bytes=rss,available_memory_bytes=psutil.virtual_memory().available)
