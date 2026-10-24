"""Bounded retry for Windows atomic report replacement during readers' locks."""
import time
from scripts.common import write_json as atomic_write_json

def write_json(path,value):
    for attempt in range(8):
        try:return atomic_write_json(path,value)
        except PermissionError as error:
            if attempt==7 or getattr(error,'winerror',None) not in (None,5,32):raise
            time.sleep(min(.25,.02*2**attempt))

def replace(source,target):
    for attempt in range(8):
        try:return source.replace(target)
        except PermissionError as error:
            if attempt==7 or getattr(error,'winerror',None) not in (None,5,32):raise
            time.sleep(min(.25,.02*2**attempt))
