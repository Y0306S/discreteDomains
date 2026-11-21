"""Provision the same unprivileged PostgreSQL UID used by the existing fleet."""

import ctypes
import grp
import json
import pwd
import socket
import subprocess
from pathlib import Path

if socket.gethostname() != 'cs-fa498-8793f-server':
    raise RuntimeError('Provisioning belongs on LLMOptimizer4')
try:
    grp.getgrgid(10000)
except KeyError:
    subprocess.run(['groupadd', '-g', '10000', 'pgimdb'], check=True)
try:
    pwd.getpwuid(10000)
except KeyError:
    subprocess.run(['useradd', '-M', '-u', '10000', '-g', '10000',
                    '-d', '/nonexistent', '-s', '/usr/sbin/nologin', 'pgimdb'], check=True)

acl = ctypes.CDLL('libacl.so.1', use_errno=True)
acl.acl_get_file.argtypes = [ctypes.c_char_p, ctypes.c_int]
acl.acl_get_file.restype = ctypes.c_void_p
acl.acl_to_text.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
acl.acl_to_text.restype = ctypes.c_void_p
acl.acl_from_text.argtypes = [ctypes.c_char_p]
acl.acl_from_text.restype = ctypes.c_void_p
acl.acl_set_file.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p]
acl.acl_free.argtypes = [ctypes.c_void_p]
existing = acl.acl_get_file(b'/root', 0x8000)
if not existing:
    raise OSError(ctypes.get_errno(), 'Cannot read /root access ACL')
raw = acl.acl_to_text(existing, None)
before = ctypes.string_at(raw).decode()
acl.acl_free(raw)
acl.acl_free(existing)
lines = set(before.strip().splitlines())
if lines == {'user::rwx', 'group::---', 'other::---'}:
    replacement = acl.acl_from_text(b'user::rwx,user:10000:--x,group::---,mask::--x,other::---')
    if not replacement or acl.acl_set_file(b'/root', 0x8000, replacement):
        raise OSError(ctypes.get_errno(), 'Cannot grant PostgreSQL directory traversal')
    acl.acl_free(replacement)
elif not any(line.startswith(('user:10000:--x', 'user:pgimdb:--x')) for line in lines):
    raise RuntimeError('Unexpected existing ACL; preserve it for review')
subprocess.run(['setpriv', '--reuid=10000', '--regid=10000', '--clear-groups',
    '/root/data/cys/envs/pg16/bin/pg_ctl', '--version'], check=True)
print(json.dumps({'postgres_uid': 10000, 'root_access_granted': 'traversal only',
                  'previous_root_acl': before}))
