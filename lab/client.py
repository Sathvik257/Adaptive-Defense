"""Bounded traffic only against the fixed Compose target; no target argument."""
import http.client
import json
import socket
import time
from pathlib import Path

time.sleep(5)
host='target'
events=[]

def record(family,operation):
    start=time.time()
    try:
        operation()
        outcome='completed'
    except OSError as exc:
        outcome=type(exc).__name__
    events.append({'start':start,'end':time.time(),'family':family,'outcome':outcome})
    time.sleep(.2)

def request(method):
    connection=http.client.HTTPConnection(host,8080,timeout=2)
    try:
        connection.request(method,'/login' if method=='POST' else '/',body='user=lab&password=invalid' if method=='POST' else None)
        connection.getresponse().read()
    finally:
        connection.close()

for _ in range(30):
    record('Normal',lambda:request('GET'))
time.sleep(2)
for port in range(8070,8091):
    def probe(port=port):
        with socket.create_connection((host,port),timeout=.5):
            pass
    record('PortScan',probe)
time.sleep(2)
for _ in range(30):
    record('FailedLoginAttempts',lambda:request('POST'))
Path('/capture/events.json').write_text(json.dumps(events,indent=2))
print('Finished bounded local scenarios; capture stops after 120 seconds.',flush=True)
