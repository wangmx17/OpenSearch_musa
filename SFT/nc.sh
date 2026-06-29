ssh 10.121.33.16 "python3 - <<'PY'
import socket
host='10.121.33.67'
port=34237
s=socket.socket()
s.settimeout(5)
try:
    s.connect((host, port))
    print(f'OK: {host}:{port} reachable')
except Exception as e:
    print(f'FAIL: {host}:{port} not reachable: {e}')
finally:
    s.close()
PY"
