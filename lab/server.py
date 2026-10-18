"""Disposable local HTTP target; no accounts, commands or persistence."""
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200);self.end_headers();self.wfile.write(b'lab example page\n'*30)
    def do_POST(self):
        self.rfile.read(min(int(self.headers.get('Content-Length','0')),4096))
        self.send_response(401);self.end_headers();self.wfile.write(b'Lab login rejected')

HTTPServer(('0.0.0.0',8080),Handler).serve_forever()
