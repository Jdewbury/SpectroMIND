import subprocess
import os
import threading
import sys

def stream_output(process, prefix):
    for line in iter(process.stdout.readline, ''):
        sys.stdout.write(f"{prefix}: {line}")
        sys.stdout.flush()

os.environ['NODE_OPTIONS'] = '--openssl-legacy-provider'

backend_path = os.path.join(os.getcwd(), 'backend')
frontend_path = os.path.join(os.getcwd(), 'frontend')

os.chdir(backend_path)
flask_command = ["python", "app.py"]
flask_process = subprocess.Popen(flask_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

os.chdir(frontend_path)
npm_command = ["npm", "start"]
npm_process = subprocess.Popen(npm_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

os.chdir(backend_path)

# Start threads to stream output
flask_thread = threading.Thread(target=stream_output, args=(flask_process, "Backend"))
npm_thread = threading.Thread(target=stream_output, args=(npm_process, "Frontend"))

flask_thread.start()
npm_thread.start()

try:
    flask_process.wait()
    npm_process.wait()
except KeyboardInterrupt:
    print("Stopping processes...")
    flask_process.terminate()
    npm_process.terminate()

flask_thread.join()
npm_thread.join()

print("All processes terminated.")
