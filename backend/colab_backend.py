import os
import threading
import sys
import time
import logging
import subprocess

def run_spectromind_backend(authtoken, pull_branch=None, port=5000):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    class SuppressOutput:
        def __enter__(self):
            self._stdout = sys.stdout
            self._stderr = sys.stderr
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = self._stdout
            sys.stderr = self._stderr

    os.chdir('/content/drive/MyDrive/SpectroMIND/backend/')

    if pull_branch:
        with SuppressOutput():
                os.system(f'git pull origin {pull_branch}')

    with SuppressOutput():
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "flask", "flask_cors", "numpy", "torch", "scikit-learn",
                               "flask-ngrok", "pyngrok"])

    with SuppressOutput():
        os.system(f'ngrok authtoken {authtoken}')

    from app import app
    from pyngrok import ngrok

    app.logger.setLevel(logging.INFO)

    def run_app():
        app.logger.info("Starting Flask app...")
        app.run(host='0.0.0.0', port=port)
        app.logger.info("Flask app stopped.")

    def monitor_app():
        while True:
            time.sleep(10)
            if not any(thread.name == 'FlaskThread' for thread in threading.enumerate()):
                logging.error("Flask app is not running. Restarting...")
                flask_thread = threading.Thread(target=run_app, name='FlaskThread', daemon=True)
                flask_thread.start()
            else:
                logging.info("Flask app is running.")

    flask_thread = threading.Thread(target=run_app, name='FlaskThread', daemon=True)
    flask_thread.start()

    monitor_thread = threading.Thread(target=monitor_app, name='MonitorThread', daemon=True)
    monitor_thread.start()

    with SuppressOutput():
        public_url = ngrok.connect(port)

    print(f"Public URL: {public_url}")

    while True:
        time.sleep(1)