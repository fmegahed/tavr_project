import os
import subprocess

# Set the environment variable
os.environ["SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL"] = "True"

# Run pip install
subprocess.check_call(["pip", "install", "--no-cache-dir", "-r", "requirements.txt"])