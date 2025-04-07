ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
RUN pip install --no-cache-dir -r /tmp/requirements.txt
