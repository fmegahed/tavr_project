RUN pip install scikit-learn<=1.1.2
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
RUN pip install --no-cache-dir -r requirements.txt