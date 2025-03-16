FROM coral-api-base-image
RUN apt-get update && apt-get install -y python3-dev build-essential
# Install the latest version of CMake
RUN wget https://cmake.org/files/v3.24/cmake-3.24.0-linux-aarch64.tar.gz -P /tmp && \
    tar -xzvf /tmp/cmake-3.24.0-linux-aarch64.tar.gz -C /opt && \
    ln -s /opt/cmake-3.24.0-linux-aarch64/bin/cmake /usr/local/bin/cmake && \
    cmake --version
RUN pip3 install dm-tree
RUN pip3 install -q tensorflow-model-optimization==0.5.0
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
WORKDIR /app
COPY . .
EXPOSE 8000
CMD ["python3", "api.py"]
