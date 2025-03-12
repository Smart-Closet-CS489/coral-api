FROM debian:buster

RUN apt-get update --allow-releaseinfo-change && \
apt-get install curl gnupg ca-certificates zlib1g-dev libjpeg-dev git apt-utils -y wget

RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

RUN apt-get update --allow-releaseinfo-change && \
    apt-get install python3 python3-pip -y

RUN apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev gcc gfortran libgfortran5 libatlas3-base libatlas-base-dev libopenblas-dev libopenblas-base libblas-dev liblapack-dev cython3 libatlas-base-dev openmpi-bin libopenmpi-dev python3-dev
RUN pip3 install -U wheel mock six

RUN pip3 install numpy==1.19.5
COPY tensorflow-2.5.0-cp37-cp37m-linux_aarch64.whl /tmp/
RUN pip3 install /tmp/tensorflow-2.5.0-cp37-cp37m-linux_aarch64.whl

RUN apt-get install libedgetpu1-legacy-std python3-edgetpu -y
RUN apt-get install python3-pycoral -y

#RUN pip3 install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp37-cp37m-linux_aarch64.whl
RUN pip3 install pillow
RUN pip3 install flask

WORKDIR /app
