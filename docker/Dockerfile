FROM python:3.9.20-slim-bullseye

COPY sources.list /etc/apt/sources.list
RUN apt update -y && apt upgrade -y && DEBIAN_FRONTEND=noninteractive apt install -y build-essential zlib1g-dev make cmake g++ git nano libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev libmkl-full-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
RUN pip install numpy tqdm 

RUN git clone --recurse-submodules https://github.com/MRandl/vector_search_perf.git vectors

WORKDIR /vectors/DiskANN
RUN mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j 
RUN pip install build && python -m build && pip install dist/diskannpy-0.7.0-cp39-cp39-linux_x86_64.whl
WORKDIR /
ENTRYPOINT ["/bin/bash", "-l", "-c"]

