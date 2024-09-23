FROM python:3.9.20-slim-bullseye

COPY sources.list /etc/apt/sources.list
RUN apt update -y && apt upgrade -y && DEBIAN_FRONTEND=noninteractive apt install -y build-essential zlib1g-dev make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev libmkl-full-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget nano 

ENTRYPOINT ["bash"]

