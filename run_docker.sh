docker build -t vectors ./docker && docker run -it --cap-add PERFMON --rm -v /home/matrix/Documents/disk:/dochost --runtime=nvidia --gpus all vectors
