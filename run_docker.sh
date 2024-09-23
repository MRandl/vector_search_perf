docker build -t nvidia_randl ./docker && docker run -it --cap-add PERFMON --rm -v /home/matrix/Documents/disk:/dochost --runtime=nvidia --gpus all nvidia_randl
