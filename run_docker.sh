docker build -t vectors ./docker && docker run -it --rm -v /home/matrix/Documents/disk:/dochost --runtime=nvidia --gpus all vectors
