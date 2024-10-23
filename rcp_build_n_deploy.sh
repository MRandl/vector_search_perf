docker build -t ic-registry.epfl.ch/sacs/randl/vectors_perf ./docker
docker push ic-registry.epfl.ch/sacs/randl/vectors_perf
runai_rcp submit --name vecperf \
    -i ic-registry.epfl.ch/sacs/randl/vectors_perf  \
    --gpu 0 --cpu 32 \
    --memory 450G \
    --pvc runai-sacs-randl-scratch:/mnt/nfs \
    --interactive -- \"sleep infinity\"

