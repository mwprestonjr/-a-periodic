# hash:sha256:6d0239b34855770e1a46c0e93a9075ae744848b54854aebc5e56a2ca76ad8136
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    allensdk==2.16.2 \
    brain-observatory-utilities==0.4.0 \
    pandas==1.5.3 \
    scikit-learn==1.5.1 \
    seaborn==0.13.2
