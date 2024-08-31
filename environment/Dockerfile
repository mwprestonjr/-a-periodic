# hash:sha256:99f27b9f934ac83c16cfb96853aa44d732177692d586d7312052c07e73156442
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    allensdk==2.16.2 \
    brain-observatory-utilities==0.4.0 \
    mne==1.8.0 \
    pandas==1.5.3 \
    scikit-learn==1.5.1 \
    seaborn==0.13.2 \
    specparam==2.0.0rc2
