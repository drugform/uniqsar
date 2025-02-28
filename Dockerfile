FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
RUN apt update && apt -y install build-essential curl

RUN pip install rdkit==2023.9.5 molvs==0.1.1 base58==2.1.1 pyyaml==6.0.1 numpy==1.26.4 matplotlib==3.8.4 scikit-learn==1.4.1.post1 pandas==2.2.1 scipy==1.13.0 permetrics==2.0.0 sigfig==1.3.3

# requirements for Chemformer
RUN apt install -y git
RUN pip install git+https://github.com/MolecularAI/pysmilesutils.git 
RUN pip install deepspeed==0.14.0 pytorch_lightning==1.9.4 torch==1.13.1


# requirements for ESM
RUN pip install fairseq==0.12.2 torch==1.13.1 torchaudio==0.13.1

WORKDIR /app
ENV HOME /app
COPY bin /app/bin
COPY src /app/src
RUN mkdir /app/.{triton,config,cache}
RUN chmod a+rwx -R /app
WORKDIR /app/src
