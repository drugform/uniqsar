FROM ubuntu:20.04

###
RUN apt-get update && apt-get -y install git curl wget
RUN curl -LO http://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN apt-get -y install libxrender1

###
RUN pip install zerorpc==0.6.3 msgpack_numpy==0.4.8 rdkit==2023.9.5 pymongo==4.6.3 molvs==0.1.1 base58==2.1.1 pyyaml==6.0.1 numpy==1.26.4
RUN pip install flask==3.0.3 flask_cors==4.0.0 waitress==3.0.0 Flask-HTTPAuth==4.8.0
RUN pip install pandas==2.2.3 matplotlib==3.9.2 scipy==1.14.1

###
WORKDIR /app
COPY backend /app
COPY lib /app/drugform/lib
