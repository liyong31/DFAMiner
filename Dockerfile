FROM ubuntu:latest

# Installing required packages
RUN apt-get update \
	&& DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y python3-pip \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /home/ubuntu

# Copy DFAMiner
COPY --chown=ubuntu . DFAMiner/

WORKDIR /home/ubuntu/DFAMiner

# install required dependencies
RUN pip3 install --break-system-packages -r requirements.txt

USER ubuntu

RUN mkdir /home/ubuntu/inputs /home/ubuntu/outputs
