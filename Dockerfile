FROM pytorch/pytorch:latest

RUN apt update
# Install curl and wget
RUN apt-get update && apt-get install -y curl && \
    apt-get update && apt-get install -y wget && \
    apt-get install -y git

# Set up the working directory
WORKDIR /app

# Install LibGL used by OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get update && apt-get install -y libglib2.0-0
RUN apt-get update && apt-get install -y libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev libtbb-dev

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    rm get-pip.py    

COPY . .

# Set up Python environment
ENV PATH=/opt/conda/bin:$PATH
RUN pip install --no-cache-dir -r requirements.txt



# Set the DISPLAY environment variable to use the host's X server for GUI applications
ENV DISPLAY=host.docker.internal:0.0