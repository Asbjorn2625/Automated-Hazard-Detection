FROM pytorch/pytorch:latest

# Install curl and wget
RUN apt-get update && apt-get install -y curl && \
    apt-get update && apt-get install -y wget

# Install LibGL used by OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get update && apt-get install -y libglib2.0-0
RUN apt-get update && apt-get install -y libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev libtbb-dev



# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Set up Python environment
ENV PATH=/opt/conda/bin:$PATH
RUN pip install numpy opencv-python matplotlib tqdm scikit-learn tensorflow keras tensorflow-hub pytesseract craft-text-detector

COPY . .
