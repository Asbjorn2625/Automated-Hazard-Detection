FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Install curl, wget, and other dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbb-dev \
    pkg-config


# Add Tesseract repository and install Tesseract
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get update \
    && apt-get install -y tesseract-ocr \
    tesseract-ocr-eng

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Set up the Conda environment
ENV PATH=/opt/conda/bin:$PATH
RUN conda update conda -y && \
    conda create -n myenv python=3.9 -y && \
    echo "source activate myenv" > ~/.bashrc

# Clean up package lists to reduce image size
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and the rest of the project
COPY . .

# Install pip dependencies
RUN /bin/bash -c "source activate myenv && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install torch==1.13.1 -f https://download.pytorch.org/whl/cu118/torch_stable.html && \
    pip install torchvision==0.14.1 torchaudio==0.13.1 && \
    pip install albumentations && \
    pip uninstall -y opencv-python opencv-python-headless && \
    pip install opencv-python==4.5.4.60"

# Set the DISPLAY environment variable to use the host's X server for GUI applications
ENV DISPLAY=host.docker.internal:0.0