FROM dealii/dealii:v9.6.0-jammy

USER root

# Set default password for SSH (optional, for debugging)
RUN echo 'dealii:dealii' | chpasswd

# Expose SSH port (if you intend to run SSH later)
EXPOSE 22

# Install development packages and clean up
RUN apt-get update && \
    apt-get install -y \
    lldb \
    lld \
    doxygen \
    gdb \
    git \
    liblapacke-dev \
    python3-pip \
    python3-venv \
    libboost-all-dev \
    libomp-dev \
    libeigen3-dev && \
    rm -rf /var/lib/apt/lists/*

# Create a virtual environment so pip does not interfere with
# the system Python that deal.II tools may depend on.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies.
# numpy<2  : keeps ABI compatibility with any C-extensions in the base image
#             that were built against NumPy 1.x.
# matplotlib>=3.8 : matplot2tikz 0.5+ requires at least 3.6; 3.8 is safe.
# matplot2tikz   : the active fork (replaces the archived matplotlib2tikz).
RUN pip install --no-cache-dir \
    "numpy<2" \
    "matplotlib>=3.8" \
    matplot2tikz

# Tell matplotlib to use the non-interactive Agg backend (no display needed).
ENV MPLBACKEND=Agg

# Install Spectra (header-only C++ library)
RUN git clone https://github.com/yixuan/spectra.git /tmp/spectra && \
    mkdir -p /usr/local/include/Spectra && \
    cp -r /tmp/spectra/include/Spectra/* /usr/local/include/Spectra/ && \
    rm -rf /tmp/spectra

# Set user and working directory
USER dealii
WORKDIR /home/dealii
