FROM renku/renkulab-py:3.11-0.25.0

USER root

# Minimal system deps (git is handy; build-essential not needed if you don't compile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create conda env (includes laynii binaries from conda-forge)
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -a -y

# Install ENIGMA into the env explicitly
RUN conda run -n layerfmri-course pip install --no-cache-dir git+https://github.com/MICA-MNI/ENIGMA.git

# Make env default for shells + notebooks
ENV PATH=/opt/conda/envs/layerfmri-course/bin:$PATH

USER ${NB_USER}