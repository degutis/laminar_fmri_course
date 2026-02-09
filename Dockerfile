FROM renkulab/renku:latest

USER root

# System dependencies for building C++ tools like LAYNII
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Python environment (conda) ----
# If your base image already has conda/mamba, use it; otherwise install miniconda here.
# Many Renku images are already prepared for common workflows.

COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -a -y

# Make the env the default
ENV PATH=/opt/conda/envs/layerfmri-course/bin:$PATH

# ---- Install ENIGMA Toolbox from GitHub (recommended) ----
# ENIGMA Toolbox docs recommend GitHub download and list dependencies.  [oai_citation:5‡enigma-toolbox.readthedocs.io](https://enigma-toolbox.readthedocs.io/en/latest/pages/01.install/?utm_source=chatgpt.com)
RUN pip install --no-cache-dir git+https://github.com/MICA-MNI/ENIGMA.git

# ---- Build LAYNII ----
# LAYNII includes straightforward compile instructions; appendix shows direct c++ compile lines.  [oai_citation:6‡GitHub](https://github.com/layerfMRI/LAYNII/blob/master/README_APPENDIX.md?utm_source=chatgpt.com)
RUN git clone --depth 1 https://github.com/layerfMRI/LAYNII.git /opt/LAYNII && \
    cd /opt/LAYNII && \
    make all && \
    ln -s /opt/LAYNII/bin/* /usr/local/bin/

USER ${NB_USER}