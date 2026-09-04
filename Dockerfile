# NACRE-MLH: consensus spatial cell-type annotation with a resolution certificate.
#
# The image carries R and Python and every annotator dependency, so nothing is installed at run
# time and the container needs no network unless you ask it to extend the hierarchy with an LLM.
#
# Base is the Bioconductor 3.18 image because that release is what pins zellkonverter 1.12,
# SingleR 2.4, scuttle 1.12 and SingleCellExperiment 1.24 together on R 4.3, which is the
# combination the pipeline was validated against.
#
#   docker build -t nacre-mlh .
#   docker run --rm -m 32g \
#     -v /path/to/data:/data:ro -v /path/to/results:/results -v $PWD/runs.csv:/work/runs.csv:ro \
#     nacre-mlh --config /work/runs.csv --out /results

FROM bioconductor/bioconductor_docker:RELEASE_3_18

# image.source is what makes a registry link the published package back to its repository; without
# it a GHCR package appears orphaned, with no README and no route back to the code.
LABEL org.opencontainers.image.title="NACRE-MLH" \
      org.opencontainers.image.description="Consensus cell-type annotation with a resolution \
certificate, for Xenium, MERSCOPE and CosMx." \
      org.opencontainers.image.source="https://github.com/njabuloshong1/NACRE-MLH" \
      org.opencontainers.image.url="https://github.com/njabuloshong1/NACRE-MLH" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    OPENBLAS_NUM_THREADS=4 \
    OMP_NUM_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev python3-venv \
        libhdf5-dev libglpk40 libxml2-dev libcurl4-openssl-dev libssl-dev \
        libfontconfig1-dev libharfbuzz-dev libfribidi-dev \
        libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# R first. It is by far the slowest layer (spacexr and Seurat compile from source, and the
# zellkonverter python environment is provisioned here), and it changes least often. Putting it
# ahead of the Python layer means adding or bumping a Python package does not rebuild any of it.
COPY docker/install_r_packages.R /tmp/install_r_packages.R
RUN Rscript /tmp/install_r_packages.R

# Python. torch is the CUDA 12.4 build so the resolver can use a GPU when one is passed in with
# `docker run --gpus all`. The wheel bundles its own CUDA runtime, so the base image needs no CUDA
# libraries; only the host driver and the NVIDIA Container Toolkit are required. Without --gpus the
# same image still runs, falling back to CPU, so one image covers both cases.
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --index-url https://download.pytorch.org/whl/cu124 torch \
    && python3 -m pip install -r /tmp/requirements.txt

WORKDIR /opt/nacre
COPY nacre_mlh_cp.py run_pipeline.py ./
COPY nacre/      ./nacre/
COPY annotators/ ./annotators/
COPY assets/     ./assets/
COPY pipeline/   ./pipeline/
COPY runs.template.csv README.md ./

# /work is where a run sheet is mounted, /data and /results are the user's volumes.
RUN mkdir -p /work /data /results
VOLUME ["/data", "/results"]
WORKDIR /work

# Fail the build rather than ship an image whose annotators cannot run.
RUN python3 -c "import sys; sys.path.insert(0,'/opt/nacre'); \
from pipeline import resources as r; \
f,m,o = r.check_python(); assert not m, m; \
fr,mr = r.check_r('Rscript'); assert not mr, mr; \
print('environment OK'); print(r.describe(f,'Rscript',fr))"

ENTRYPOINT ["python3", "/opt/nacre/run_pipeline.py"]
CMD ["--help"]
