# A tiny GPU memory hog for rampgate's integration tests and real-hardware runs.
# No CUDA toolkit: hog.py calls the CUDA driver API (libcuda.so.1) through ctypes;
# the NVIDIA container runtime mounts the driver library (docker run --gpus ...).
FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 NVIDIA_DRIVER_CAPABILITIES=compute,utility
COPY hog.py /hog.py
ENTRYPOINT ["python", "/hog.py"]
