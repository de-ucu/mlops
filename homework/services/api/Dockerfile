FROM nvcr.io/nvidia/tensorflow:21.02-tf2-py3

RUN python -m pip install nvidia-pyindex
RUN python -m pip install tritonclient geventhttpclient flask

ADD ./src/ /app

ENTRYPOINT ["python", "/app/server.py"]
