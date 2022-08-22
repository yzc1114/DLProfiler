FROM nvcr.io/nvidia/pytorch:22.07-py3

WORKDIR /code

COPY . .

RUN mkdir /data-dir && pip install -r requirements.txt

ENTRYPOINT ["python"]
