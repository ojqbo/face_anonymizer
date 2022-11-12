FROM nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04
ENV TZ=Europe/Warsaw
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update
RUN apt-get install -y libffmpeg-nvenc-dev ffmpeg
RUN apt-get install -y curl

RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.10
RUN apt-get install -y python3.10-distutils
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN pip install poetry

WORKDIR /app
COPY ./poetry.lock ./poetry.toml ./pyproject.toml /app/
RUN poetry install --no-root --without dev && rm poetry.lock poetry.toml pyproject.toml
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN groupadd -r user && useradd -r -g user user
RUN chown user:user $(pwd)
COPY ./frontend /app/frontend
COPY ./face_anonymizer /app/face_anonymizer
USER user
EXPOSE 80
