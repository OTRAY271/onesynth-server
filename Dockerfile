FROM public.ecr.aws/lambda/python:3.11

WORKDIR /code

COPY poetry.lock pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

COPY ./app ./app
ENV PYTHONPATH code/app

CMD [ "app.main.handler" ]