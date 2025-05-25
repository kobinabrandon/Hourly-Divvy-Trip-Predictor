FROM python:3.12.5-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends pip gcc make python3-dev 
RUN pip install uv 

WORKDIR /app
COPY . /app/

RUN uv venv
RUN pip install -e "."
EXPOSE 8501

ENV EMAIL=email 
ENV COMET_API_KEY=some_code
ENV COMET_PROJECT_NAME=some_name
ENV COMET_WORKSPACE=some_other_name
ENV HOPSWORKS_API_KEY=some_other_code
ENV HOPSWORKS_PROJECT_NAME=who_knows

# Prevents the streamlit app from asking for an email address
RUN mkdir -p ~/.streamlit/
RUN echo "[general]"  > ~/.streamlit/credentials.toml
RUN echo "email = \"\""  >> ~/.streamlit/credentials.toml

RUN touch /var/log/cron.log /var/log/supervisord.log
ENTRYPOINT ["uv", "run", "streamlit", "run", "src/inference_pipeline/frontend/main.py", "--server.port", "8501"]

