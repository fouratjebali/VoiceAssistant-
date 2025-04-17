FROM python:3.10

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt



COPY certs/rootCA.crt /usr/local/share/ca-certificates/rootCA.crt

RUN chmod 644 /usr/local/share/ca-certificates/rootCA.crt && \
  update-ca-certificates

ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
# -----------------------------------------------------------------

CMD ["python", "-u", "server.py"]