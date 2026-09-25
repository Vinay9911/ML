# M20 - Network Capacity
FROM python:3.11-slim

WORKDIR /app

# twin_common is vendored into the image by scripts/package_delivery.py, so the build needs
# no access to the rest of the repository.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Offline by design: a container must never reach for the network mid-request.
ENV TWIN_OFFLINE=1
EXPOSE 8020

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://127.0.0.1:8020/health').status_code==200 else 1)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8020"]
