FROM coral-api-base-image
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
WORKDIR /app
COPY . .
EXPOSE 8000
CMD ["python3", "api.py"]
