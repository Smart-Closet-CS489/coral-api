FROM coral-api-base-image
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
COPY nginx.conf /etc/nginx/nginx.conf
COPY api /app/api
ENV DOCKER_MODEL_DIR=/app/models
WORKDIR /app
EXPOSE 8000

CMD gunicorn --timeout 3600 --bind 0.0.0.0:5001 api.api:app & \
    gunicorn --timeout 3600 --bind 0.0.0.0:5002 api.api_helper:app_helper & \
    /usr/sbin/nginx -g "daemon off;"
