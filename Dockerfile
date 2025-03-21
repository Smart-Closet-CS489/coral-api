FROM coral-api-base-image
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
COPY nginx.conf /etc/nginx/nginx.conf
COPY api /app/api
ENV DOCKER_MODEL_DIR=/app/models
WORKDIR /app
EXPOSE 8000

CMD gunicorn --timeout 200 --bind 0.0.0.0:5001 api.model_api.model:app & \
    gunicorn --timeout 200 --bind 0.0.0.0:5002 api.inference_api.inference:app & \
    /usr/sbin/nginx -g "daemon off;"
