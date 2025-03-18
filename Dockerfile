FROM coral-api-base-image
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
COPY nginx.conf /etc/nginx/nginx.conf
COPY inference.py /app/inference.py
COPY model.py /app/model.py
WORKDIR /app
EXPOSE 8000
# CMD ["python3", "api.py"]
CMD gunicorn --timeout 200 --bind 0.0.0.0:5001 model:app & \
    gunicorn --timeout 200 --bind 0.0.0.0:5002 inference:app & \
    /usr/sbin/nginx -g "daemon off;"
