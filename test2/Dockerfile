# Use the existing base image
FROM coral-usb-base-image
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

# Set the working directory in the container (optional, but a good practice)
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Expose the port Flask will run on
EXPOSE 5000

# Define the command to run your Flask app
CMD ["python3", "api.py"]
