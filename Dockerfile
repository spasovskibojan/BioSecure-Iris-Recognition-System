# Use Python 3.9 as the base image
FROM python:3.9

# Set the working directory to /code
WORKDIR /code

# Copy the requirements file specifically to avoid cache invalidation
COPY ./requirements.txt /code/requirements.txt

# Install dependencies using pip
# Using opencv-python-headless to avoid GUI dependency issues in cloud
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application code
COPY . /code

# Create the uploads directory if it doesn't exist
RUN mkdir -p static/uploads

# Set permissions to ensure the app can write to the uploads directory
# Hugging Face Spaces runs as a non-root user (usually 1000), so 777 ensures write access
RUN chmod -R 777 static/uploads

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Command to run the application using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
