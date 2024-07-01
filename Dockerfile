# Use an official Python runtime as a parent image
FROM python:3.9.6-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Streamlit uses
EXPOSE 8501

# Run the Streamlit app when the container launches
CMD ["streamlit", "run", "src/app.py"]
