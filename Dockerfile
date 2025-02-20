#Attempting containerization

FROM Python:3.9
COPY . /app
WORKDIR /app
RUN pip install --no--cache-dir -r requirements.txt
CMD python app.py