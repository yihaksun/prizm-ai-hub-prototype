FROM python:3.11-slim
WORKDIR /app

COPY portal/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Portal도 ML 개발자 Notebook과 동일한 mlops SDK를 그대로 사용한다.
COPY sdk /opt/mlops-sdk
RUN pip install --no-cache-dir -e /opt/mlops-sdk

COPY portal/app.py portal/rule_validator.py ./

EXPOSE 8090
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8090"]
