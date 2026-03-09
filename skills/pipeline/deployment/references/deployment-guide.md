# Deployment Guide

## Local Deployment (default)
The API runs on your local machine using uvicorn.
- Accessible at: http://localhost:8000
- Only accessible from your machine
- Stops when you close the terminal
- Good for: development, testing, personal use

## Moving to Production

### Option 1 — Docker Container
The Dockerfile in your api/ folder packages everything needed to run the API.

```bash
cd sessions/{session_id}/api
docker build -t my-model-api .
docker run -p 8000:8000 my-model-api
```

Benefits:
- Portable — runs the same anywhere Docker is installed
- Isolated — no dependency conflicts
- Easy to move to a cloud provider

### Option 2 — Cloud Deployment
Once containerised, the Docker image can be deployed to:

| Provider | Service | Plain English |
|---|---|---|
| AWS | ECS / App Runner | Amazon's container hosting |
| Google Cloud | Cloud Run | Google's serverless container platform |
| Azure | Container Apps | Microsoft's container hosting |
| Railway / Render | Web Service | Simple, low-cost cloud hosting |

All of these accept a Docker image and handle the rest.

### Option 3 — Always-On Local Server
To keep the API running even after closing the terminal:

```bash
# Using nohup
nohup uvicorn app:app --host 0.0.0.0 --port 8000 &

# Using screen
screen -S model-api
uvicorn app:app --host 0.0.0.0 --port 8000
# Ctrl+A, D to detach
```

---

## How to Make a Prediction — Plain English

Send a POST request to /predict with your data:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": {"age": 35, "contract_type": "monthly", "tenure": 6}}'
```

Response:
```json
{
  "prediction": 1,
  "probability": 0.73,
  "message": "Prediction complete."
}
```

The /docs endpoint provides an interactive interface for testing without
any code — open it in a browser and fill in the form fields.

---

## Versioning and Updates

When you retrain or update the model:
1. Replace the .pkl files in the api/models/ directory
2. Restart the API
3. Run a health check to confirm it loaded correctly
4. Run a test prediction to verify predictions are as expected

The API code itself does not need to change unless the feature columns change.
If feature columns change, regenerate the API using the Deployment skill.
