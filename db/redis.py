from google.cloud import secretmanager
import json

def get_redis_credentials():
    client = secretmanager.SecretManagerServiceClient()
    secret_name = "projects/YOUR_PROJECT_ID/secrets/YOUR_SECRET_NAME/versions/latest"
    response = client.access_secret_version(request={"name": secret_name})
    secret_payload = response.payload.data.decode("UTF-8")
    return json.loads(secret_payload)
