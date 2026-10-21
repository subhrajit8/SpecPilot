from google.cloud import secretmanager
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from main import config
import json
import jwt

def get_authentication_credentials():
    client = secretmanager.SecretManagerServiceClient()
    secret_name = "projects/YOUR_PROJECT_ID/secrets/YOUR_SECRET_NAME/versions/latest"
    response = client.access_secret_version(request={"name": secret_name})  
    secret_payload = response.payload.data.decode("UTF-8")
    return json.loads(secret_payload)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
def get_current_user(token: str = Depends(oauth2_scheme)):
    secret_key = config.get("JWT_SECRET_KEY")
    algorithm = config.get("JWT_ALGORITHM")
    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise ValueError("User ID not found in token")
        return user_id
    except jwt.PyJWTError:
        raise ValueError("Invalid token")
