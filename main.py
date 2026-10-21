from fastapi import FastAPI, Depends, Request, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.securty import OAuth2PasswordBearer
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from redis.asyncio import Redis
import uvicorn
import asyncio

from db.postgres import get_postgres_credentials
from db.redis import get_redis_credentials
from utils.authentication import get_authentication_credentials
from graph import build_graph
from Router import routers

try:
    postgres_credentials = get_postgres_credentials()
    PostgresURL = f"postgresql+asyncpg://{postgres_credentials['username']}:{postgres_credentials['password']}@{postgres_credentials['host']}:{postgres_credentials['port']}/{postgres_credentials['database']}"
except Exception as e:
    raise RuntimeError(f"Failed to retrieve PostgreSQL credentials: {e}")

try:
    redis_credentials = get_redis_credentials()
    RedisURL = f"redis://:{redis_credentials['password']}@{redis_credentials['host']}:{redis_credentials['port']}/{redis_credentials['db']}"
    # Expire both stream and metadata after one day to bound retained run data.
    RUN_RETENTION_SECONDS = 24 * 60 * 60
    # Keep stream and metadata keys together under one run-specific namespace.
    RUN_KEY_PREFIX = "specpilot:run:"
except Exception as e:
    raise RuntimeError(f"Failed to retrieve Redis credentials: {e}")

config={}

@asynccontextmanager
async def life(app: FastAPI):
    # Create the shared Redis connection once at startup; request handlers and
    # background generation tasks use this connection through app.state.
    redis_client = Redis.from_url(RedisURL, decode_responses=True)
    app.state.redis = redis_client
    app.state.run_tasks = set()
    try:
        # Check Redis before accepting requests because generation depends on it.
        await redis_client.ping()
        # Load JWT secret key from Google Secret Manager and store it in the config dictionary
        config['JWT_SECRET_KEY'] = get_authentication_credentials().get("JWT_SECRET_KEY")
        config['JWT_ALGORITHM'] = "HS256"

        # Keep the Postgres checkpointer open for the entire serving period.
        # It persists chat state and is shared by requests handled by this app.
        async with AsyncPostgresSaver.from_conn_string(PostgresURL) as saver:
            # Ensure the checkpointer's tables exist, then build the graph with
            # this live saver and make it available to request handlers.
            await saver.setup()
            app.state.graph = build_graph(saver)

            # FastAPI serves requests while execution is paused here.
            yield
    finally:
        config.clear()
        # On shutdown, stop background workers before closing their Redis connection.
        tasks = list(app.state.run_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await redis_client.aclose()

app = FastAPI(lifespan=life)

# @app.middleware("http")
# def rate_limit_middleware(request, call_next):
#     # Implement rate limiting logic here
#     response = call_next(request)
#     return response

app.add_middleware(CORSMiddleware,
    app,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routers.routes,prefix="/api")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)