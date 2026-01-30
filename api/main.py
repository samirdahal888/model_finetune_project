from api.factory import create_app
import uvicorn
from config.config import Config

app = create_app()

if __name__ == "__main__":
    uvicorn.run("api.main:app", host=Config.API_HOST, port=Config.API_PORT, reload=True)
