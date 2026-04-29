from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import SettingsConfigDict
from .settings import APISettings, CORSSettings, ProvidersSettings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"


class AppSettings(APISettings, CORSSettings, ProvidersSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="allow"
    )


settings = AppSettings()
