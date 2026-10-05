from authlib.integrations.starlette_client import OAuth

from app.core.config import GMAIL_SCOPE, Settings

GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"


def build_oauth(settings: Settings) -> OAuth:
    oauth = OAuth()
    oauth.register(
        name="google",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret.get_secret_value(),
        server_metadata_url=GOOGLE_DISCOVERY_URL,
        client_kwargs={"scope": f"openid email profile {GMAIL_SCOPE}"},
    )
    return oauth
