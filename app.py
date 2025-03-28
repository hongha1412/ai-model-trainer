import os
import logging
from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# Register blueprints
from api.openai_compatible import api_bp
app.register_blueprint(api_bp, url_prefix='/api/v1')

# Register OpenAPI documentation blueprint
from api.openapi_spec import openapi_bp
app.register_blueprint(openapi_bp, url_prefix='/api/docs')

# Import routes after app is created to avoid circular imports
from routes import *

# Load default configuration
from models.config import ModelConfig
ModelConfig.load_default_config()

logger.info("AI Server initialized successfully")
