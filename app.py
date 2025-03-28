import os
import logging
from flask import Flask, send_from_directory
try:
    from werkzeug.middleware.proxy_fix import ProxyFix
    proxy_fix_available = True
except ImportError:
    proxy_fix_available = False
    print("WARNING: ProxyFix middleware not available. Some functionality may be limited.")

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Check if we're in development or production
is_development = os.environ.get('FLASK_ENV') == 'development'

# Create Flask app
app = Flask(__name__, 
            static_folder='static', 
            static_url_path='/static')
            
if proxy_fix_available:
    app.wsgi_app = ProxyFix(app.wsgi_app)
    
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# Register API blueprints
from api.openai_compatible import api_bp
app.register_blueprint(api_bp, url_prefix='/api/v1')

# Register OpenAPI documentation blueprint
from api.openapi_spec import openapi_bp
app.register_blueprint(openapi_bp)

# Register monitor API blueprint
try:
    from api.monitor import monitor_bp
    app.register_blueprint(monitor_bp)
    logger.info("Monitor API blueprint registered successfully")
except ImportError as e:
    logger.warning(f"Monitor API could not be registered: {str(e)}")

# Try to load the training API blueprint
try:
    from api.training import training_bp
    app.register_blueprint(training_bp)
    logger.info("Training API blueprint registered successfully")
except ImportError as e:
    logger.warning(f"Training API could not be registered due to missing dependencies: {str(e)}")
    logger.warning("Some training-related functionality will be limited. To enable full functionality, install required packages.")
    # Create a minimal training blueprint that returns helpful error messages
    from flask import Blueprint, jsonify
    training_bp = Blueprint("training", __name__)
    
    @training_bp.route("/api/training/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
    def training_missing_dependencies(path):
        return jsonify({
            "error": "Training functionality is not available due to missing dependencies.",
            "message": "Please install required packages (pandas, scikit-learn, transformers) to enable training functionality."
        }), 503
        
    app.register_blueprint(training_bp)

# Import routes after app is created to avoid circular imports
from routes import *

# Load default configuration
from models.config import ModelConfig
ModelConfig.load_default_config()

# Serve Vue App
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_vue_app(path):
    # Skip API and static routes - but return a 404 instead of None
    if path and (path.startswith('api/') or path.startswith('static/')):
        return "Not Found", 404
    try:
        # For production, serve the built Vue app
        if os.path.exists(os.path.join('static', 'vue')):
            if path and os.path.exists(os.path.join('static', 'vue', path)):
                return send_from_directory(os.path.join('static', 'vue'), path)
            else:
                return send_from_directory(os.path.join('static', 'vue'), 'index.html')
        else:
            # If frontend isn't built yet, show a message
            return """
            <html>
                <body style="font-family: sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; flex-direction: column;">
                    <h1>AI Model Server API</h1>
                    <p>The frontend is not built yet. Please build the Vue.js frontend:</p>
                    <pre style="background: #f4f4f4; padding: 20px;">
./build_frontend.sh
                    </pre>
                    <p>API endpoints are available at <a href="/api/models">/api/models</a></p>
                    <p>OpenAPI documentation is available at <a href="/openapi.json">/openapi.json</a></p>
                </body>
            </html>
            """
    except Exception as e:
        logger.exception(f"Error serving Vue app: {e}")
        return """
        <html>
            <body style="font-family: sans-serif; text-align: center; padding: 50px;">
                <h1>Error</h1>
                <p>An error occurred while serving the frontend application.</p>
            </body>
        </html>
        """, 500

logger.info("AI Server initialized successfully")
