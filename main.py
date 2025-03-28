from app import app
import os

if __name__ == "__main__":
    # Use the port from the environment or the Replit configuration (5050)
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=True)
