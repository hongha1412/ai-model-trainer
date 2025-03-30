from app import app
import os

if __name__ == "__main__":
    # Use port 5000 to avoid conflicts
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
