# AI Model Server

A Python Flask server providing OpenAI-compatible API endpoints for custom T5 model inference with a modern web interface for model management.

![AI Model Server Logo](frontend/public/favicon.svg)

## Overview

The AI Model Server provides an interface for managing and running inference on custom language models with an API that's compatible with OpenAI's standard. This allows for seamless integration with tools and libraries that already support OpenAI's API format.

Key Features:
- OpenAI-compatible API endpoints for model inference
- Support for multiple model types (T5, GPT-2, with more planned)
- Modern Vue.js-based frontend for model management
- API documentation with interactive interface
- Simple model configuration system

## Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher (for frontend development)
- Git (for version control)

## Installation & Setup

### Backend Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-model-server.git
   cd ai-model-server
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure your environment variables:
   ```bash
   # For development (example values)
   export FLASK_ENV=development
   export SESSION_SECRET=your-secret-key
   
   # On Windows:
   # set FLASK_ENV=development
   # set SESSION_SECRET=your-secret-key
   ```

### Frontend Setup

1. Install the Node.js dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. For development, start the Vue dev server:
   ```bash
   npm run serve
   ```

3. For production, build the frontend:
   ```bash
   # From the project root:
   ./build_frontend.sh
   ```

### Running the Server

1. Start the Flask server:
   ```bash
   python main.py
   ```

2. Access the application:
   - Web interface: http://localhost:5000
   - API documentation: http://localhost:5000/api/openapi
   - Models endpoint: http://localhost:5000/api/models

## Development Guidelines

### Project Structure

```
.
├── api/                  # API endpoints
│   ├── openai_compatible.py   # OpenAI-compatible API implementations
│   └── openapi_spec.py        # OpenAPI specification and documentation
├── config/               # Configuration files
│   └── model_config.json      # Model configuration
├── frontend/             # Vue.js frontend
│   ├── public/               # Static assets
│   └── src/                  # Source files
│       ├── components/         # Vue components
│       ├── router/             # Vue Router configuration
│       ├── store/              # Pinia state management
│       ├── views/              # Vue views/pages
│       ├── App.vue             # Main app component
│       └── main.ts             # Entry point
├── models/               # Model-related code
│   ├── ai_model.py            # Model registry and interface
│   └── config.py              # Configuration management
├── static/               # Static files
│   └── vue/                  # Built frontend (after running build_frontend.sh)
├── templates/            # HTML templates (for legacy views)
├── utils/                # Utility functions
│   ├── model_loader.py        # Model loading functionality
│   └── validation.py          # Request validation
├── app.py                # Flask application setup
├── build_frontend.sh     # Script to build the frontend
├── main.py               # Application entry point
└── routes.py             # Flask routes and endpoints
```

### Backend Development

1. Follow Flask conventions for route handlers.
2. Maintain compatibility with OpenAI API format.
3. Add comprehensive error handling and logging.
4. Use type annotations where possible.
5. Document new endpoints in the OpenAPI specification.

#### Adding a New API Endpoint

1. Create a new route handler in `routes.py` or a new file in `api/`.
2. Update the OpenAPI specification in `api/openapi_spec.py`.
3. Add any necessary validation in `utils/validation.py`.
4. Test the endpoint with curl or an API client.

### Frontend Development

1. Follow Vue 3 Composition API patterns.
2. Use TypeScript for type safety.
3. Use Pinia for state management.
4. Maintain consistent UI styling according to the design system.
5. Ensure responsiveness for mobile devices.

#### Adding a New View

1. Create a Vue component in `frontend/src/views/`.
2. Add the route in `frontend/src/router/index.ts`.
3. Include necessary state management in Pinia stores.
4. Update any navigation components to include the new view.

### API Changes

When modifying the API:
1. Maintain backwards compatibility when possible.
2. Update the OpenAPI specification.
3. Update frontend API calls.
4. Add version information for breaking changes.

### Testing

1. Write unit tests for backend functionality.
2. Test API endpoints with curl or Postman.
3. Test the frontend on multiple browsers.
4. Verify mobile responsiveness.

## Adding a New Model Type

To add support for a new model architecture (beyond T5 and GPT-2):

1. Implement a new model wrapper class in `utils/model_loader.py`.
2. Update the `load_model_from_path` function to handle the new model type.
3. Add appropriate handling in inference endpoints.
4. Update the frontend model type selector in `frontend/src/views/ModelConfig.vue`.

## Roadmap

### Phase 1: Core Functionality (Complete)
- ✅ Basic Flask server setup
- ✅ Model registry and management
- ✅ Initial OpenAI-compatible API endpoints
- ✅ Basic web interface
- ✅ OpenAPI documentation

### Phase 2: Frontend Modernization (Current)
- ✅ Vue.js frontend with TypeScript
- ✅ Modern UI design
- ✅ Interactive model testing interface
- ✅ Responsive design for mobile devices
- ✅ API documentation browser

### Phase 3: Enhanced Model Support (Next)
- 🔲 Support for more model architectures:
  - 🔲 BERT and RoBERTa models
  - 🔲 GPT-J and BLOOM models
  - 🔲 LLaMA and Mistral models
- 🔲 Custom model training interface
- 🔲 Model performance metrics
- 🔲 Batch inference support
- 🔲 File upload for model artifacts

### Phase 4: Advanced Features
- 🔲 User authentication and multi-user support
- 🔲 Role-based access control
- 🔲 API key management
- 🔲 Usage tracking and quotas
- 🔲 Model versioning
- 🔲 A/B testing for models

### Phase 5: Enterprise Features
- 🔲 Horizontal scaling and load balancing
- 🔲 Containerization with Docker
- 🔲 Kubernetes deployment configuration
- 🔲 Monitoring and alerting
- 🔲 Audit logging
- 🔲 Backup and restore functionality

### Phase 6: Advanced ML Features
- 🔲 Model fine-tuning API
- 🔲 Dataset management
- 🔲 Automated model evaluation
- 🔲 Hyperparameter optimization
- 🔲 Model distillation
- 🔲 Adapter-based fine-tuning

## API Compatibility

The API is designed to be compatible with OpenAI's API format. Supported endpoints:

- `/v1/models` - List available models
- `/v1/models/{model_id}` - Get model information
- `/v1/completions` - Generate text completions
- `/v1/chat/completions` - Generate chat completions

For specific request/response formats, refer to the OpenAPI documentation.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Submit a pull request

## License

[MIT License](LICENSE)

## Acknowledgements

- OpenAI API design for inspiration
- Hugging Face for transformer models
- Vue.js team for the frontend framework
- Flask team for the backend framework