# AI Model Server

A Python Flask server providing OpenAI-compatible API endpoints for custom machine learning model inference, designed to support advanced AI workflows with a modern web interface for model management, configuration, and training.

![AI Model Server Logo](frontend/public/favicon.svg)

## Overview

The AI Model Server provides a flexible platform for managing, training, and running inference on custom language models with an API that's compatible with OpenAI's standard. This allows for seamless integration with tools and libraries that already support OpenAI's API format.

Key Features:
- OpenAI-compatible API endpoints for model inference
- Support for multiple model types with extensible architecture
- Model training capabilities with various learning approaches
- Dataset management for training workflows
- HuggingFace model integration
- Modern Vue.js-based frontend with TypeScript
- API documentation with interactive interface
- Comprehensive model configuration system
- SafeTensor model file support

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
   pip install flask flask-cors gunicorn numpy pandas requests scikit-learn scipy werkzeug sentencepiece
   # Optional: install PyTorch and transformers if you plan to use ML models
   # pip install torch transformers
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

2. For development, start the Vue.js dev server:
   ```bash
   npm run dev
   ```

3. For production, build the frontend:
   ```bash
   # From the project root:
   ./build_frontend.sh
   ```

### Running the Application

#### Development Mode (Two Separate Servers)

1. Start the Flask backend server (port 5000):
   ```bash
   python main.py
   ```

2. In a separate terminal, start the Vue.js dev server (port 3000):
   ```bash
   cd frontend
   npm run dev
   ```

3. Access the application:
   - Frontend interface: http://localhost:3000
   - Backend API (direct): http://localhost:5000/api/
   - OpenAI-compatible endpoints: http://localhost:5000/v1/

#### Production Mode (Single Server)

1. Build the frontend:
   ```bash
   ./build_frontend.sh
   ```

2. Start the Flask server:
   ```bash
   python main.py
   ```

3. Access the application:
   - Web interface: http://localhost:5000
   - API documentation: http://localhost:5000/api/openapi
   - Models endpoint: http://localhost:5000/api/models

## Development Guidelines

### Project Structure

```
.
├── api/                  # API endpoints
│   ├── openai_compatible.py   # OpenAI-compatible API implementations
│   ├── openapi_spec.py        # OpenAPI specification and documentation
│   └── training.py            # Training API endpoints
├── config/               # Configuration files
│   ├── model_config.json      # Model configuration
│   └── training_config.json   # Training configuration
├── frontend/             # Vue.js frontend
│   ├── public/               # Static assets
│   └── src/                  # Source files
│       ├── components/         # Vue components
│       │   └── training/         # Training-specific components
│       ├── router/             # Vue Router configuration
│       ├── store/              # State management
│       │   ├── models.ts         # Models store
│       │   └── training.ts       # Training store
│       ├── views/              # Vue views/pages
│       ├── App.vue             # Main app component
│       └── main.ts             # Entry point
├── models/               # Model-related code
│   ├── downloaded/           # Models downloaded from HuggingFace
│   ├── outputs/              # Model output files
│   ├── trained/              # Models trained within the system
│   ├── ai_model.py            # Model registry and interface
│   └── config.py              # Configuration management
├── static/               # Static files
│   └── vue/                  # Built frontend (after running build_frontend.sh)
├── templates/            # HTML templates (for legacy views)
├── training/             # Training-related code
│   ├── dataset_handler.py     # Handles different dataset formats
│   ├── huggingface.py         # HuggingFace API integration
│   └── trainer.py             # Model training functionality
├── uploads/              # User-uploaded files
│   └── datasets/             # Uploaded datasets for training
├── utils/                # Utility functions
│   ├── dependency_checker.py  # Checks and handles dependencies
│   ├── model_loader.py        # Model loading functionality
│   └── validation.py          # Request validation
├── app.py                # Flask application setup
├── build_frontend.sh     # Script to build the frontend
├── main.py               # Application entry point
├── routes.py             # Flask routes and endpoints
├── start_server.sh       # Script to start the server
└── start_frontend.sh     # Script to start the frontend dev server
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

## Model Training

The system includes a comprehensive model training module supporting various learning approaches:

### Training Approaches

- **Supervised Learning**: For tasks with labeled data (classification, regression, etc.)
- **Unsupervised Learning**: For tasks with unlabeled data (clustering, dimensionality reduction, etc.)
- **Reinforcement Learning**: For training models through rewards and punishments
- **Semi-Supervised Learning**: For training with both labeled and unlabeled data
- **Self-Supervised Learning**: For training with artificially created labels
- **Online Learning**: For incremental learning with streaming data
- **Federated Learning**: For training across multiple decentralized devices

### Dataset Support

The training module supports various dataset formats:

- **CSV**: For tabular data with headers
- **JSON**: For structured data
- **Text**: For plain text datasets

### Training Configuration

Training configuration is managed by the `TrainingConfig` class in `training/config.py`. Each learning approach has default hyperparameter configurations that can be customized.

### HuggingFace Integration

The system integrates with HuggingFace's model hub, allowing you to:

1. Search for pre-trained models
2. Download models for fine-tuning
3. Browse available model tasks
4. Access model information and metadata

### SafeTensor Support

Models trained with the system can be saved in SafeTensor format, providing:

- Better security (no arbitrary code execution)
- Improved serialization/deserialization performance
- Compatibility with different frameworks
- Smaller file sizes

### Training Workflow

1. Select or upload a dataset
2. Choose a model from HuggingFace or local directory
3. Select a learning approach and customize training parameters
4. Start the training process
5. Monitor training progress
6. Save and test the trained model

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

### Phase 3: Model Training & Management (Current)
- ✅ Custom model training interface with multiple learning approaches
- ✅ Dataset management for training (support for CSV, JSON, text formats)
- ✅ HuggingFace integration for model discovery and download
- ✅ Training configuration management
- ✅ Support for SafeTensor model files
- ✅ Directory browsing for model path selection
- 🔲 Support for more model architectures:
  - 🔲 BERT and RoBERTa models
  - 🔲 GPT-J and BLOOM models
  - 🔲 LLaMA and Mistral models
- 🔲 Model performance metrics
- 🔲 Batch inference support

### Phase 4: Advanced Features (Next)
- 🔲 User authentication and multi-user support
- 🔲 Role-based access control
- 🔲 API key management
- 🔲 Usage tracking and quotas
- 🔲 Model versioning
- 🔲 A/B testing for models
- 🔲 Automated model evaluation

### Phase 5: Enterprise Features
- 🔲 Horizontal scaling and load balancing
- 🔲 Containerization with Docker
- 🔲 Kubernetes deployment configuration
- 🔲 Monitoring and alerting
- 🔲 Audit logging
- 🔲 Backup and restore functionality

### Phase 6: Advanced ML Features
- 🔲 Advanced training techniques:
  - 🔲 Hyperparameter optimization
  - 🔲 Model distillation
  - 🔲 Adapter-based fine-tuning
  - 🔲 Quantization support
- 🔲 Multi-modal model support
- 🔲 Transfer learning workflows
- 🔲 Training visualization and analytics

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