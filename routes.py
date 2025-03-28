import os
import logging
import json
from flask import render_template, request, jsonify, redirect, url_for, flash
from app import app
from models.config import ModelConfig
from models.ai_model import get_loaded_model_ids, get_model_info
from utils.model_loader import load_model_from_path, unload_model

logger = logging.getLogger(__name__)

# Legacy routes for backwards compatibility
@app.route('/')
def index():
    """Render the main dashboard page"""
    # This route now redirects to the Vue.js frontend
    # The actual rendering is done via the catchall route in app.py
    return render_template('base.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard showing all loaded models and their status"""
    # Legacy route for backwards compatibility
    loaded_models = get_loaded_model_ids()
    model_infos = [get_model_info(model_id) for model_id in loaded_models]
    return render_template('dashboard.html', models=model_infos)

# New JSON API endpoints for Vue.js frontend

@app.route('/api/models', methods=['GET'])
def api_models():
    """API endpoint to get all loaded models"""
    loaded_models = get_loaded_model_ids()
    model_infos = [get_model_info(model_id) for model_id in loaded_models]
    
    # Format for frontend compatibility
    formatted_models = []
    for model in model_infos:
        if model:  # Skip None values
            formatted_models.append({
                'id': model.get('id', ''),
                'owner': 'ai-model-server',
                'ready': True,  # Assume loaded means ready
                'type': model.get('type', 'unknown'),
                'created': model.get('created', 0)
            })
    
    return jsonify({'data': formatted_models})

@app.route('/api/model-config', methods=['GET'])
def api_model_config():
    """API endpoint to get model configuration"""
    config = ModelConfig.get_config()
    return jsonify(config)

@app.route('/api/model-config', methods=['POST'])
def api_save_model_config():
    """API endpoint to save a model configuration"""
    try:
        data = request.json
        model_id = data.get('model_id')
        model_path = data.get('model_path')
        model_type = data.get('model_type', 't5')
        
        if not model_id or not model_path:
            return jsonify({'error': 'Model ID and path are required'}), 400
        
        # For demo purposes, we'll accept any path
        # Create directory if it doesn't exist (for demo)
        os.makedirs(model_path, exist_ok=True)
        
        # Add model to configuration
        config = ModelConfig.get_config()
        config['models'][model_id] = {
            'path': model_path,
            'type': model_type
        }
        ModelConfig.save_config(config)
        
        return jsonify({'success': True, 'message': f'Model {model_id} configuration saved'})
    except Exception as e:
        logger.error(f"Error saving model config: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-config/<model_id>', methods=['DELETE'])
def api_remove_model_config(model_id):
    """API endpoint to remove a model configuration"""
    try:
        if not model_id:
            return jsonify({'error': 'Model ID is required'}), 400
        
        # Remove model from configuration
        config = ModelConfig.get_config()
        if model_id in config['models']:
            del config['models'][model_id]
            ModelConfig.save_config(config)
            return jsonify({'success': True, 'message': f'Model {model_id} configuration removed'})
        else:
            return jsonify({'error': f'Model {model_id} not found in configuration'}), 404
    except Exception as e:
        logger.error(f"Error removing model config: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<model_id>/load', methods=['POST'])
def api_load_model(model_id):
    """API endpoint to load a model"""
    try:
        if not model_id:
            return jsonify({'error': 'Model ID is required'}), 400
        
        # Get model details from config
        config = ModelConfig.get_config()
        if model_id in config['models']:
            model_config = config['models'][model_id]
            model_path = model_config['path']
            model_type = model_config.get('type', 't5')
        else:
            return jsonify({'error': f'Model {model_id} not found in configuration'}), 400
        
        # Load the model
        load_model_from_path(model_id, model_path, model_type)
        
        return jsonify({'success': True, 'message': f'Model {model_id} loaded successfully'})
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<model_id>/unload', methods=['POST'])
def api_unload_model(model_id):
    """API endpoint to unload a model"""
    try:
        if not model_id:
            return jsonify({'error': 'Model ID is required'}), 400
        
        unload_model(model_id)
        return jsonify({'success': True, 'message': f'Model {model_id} unloaded successfully'})
    except Exception as e:
        logger.error(f"Error unloading model: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Legacy routes for backwards compatibility - keep these for now
@app.route('/model/config', methods=['GET', 'POST'])
def model_config():
    """Page for configuring model settings"""
    if request.method == 'POST':
        # Save model configuration
        model_id = request.form.get('model_id')
        model_path = request.form.get('model_path')
        model_type = request.form.get('model_type', 't5')
        
        if not model_id or not model_path:
            flash('Model ID and path are required', 'error')
            return redirect(url_for('model_config'))
        
        # For demo purposes, we'll accept any path
        # Create directory if it doesn't exist (for demo)
        os.makedirs(model_path, exist_ok=True)
        
        # Add model to configuration
        config = ModelConfig.get_config()
        config['models'][model_id] = {
            'path': model_path,
            'type': model_type
        }
        ModelConfig.save_config(config)
        flash(f'Model {model_id} added to configuration', 'success')
        return redirect(url_for('dashboard'))
    
    # GET request - show configuration form
    config = ModelConfig.get_config()
    return render_template('model_config.html', config=config)

@app.route('/model/load', methods=['POST'])
def load_model():
    """API endpoint to load a model"""
    model_id = request.form.get('model_id')
    
    if not model_id:
        flash('Model ID is required', 'error')
        return redirect(url_for('dashboard'))
    
    config = ModelConfig.get_config()
    if model_id not in config['models']:
        flash(f'Model {model_id} not found in configuration', 'error')
        return redirect(url_for('dashboard'))
    
    model_config = config['models'][model_id]
    try:
        load_model_from_path(model_id, model_config['path'], model_config['type'])
        flash(f'Model {model_id} loaded successfully', 'success')
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {str(e)}")
        flash(f'Error loading model: {str(e)}', 'error')
    
    return redirect(url_for('dashboard'))

@app.route('/model/unload', methods=['POST'])
def unload_model_route():
    """API endpoint to unload a model"""
    model_id = request.form.get('model_id')
    
    if not model_id:
        flash('Model ID is required', 'error')
        return redirect(url_for('dashboard'))
    
    try:
        unload_model(model_id)
        flash(f'Model {model_id} unloaded successfully', 'success')
    except Exception as e:
        logger.error(f"Error unloading model {model_id}: {str(e)}")
        flash(f'Error unloading model: {str(e)}', 'error')
    
    return redirect(url_for('dashboard'))

@app.route('/model/test/<model_id>')
def model_test(model_id):
    """Page for testing a specific model"""
    model_info = get_model_info(model_id)
    if not model_info:
        flash(f'Model {model_id} not found', 'error')
        return redirect(url_for('dashboard'))
    
    return render_template('model_test.html', model=model_info)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('base.html', error="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('base.html', error="Internal server error"), 500
