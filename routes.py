import os
import logging
from flask import render_template, request, jsonify, redirect, url_for, flash
from app import app
from models.config import ModelConfig
from models.ai_model import get_loaded_model_ids, get_model_info
from utils.model_loader import load_model_from_path, unload_model

logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Render the main dashboard page"""
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    """Dashboard showing all loaded models and their status"""
    loaded_models = get_loaded_model_ids()
    model_infos = [get_model_info(model_id) for model_id in loaded_models]
    return render_template('dashboard.html', models=model_infos)

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
        # In a production environment, you would validate that the path exists:
        # if not os.path.exists(model_path):
        #     flash(f'Model path does not exist: {model_path}', 'error')
        #     return redirect(url_for('model_config'))
        
        # Create directory if it doesn't exist (for demo)
        os.makedirs(model_path, exist_ok=True)
        
        # Add model to configuration
        config = ModelConfig.get_config()
        config['models'][model_id] = {
            'path': model_path,
            'type': model_type,
            'loaded': False
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
