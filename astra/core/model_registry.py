"""
Model Registry for nebula-trade.
Handles model versioning, metadata tracking, and model loading/saving.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


class ModelRegistry:
    """Registry for tracking and managing trained models."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize model registry.
        
        Args:
            project_root: Root directory of the project. If None, auto-detect.
        """
        if project_root is None:
            # Auto-detect project root
            self.project_root = Path(__file__).resolve().parent.parent.parent
        else:
            self.project_root = Path(project_root)
        
        # Model directories
        self.models_dir = self.project_root / "models"
        self.production_dir = self.models_dir / "production"
        self.archive_dir = self.models_dir / "archive"
        
        # Ensure directories exist
        self.production_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Registry file
        self.registry_file = self.models_dir / "registry.json"
        self._load_registry()
    
    def _load_registry(self):
        """Load model registry from file."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                'models': {},
                'last_updated': None
            }
    
    def _save_registry(self):
        """Save model registry to file."""
        self.registry['last_updated'] = datetime.now().isoformat()
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(
        self,
        model_name: str,
        model_version: str,
        model_dir: Path,
        metadata: Dict[str, Any],
        training_config: Dict[str, Any],
        performance: Optional[Dict[str, Any]] = None
    ) -> str:
        """Register a trained model.
        
        Args:
            model_name: Model identifier (e.g., 'v5_tuned')
            model_version: Version string (e.g., '20251004_120629')
            model_dir: Directory containing the trained model
            metadata: Model metadata
            training_config: Training configuration used
            performance: Performance metrics (if available)
            
        Returns:
            Model ID (unique identifier)
        """
        model_id = f"{model_name}_{model_version}"
        
        # Check if already registered
        if model_id in self.registry['models']:
            raise ValueError(f"Model already registered: {model_id}")
        
        # Store model info
        model_info = {
            'id': model_id,
            'name': model_name,
            'version': model_version,
            'path': str(model_dir.relative_to(self.project_root)),
            'metadata': metadata,
            'training_config': {
                'total_timesteps': training_config.get('training', {}).get('total_timesteps'),
                'data_period': training_config.get('model', {}).get('data', {}).get('period'),
                'reward_type': training_config.get('model', {}).get('reward', {}).get('type'),
            },
            'performance': performance or {},
            'registered_at': datetime.now().isoformat(),
            'status': 'active'
        }
        
        self.registry['models'][model_id] = model_info
        self._save_registry()
        
        return model_id
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a registered model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model information
        """
        if model_id not in self.registry['models']:
            raise ValueError(f"Model not found: {model_id}")
        
        return self.registry['models'][model_id]
    
    def get_model_path(self, model_id: str) -> Path:
        """Get path to a registered model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Path to model directory
        """
        model_info = self.get_model_info(model_id)
        return self.project_root / model_info['path']
    
    def list_models(
        self,
        model_name: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List registered models.
        
        Args:
            model_name: Filter by model name (e.g., 'v1_momentum')
            status: Filter by status ('active', 'archived', 'deprecated')
            
        Returns:
            List of model information dictionaries
        """
        models = []
        
        for model_id, model_info in self.registry['models'].items():
            # Apply filters
            if model_name and model_info['name'] != model_name:
                continue
            if status and model_info.get('status') != status:
                continue
            
            models.append(model_info)
        
        # Sort by registration date (newest first)
        models.sort(key=lambda x: x['registered_at'], reverse=True)
        
        return models
    
    def get_latest_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get the latest version of a model.
        
        Args:
            model_name: Model name (e.g., 'v1_momentum')
            
        Returns:
            Latest model info or None if not found
        """
        models = self.list_models(model_name=model_name, status='active')
        
        if not models:
            return None
        
        return models[0]  # Already sorted by date
    
    def update_performance(self, model_id: str, performance: Dict[str, Any]):
        """Update performance metrics for a model.
        
        Args:
            model_id: Model identifier
            performance: Performance metrics to add/update
        """
        if model_id not in self.registry['models']:
            raise ValueError(f"Model not found: {model_id}")
        
        self.registry['models'][model_id]['performance'].update(performance)
        self._save_registry()
    
    def set_model_status(self, model_id: str, status: str):
        """Update model status.
        
        Args:
            model_id: Model identifier
            status: New status ('active', 'archived', 'deprecated', 'retired')
        """
        if model_id not in self.registry['models']:
            raise ValueError(f"Model not found: {model_id}")
        
        valid_statuses = ['active', 'archived', 'deprecated', 'retired']
        if status not in valid_statuses:
            raise ValueError(f"Invalid status. Must be one of: {valid_statuses}")
        
        self.registry['models'][model_id]['status'] = status
        self._save_registry()
    
    def archive_model(self, model_id: str, reason: Optional[str] = None):
        """Archive a model (move to archive directory).
        
        Args:
            model_id: Model identifier
            reason: Optional reason for archiving
        """
        model_info = self.get_model_info(model_id)
        model_path = self.get_model_path(model_id)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        # Create archive destination
        archive_dest = self.archive_dir / model_path.name
        
        # Move model
        shutil.move(str(model_path), str(archive_dest))
        
        # Update registry
        self.registry['models'][model_id]['path'] = str(archive_dest.relative_to(self.project_root))
        self.registry['models'][model_id]['status'] = 'archived'
        self.registry['models'][model_id]['archived_at'] = datetime.now().isoformat()
        
        if reason:
            self.registry['models'][model_id]['archive_reason'] = reason
        
        self._save_registry()
    
    def get_production_models(self) -> List[Dict[str, Any]]:
        """Get all models in production directory.
        
        Returns:
            List of production models
        """
        models = []
        
        for model_info in self.registry['models'].values():
            model_path = self.project_root / model_info['path']
            if model_path.is_relative_to(self.production_dir):
                models.append(model_info)
        
        return models
    
    def find_model_by_path(self, model_path: Path) -> Optional[Dict[str, Any]]:
        """Find a model by its directory path.
        
        Args:
            model_path: Path to model directory
            
        Returns:
            Model info or None if not found
        """
        # Normalize path to relative
        try:
            rel_path = model_path.relative_to(self.project_root)
        except ValueError:
            rel_path = model_path
        
        rel_path_str = str(rel_path)
        
        for model_info in self.registry['models'].values():
            if model_info['path'] == rel_path_str:
                return model_info
        
        return None
    
    def export_registry(self, output_path: Path):
        """Export registry to a file.
        
        Args:
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def import_registry(self, input_path: Path, merge: bool = True):
        """Import registry from a file.
        
        Args:
            input_path: Input file path
            merge: If True, merge with existing registry. If False, replace.
        """
        with open(input_path, 'r') as f:
            imported = json.load(f)
        
        if merge:
            # Merge models
            self.registry['models'].update(imported['models'])
        else:
            # Replace entirely
            self.registry = imported
        
        self._save_registry()
