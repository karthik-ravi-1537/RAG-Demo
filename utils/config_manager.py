"""Configuration management utilities."""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
try:
    from core.data_models import RAGConfig
    from core.exceptions import ConfigurationError
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from core.data_models import RAGConfig
    from core.exceptions import ConfigurationError


class ConfigManager:
    """Manages configuration loading, validation, and saving."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else None
        self._config: Optional[RAGConfig] = None
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> RAGConfig:
        """Load configuration from file."""
        path = Path(config_path) if config_path else self.config_path
        
        if not path or not path.exists():
            # Return default configuration
            return RAGConfig()
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    config_dict = yaml.safe_load(f)
                elif path.suffix.lower() == '.json':
                    config_dict = json.load(f)
                else:
                    raise ConfigurationError(f"Unsupported config file format: {path.suffix}")
            
            self._config = RAGConfig.from_dict(config_dict)
            return self._config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load config from {path}: {str(e)}")
    
    def save_config(self, config: RAGConfig, config_path: Optional[Union[str, Path]] = None) -> None:
        """Save configuration to file."""
        path = Path(config_path) if config_path else self.config_path
        
        if not path:
            raise ConfigurationError("No config path specified")
        
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            config_dict = config.to_dict()
            
            with open(path, 'w', encoding='utf-8') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif path.suffix.lower() == '.json':
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ConfigurationError(f"Unsupported config file format: {path.suffix}")
            
            self._config = config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save config to {path}: {str(e)}")
    
    def get_config(self) -> RAGConfig:
        """Get current configuration."""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def update_config(self, updates: Dict[str, Any]) -> RAGConfig:
        """Update configuration with new values."""
        current_config = self.get_config()
        current_dict = current_config.to_dict()
        
        # Deep merge updates
        self._deep_merge(current_dict, updates)
        
        # Create new config from merged dictionary
        self._config = RAGConfig.from_dict(current_dict)
        return self._config
    
    def _deep_merge(self, base: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Deep merge updates into base dictionary."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def validate_config(self, config: RAGConfig) -> bool:
        """Validate configuration parameters."""
        try:
            # Validate chunking config
            if config.chunking.chunk_size <= 0:
                raise ConfigurationError("Chunk size must be positive")
            
            if config.chunking.overlap < 0:
                raise ConfigurationError("Overlap cannot be negative")
            
            if config.chunking.overlap >= config.chunking.chunk_size:
                raise ConfigurationError("Overlap must be less than chunk size")
            
            # Validate embedding config
            if config.embedding.dimension <= 0:
                raise ConfigurationError("Embedding dimension must be positive")
            
            if config.embedding.batch_size <= 0:
                raise ConfigurationError("Batch size must be positive")
            
            # Validate RAG config
            if config.retrieval_top_k <= 0:
                raise ConfigurationError("Retrieval top_k must be positive")
            
            if not (0 <= config.similarity_threshold <= 1):
                raise ConfigurationError("Similarity threshold must be between 0 and 1")
            
            if config.max_context_tokens <= 0:
                raise ConfigurationError("Max context tokens must be positive")
            
            return True
            
        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(f"Config validation failed: {str(e)}")
    
    @staticmethod
    def create_default_config(config_path: Union[str, Path]) -> RAGConfig:
        """Create and save a default configuration file."""
        config = RAGConfig()
        manager = ConfigManager(config_path)
        manager.save_config(config)
        return config