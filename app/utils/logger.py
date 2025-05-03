import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import json
from datetime import datetime

def setup_logging():
    """Configure logging for the application."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_format_json = {
        "timestamp": "%(asctime)s",
        "service": "%(name)s",
        "level": "%(levelname)s",
        "message": "%(message)s",
        "context": "%(context)s" if "%(context)s" in log_format else ""
    }
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # File handler
    file_handler = RotatingFileHandler(
        log_dir / "soko_yetu_ai.log",
        maxBytes=1024 * 1024 * 5,  # 5MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # JSON handler for structured logging
    json_handler = RotatingFileHandler(
        log_dir / "soko_yetu_ai.json",
        maxBytes=1024 * 1024 * 5,  # 5MB
        backupCount=5
    )
    json_handler.setLevel(logging.DEBUG)
    
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                "logger": record.name,
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }
            
            if hasattr(record, "context"):
                log_record["context"] = record.context
                
            if record.exc_info:
                log_record["exception"] = self.formatException(record.exc_info)
                
            return json.dumps(log_record)
            
    json_handler.setFormatter(JSONFormatter())
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[console_handler, file_handler, json_handler]
    )
    
    # Reduce noise from some libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)

logger = logging.getLogger("soko_yetu_ai")