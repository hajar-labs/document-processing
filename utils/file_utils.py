# File utilities for document processing
"""
Comprehensive file handling utilities for the document processing pipeline.
Handles file operations, batch processing, and format detection.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Generator, Tuple
import mimetypes
import hashlib
import json
import shutil
from datetime import datetime
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)


class FileManager:
    """
    Comprehensive file manager for document processing pipeline.
    Handles file operations, validation, and batch processing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Configuration
        self.max_file_size = self.config.get('max_file_size', 100 * 1024 * 1024)  # 100MB
        self.allowed_extensions = self.config.get('allowed_extensions', {
            '.pdf', '.docx', '.doc', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'
        })
        self.temp_dir = Path(self.config.get('temp_dir', tempfile.gettempdir()))
        self.backup_enabled = self.config.get('backup_enabled', True)
        self.parallel_processing = self.config.get('parallel_processing', True)
        self.max_workers = self.config.get('max_workers', 4)
        
        # Ensure temp directory exists
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # MIME type mappings
        self.mime_mappings = {
            'application/pdf': '.pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
            'application/msword': '.doc',
            'image/png': '.png',
            'image/jpeg': '.jpg',
            'image/jpg': '.jpg',
            'image/bmp': '.bmp',
            'image/tiff': '.tiff',
            'image/tif': '.tif'
        }
    
    def validate_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate file for processing.
        
        Args:
            file_path: Path to file
            
        Returns:
            Validation result dictionary
        """
        file_path = Path(file_path)
        result = {
            'is_valid': False,
            'file_exists': False,
            'file_size': 0,
            'file_extension': '',
            'mime_type': '',
            'is_readable': False,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check if file exists
            if not file_path.exists():
                result['errors'].append(f"File does not exist: {file_path}")
                return result
            
            result['file_exists'] = True
            
            # Check if it's a file (not directory)
            if not file_path.is_file():
                result['errors'].append(f"Path is not a file: {file_path}")
                return result
            
            # Get file size
            file_size = file_path.stat().st_size
            result['file_size'] = file_size
            
            # Check file size
            if file_size == 0:
                result['errors'].append("File is empty")
                return result
            
            if file_size > self.max_file_size:
                result['errors'].append(
                    f"File size ({file_size:,} bytes) exceeds maximum allowed "
                    f"size ({self.max_file_size:,} bytes)"
                )
                return result
            
            # Check file extension
            file_extension = file_path.suffix.lower()
            result['file_extension'] = file_extension
            
            if file_extension not in self.allowed_extensions:
                result['errors'].append(
                    f"File extension '{file_extension}' is not supported. "
                    f"Allowed extensions: {sorted(self.allowed_extensions)}"
                )
                return result
            
            # Check MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            result['mime_type'] = mime_type or 'unknown'
            
            if mime_type and mime_type not in self.mime_mappings:
                result['warnings'].append(f"MIME type '{mime_type}' may not be fully supported")
            
            # Check if file is readable
            try:
                with open(file_path, 'rb') as f:
                    f.read(1024)  # Try to read first 1KB
                result['is_readable'] = True
            except PermissionError:
                result['errors'].append("Permission denied: cannot read file")
                return result
            except Exception as e:
                result['errors'].append(f"File read error: {str(e)}")
                return result
            
            # Additional checks based on file type
            if file_extension == '.pdf':
                pdf_validation = self._validate_pdf_file(file_path)
                result.update(pdf_validation)
            elif file_extension in ['.docx', '.doc']:
                word_validation = self._validate_word_file(file_path)
                result.update(word_validation)
            elif file_extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
                image_validation = self._validate_image_file(file_path)
                result.update(image_validation)
            
            # If no errors, file is valid
            if not result['errors']:
                result['is_valid'] = True
            
        except Exception as e:
            result['errors'].append(f"Validation error: {str(e)}")
            logger.error(f"File validation failed for {file_path}: {e}")
        
        return result
    
    def _validate_pdf_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate PDF file specifically."""
        result = {}
        
        try:
            # Check PDF header
            with open(file_path, 'rb') as f:
                header = f.read(8)
                if not header.startswith(b'%PDF-'):
                    result['warnings'] = result.get('warnings', []) + ["File may not be a valid PDF"]
            
        except Exception as e:
            result['warnings'] = result.get('warnings', []) + [f"PDF validation warning: {str(e)}"]
        
        return result
    
    def _validate_word_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate Word document file specifically."""
        result = {}
        
        try:
            if file_path.suffix.lower() == '.docx':
                # Check if it's a valid ZIP file (DOCX format)
                try:
                    with zipfile.ZipFile(file_path, 'r') as docx_zip:
                        # Check for required DOCX files
                        required_files = ['[Content_Types].xml', 'word/document.xml']
                        file_list = docx_zip.namelist()
                        
                        for required in required_files:
                            if required not in file_list:
                                result['warnings'] = result.get('warnings', []) + [
                                    f"DOCX file may be corrupted: missing {required}"
                                ]
                except zipfile.BadZipFile:
                    result['warnings'] = result.get('warnings', []) + ["DOCX file appears to be corrupted"]
            
        except Exception as e:
            result['warnings'] = result.get('warnings', []) + [f"Word validation warning: {str(e)}"]
        
        return result
    
    def _validate_image_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate image file specifically."""
        result = {}
        
        try:
            from PIL import Image
            
            # Try to open and verify the image
            with Image.open(file_path) as img:
                result['image_format'] = img.format
                result['image_mode'] = img.mode
                result['image_size'] = img.size
                
                # Check for very small images
                if img.size[0] < 100 or img.size[1] < 100:
                    result['warnings'] = result.get('warnings', []) + [
                        f"Image is very small ({img.size[0]}x{img.size[1]}) - OCR may not work well"
                    ]
                
                # Check for very large images
                if img.size[0] > 4000 or img.size[1] > 4000:
                    result['warnings'] = result.get('warnings', []) + [
                        f"Image is very large ({img.size[0]}x{img.size[1]}) - processing may be slow"
                    ]
        
        except ImportError:
            result['warnings'] = result.get('warnings', []) + ["PIL not available for image validation"]
        except Exception as e:
            result['warnings'] = result.get('warnings', []) + [f"Image validation warning: {str(e)}"]
        
        return result
    
    def calculate_file_hash(self, file_path: Union[str, Path], algorithm: str = 'md5') -> str:
        """
        Calculate file hash for deduplication.
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm ('md5', 'sha256')
            
        Returns:
            File hash string
        """
        file_path = Path(file_path)
        
        if algorithm == 'md5':
            hash_obj = hashlib.md5()
        elif algorithm == 'sha256':
            hash_obj = hashlib.sha256()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            
            return hash_obj.hexdigest()
        
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def find_files(self, directory: Union[str, Path], 
                   recursive: bool = True, 
                   pattern: str = "*") -> List[Path]:
        """
        Find files in directory with optional filtering.
        
        Args:
            directory: Directory to search
            recursive: Search recursively in subdirectories
            pattern: File pattern to match
            
        Returns:
            List of file paths
        """
        directory = Path(directory)
        
        if not directory.exists() or not directory.is_dir():
            logger.error(f"Directory does not exist: {directory}")
            return []
        
        files = []
        
        try:
            if recursive:
                pattern_path = f"**/{pattern}"
                found_files = directory.glob(pattern_path)
            else:
                found_files = directory.glob(pattern)
            
            for file_path in found_files:
                if file_path.is_file() and file_path.suffix.lower() in self.allowed_extensions:
                    files.append(file_path)
        
        except Exception as e:
            logger.error(f"Error finding files in {directory}: {e}")
        
        return sorted(files)
    
    def batch_validate_files(self, file_paths: List[Union[str, Path]]) -> Dict[str, Any]:
        """
        Validate multiple files in batch.
        
        Args:
            file_paths: List of file paths to validate
            
        Returns:
            Batch validation results
        """
        results = {
            'total_files': len(file_paths),
            'valid_files': [],
            'invalid_files': [],
            'validation_details': {},
            'summary': {
                'valid_count': 0,
                'invalid_count': 0,
                'total_size': 0,
                'file_types': {}
            }
        }
        
        if self.parallel_processing and len(file_paths) > 1:
            # Parallel validation
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_path = {
                    executor.submit(self.validate_file, path): path 
                    for path in file_paths
                }
                
                for future in as_completed(future_to_path):
                    file_path = future_to_path[future]
                    try:
                        validation_result = future.result()
                        results['validation_details'][str(file_path)] = validation_result
                        
                        if validation_result['is_valid']:
                            results['valid_files'].append(str(file_path))
                            results['summary']['valid_count'] += 1
                        else:
                            results['invalid_files'].append(str(file_path))
                            results['summary']['invalid_count'] += 1
                        
                        # Update summary statistics
                        results['summary']['total_size'] += validation_result['file_size']
                        
                        file_ext = validation_result['file_extension']
                        if file_ext:
                            results['summary']['file_types'][file_ext] = \
                                results['summary']['file_types'].get(file_ext, 0) + 1
                    
                    except Exception as e:
                        logger.error(f"Validation failed for {file_path}: {e}")
                        results['invalid_files'].append(str(file_path))
                        results['summary']['invalid_count'] += 1
                        results['validation_details'][str(file_path)] = {
                            'is_valid': False,
                            'errors': [f"Validation exception: {str(e)}"]
                        }
        else:
            # Sequential validation
            for file_path in file_paths:
                try:
                    validation_result = self.validate_file(file_path)
                    results['validation_details'][str(file_path)] = validation_result
                    
                    if validation_result['is_valid']:
                        results['valid_files'].append(str(file_path))
                        results['summary']['valid_count'] += 1
                    else:
                        results['invalid_files'].append(str(file_path))
                        results['summary']['invalid_count'] += 1
                    
                    # Update summary statistics
                    results['summary']['total_size'] += validation_result['file_size']
                    
                    file_ext = validation_result['file_extension']
                    if file_ext:
                        results['summary']['file_types'][file_ext] = \
                            results['summary']['file_types'].get(file_ext, 0) + 1
                
                except Exception as e:
                    logger.error(f"Validation failed for {file_path}: {e}")
                    results['invalid_files'].append(str(file_path))
                    results['summary']['invalid_count'] += 1
                    results['validation_details'][str(file_path)] = {
                        'is_valid': False,
                        'errors': [f"Validation exception: {str(e)}"]
                    }
        
        return results
    
    def create_temp_file(self, suffix: str = '', prefix: str = 'doc_process_') -> Path:
        """
        Create a temporary file for processing.
        
        Args:
            suffix: File suffix
            prefix: File prefix
            
        Returns:
            Path to temporary file
        """
        try:
            temp_fd, temp_path = tempfile.mkstemp(
                suffix=suffix, 
                prefix=prefix, 
                dir=self.temp_dir
            )
            os.close(temp_fd)  # Close file descriptor
            return Path(temp_path)
        
        except Exception as e:
            logger.error(f"Failed to create temp file: {e}")
            # Fallback to simple path generation
            timestamp = int(time.time() * 1000)
            temp_name = f"{prefix}{timestamp}{suffix}"
            return self.temp_dir / temp_name
    
    def cleanup_temp_files(self, older_than_hours: int = 24):
        """
        Clean up old temporary files.
        
        Args:
            older_than_hours: Remove files older than this many hours
        """
        try:
            current_time = time.time()
            cutoff_time = current_time - (older_than_hours * 3600)
            
            temp_files = list(self.temp_dir.glob('doc_process_*'))
            
            removed_count = 0
            for temp_file in temp_files:
                try:
                    if temp_file.stat().st_mtime < cutoff_time:
                        temp_file.unlink()
                        removed_count += 1
                except Exception as e:
                    logger.warning(f"Could not remove temp file {temp_file}: {e}")
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} temporary files")
        
        except Exception as e:
            logger.error(f"Error during temp file cleanup: {e}")
    
    def backup_file(self, file_path: Union[str, Path], 
                   backup_dir: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """
        Create a backup of a file.
        
        Args:
            file_path: Path to file to backup
            backup_dir: Directory for backup (default: same directory as original)
            
        Returns:
            Path to backup file or None if backup failed
        """
        if not self.backup_enabled:
            return None
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"Cannot backup non-existent file: {file_path}")
            return None
        
        try:
            if backup_dir:
                backup_dir = Path(backup_dir)
                backup_dir.mkdir(parents=True, exist_ok=True)
            else:
                backup_dir = file_path.parent
            
            # Generate backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{file_path.stem}_{timestamp}_backup{file_path.suffix}"
            backup_path = backup_dir / backup_name
            
            # Copy file
            shutil.copy2(file_path, backup_path)
            
            logger.info(f"Created backup: {backup_path}")
            return backup_path
        
        except Exception as e:
            logger.error(f"Failed to create backup for {file_path}: {e}")
            return None
    
    def organize_files_by_type(self, file_paths: List[Union[str, Path]]) -> Dict[str, List[Path]]:
        """
        Organize files by type/extension.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Dictionary with file types as keys and lists of paths as values
        """
        organized = {}
        
        for file_path in file_paths:
            file_path = Path(file_path)
            file_extension = file_path.suffix.lower()
            
            if file_extension not in organized:
                organized[file_extension] = []
            
            organized[file_extension].append(file_path)
        
        return organized
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get comprehensive file information.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {'error': 'File does not exist'}
        
        try:
            stat = file_path.stat()
            mime_type, encoding = mimetypes.guess_type(str(file_path))
            
            info = {
                'path': str(file_path),
                'name': file_path.name,
                'stem': file_path.stem,
                'suffix': file_path.suffix,
                'size': stat.st_size,
                'size_human': self._format_size(stat.st_size),
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'accessed': datetime.fromtimestamp(stat.st_atime).isoformat(),
                'mime_type': mime_type,
                'encoding': encoding,
                'is_readable': os.access(file_path, os.R_OK),
                'is_writable': os.access(file_path, os.W_OK),
                'hash_md5': self.calculate_file_hash(file_path, 'md5'),
            }
            
            return info
        
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            return {'error': str(e)}
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        
        return f"{s} {size_names[i]}"
    
    def create_processing_manifest(self, file_paths: List[Union[str, Path]], 
                                 output_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Create a manifest file for batch processing.
        
        Args:
            file_paths: List of files to process
            output_path: Path to save manifest
            
        Returns:
            Manifest data
        """
        manifest = {
            'created': datetime.now().isoformat(),
            'total_files': len(file_paths),
            'files': []
        }
        
        for i, file_path in enumerate(file_paths):
            file_info = self.get_file_info(file_path)
            validation = self.validate_file(file_path)
            
            file_entry = {
                'index': i,
                'path': str(file_path),
                'info': file_info,
                'validation': validation,
                'processed': False,
                'processing_time': None,
                'output_files': []
            }
            
            manifest['files'].append(file_entry)
        
        # Save manifest
        output_path = Path(output_path)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Created processing manifest: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")
        
        return manifest
    
    def load_processing_manifest(self, manifest_path: Union[str, Path]) -> Dict[str, Any]:
        """Load processing manifest from file."""
        manifest_path = Path(manifest_path)
        
        if not manifest_path.exists():
            logger.error(f"Manifest file does not exist: {manifest_path}")
            return {}
        
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            
            logger.info(f"Loaded processing manifest: {manifest_path}")
            return manifest
        
        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")
            return {}
    
    def update_manifest_entry(self, manifest: Dict[str, Any], file_index: int, 
                            processed: bool = True, processing_time: float = None, 
                            output_files: List[str] = None) -> Dict[str, Any]:
        """Update a manifest entry with processing results."""
        if 'files' not in manifest or file_index >= len(manifest['files']):
            logger.error(f"Invalid manifest entry index: {file_index}")
            return manifest
        
        entry = manifest['files'][file_index]
        entry['processed'] = processed
        
        if processing_time is not None:
            entry['processing_time'] = processing_time
        
        if output_files:
            entry['output_files'] = output_files
        
        return manifest
    
    def save_manifest(self, manifest: Dict[str, Any], manifest_path: Union[str, Path]):
        """Save updated manifest to file."""
        manifest_path = Path(manifest_path)
        
        try:
            # Add last updated timestamp
            manifest['last_updated'] = datetime.now().isoformat()
            
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Updated manifest saved: {manifest_path}")
        
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")
    
    def get_processing_statistics(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Generate processing statistics from manifest."""
        if 'files' not in manifest:
            return {}
        
        files = manifest['files']
        total_files = len(files)
        processed_files = len([f for f in files if f.get('processed', False)])
        
        # Calculate processing times
        processing_times = [f.get('processing_time', 0) for f in files if f.get('processing_time')]
        
        stats = {
            'total_files': total_files,
            'processed_files': processed_files,
            'remaining_files': total_files - processed_files,
            'completion_rate': (processed_files / total_files * 100) if total_files > 0 else 0,
            'processing_times': {
                'total_time': sum(processing_times),
                'average_time': sum(processing_times) / len(processing_times) if processing_times else 0,
                'min_time': min(processing_times) if processing_times else 0,
                'max_time': max(processing_times) if processing_times else 0
            }
        }
        
        # File type statistics
        file_types = {}
        for file_entry in files:
            path = Path(file_entry['path'])
            ext = path.suffix.lower()
            if ext not in file_types:
                file_types[ext] = {'total': 0, 'processed': 0}
            
            file_types[ext]['total'] += 1
            if file_entry.get('processed', False):
                file_types[ext]['processed'] += 1
        
        stats['file_types'] = file_types
        
        return stats


class BatchProcessor:
    """
    Batch processor for handling multiple documents.
    """
    
    def __init__(self, file_manager: FileManager, config: Optional[Dict[str, Any]] = None):
        self.file_manager = file_manager
        self.config = config or {}
        
        self.max_workers = self.config.get('max_workers', 4)
        self.continue_on_error = self.config.get('continue_on_error', True)
        self.save_progress = self.config.get('save_progress', True)
        
    def process_files(self, file_paths: List[Union[str, Path]], 
                     processing_function, output_directory: Union[str, Path],
                     manifest_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Process multiple files using provided processing function.
        
        Args:
            file_paths: List of files to process
            processing_function: Function to process each file
            output_directory: Directory for output files
            manifest_path: Optional path to save processing manifest
            
        Returns:
            Processing results
        """
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        
        # Create manifest if path provided
        manifest = None
        if manifest_path:
            manifest = self.file_manager.create_processing_manifest(file_paths, manifest_path)
        
        # Validate all files first
        logger.info(f"Validating {len(file_paths)} files...")
        validation_results = self.file_manager.batch_validate_files(file_paths)
        valid_files = [Path(p) for p in validation_results['valid_files']]
        
        logger.info(f"Processing {len(valid_files)} valid files...")
        
        results = {
            'total_files': len(file_paths),
            'valid_files': len(valid_files),
            'invalid_files': len(validation_results['invalid_files']),
            'processed_successfully': 0,
            'processing_errors': 0,
            'results': [],
            'errors': [],
            'start_time': datetime.now().isoformat()
        }
        
        # Process files
        start_time = time.time()
        
        if self.file_manager.parallel_processing and len(valid_files) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(self._process_single_file, 
                                  file_path, processing_function, output_directory, i): 
                    (file_path, i)
                    for i, file_path in enumerate(valid_files)
                }
                
                for future in as_completed(future_to_file):
                    file_path, file_index = future_to_file[future]
                    
                    try:
                        result = future.result()
                        results['results'].append(result)
                        
                        if result['success']:
                            results['processed_successfully'] += 1
                        else:
                            results['processing_errors'] += 1
                            results['errors'].append({
                                'file': str(file_path),
                                'error': result.get('error', 'Unknown error')
                            })
                        
                        # Update manifest if available
                        if manifest:
                            self.file_manager.update_manifest_entry(
                                manifest, file_index, 
                                processed=result['success'],
                                processing_time=result.get('processing_time'),
                                output_files=result.get('output_files', [])
                            )
                    
                    except Exception as e:
                        logger.error(f"Processing failed for {file_path}: {e}")
                        results['processing_errors'] += 1
                        results['errors'].append({
                            'file': str(file_path),
                            'error': str(e)
                        })
                        
                        if not self.continue_on_error:
                            break
        else:
            # Sequential processing
            for i, file_path in enumerate(valid_files):
                try:
                    result = self._process_single_file(
                        file_path, processing_function, output_directory, i
                    )
                    results['results'].append(result)
                    
                    if result['success']:
                        results['processed_successfully'] += 1
                    else:
                        results['processing_errors'] += 1
                        results['errors'].append({
                            'file': str(file_path),
                            'error': result.get('error', 'Unknown error')
                        })
                    
                    # Update manifest if available
                    if manifest:
                        self.file_manager.update_manifest_entry(
                            manifest, i,
                            processed=result['success'],
                            processing_time=result.get('processing_time'),
                            output_files=result.get('output_files', [])
                        )
                        
                        # Save progress periodically
                        if self.save_progress and i % 10 == 0 and manifest_path:
                            self.file_manager.save_manifest(manifest, manifest_path)
                
                except Exception as e:
                    logger.error(f"Processing failed for {file_path}: {e}")
                    results['processing_errors'] += 1
                    results['errors'].append({
                        'file': str(file_path),
                        'error': str(e)
                    })
                    
                    if not self.continue_on_error:
                        break
        
        # Finalize results
        end_time = time.time()
        results['end_time'] = datetime.now().isoformat()
        results['total_processing_time'] = end_time - start_time
        
        # Save final manifest
        if manifest and manifest_path:
            self.file_manager.save_manifest(manifest, manifest_path)
        
        logger.info(f"Batch processing completed: {results['processed_successfully']}/{len(valid_files)} files processed successfully")
        
        return results
    
    def _process_single_file(self, file_path: Path, processing_function, 
                           output_directory: Path, file_index: int) -> Dict[str, Any]:
        """Process a single file."""
        start_time = time.time()
        
        try:
            # Call the processing function
            result = processing_function(file_path, output_directory, file_index)
            
            processing_time = time.time() - start_time
            
            return {
                'file_path': str(file_path),
                'file_index': file_index,
                'success': True,
                'processing_time': processing_time,
                'result': result,
                'output_files': result.get('output_files', []) if isinstance(result, dict) else []
            }
        
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing {file_path}: {e}")
            
            return {
                'file_path': str(file_path),
                'file_index': file_index,
                'success': False,
                'processing_time': processing_time,
                'error': str(e)
            }