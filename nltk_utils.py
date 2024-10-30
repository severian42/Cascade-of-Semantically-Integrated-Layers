import nltk
from typing import List
import os
import ssl

def ensure_nltk_resources() -> None:
    """Ensure all required NLTK resources are downloaded."""
    try:
        # Handle SSL certificate verification issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # Define required resources
        required_resources = [
            'punkt',
            'stopwords',
            'averaged_perceptron_tagger',
            'punkt_tab'
        ]
        
        # Download all required resources
        for resource in required_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
                print(f"Found {resource}")
            except LookupError:
                print(f"Downloading {resource}...")
                nltk.download(resource, quiet=True)
                print(f"Successfully downloaded {resource}")
                
    except Exception as e:
        raise RuntimeError(f"Failed to initialize NLTK resources: {str(e)}")
