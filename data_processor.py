"""
Data Processor for MyFixit Dataset
Loads JSON repair guides, extracts text, and chunks documents for RAG pipeline
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import re


class DataProcessor:
    """Process MyFixit JSON files into chunked documents for RAG"""
    
    def __init__(self, json_dir: str = "MyFixit-Dataset-master/jsons", chunk_size: int = 150):
        """
        Initialize DataProcessor
        
        Args:
            json_dir: Directory containing JSON files
            chunk_size: Target number of words per chunk (100-200 recommended)
        """
        self.json_dir = Path(json_dir)
        self.chunk_size = chunk_size
        self.documents = []
        
    def load_json_files(self) -> List[Dict[str, Any]]:
        """
        Load all JSON files from the jsons directory
        
        Returns:
            List of parsed JSON guide objects
        """
        guides = []
        
        if not self.json_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.json_dir}")
        
        json_files = list(self.json_dir.glob("*.json"))
        print(f"Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            print(f"Loading {json_file.name}...")
            with open(json_file, 'r', encoding='utf-8') as f:
                # Each line is a separate JSON object (newline-delimited JSON)
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            guide = json.loads(line)
                            guides.append(guide)
                        except json.JSONDecodeError as e:
                            print(f"Error parsing line {line_num} in {json_file.name}: {e}")
        
        print(f"Loaded {len(guides)} repair guides total")
        return guides
    
    def extract_text_from_guide(self, guide: Dict[str, Any]) -> str:
        """
        Extract all relevant text from a guide
        
        Args:
            guide: A single repair guide dictionary
            
        Returns:
            Concatenated text from the guide
        """
        text_parts = []
        
        # Add title
        if 'Title' in guide:
            text_parts.append(f"Title: {guide['Title']}")
        
        # Add subject/summary
        if 'Subject' in guide:
            text_parts.append(f"Summary: {guide['Subject']}")
        
        # Add category information
        if 'Category' in guide:
            text_parts.append(f"Category: {guide['Category']}")
        
        # Add ancestor hierarchy for context
        if 'Ancestors' in guide and isinstance(guide['Ancestors'], list):
            hierarchy = " > ".join([str(a) for a in guide['Ancestors']])
            text_parts.append(f"Device Type: {hierarchy}")
        
        # Extract tools needed
        if 'Toolbox' in guide and isinstance(guide['Toolbox'], list):
            tools = [tool.get('Name', '') for tool in guide['Toolbox'] if 'Name' in tool]
            if tools:
                text_parts.append(f"Tools Required: {', '.join(tools)}")
        
        # Extract step-by-step instructions
        if 'Steps' in guide and isinstance(guide['Steps'], list):
            text_parts.append("\nStep-by-Step Instructions:")
            
            for step in guide['Steps']:
                step_num = step.get('Order', 'N/A')
                
                # Extract text from Lines array
                if 'Lines' in step and isinstance(step['Lines'], list):
                    for line in step['Lines']:
                        if 'Text' in line:
                            text_parts.append(f"Step {step_num}: {line['Text']}")
                
                # Also use Text_raw as fallback or additional context
                elif 'Text_raw' in step:
                    text_parts.append(f"Step {step_num}: {step['Text_raw']}")
                
                # Extract tools used in this step
                if 'Tools_extracted' in step and isinstance(step['Tools_extracted'], list):
                    tools_in_step = [tool.get('Name', '') for tool in step['Tools_extracted'] if 'Name' in tool]
                    if tools_in_step:
                        text_parts.append(f"  Tools for this step: {', '.join(tools_in_step)}")
        
        return "\n".join(text_parts)
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split text into chunks of approximately chunk_size words
        
        Args:
            text: Full text to chunk
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Split into words
        words = text.split()
        chunks = []
        
        # Create chunks
        for i in range(0, len(words), self.chunk_size):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunk = {
                'text': chunk_text,
                'word_count': len(chunk_words),
                'metadata': metadata.copy(),
                'chunk_index': len(chunks)
            }
            chunks.append(chunk)
        
        return chunks
    
    def process_all_guides(self) -> List[Dict[str, Any]]:
        """
        Load all guides, extract text, and create chunks
        
        Returns:
            List of document chunks with metadata
        """
        # Load all JSON files
        guides = self.load_json_files()
        
        all_chunks = []
        
        for guide_idx, guide in enumerate(guides):
            # Extract metadata
            metadata = {
                'title': guide.get('Title', 'Unknown'),
                'category': guide.get('Category', 'Unknown'),
                'url': guide.get('Url', ''),
                'guide_id': guide.get('Guidid', ''),
                'ancestors': guide.get('Ancestors', []),
                'source_guide_index': guide_idx
            }
            
            # Extract text
            text = self.extract_text_from_guide(guide)
            
            # Create chunks
            chunks = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)
        
        self.documents = all_chunks
        print(f"Created {len(all_chunks)} chunks from {len(guides)} guides")
        
        return all_chunks
    
    def get_documents(self) -> List[Dict[str, Any]]:
        """Get processed document chunks"""
        return self.documents
    
    def save_chunks(self, output_file: str = "processed_chunks.json"):
        """
        Save processed chunks to a JSON file for inspection
        
        Args:
            output_file: Path to output file
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, indent=2)
        print(f"Saved {len(self.documents)} chunks to {output_file}")


def main():
    """Test the data processor"""
    processor = DataProcessor(json_dir="MyFixit-Dataset-master/jsons", chunk_size=150)
    chunks = processor.process_all_guides()
    
    # Display sample chunks
    print("\n" + "="*80)
    print("SAMPLE CHUNKS:")
    print("="*80)
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(f"Title: {chunk['metadata']['title']}")
        print(f"Category: {chunk['metadata']['category']}")
        print(f"Word Count: {chunk['word_count']}")
        print(f"Text Preview: {chunk['text'][:200]}...")
        print("-"*80)
    
    # Save for inspection
    processor.save_chunks("processed_chunks.json")
    
    print(f"\nTotal statistics:")
    print(f"- Total guides processed: {len(set(c['metadata']['source_guide_index'] for c in chunks))}")
    print(f"- Total chunks created: {len(chunks)}")
    print(f"- Average words per chunk: {sum(c['word_count'] for c in chunks) / len(chunks):.1f}")


if __name__ == "__main__":
    main()
