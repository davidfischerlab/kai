"""Hierarchical workflow parser for extracting notebooks in a hierarchical chunking structure."""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from kai.utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class HierarchicalChunk:
    """Represents a chunk in the hierarchical structure."""
    content: str
    level: str  # "document", "section", "code_cell" 
    chunk_id: str
    parent_id: Optional[str]
    chunk_index: int
    metadata: Dict[str, Any]


class HierarchicalWorkflowParser:
    """Parser that creates hierarchical chunks from Jupyter notebooks."""
    
    def __init__(self):
        """Initialize the hierarchical parser."""
        self.section_header_patterns = [
            r'^#+\s+(.+)$',  # Markdown headers
            r'^## (.+)$',    # Level 2 headers
            r'^### (.+)$',   # Level 3 headers
            r'^#### (.+)$',  # Level 4 headers
        ]
    
    def parse_notebook(self, notebook_path: Path, tool_name: str, repo_name: str) -> List[HierarchicalChunk]:
        """Parse a notebook into hierarchical chunks.
        
        Args:
            notebook_path: Path to the notebook
            tool_name: Name of the tool (e.g., "scanpy", "anndata")
            repo_name: Repository name (e.g., "scverse/scanpy-tutorials")
            
        Returns:
            List of hierarchical chunks
        """
        logger.debug(f"Parsing notebook hierarchically: {notebook_path}")
        
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook_data = json.load(f)
        except Exception as e:
            logger.debug(f"Failed to load notebook {notebook_path}: {e}")
            return []
        
        cells = notebook_data.get('cells', [])
        # Create unique document ID using full notebook path for absolute uniqueness
        # Use relative path from repository root to ensure uniqueness across repos
        relative_path = str(notebook_path).replace('/', '_').replace('\\', '_')
        document_id = re.sub(r'[^\w\-]', '_', relative_path)
        # Limit length and ensure uniqueness with path info
        if len(document_id) > 100:
            import hashlib
            path_hash = hashlib.md5(str(notebook_path).encode()).hexdigest()[:8]
            document_id = f"{notebook_path.stem}_{path_hash}"
            document_id = re.sub(r'[^\w\-]', '_', document_id)
        
        # Extract hierarchical structure
        chunks = []
        
        # 1. Create document-level chunk
        document_chunk = self._create_document_chunk(
            cells, document_id, tool_name, repo_name
        )
        chunks.append(document_chunk)
        
        # 2. Parse sections and code cells with hierarchical context
        sections = self._parse_sections(cells)
        
        # Extract document context for embedding in child chunks
        document_context = f"{document_chunk.content[:200]}..."  # First 200 chars as context
        
        section_index = 0
        for section in sections:
            # Create section chunk with document context
            section_chunk = self._create_section_chunk(
                section, document_id, section_index, tool_name, repo_name, document_context
            )
            chunks.append(section_chunk)
            
            # Extract section context for embedding in code cell chunks
            section_context = f"{section['title']}"
            
            # Create code cell chunks within section with section context
            code_cell_index = 0
            for code_cell in section['code_cells']:
                code_chunk = self._create_code_cell_chunk(
                    code_cell, section_chunk.chunk_id, code_cell_index, tool_name, repo_name, section_context
                )
                chunks.append(code_chunk)
                code_cell_index += 1
            
            section_index += 1
        
        logger.debug(f"Created {len(chunks)} hierarchical chunks from {notebook_path}")
        return chunks
    
    def _create_document_chunk(self, cells: List[Dict[str, Any]], document_id: str, 
                             tool_name: str, repo_name: str) -> HierarchicalChunk:
        """Create a document-level chunk summarizing the entire notebook."""
        
        # Extract title from first markdown cell or filename
        title = document_id.replace('_', ' ').replace('-', ' ').title()
        for cell in cells:
            if cell.get('cell_type') == 'markdown':
                source = ''.join(cell.get('source', []))
                # Look for title in first markdown cell
                lines = source.split('\n')
                for line in lines:
                    if line.strip().startswith('#'):
                        title = line.strip('#').strip()
                        break
                break
        
        # Create document summary
        content_parts = [f"Tutorial: {title}"]
        content_parts.append(f"Tool: {tool_name}")
        
        # Extract workflow overview from first few markdown cells
        markdown_content = []
        for cell in cells[:10]:  # Look at first 10 cells
            if cell.get('cell_type') == 'markdown':
                source = ''.join(cell.get('source', []))
                if source.strip() and not source.strip().startswith('#'):
                    markdown_content.append(source.strip())
        
        if markdown_content:
            overview = ' '.join(markdown_content)[:400]  # Limit overview length
            content_parts.append(f"Overview: {overview}")
        
        # Identify key computational steps
        code_patterns = self._identify_key_patterns(cells)
        if code_patterns:
            content_parts.append(f"Key operations: {', '.join(code_patterns)}")
        
        content = '\n\n'.join(content_parts)
        
        return HierarchicalChunk(
            content=content,
            level="document",
            chunk_id=document_id,
            parent_id=None,
            chunk_index=0,
            metadata={
                "tool": tool_name,
                "repo": repo_name
            }
        )
    
    def _parse_sections(self, cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse notebook cells into sections based on headers."""
        sections = []
        current_section = None
        
        for cell in cells:
            cell_type = cell.get('cell_type', '')
            source = ''.join(cell.get('source', []))
            
            if cell_type == 'markdown':
                # Check if this is a section header
                header_match = self._is_section_header(source)
                if header_match:
                    # Save previous section
                    if current_section:
                        sections.append(current_section)
                    
                    # Start new section
                    current_section = {
                        'title': header_match,
                        'markdown_cells': [source],
                        'code_cells': []
                    }
                elif current_section:
                    current_section['markdown_cells'].append(source)
                    
            elif cell_type == 'code' and current_section:
                # Add code cell with surrounding context
                current_section['code_cells'].append({
                    'source': source,
                    'context': current_section['markdown_cells'][-1] if current_section['markdown_cells'] else ""
                })
        
        # Add final section
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _is_section_header(self, markdown_source: str) -> Optional[str]:
        """Check if markdown source is a section header."""
        for pattern in self.section_header_patterns:
            match = re.search(pattern, markdown_source, re.MULTILINE)
            if match:
                return match.group(1).strip()
        return None
    
    def _create_section_chunk(self, section: Dict[str, Any], document_id: str, 
                            section_index: int, tool_name: str, repo_name: str,
                            document_context: str = "") -> HierarchicalChunk:
        """Create a section-level chunk with hierarchical context embedded."""
        
        section_id = f"{document_id}_section_{section_index}"
        title = section['title']
        
        # Include hierarchical context in the content itself (contextual embeddings)
        content_parts = []
        
        # Add document context for hierarchical awareness
        if document_context:
            content_parts.append(f"Document Context: {document_context}")
        
        content_parts.append(f"Section: {title}")
        
        # Add section description from markdown
        markdown_content = []
        for md_cell in section['markdown_cells']:
            # Skip header lines
            lines = md_cell.split('\n')
            non_header_lines = [line for line in lines if not line.strip().startswith('#')]
            if non_header_lines:
                markdown_content.append('\n'.join(non_header_lines).strip())
        
        if markdown_content:
            description = ' '.join(markdown_content)[:300]  # Limit description
            content_parts.append(f"Description: {description}")
        
        # Add summary of code operations
        code_summaries = []
        for code_cell in section['code_cells']:
            code_source = code_cell['source']
            # Extract key function calls
            func_calls = re.findall(r'(\w+\.\w+\()', code_source)
            if func_calls:
                code_summaries.extend(func_calls[:3])  # Top 3 function calls
        
        if code_summaries:
            content_parts.append(f"Operations: {', '.join(set(code_summaries))}")
        
        content = '\n\n'.join(content_parts)
        
        return HierarchicalChunk(
            content=content,
            level="section",
            chunk_id=section_id,
            parent_id=document_id,
            chunk_index=section_index,
            metadata={
                "tool": tool_name,
                "repo": repo_name
            }
        )
    
    def _create_code_cell_chunk(self, code_cell: Dict[str, Any], section_id: str, 
                              code_index: int, tool_name: str, repo_name: str,
                              section_context: str = "") -> HierarchicalChunk:
        """Create a code cell chunk with hierarchical context embedded."""
        
        chunk_id = f"{section_id}_code_{code_index}"
        code_source = code_cell['source']
        context = code_cell['context']
        
        # Include hierarchical context in the content itself (contextual embeddings)
        content_parts = []
        
        # Add section context for hierarchical awareness
        if section_context:
            content_parts.append(f"Section Context: {section_context}")
        
        # Add immediate context if available
        if context:
            # Clean context (remove headers, limit length)
            clean_context = re.sub(r'^#+\s+.*$', '', context, flags=re.MULTILINE)
            clean_context = clean_context.strip()
            if clean_context:
                content_parts.append(f"Context: {clean_context[:150]}")
        
        # Add the code
        if code_source.strip():
            content_parts.append(f"Code:\n{code_source.strip()}")
        
        content = '\n\n'.join(content_parts)
        
        return HierarchicalChunk(
            content=content,
            level="code_cell",
            chunk_id=chunk_id,
            parent_id=section_id,
            chunk_index=code_index,
            metadata={
                "tool": tool_name,
                "repo": repo_name
            }
        )
    
    def _identify_key_patterns(self, cells: List[Dict[str, Any]]) -> List[str]:
        """Identify key computational patterns in the notebook."""
        patterns = []
        
        # Common bioinformatics patterns
        pattern_mapping = {
            r'\.filter_': 'filtering',
            r'\.normalize_': 'normalization',
            r'\.log1p': 'log transformation',
            r'\.pca': 'PCA',
            r'\.neighbors': 'neighborhood graph',
            r'\.leiden': 'leiden clustering',
            r'\.umap': 'UMAP',
            r'\.rank_genes_groups': 'differential expression',
            r'\.plot': 'visualization',
            r'\.read_': 'data loading'
        }
        
        for cell in cells:
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                for pattern, description in pattern_mapping.items():
                    if re.search(pattern, source) and description not in patterns:
                        patterns.append(description)
        
        return patterns[:5]  # Top 5 patterns
    
    def chunks_to_documents(self, chunks: List[HierarchicalChunk]) -> List[Dict[str, Any]]:
        """Convert hierarchical chunks to ChromaDB document format."""
        documents = []
        
        for chunk in chunks:
            metadata = {
                # Essential metadata only
                "tool": chunk.metadata["tool"],
                "repo": chunk.metadata["repo"],
                "doc_type": "workflow",
                # Hierarchical structure metadata
                "chunk_level": chunk.level,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index
            }
            
            # Only add parent_id if it's not None (ChromaDB can't handle None values)
            if chunk.parent_id is not None:
                metadata["parent_id"] = chunk.parent_id
            
            doc = {
                "content": chunk.content,
                "metadata": metadata
            }
            documents.append(doc)
        
        return documents