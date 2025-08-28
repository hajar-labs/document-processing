# Document structure analysis
# Document structure analysis and hierarchy detection
"""
Advanced document structure analyzer for hierarchical content organization.
Detects sections, subsections, lists, and logical document flow.
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class DocumentSection:
    """Represents a document section with hierarchical information."""
    id: str
    title: str
    content: str
    level: int  # Hierarchy level (1 = top level)
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    section_type: str = "content"  # content, header, footer, toc, etc.
    start_position: int = 0
    end_position: int = 0
    language: str = "unknown"
    keywords: List[str] = None
    summary: str = ""
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
        if self.keywords is None:
            self.keywords = []


@dataclass
class DocumentStructure:
    """Complete document structure representation."""
    sections: List[DocumentSection]
    hierarchy: Dict[str, List[str]]  # parent_id -> children_ids
    reading_order: List[str]  # Section IDs in reading order
    table_of_contents: List[Dict[str, Any]]
    document_type: str
    max_depth: int
    total_sections: int
    structure_quality: float
    language_distribution: Dict[str, float]


class HierarchicalAnalyzer:
    """Analyzes document hierarchical structure and organization."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Heading patterns for different languages
        self.heading_patterns = {
            'french': [
                r'^(TITRE|CHAPITRE|SECTION|ARTICLE|ANNEXE)\s+([IVXLCDM]+|\d+)',
                r'^(\d+\.(?:\d+\.)*)\s+(.+)',  # Numbered headings
                r'^([A-Z][^.!?]*):?\s*$',  # All caps headings
                r'^(Introduction|Conclusion|Résumé|Préambule)',
            ],
            'arabic': [
                r'^(العنوان|الفصل|القسم|المادة|الملحق)\s+(\d+)',
                r'^(\d+\.(?:\d+\.)*)\s+(.+)',  # Numbered headings
                r'^(المقدمة|الخاتمة|الملخص|التمهيد)',
            ],
            'universal': [
                r'^(\d+\.(?:\d+\.)*)\s+(.+)',  # Numbered sections
                r'^([A-Z\u0600-\u06FF][\w\s]{2,50}):?\s*$',  # Title case
            ]
        }
        
        # List patterns
        self.list_patterns = [
            r'^\s*[-•▪▫◦‣⁃*+]\s+(.+)',  # Bullet lists
            r'^\s*(\d+[.)]\s+.+)',  # Numbered lists
            r'^\s*([a-zA-Z][.)]\s+.+)',  # Letter lists
            r'^\s*([\u0600-\u06FF]+[.)]\s+.+)',  # Arabic lists
        ]
        
        # Special section indicators
        self.special_sections = {
            'toc': [
                r'table\s+des?\s+matières?',
                r'sommaire',
                r'index',
                r'فهرس\s+المحتويات?',
                r'المحتويات'
            ],
            'bibliography': [
                r'bibliographie',
                r'références?',
                r'sources?',
                r'المراجع',
                r'المصادر'
            ],
            'appendix': [
                r'annexe\w*',
                r'appendice\w*',
                r'الملاحق?',
                r'المرفقات?'
            ]
        }


class StructureAnalyzer:
    """Main document structure analyzer."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.hierarchical_analyzer = HierarchicalAnalyzer(config)
        
        # Analysis parameters
        self.min_section_length = self.config.get('min_section_length', 50)
        self.max_section_length = self.config.get('max_section_length', 5000)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.3)
        self.enable_semantic_grouping = self.config.get('enable_semantic_grouping', True)
    
    def analyze_document_structure(self, text: str, 
                                  language: str = None) -> DocumentStructure:
        """
        Comprehensive document structure analysis.
        
        Args:
            text: Document text content
            language: Primary document language
            
        Returns:
            DocumentStructure object with complete analysis
        """
        if not text or len(text.strip()) < 100:
            return self._create_empty_structure()
        
        # Step 1: Detect sections and headings
        sections = self._detect_sections(text, language)
        
        # Step 2: Build hierarchy
        hierarchy = self._build_hierarchy(sections)
        
        # Step 3: Determine reading order
        reading_order = self._determine_reading_order(sections, hierarchy)
        
        # Step 4: Generate table of contents
        toc = self._generate_table_of_contents(sections, hierarchy)
        
        # Step 5: Analyze structure quality
        structure_quality = self._assess_structure_quality(sections, hierarchy)
        
        # Step 6: Language distribution analysis
        language_dist = self._analyze_language_distribution(sections)
        
        # Step 7: Document type classification
        document_type = self._classify_document_type(sections, text)
        
        return DocumentStructure(
            sections=sections,
            hierarchy=hierarchy,
            reading_order=reading_order,
            table_of_contents=toc,
            document_type=document_type,
            max_depth=self._calculate_max_depth(hierarchy),
            total_sections=len(sections),
            structure_quality=structure_quality,
            language_distribution=language_dist
        )
    
    def _detect_sections(self, text: str, language: str = None) -> List[DocumentSection]:
        """Detect document sections and headings."""
        sections = []
        lines = text.split('\n')
        
        current_section_content = []
        current_section_start = 0
        section_counter = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            if not line_stripped:
                current_section_content.append('')
                continue
            
            # Check if line is a heading
            heading_info = self._is_heading(line_stripped, language)
            
            if heading_info['is_heading']:
                # Save previous section if exists
                if current_section_content:
                    content = '\n'.join(current_section_content).strip()
                    if len(content) >= self.min_section_length:
                        section = DocumentSection(
                            id=f"section_{section_counter}",
                            title=f"Section {section_counter}",
                            content=content,
                            level=1,
                            start_position=current_section_start,
                            end_position=i,
                            section_type='content'
                        )
                        sections.append(section)
                        section_counter += 1
                
                # Start new section with this heading
                section = DocumentSection(
                    id=f"section_{section_counter}",
                    title=heading_info['title'],
                    content=line_stripped,
                    level=heading_info['level'],
                    start_position=i,
                    end_position=i + 1,
                    section_type='heading'
                )
                sections.append(section)
                section_counter += 1
                
                current_section_content = []
                current_section_start = i + 1
            else:
                current_section_content.append(line)
        
        # Handle final section
        if current_section_content:
            content = '\n'.join(current_section_content).strip()
            if len(content) >= self.min_section_length:
                section = DocumentSection(
                    id=f"section_{section_counter}",
                    title=f"Final Section",
                    content=content,
                    level=1,
                    start_position=current_section_start,
                    end_position=len(lines),
                    section_type='content'
                )
                sections.append(section)
        
        # Post-process sections
        sections = self._post_process_sections(sections, language)
        
        return sections
    
    def _is_heading(self, line: str, language: str = None) -> Dict[str, Any]:
        """Determine if a line is a heading and extract information."""
        result = {
            'is_heading': False,
            'title': line,
            'level': 1,
            'type': 'unknown'
        }
        
        # Check against language-specific patterns
        patterns_to_check = []
        
        if language in self.hierarchical_analyzer.heading_patterns:
            patterns_to_check.extend(self.hierarchical_analyzer.heading_patterns[language])
        
        patterns_to_check.extend(self.hierarchical_analyzer.heading_patterns['universal'])
        
        for pattern in patterns_to_check:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                result['is_heading'] = True
                result['title'] = line
                
                # Determine level based on pattern
                if '\\d+\\.' in pattern:
                    # Count dots to determine level
                    dots = line.count('.')
                    result['level'] = min(dots + 1, 6)
                elif any(word in pattern.upper() for word in ['TITRE', 'CHAPTER']):
                    result['level'] = 1
                elif any(word in pattern.upper() for word in ['SECTION', 'ARTICLE']):
                    result['level'] = 2
                else:
                    result['level'] = 1
                
                break
        
        # Additional heuristics
        if not result['is_heading']:
            # Check for all caps (might be heading)
            if (line.isupper() and 
                10 <= len(line) <= 100 and 
                not line.endswith('.') and
                len(line.split()) <= 10):
                result['is_heading'] = True
                result['level'] = 2
                result['type'] = 'caps_heading'
            
            # Check for numbered sections
            elif re.match(r'^\d+\.\s+', line):
                result['is_heading'] = True
                result['level'] = 2
                result['type'] = 'numbered'
        
        return result
    
    def _post_process_sections(self, sections: List[DocumentSection], 
                              language: str = None) -> List[DocumentSection]:
        """Post-process sections to improve quality and add metadata."""
        processed_sections = []
        
        for section in sections:
            # Detect section language
            section.language = self._detect_section_language(section.content)
            
            # Extract keywords
            section.keywords = self._extract_section_keywords(section.content, language)
            
            # Generate summary for long sections
            if len(section.content) > 500:
                section.summary = self._generate_section_summary(section.content)
            
            # Classify special sections
            section.section_type = self._classify_section_type(section.title, section.content)
            
            processed_sections.append(section)
        
        return processed_sections
    
    def _detect_section_language(self, text: str) -> str:
        """Detect language of a section."""
        if not text:
            return "unknown"
        
        # Count Arabic characters
        arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text))
        
        # Count French-specific characters
        french_chars = len(re.findall(r'[àâäçéèêëïîôùûüÿñæœÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑÆŒ]', text))
        
        # Count general Latin characters
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        
        total_chars = len(re.findall(r'\S', text))
        
        if total_chars == 0:
            return "unknown"
        
        arabic_ratio = arabic_chars / total_chars
        french_ratio = (french_chars + latin_chars) / total_chars
        
        if arabic_ratio > 0.1:
            if french_ratio > 0.1:
                return "mixed"
            return "arabic"
        elif french_chars > 0 or french_ratio > 0.7:
            return "french"
        else:
            return "unknown"
    
    def _extract_section_keywords(self, text: str, language: str = None, 
                                 max_keywords: int = 10) -> List[str]:
        """Extract keywords from section content."""
        if not text or len(text) < 50:
            return []
        
        # Simple keyword extraction based on frequency
        words = re.findall(r'\b[a-zA-ZàâäçéèêëïîôùûüÿñæœÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑÆŒ\u0600-\u06FF]{4,}\b', text)
        
        if not words:
            return []
        
        # Count word frequencies
        word_freq = Counter(word.lower() for word in words)
        
        # Filter common stop words
        stop_words = {'dans', 'avec', 'pour', 'sont', 'cette', 'plus', 'tout', 'comme',
                     'على', 'في', 'من', 'إلى', 'هذا', 'هذه', 'التي', 'الذي'}
        
        keywords = []
        for word, freq in word_freq.most_common(max_keywords * 2):
            if word not in stop_words and len(word) > 3 and freq >= 2:
                keywords.append(word)
                if len(keywords) >= max_keywords:
                    break
        
        return keywords
    
    def _generate_section_summary(self, text: str, max_length: int = 200) -> str:
        """Generate a brief summary of section content."""
        if not text or len(text) <= max_length:
            return text[:max_length]
        
        # Simple extractive summarization
        sentences = re.split(r'[.!?؟۔]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s) > 20]
        
        if not sentences:
            return text[:max_length]
        
        # Score sentences by position and length
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            # Prefer sentences at the beginning
            position_score = 1.0 / (i + 1)
            # Prefer medium-length sentences
            length_score = min(len(sentence) / 100, 1.0)
            total_score = position_score * 0.7 + length_score * 0.3
            sentence_scores.append((sentence, total_score))
        
        # Sort by score and take top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        summary = ""
        for sentence, score in sentence_scores:
            if len(summary + sentence) <= max_length:
                summary += sentence + ". "
            else:
                break
        
        return summary.strip()
    
    def _classify_section_type(self, title: str, content: str) -> str:
        """Classify section type based on title and content."""
        title_lower = title.lower()
        content_lower = content.lower()
        
        # Check for special sections
        for section_type, patterns in self.hierarchical_analyzer.special_sections.items():
            for pattern in patterns:
                if re.search(pattern, title_lower, re.IGNORECASE) or \
                   re.search(pattern, content_lower[:200], re.IGNORECASE):
                    return section_type
        
        # Content-based classification
        if 'introduction' in title_lower or 'مقدمة' in title_lower:
            return 'introduction'
        elif 'conclusion' in title_lower or 'خاتمة' in title_lower:
            return 'conclusion'
        elif re.search(r'chapitre\s+\d+|فصل\s+\d+', title_lower):
            return 'chapter'
        elif re.search(r'section\s+\d+|قسم\s+\d+', title_lower):
            return 'section'
        elif re.search(r'article\s+\d+|مادة\s+\d+', title_lower):
            return 'article'
        elif len(content.split()) > 500:
            return 'main_content'
        else:
            return 'content'
    
    def _build_hierarchy(self, sections: List[DocumentSection]) -> Dict[str, List[str]]:
        """Build hierarchical relationships between sections."""
        hierarchy = defaultdict(list)
        
        if not sections:
            return dict(hierarchy)
        
        # Stack to track parent sections at each level
        parent_stack = [None]  # Stack of (section_id, level)
        
        for section in sections:
            current_level = section.level
            
            # Find appropriate parent
            while len(parent_stack) > current_level:
                parent_stack.pop()
            
            while len(parent_stack) < current_level:
                parent_stack.append(None)
            
            # Set parent relationship
            if current_level > 1 and len(parent_stack) > 1:
                parent_id = parent_stack[-2]
                if parent_id:
                    section.parent_id = parent_id
                    hierarchy[parent_id].append(section.id)
            
            # Update stack
            if len(parent_stack) == current_level:
                parent_stack.append(section.id)
            else:
                parent_stack[current_level - 1] = section.id
        
        return dict(hierarchy)
    
    def _determine_reading_order(self, sections: List[DocumentSection], 
                                hierarchy: Dict[str, List[str]]) -> List[str]:
        """Determine logical reading order of sections."""
        if not sections:
            return []
        
        # Create a graph of section relationships
        G = nx.DiGraph()
        
        # Add nodes
        for section in sections:
            G.add_node(section.id, 
                      level=section.level, 
                      position=section.start_position,
                      type=section.section_type)
        
        # Add edges based on hierarchy and position
        for parent_id, children_ids in hierarchy.items():
            for child_id in children_ids:
                G.add_edge(parent_id, child_id, weight=1)
        
        # Add sequential edges based on position
        sections_by_position = sorted(sections, key=lambda s: s.start_position)
        for i in range(len(sections_by_position) - 1):
            current = sections_by_position[i]
            next_section = sections_by_position[i + 1]
            
            # Add edge if they're at similar levels or sequential
            if abs(current.level - next_section.level) <= 1:
                G.add_edge(current.id, next_section.id, weight=0.5)
        
        # Perform topological sort to get reading order
        try:
            reading_order = list(nx.topological_sort(G))
        except nx.NetworkXError:
            # Fallback: sort by position
            reading_order = [s.id for s in sections_by_position]
        
        return reading_order
    
    def _generate_table_of_contents(self, sections: List[DocumentSection], 
                                   hierarchy: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Generate table of contents from document structure."""
        toc = []
        
        # Find top-level sections (no parent)
        top_level_sections = [s for s in sections if not s.parent_id and s.section_type in ['heading', 'chapter', 'section']]
        
        def build_toc_entry(section: DocumentSection) -> Dict[str, Any]:
            entry = {
                'id': section.id,
                'title': section.title,
                'level': section.level,
                'page': section.start_position // 50 + 1,  # Rough page estimation
                'children': []
            }
            
            # Add children recursively
            if section.id in hierarchy:
                for child_id in hierarchy[section.id]:
                    child_section = next((s for s in sections if s.id == child_id), None)
                    if child_section and child_section.section_type in ['heading', 'chapter', 'section']:
                        entry['children'].append(build_toc_entry(child_section))
            
            return entry
        
        # Build TOC recursively
        for section in sorted(top_level_sections, key=lambda s: s.start_position):
            toc.append(build_toc_entry(section))
        
        return toc
    
    def _assess_structure_quality(self, sections: List[DocumentSection], 
                                 hierarchy: Dict[str, List[str]]) -> float:
        """Assess the quality of document structure."""
        if not sections:
            return 0.0
        
        quality_factors = []
        
        # Factor 1: Presence of headings
        heading_sections = [s for s in sections if s.section_type in ['heading', 'chapter', 'section']]
        heading_ratio = len(heading_sections) / len(sections)
        quality_factors.append(min(heading_ratio * 2, 1.0))
        
        # Factor 2: Hierarchical consistency
        levels = [s.level for s in sections]
        if levels:
            level_consistency = 1.0 - (np.std(levels) / np.mean(levels)) if np.mean(levels) > 0 else 0
            quality_factors.append(max(0, level_consistency))
        
        # Factor 3: Section length consistency
        content_sections = [s for s in sections if s.section_type == 'content']
        if content_sections:
            lengths = [len(s.content) for s in content_sections]
            avg_length = np.mean(lengths)
            length_variance = np.var(lengths)
            length_consistency = max(0, 1.0 - (length_variance / (avg_length ** 2)))
            quality_factors.append(length_consistency)
        
        # Factor 4: Presence of special sections
        special_sections = [s for s in sections if s.section_type in ['introduction', 'conclusion', 'toc']]
        special_bonus = min(len(special_sections) * 0.2, 0.4)
        quality_factors.append(special_bonus)
        
        return np.mean(quality_factors) if quality_factors else 0.0
    
    def _analyze_language_distribution(self, sections: List[DocumentSection]) -> Dict[str, float]:
        """Analyze language distribution across sections."""
        if not sections:
            return {}
        
        language_counts = Counter(s.language for s in sections)
        total_sections = len(sections)
        
        return {lang: count / total_sections for lang, count in language_counts.items()}
    
    def _classify_document_type(self, sections: List[DocumentSection], 
                               full_text: str) -> str:
        """Classify document type based on structure analysis."""
        # Analyze section types
        section_types = Counter(s.section_type for s in sections)
        
        # Check for specific patterns
        has_articles = any('article' in s.section_type for s in sections)
        has_chapters = any('chapter' in s.section_type for s in sections)
        has_toc = any('toc' in s.section_type for s in sections)
        
        # Classification logic
        if has_articles and len(sections) > 10:
            return 'regulation'
        elif has_chapters or (has_toc and len(sections) > 15):
            return 'book'
        elif 'bibliography' in section_types:
            return 'report'
        elif len(sections) < 5:
            return 'memo'
        else:
            return 'document'
    
    def _calculate_max_depth(self, hierarchy: Dict[str, List[str]]) -> int:
        """Calculate maximum hierarchy depth."""
        if not hierarchy:
            return 1
        
        def get_depth(node_id: str, current_depth: int = 1) -> int:
            if node_id not in hierarchy or not hierarchy[node_id]:
                return current_depth
            
            max_child_depth = current_depth
            for child_id in hierarchy[node_id]:
                child_depth = get_depth(child_id, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
            
            return max_child_depth
        
        # Find root nodes (nodes that are not children of other nodes)
        all_children = set()
        for children in hierarchy.values():
            all_children.update(children)
        
        root_nodes = set(hierarchy.keys()) - all_children
        
        if not root_nodes:
            return 1
        
        max_depth = 1
        for root in root_nodes:
            depth = get_depth(root)
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _create_empty_structure(self) -> DocumentStructure:
        """Create empty document structure for invalid input."""
        return DocumentStructure(
            sections=[],
            hierarchy={},
            reading_order=[],
            table_of_contents=[],
            document_type='unknown',
            max_depth=0,
            total_sections=0,
            structure_quality=0.0,
            language_distribution={}
        )
    
    def export_structure(self, structure: DocumentStructure, 
                        format: str = 'json') -> str:
        """Export document structure in specified format."""
        if format.lower() == 'json':
            import json
            from dataclasses import asdict
            
            # Convert to dictionary
            structure_dict = asdict(structure)
            
            return json.dumps(structure_dict, indent=2, ensure_ascii=False)
        
        elif format.lower() == 'markdown':
            return self._to_markdown(structure)
        
        elif format.lower() == 'html':
            return self._to_html(structure)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _to_markdown(self, structure: DocumentStructure) -> str:
        """Convert structure to Markdown format."""
        md_content = []
        
        md_content.append(f"# Document Structure Analysis")
        md_content.append(f"- **Document Type**: {structure.document_type}")
        md_content.append(f"- **Total Sections**: {structure.total_sections}")
        md_content.append(f"- **Max Depth**: {structure.max_depth}")
        md_content.append(f"- **Structure Quality**: {structure.structure_quality:.2f}")
        md_content.append("")
        
        # Table of Contents
        if structure.table_of_contents:
            md_content.append("## Table of Contents")
            
            def format_toc_entry(entry: Dict[str, Any], indent: int = 0) -> str:
                prefix = "  " * indent + "- "
                result = f"{prefix}[{entry['title']}](#section-{entry['id']})"
                
                for child in entry.get('children', []):
                    result += "\n" + format_toc_entry(child, indent + 1)
                
                return result
            
            for entry in structure.table_of_contents:
                md_content.append(format_toc_entry(entry))
            
            md_content.append("")
        
        # Sections
        if structure.sections:
            md_content.append("## Sections")
            
            for section in structure.sections:
                md_content.append(f"### {section.title} {{#section-{section.id}}}")
                md_content.append(f"- **Type**: {section.section_type}")
                md_content.append(f"- **Level**: {section.level}")
                md_content.append(f"- **Language**: {section.language}")
                
                if section.keywords:
                    md_content.append(f"- **Keywords**: {', '.join(section.keywords)}")
                
                if section.summary:
                    md_content.append(f"- **Summary**: {section.summary}")
                
                md_content.append("")
        
        return "\n".join(md_content)
    
    def _to_html(self, structure: DocumentStructure) -> str:
        """Convert structure to HTML format."""
        html_content = []
        
        html_content.append("<!DOCTYPE html>")
        html_content.append("<html><head><title>Document Structure</title></head><body>")
        html_content.append("<h1>Document Structure Analysis</h1>")
        
        # Summary
        html_content.append("<div class='summary'>")
        html_content.append(f"<p><strong>Document Type:</strong> {structure.document_type}</p>")
        html_content.append(f"<p><strong>Total Sections:</strong> {structure.total_sections}</p>")
        html_content.append(f"<p><strong>Max Depth:</strong> {structure.max_depth}</p>")
        html_content.append(f"<p><strong>Structure Quality:</strong> {structure.structure_quality:.2f}</p>")
        html_content.append("</div>")
        
        # Table of Contents
        if structure.table_of_contents:
            html_content.append("<h2>Table of Contents</h2>")
            html_content.append("<ul>")
            
            def format_toc_html(entry: Dict[str, Any]) -> str:
                result = f"<li><a href='#section-{entry['id']}'>{entry['title']}</a>"
                
                if entry.get('children'):
                    result += "<ul>"
                    for child in entry['children']:
                        result += format_toc_html(child)
                    result += "</ul>"
                
                result += "</li>"
                return result
            
            for entry in structure.table_of_contents:
                html_content.append(format_toc_html(entry))
            
            html_content.append("</ul>")
        
        html_content.append("</body></html>")
        
        return "\n".join(html_content)