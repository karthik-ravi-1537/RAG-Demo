"""Entity and relationship extraction for Graph-RAG implementation."""

import re
from typing import List, Dict, Any, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter

from core.data_models import Entity, Relationship, Chunk, ProcessedDocument
from core.exceptions import ProcessingError
from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractionResult:
    """Result of entity and relationship extraction."""
    entities: List[Entity]
    relationships: List[Relationship]
    extraction_stats: Dict[str, Any]


class EntityExtractor:
    """
    Entity and relationship extractor using rule-based and pattern-based approaches.
    
    This implementation uses NLTK and regex patterns for extraction, making it
    dependency-light while still effective for demonstration purposes.
    """
    
    def __init__(self):
        """Initialize the entity extractor."""
        self.entity_patterns = self._build_entity_patterns()
        self.relationship_patterns = self._build_relationship_patterns()
        
        # Entity type mappings
        self.entity_types = {
            'PERSON': ['person', 'people', 'individual', 'researcher', 'scientist', 'author'],
            'ORGANIZATION': ['company', 'corporation', 'university', 'institution', 'organization'],
            'TECHNOLOGY': ['algorithm', 'model', 'system', 'framework', 'method', 'technique'],
            'CONCEPT': ['concept', 'idea', 'theory', 'principle', 'approach'],
            'LOCATION': ['country', 'city', 'place', 'location', 'region'],
            'PRODUCT': ['product', 'software', 'application', 'tool', 'platform']
        }
        
        # Common relationship indicators
        self.relationship_indicators = {
            'DEVELOPS': ['develops', 'created', 'built', 'designed', 'invented'],
            'USES': ['uses', 'utilizes', 'employs', 'applies', 'implements'],
            'PART_OF': ['part of', 'component of', 'belongs to', 'within', 'inside'],
            'RELATED_TO': ['related to', 'associated with', 'connected to', 'linked to'],
            'IMPROVES': ['improves', 'enhances', 'optimizes', 'increases', 'boosts'],
            'ENABLES': ['enables', 'allows', 'facilitates', 'supports', 'helps']
        }
        
        logger.info("Initialized EntityExtractor with rule-based patterns")
    
    def extract_from_chunk(self, chunk: Chunk) -> ExtractionResult:
        """Extract entities and relationships from a single chunk."""
        try:
            # Extract entities
            entities = self._extract_entities(chunk.content, chunk.id)
            
            # Extract relationships
            relationships = self._extract_relationships(chunk.content, entities, chunk.id)
            
            # Calculate extraction statistics
            stats = self._calculate_stats(entities, relationships, chunk.content)
            
            logger.debug(f"Extracted {len(entities)} entities and {len(relationships)} relationships from chunk {chunk.id}")
            
            return ExtractionResult(
                entities=entities,
                relationships=relationships,
                extraction_stats=stats
            )
            
        except Exception as e:
            logger.error(f"Failed to extract from chunk {chunk.id}: {str(e)}")
            raise ProcessingError(f"Entity extraction failed: {str(e)}")
    
    def extract_from_document(self, document: ProcessedDocument, chunks: List[Chunk]) -> ExtractionResult:
        """Extract entities and relationships from a document and its chunks."""
        try:
            all_entities = []
            all_relationships = []
            combined_stats = defaultdict(int)
            entity_type_counts = defaultdict(int)
            relationship_type_counts = defaultdict(int)
            
            # Extract from each chunk
            for chunk in chunks:
                result = self.extract_from_chunk(chunk)
                all_entities.extend(result.entities)
                all_relationships.extend(result.relationships)
                
                # Combine statistics
                for key, value in result.extraction_stats.items():
                    if key == 'entity_types':
                        # Combine entity type counts
                        for entity_type, count in value.items():
                            entity_type_counts[entity_type] += count
                    elif key == 'relationship_types':
                        # Combine relationship type counts
                        for rel_type, count in value.items():
                            relationship_type_counts[rel_type] += count
                    else:
                        # Add numeric values
                        combined_stats[key] += value
            
            # Merge duplicate entities
            merged_entities = self._merge_entities(all_entities)
            
            # Filter and score relationships
            filtered_relationships = self._filter_relationships(all_relationships, merged_entities)
            
            # Update statistics
            combined_stats['merged_entities'] = len(merged_entities)
            combined_stats['filtered_relationships'] = len(filtered_relationships)
            combined_stats['document_id'] = document.id
            combined_stats['entity_types'] = dict(entity_type_counts)
            combined_stats['relationship_types'] = dict(relationship_type_counts)
            
            logger.info(f"Document {document.id}: {len(merged_entities)} entities, {len(filtered_relationships)} relationships")
            
            return ExtractionResult(
                entities=merged_entities,
                relationships=filtered_relationships,
                extraction_stats=dict(combined_stats)
            )
            
        except Exception as e:
            logger.error(f"Failed to extract from document {document.id}: {str(e)}")
            raise ProcessingError(f"Document extraction failed: {str(e)}")
    
    def _extract_entities(self, text: str, chunk_id: str) -> List[Entity]:
        """Extract entities from text using pattern matching."""
        entities = []
        text_lower = text.lower()
        
        # Extract different types of entities
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_text = match.group().strip()
                    if len(entity_text) > 2:  # Filter very short matches
                        entity = Entity(
                            name=entity_text,
                            entity_type=entity_type,
                            mentions=[entity_text],
                            chunk_ids=[chunk_id],
                            confidence=self._calculate_entity_confidence(entity_text, entity_type, text),
                            properties={
                                'position': match.span(),
                                'context': text[max(0, match.start()-50):match.end()+50]
                            }
                        )
                        entities.append(entity)
        
        # Extract capitalized words as potential entities
        capitalized_entities = self._extract_capitalized_entities(text, chunk_id)
        entities.extend(capitalized_entities)
        
        return entities
    
    def _extract_capitalized_entities(self, text: str, chunk_id: str) -> List[Entity]:
        """Extract capitalized words/phrases as potential entities."""
        entities = []
        
        # Pattern for capitalized words (potential proper nouns)
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.finditer(pattern, text)
        
        for match in matches:
            entity_text = match.group().strip()
            
            # Skip common words and short entities
            if (len(entity_text) > 3 and 
                entity_text not in ['The', 'This', 'That', 'These', 'Those', 'When', 'Where', 'What', 'How'] and
                not entity_text.startswith(('Figure', 'Table', 'Section', 'Chapter'))):
                
                # Determine entity type based on context
                entity_type = self._classify_entity_type(entity_text, text)
                
                entity = Entity(
                    name=entity_text,
                    entity_type=entity_type,
                    mentions=[entity_text],
                    chunk_ids=[chunk_id],
                    confidence=0.6,  # Lower confidence for capitalized entities
                    properties={
                        'position': match.span(),
                        'extraction_method': 'capitalization',
                        'context': text[max(0, match.start()-30):match.end()+30]
                    }
                )
                entities.append(entity)
        
        return entities
    
    def _extract_relationships(self, text: str, entities: List[Entity], chunk_id: str) -> List[Relationship]:
        """Extract relationships between entities."""
        relationships = []
        
        if len(entities) < 2:
            return relationships
        
        # Create entity position mapping
        entity_positions = {}
        for entity in entities:
            if 'position' in entity.properties:
                entity_positions[entity.name] = entity.properties['position']
        
        # Look for relationships using patterns
        for rel_type, indicators in self.relationship_indicators.items():
            for indicator in indicators:
                pattern = rf'(\w+(?:\s+\w+)*)\s+{re.escape(indicator)}\s+(\w+(?:\s+\w+)*)'
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    source_text = match.group(1).strip()
                    target_text = match.group(2).strip()
                    
                    # Find matching entities
                    source_entity = self._find_matching_entity(source_text, entities)
                    target_entity = self._find_matching_entity(target_text, entities)
                    
                    if source_entity and target_entity and source_entity != target_entity:
                        confidence = self._calculate_relationship_confidence(
                            source_entity, target_entity, rel_type, indicator, text
                        )
                        
                        relationship = Relationship(
                            source_entity=source_entity.name,
                            target_entity=target_entity.name,
                            relation_type=rel_type,
                            confidence=confidence,
                            evidence_chunks=[chunk_id],
                            properties={
                                'indicator': indicator,
                                'context': match.group(),
                                'position': match.span()
                            }
                        )
                        relationships.append(relationship)
        
        # Extract proximity-based relationships
        proximity_relationships = self._extract_proximity_relationships(entities, text, chunk_id)
        relationships.extend(proximity_relationships)
        
        return relationships
    
    def _extract_proximity_relationships(self, entities: List[Entity], text: str, chunk_id: str) -> List[Relationship]:
        """Extract relationships based on entity proximity in text."""
        relationships = []
        
        # Sort entities by position
        positioned_entities = []
        for entity in entities:
            if 'position' in entity.properties:
                positioned_entities.append((entity, entity.properties['position'][0]))
        
        positioned_entities.sort(key=lambda x: x[1])
        
        # Look for entities that appear close to each other
        for i in range(len(positioned_entities)):
            for j in range(i + 1, min(i + 4, len(positioned_entities))):  # Check next 3 entities
                entity1, pos1 = positioned_entities[i]
                entity2, pos2 = positioned_entities[j]
                
                distance = pos2 - pos1
                if distance < 200:  # Within 200 characters
                    # Extract context between entities
                    context = text[pos1:pos2 + 50]
                    
                    # Determine relationship type based on context
                    rel_type = self._infer_relationship_type(entity1, entity2, context)
                    
                    if rel_type:
                        confidence = max(0.3, 0.8 - (distance / 200) * 0.5)  # Distance-based confidence
                        
                        relationship = Relationship(
                            source_entity=entity1.name,
                            target_entity=entity2.name,
                            relation_type=rel_type,
                            confidence=confidence,
                            evidence_chunks=[chunk_id],
                            properties={
                                'extraction_method': 'proximity',
                                'distance': distance,
                                'context': context
                            }
                        )
                        relationships.append(relationship)
        
        return relationships
    
    def _merge_entities(self, entities: List[Entity]) -> List[Entity]:
        """Merge duplicate entities based on name similarity."""
        if not entities:
            return []
        
        # Group entities by normalized name
        entity_groups = defaultdict(list)
        for entity in entities:
            normalized_name = self._normalize_entity_name(entity.name)
            entity_groups[normalized_name].append(entity)
        
        merged_entities = []
        for group in entity_groups.values():
            if len(group) == 1:
                merged_entities.append(group[0])
            else:
                # Merge multiple entities
                merged = self._merge_entity_group(group)
                merged_entities.append(merged)
        
        return merged_entities
    
    def _merge_entity_group(self, entities: List[Entity]) -> Entity:
        """Merge a group of similar entities."""
        # Use the most common name
        names = [e.name for e in entities]
        most_common_name = Counter(names).most_common(1)[0][0]
        
        # Combine all mentions
        all_mentions = []
        all_chunk_ids = []
        all_properties = {}
        
        for entity in entities:
            all_mentions.extend(entity.mentions)
            all_chunk_ids.extend(entity.chunk_ids)
            all_properties.update(entity.properties)
        
        # Use the highest confidence and most specific type
        max_confidence = max(e.confidence for e in entities)
        entity_types = [e.entity_type for e in entities]
        most_common_type = Counter(entity_types).most_common(1)[0][0]
        
        return Entity(
            name=most_common_name,
            entity_type=most_common_type,
            mentions=list(set(all_mentions)),
            chunk_ids=list(set(all_chunk_ids)),
            confidence=max_confidence,
            properties=all_properties
        )
    
    def _filter_relationships(self, relationships: List[Relationship], entities: List[Entity]) -> List[Relationship]:
        """Filter and score relationships."""
        if not relationships:
            return []
        
        # Create entity name set for validation
        entity_names = {e.name for e in entities}
        
        # Filter relationships with valid entities
        valid_relationships = []
        for rel in relationships:
            if rel.source_entity in entity_names and rel.target_entity in entity_names:
                valid_relationships.append(rel)
        
        # Remove duplicate relationships
        unique_relationships = []
        seen = set()
        
        for rel in valid_relationships:
            # Create a key for deduplication
            key = (rel.source_entity, rel.target_entity, rel.relation_type)
            reverse_key = (rel.target_entity, rel.source_entity, rel.relation_type)
            
            if key not in seen and reverse_key not in seen:
                seen.add(key)
                unique_relationships.append(rel)
            else:
                # If we've seen this relationship, keep the one with higher confidence
                existing_rel = next((r for r in unique_relationships 
                                   if (r.source_entity, r.target_entity, r.relation_type) == key or
                                      (r.target_entity, r.source_entity, r.relation_type) == key), None)
                if existing_rel and rel.confidence > existing_rel.confidence:
                    unique_relationships.remove(existing_rel)
                    unique_relationships.append(rel)
        
        # Sort by confidence
        unique_relationships.sort(key=lambda x: x.confidence, reverse=True)
        
        return unique_relationships
    
    def _build_entity_patterns(self) -> Dict[str, List[str]]:
        """Build regex patterns for entity extraction."""
        return {
            'TECHNOLOGY': [
                r'\b(?:machine learning|artificial intelligence|deep learning|neural network|algorithm|model|framework|system|platform|API|database|software|application|tool|library|package)\b',
                r'\b(?:Python|Java|JavaScript|C\+\+|TensorFlow|PyTorch|scikit-learn|BERT|GPT|transformer|CNN|RNN|LSTM|GAN)\b',
                r'\b(?:RAG|NLP|AI|ML|DL|API|SQL|NoSQL|REST|GraphQL|JSON|XML|HTTP|HTTPS|TCP|UDP)\b'
            ],
            'CONCEPT': [
                r'\b(?:concept|principle|theory|approach|method|technique|strategy|process|procedure|workflow|pipeline|architecture|design pattern|best practice)\b',
                r'\b(?:supervised learning|unsupervised learning|reinforcement learning|classification|regression|clustering|dimensionality reduction|feature engineering|data preprocessing)\b'
            ],
            'ORGANIZATION': [
                r'\b(?:Google|Microsoft|Amazon|Facebook|Meta|Apple|IBM|OpenAI|Anthropic|DeepMind|Tesla|Netflix|Uber|Airbnb)\b',
                r'\b(?:Stanford|MIT|Harvard|Berkeley|CMU|University of|Institute of|College of)\b',
                r'\b(?:company|corporation|startup|organization|institution|foundation|consortium|alliance)\b'
            ],
            'PERSON': [
                r'\b(?:Dr\.|Prof\.|Professor|Mr\.|Ms\.|Mrs\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:PhD|MD|MSc|BSc))\b'
            ],
            'PRODUCT': [
                r'\b(?:ChatGPT|GPT-4|BERT|RoBERTa|T5|BART|XLNet|ELECTRA|DeBERTa|BigBird)\b',
                r'\b(?:TensorFlow|PyTorch|Keras|scikit-learn|pandas|numpy|matplotlib|seaborn|plotly)\b',
                r'\b(?:AWS|Azure|GCP|Docker|Kubernetes|Jenkins|Git|GitHub|GitLab|Jupyter|Colab)\b'
            ]
        }
    
    def _build_relationship_patterns(self) -> Dict[str, List[str]]:
        """Build patterns for relationship extraction."""
        return {
            'USES': [r'uses?', r'utilizes?', r'employs?', r'applies?', r'implements?', r'leverages?'],
            'DEVELOPS': [r'develops?', r'creates?', r'builds?', r'designs?', r'invents?', r'constructs?'],
            'IMPROVES': [r'improves?', r'enhances?', r'optimizes?', r'increases?', r'boosts?', r'upgrades?'],
            'ENABLES': [r'enables?', r'allows?', r'facilitates?', r'supports?', r'helps?', r'assists?'],
            'PART_OF': [r'part of', r'component of', r'belongs to', r'within', r'inside', r'contains?'],
            'RELATED_TO': [r'related to', r'associated with', r'connected to', r'linked to', r'similar to']
        }
    
    def _calculate_entity_confidence(self, entity_text: str, entity_type: str, context: str) -> float:
        """Calculate confidence score for an entity."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence for specific patterns
        if entity_type in ['TECHNOLOGY', 'PRODUCT'] and entity_text.upper() == entity_text:
            confidence += 0.2  # Acronyms are often technical terms
        
        if len(entity_text) > 10:
            confidence += 0.1  # Longer entities are often more specific
        
        # Context-based confidence
        context_lower = context.lower()
        if any(keyword in context_lower for keyword in ['algorithm', 'model', 'system', 'framework']):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_relationship_confidence(self, source: Entity, target: Entity, rel_type: str, indicator: str, context: str) -> float:
        """Calculate confidence score for a relationship."""
        confidence = 0.6  # Base confidence
        
        # Boost confidence for strong indicators
        strong_indicators = ['develops', 'creates', 'uses', 'implements']
        if indicator in strong_indicators:
            confidence += 0.2
        
        # Boost confidence for compatible entity types
        compatible_pairs = [
            ('PERSON', 'TECHNOLOGY'),
            ('ORGANIZATION', 'PRODUCT'),
            ('TECHNOLOGY', 'CONCEPT'),
            ('PRODUCT', 'TECHNOLOGY')
        ]
        
        if (source.entity_type, target.entity_type) in compatible_pairs:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _classify_entity_type(self, entity_text: str, context: str) -> str:
        """Classify entity type based on text and context."""
        entity_lower = entity_text.lower()
        context_lower = context.lower()
        
        # Check for technology indicators
        tech_indicators = ['algorithm', 'model', 'system', 'framework', 'api', 'library']
        if any(indicator in context_lower for indicator in tech_indicators):
            return 'TECHNOLOGY'
        
        # Check for organization indicators
        org_indicators = ['company', 'corporation', 'university', 'institute']
        if any(indicator in context_lower for indicator in org_indicators):
            return 'ORGANIZATION'
        
        # Check for person indicators
        person_indicators = ['researcher', 'scientist', 'professor', 'author', 'developer']
        if any(indicator in context_lower for indicator in person_indicators):
            return 'PERSON'
        
        # Default classification
        if entity_text.isupper() and len(entity_text) <= 5:
            return 'TECHNOLOGY'  # Likely an acronym
        
        return 'CONCEPT'  # Default fallback
    
    def _find_matching_entity(self, text: str, entities: List[Entity]) -> Optional[Entity]:
        """Find an entity that matches the given text."""
        text_lower = text.lower()
        
        # Exact match
        for entity in entities:
            if entity.name.lower() == text_lower:
                return entity
        
        # Partial match
        for entity in entities:
            if text_lower in entity.name.lower() or entity.name.lower() in text_lower:
                return entity
        
        return None
    
    def _infer_relationship_type(self, entity1: Entity, entity2: Entity, context: str) -> Optional[str]:
        """Infer relationship type based on entity types and context."""
        context_lower = context.lower()
        
        # Check for explicit relationship indicators
        for rel_type, indicators in self.relationship_indicators.items():
            if any(indicator in context_lower for indicator in indicators):
                return rel_type
        
        # Infer based on entity types
        type_pair = (entity1.entity_type, entity2.entity_type)
        
        if type_pair in [('PERSON', 'TECHNOLOGY'), ('ORGANIZATION', 'TECHNOLOGY')]:
            return 'DEVELOPS'
        elif type_pair in [('TECHNOLOGY', 'CONCEPT'), ('PRODUCT', 'CONCEPT')]:
            return 'IMPLEMENTS'
        elif type_pair in [('TECHNOLOGY', 'TECHNOLOGY'), ('CONCEPT', 'CONCEPT')]:
            return 'RELATED_TO'
        
        return 'RELATED_TO'  # Default relationship
    
    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for comparison."""
        return name.lower().strip().replace('-', ' ').replace('_', ' ')
    
    def _calculate_stats(self, entities: List[Entity], relationships: List[Relationship], text: str) -> Dict[str, Any]:
        """Calculate extraction statistics."""
        entity_types = Counter(e.entity_type for e in entities)
        relationship_types = Counter(r.relation_type for r in relationships)
        
        return {
            'total_entities': len(entities),
            'total_relationships': len(relationships),
            'entity_types': dict(entity_types),
            'relationship_types': dict(relationship_types),
            'text_length': len(text),
            'avg_entity_confidence': sum(e.confidence for e in entities) / max(len(entities), 1),
            'avg_relationship_confidence': sum(r.confidence for r in relationships) / max(len(relationships), 1)
        }