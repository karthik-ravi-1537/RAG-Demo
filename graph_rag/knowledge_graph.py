"""Knowledge graph construction and storage for Graph-RAG implementation."""

import json
import pickle
from typing import List, Dict, Any, Set, Optional, Tuple, Union
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None

from core.data_models import Entity, Relationship, KnowledgeGraph as KGModel, Chunk
from core.exceptions import ProcessingError
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class KnowledgeGraphBuilder:
    """
    Knowledge graph builder that constructs and manages graph structures
    from extracted entities and relationships.
    """
    
    def __init__(self):
        """Initialize the knowledge graph builder."""
        if nx is None:
            raise ImportError("NetworkX is required for knowledge graph functionality. Install with: pip install networkx")
        
        self.graph = nx.MultiDiGraph()  # Directed multigraph to allow multiple relationships
        self.entity_index: Dict[str, Entity] = {}
        self.relationship_index: Dict[str, List[Relationship]] = {}
        self.chunk_to_entities: Dict[str, List[str]] = defaultdict(list)
        self.entity_to_chunks: Dict[str, List[str]] = defaultdict(list)
        
        # Graph statistics
        self.stats = {
            'nodes': 0,
            'edges': 0,
            'connected_components': 0,
            'avg_degree': 0.0,
            'density': 0.0,
            'clustering_coefficient': 0.0
        }
        
        logger.info("Initialized KnowledgeGraphBuilder with NetworkX")
    
    def add_entities(self, entities: List[Entity]) -> None:
        """Add entities to the knowledge graph."""
        for entity in entities:
            # Add to entity index
            self.entity_index[entity.name] = entity
            
            # Add node to graph
            self.graph.add_node(
                entity.name,
                entity_type=entity.entity_type,
                confidence=entity.confidence,
                mentions=entity.mentions,
                chunk_ids=entity.chunk_ids,
                properties=entity.properties
            )
            
            # Update chunk mappings
            for chunk_id in entity.chunk_ids:
                self.chunk_to_entities[chunk_id].append(entity.name)
                self.entity_to_chunks[entity.name].append(chunk_id)
        
        logger.info(f"Added {len(entities)} entities to knowledge graph")
    
    def add_relationships(self, relationships: List[Relationship]) -> None:
        """Add relationships to the knowledge graph."""
        added_count = 0
        
        for relationship in relationships:
            # Verify entities exist
            if (relationship.source_entity not in self.entity_index or 
                relationship.target_entity not in self.entity_index):
                logger.warning(f"Skipping relationship with missing entities: {relationship.source_entity} -> {relationship.target_entity}")
                continue
            
            # Add edge to graph
            edge_id = f"{relationship.source_entity}_{relationship.target_entity}_{relationship.relation_type}"
            
            self.graph.add_edge(
                relationship.source_entity,
                relationship.target_entity,
                key=edge_id,
                relation_type=relationship.relation_type,
                confidence=relationship.confidence,
                evidence_chunks=relationship.evidence_chunks,
                properties=relationship.properties
            )
            
            # Add to relationship index
            if edge_id not in self.relationship_index:
                self.relationship_index[edge_id] = []
            self.relationship_index[edge_id].append(relationship)
            
            added_count += 1
        
        logger.info(f"Added {added_count} relationships to knowledge graph")
    
    def build_from_extractions(self, entities: List[Entity], relationships: List[Relationship]) -> 'KnowledgeGraph':
        """Build knowledge graph from extracted entities and relationships."""
        # Clear existing graph
        self.graph.clear()
        self.entity_index.clear()
        self.relationship_index.clear()
        self.chunk_to_entities.clear()
        self.entity_to_chunks.clear()
        
        # Add entities and relationships
        self.add_entities(entities)
        self.add_relationships(relationships)
        
        # Update statistics
        self._update_statistics()
        
        # Create KnowledgeGraph model
        kg_model = KGModel(
            entities=self.entity_index,
            relationships=[rel for rel_list in self.relationship_index.values() for rel in rel_list]
        )
        
        # Create and return KnowledgeGraph wrapper
        return KnowledgeGraph(self.graph, kg_model, self)
    
    def get_neighbors(self, entity_name: str, max_depth: int = 2, relation_types: Optional[List[str]] = None) -> List[str]:
        """Get neighboring entities within specified depth."""
        if entity_name not in self.graph:
            return []
        
        neighbors = set()
        current_level = {entity_name}
        
        for depth in range(max_depth):
            next_level = set()
            
            for node in current_level:
                # Get outgoing neighbors
                for neighbor in self.graph.successors(node):
                    if relation_types is None:
                        next_level.add(neighbor)
                    else:
                        # Check if any edge has the desired relation type
                        for edge_data in self.graph[node][neighbor].values():
                            if edge_data.get('relation_type') in relation_types:
                                next_level.add(neighbor)
                                break
                
                # Get incoming neighbors
                for neighbor in self.graph.predecessors(node):
                    if relation_types is None:
                        next_level.add(neighbor)
                    else:
                        # Check if any edge has the desired relation type
                        for edge_data in self.graph[neighbor][node].values():
                            if edge_data.get('relation_type') in relation_types:
                                next_level.add(neighbor)
                                break
            
            neighbors.update(next_level)
            current_level = next_level
            
            if not current_level:
                break
        
        # Remove the original entity
        neighbors.discard(entity_name)
        return list(neighbors)
    
    def find_paths(self, source: str, target: str, max_length: int = 3) -> List[List[str]]:
        """Find paths between two entities."""
        if source not in self.graph or target not in self.graph:
            return []
        
        try:
            # Find all simple paths up to max_length
            paths = list(nx.all_simple_paths(
                self.graph.to_undirected(), 
                source, 
                target, 
                cutoff=max_length
            ))
            return paths
        except nx.NetworkXNoPath:
            return []
    
    def get_subgraph(self, entities: List[str], include_neighbors: bool = True) -> nx.Graph:
        """Get subgraph containing specified entities."""
        nodes_to_include = set(entities)
        
        if include_neighbors:
            # Add immediate neighbors
            for entity in entities:
                if entity in self.graph:
                    nodes_to_include.update(self.graph.neighbors(entity))
        
        # Filter to existing nodes
        existing_nodes = [node for node in nodes_to_include if node in self.graph]
        
        return self.graph.subgraph(existing_nodes)
    
    def get_entity_importance(self, entity_name: str) -> float:
        """Calculate entity importance based on graph centrality."""
        if entity_name not in self.graph:
            return 0.0
        
        # Combine multiple centrality measures
        try:
            degree_centrality = nx.degree_centrality(self.graph)[entity_name]
            betweenness_centrality = nx.betweenness_centrality(self.graph).get(entity_name, 0.0)
            
            # Weighted combination
            importance = 0.6 * degree_centrality + 0.4 * betweenness_centrality
            
            # Boost by entity confidence
            entity = self.entity_index.get(entity_name)
            if entity:
                importance *= entity.confidence
            
            return importance
        except:
            # Fallback to simple degree centrality
            return self.graph.degree(entity_name) / max(len(self.graph.nodes()) - 1, 1)
    
    def cluster_entities(self, algorithm: str = 'louvain') -> Dict[str, int]:
        """Cluster entities using community detection."""
        if len(self.graph.nodes()) < 2:
            return {}
        
        try:
            if algorithm == 'louvain':
                # Convert to undirected for community detection
                undirected_graph = self.graph.to_undirected()
                communities = nx.community.louvain_communities(undirected_graph)
                
                # Create entity to cluster mapping
                entity_clusters = {}
                for cluster_id, community in enumerate(communities):
                    for entity in community:
                        entity_clusters[entity] = cluster_id
                
                return entity_clusters
            else:
                logger.warning(f"Unknown clustering algorithm: {algorithm}")
                return {}
        except Exception as e:
            logger.error(f"Clustering failed: {str(e)}")
            return {}
    
    def _update_statistics(self) -> None:
        """Update graph statistics."""
        if len(self.graph.nodes()) == 0:
            self.stats = {
                'nodes': 0,
                'edges': 0,
                'connected_components': 0,
                'avg_degree': 0.0,
                'density': 0.0,
                'clustering_coefficient': 0.0
            }
            return
        
        try:
            self.stats = {
                'nodes': len(self.graph.nodes()),
                'edges': len(self.graph.edges()),
                'connected_components': nx.number_weakly_connected_components(self.graph),
                'avg_degree': sum(dict(self.graph.degree()).values()) / len(self.graph.nodes()),
                'density': nx.density(self.graph),
                'clustering_coefficient': nx.average_clustering(self.graph.to_undirected())
            }
        except Exception as e:
            logger.warning(f"Failed to calculate some graph statistics: {str(e)}")
    
    def save_graph(self, file_path: Union[str, Path], format: str = 'pickle') -> None:
        """Save the knowledge graph to file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == 'pickle':
                with open(file_path, 'wb') as f:
                    pickle.dump({
                        'graph': self.graph,
                        'entity_index': self.entity_index,
                        'relationship_index': self.relationship_index,
                        'chunk_mappings': {
                            'chunk_to_entities': dict(self.chunk_to_entities),
                            'entity_to_chunks': dict(self.entity_to_chunks)
                        },
                        'stats': self.stats
                    }, f)
            
            elif format == 'json':
                # Convert graph to JSON-serializable format
                graph_data = {
                    'nodes': [
                        {
                            'id': node,
                            'attributes': data
                        }
                        for node, data in self.graph.nodes(data=True)
                    ],
                    'edges': [
                        {
                            'source': source,
                            'target': target,
                            'attributes': data
                        }
                        for source, target, data in self.graph.edges(data=True)
                    ]
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(graph_data, f, indent=2, default=str)
            
            elif format == 'gexf':
                nx.write_gexf(self.graph, file_path)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Saved knowledge graph to {file_path} in {format} format")
            
        except Exception as e:
            logger.error(f"Failed to save knowledge graph: {str(e)}")
            raise ProcessingError(f"Graph save failed: {str(e)}")
    
    def load_graph(self, file_path: Union[str, Path], format: str = 'pickle') -> None:
        """Load knowledge graph from file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ProcessingError(f"Graph file not found: {file_path}")
        
        try:
            if format == 'pickle':
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                self.graph = data['graph']
                self.entity_index = data['entity_index']
                self.relationship_index = data['relationship_index']
                
                chunk_mappings = data.get('chunk_mappings', {})
                self.chunk_to_entities = defaultdict(list, chunk_mappings.get('chunk_to_entities', {}))
                self.entity_to_chunks = defaultdict(list, chunk_mappings.get('entity_to_chunks', {}))
                
                self.stats = data.get('stats', {})
            
            elif format == 'gexf':
                self.graph = nx.read_gexf(file_path)
                # Rebuild indices from graph data
                self._rebuild_indices()
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Loaded knowledge graph from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load knowledge graph: {str(e)}")
            raise ProcessingError(f"Graph load failed: {str(e)}")
    
    def _rebuild_indices(self) -> None:
        """Rebuild indices from graph data."""
        self.entity_index.clear()
        self.relationship_index.clear()
        self.chunk_to_entities.clear()
        self.entity_to_chunks.clear()
        
        # Rebuild entity index
        for node, data in self.graph.nodes(data=True):
            entity = Entity(
                name=node,
                entity_type=data.get('entity_type', 'UNKNOWN'),
                confidence=data.get('confidence', 0.5),
                mentions=data.get('mentions', []),
                chunk_ids=data.get('chunk_ids', [])
            )
            self.entity_index[node] = entity
            
            # Rebuild chunk mappings
            for chunk_id in entity.chunk_ids:
                self.chunk_to_entities[chunk_id].append(node)
                self.entity_to_chunks[node].append(chunk_id)


class KnowledgeGraph:
    """
    Wrapper class for knowledge graph with high-level query and analysis methods.
    """
    
    def __init__(self, nx_graph: nx.MultiDiGraph, kg_model: KGModel, builder: KnowledgeGraphBuilder):
        """Initialize knowledge graph wrapper."""
        self.graph = nx_graph
        self.model = kg_model
        self.builder = builder
    
    def query_entities(self, entity_type: Optional[str] = None, min_confidence: float = 0.0) -> List[Entity]:
        """Query entities by type and confidence."""
        results = []
        
        for entity in self.model.entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue
            if entity.confidence < min_confidence:
                continue
            results.append(entity)
        
        return sorted(results, key=lambda x: x.confidence, reverse=True)
    
    def query_relationships(self, relation_type: Optional[str] = None, min_confidence: float = 0.0) -> List[Relationship]:
        """Query relationships by type and confidence."""
        results = []
        
        for relationship in self.model.relationships:
            if relation_type and relationship.relation_type != relation_type:
                continue
            if relationship.confidence < min_confidence:
                continue
            results.append(relationship)
        
        return sorted(results, key=lambda x: x.confidence, reverse=True)
    
    def find_related_entities(self, entity_name: str, max_depth: int = 2, relation_types: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """Find entities related to the given entity with importance scores."""
        related_entities = self.builder.get_neighbors(entity_name, max_depth, relation_types)
        
        # Calculate importance scores
        entity_scores = []
        for related_entity in related_entities:
            importance = self.builder.get_entity_importance(related_entity)
            entity_scores.append((related_entity, importance))
        
        return sorted(entity_scores, key=lambda x: x[1], reverse=True)
    
    def get_entity_context(self, entity_name: str) -> Dict[str, Any]:
        """Get comprehensive context for an entity."""
        if entity_name not in self.model.entities:
            return {}
        
        entity = self.model.entities[entity_name]
        
        # Get related entities
        related = self.find_related_entities(entity_name, max_depth=2)
        
        # Get relationships
        incoming_rels = []
        outgoing_rels = []
        
        for rel in self.model.relationships:
            if rel.target_entity == entity_name:
                incoming_rels.append(rel)
            elif rel.source_entity == entity_name:
                outgoing_rels.append(rel)
        
        return {
            'entity': entity,
            'importance': self.builder.get_entity_importance(entity_name),
            'related_entities': related[:10],  # Top 10 related
            'incoming_relationships': incoming_rels,
            'outgoing_relationships': outgoing_rels,
            'chunk_count': len(entity.chunk_ids),
            'mention_count': len(entity.mentions)
        }
    
    def find_connection_paths(self, source_entity: str, target_entity: str, max_length: int = 3) -> List[Dict[str, Any]]:
        """Find connection paths between two entities with relationship details."""
        paths = self.builder.find_paths(source_entity, target_entity, max_length)
        
        detailed_paths = []
        for path in paths:
            path_details = {
                'entities': path,
                'length': len(path) - 1,
                'relationships': []
            }
            
            # Get relationship details for each step
            for i in range(len(path) - 1):
                source = path[i]
                target = path[i + 1]
                
                # Find relationship between these entities
                relationship_info = None
                for rel in self.model.relationships:
                    if ((rel.source_entity == source and rel.target_entity == target) or
                        (rel.source_entity == target and rel.target_entity == source)):
                        relationship_info = {
                            'type': rel.relation_type,
                            'confidence': rel.confidence,
                            'direction': 'forward' if rel.source_entity == source else 'reverse'
                        }
                        break
                
                path_details['relationships'].append(relationship_info)
            
            detailed_paths.append(path_details)
        
        return detailed_paths
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""
        entity_type_counts = Counter(e.entity_type for e in self.model.entities.values())
        relationship_type_counts = Counter(r.relation_type for r in self.model.relationships)
        
        return {
            **self.builder.stats,
            'entity_types': dict(entity_type_counts),
            'relationship_types': dict(relationship_type_counts),
            'avg_entity_confidence': sum(e.confidence for e in self.model.entities.values()) / max(len(self.model.entities), 1),
            'avg_relationship_confidence': sum(r.confidence for r in self.model.relationships) / max(len(self.model.relationships), 1)
        }
    
    def export_for_visualization(self) -> Dict[str, Any]:
        """Export graph data in format suitable for visualization."""
        nodes = []
        for entity_name, entity in self.model.entities.items():
            nodes.append({
                'id': entity_name,
                'label': entity_name,
                'type': entity.entity_type,
                'confidence': entity.confidence,
                'importance': self.builder.get_entity_importance(entity_name),
                'size': len(entity.chunk_ids) * 5 + 10,  # Size based on chunk count
                'color': self._get_type_color(entity.entity_type)
            })
        
        edges = []
        for relationship in self.model.relationships:
            edges.append({
                'source': relationship.source_entity,
                'target': relationship.target_entity,
                'type': relationship.relation_type,
                'confidence': relationship.confidence,
                'width': relationship.confidence * 3 + 1,
                'color': self._get_relation_color(relationship.relation_type)
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'statistics': self.get_statistics()
        }
    
    def _get_type_color(self, entity_type: str) -> str:
        """Get color for entity type."""
        colors = {
            'PERSON': '#FF6B6B',
            'ORGANIZATION': '#4ECDC4',
            'TECHNOLOGY': '#45B7D1',
            'CONCEPT': '#96CEB4',
            'LOCATION': '#FFEAA7',
            'PRODUCT': '#DDA0DD'
        }
        return colors.get(entity_type, '#95A5A6')
    
    def _get_relation_color(self, relation_type: str) -> str:
        """Get color for relationship type."""
        colors = {
            'DEVELOPS': '#E74C3C',
            'USES': '#3498DB',
            'PART_OF': '#2ECC71',
            'RELATED_TO': '#95A5A6',
            'IMPROVES': '#F39C12',
            'ENABLES': '#9B59B6'
        }
        return colors.get(relation_type, '#BDC3C7')