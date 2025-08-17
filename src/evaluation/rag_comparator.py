"""RAG comparison framework for side-by-side evaluation of different RAG approaches."""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from core.data_models import RAGResult
from core.exceptions import RAGException
from core.interfaces import BaseRAG
from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ComparisonQuery:
    """A query for RAG comparison."""

    id: str
    query: str
    expected_answer: str | None = None
    category: str | None = None
    difficulty: str | None = None
    metadata: dict[str, Any] = None


@dataclass
class RAGPerformanceMetrics:
    """Performance metrics for a RAG system."""

    rag_name: str
    query_id: str

    # Timing metrics
    retrieval_time: float
    generation_time: float
    total_time: float

    # Quality metrics
    confidence_score: float
    chunks_retrieved: int
    response_length: int

    # RAG-specific metrics
    explanation: dict[str, Any]

    # Additional metrics
    memory_usage: float | None = None
    error: str | None = None


@dataclass
class ComparisonResult:
    """Result of comparing multiple RAG systems."""

    query: ComparisonQuery
    results: dict[str, RAGResult]  # rag_name -> result
    metrics: dict[str, RAGPerformanceMetrics]  # rag_name -> metrics
    winner: str | None = None
    comparison_summary: dict[str, Any] = None


class RAGComparator:
    """
    Framework for comparing multiple RAG implementations side-by-side.

    This class provides:
    1. Unified interface for all RAG implementations
    2. Performance metrics collection (timing, accuracy)
    3. Batch query processing for evaluation
    4. Comparison result visualization and analysis
    """

    def __init__(self):
        """Initialize the RAG comparator."""
        self.rag_systems: dict[str, BaseRAG] = {}
        self.comparison_history: list[ComparisonResult] = []

        # Metrics tracking
        self.aggregate_metrics: dict[str, dict[str, list[float]]] = {}

        logger.info("Initialized RAGComparator")

    def register_rag_system(self, name: str, rag_system: BaseRAG) -> None:
        """Register a RAG system for comparison."""
        self.rag_systems[name] = rag_system
        self.aggregate_metrics[name] = {
            "retrieval_times": [],
            "generation_times": [],
            "total_times": [],
            "confidence_scores": [],
            "chunks_retrieved": [],
            "response_lengths": [],
        }
        logger.info(f"Registered RAG system: {name}")

    def compare_single_query(
        self, query: str | ComparisonQuery, rag_names: list[str] | None = None
    ) -> ComparisonResult:
        """Compare RAG systems on a single query."""
        # Convert string query to ComparisonQuery
        if isinstance(query, str):
            query = ComparisonQuery(id=f"query_{len(self.comparison_history)}", query=query)

        # Use all registered systems if none specified
        if rag_names is None:
            rag_names = list(self.rag_systems.keys())

        # Validate RAG systems
        for name in rag_names:
            if name not in self.rag_systems:
                raise RAGException(f"RAG system '{name}' not registered")

        logger.info(f"Comparing {len(rag_names)} RAG systems on query: {query.query[:50]}...")

        results = {}
        metrics = {}

        # Run each RAG system
        for rag_name in rag_names:
            try:
                rag_system = self.rag_systems[rag_name]

                # Measure performance
                time.time()
                result = rag_system.query(query.query)
                time.time()

                # Store result
                results[rag_name] = result

                # Calculate metrics
                performance_metrics = RAGPerformanceMetrics(
                    rag_name=rag_name,
                    query_id=query.id,
                    retrieval_time=result.retrieval_time,
                    generation_time=result.generation_time,
                    total_time=result.total_time,
                    confidence_score=result.confidence_score,
                    chunks_retrieved=len(result.retrieved_chunks),
                    response_length=len(result.response),
                    explanation=result.explanation,
                )

                metrics[rag_name] = performance_metrics

                # Update aggregate metrics
                self._update_aggregate_metrics(rag_name, performance_metrics)

                logger.debug(f"{rag_name}: {result.total_time:.3f}s, confidence: {result.confidence_score:.3f}")

            except Exception as e:
                logger.error(f"Error running {rag_name}: {str(e)}")

                # Create error metrics
                error_metrics = RAGPerformanceMetrics(
                    rag_name=rag_name,
                    query_id=query.id,
                    retrieval_time=0.0,
                    generation_time=0.0,
                    total_time=0.0,
                    confidence_score=0.0,
                    chunks_retrieved=0,
                    response_length=0,
                    explanation={},
                    error=str(e),
                )

                metrics[rag_name] = error_metrics

        # Analyze comparison
        comparison_summary = self._analyze_comparison(results, metrics)
        winner = self._determine_winner(metrics)

        # Create comparison result
        comparison_result = ComparisonResult(
            query=query, results=results, metrics=metrics, winner=winner, comparison_summary=comparison_summary
        )

        # Store in history
        self.comparison_history.append(comparison_result)

        logger.info(f"Comparison complete. Winner: {winner}")

        return comparison_result

    def compare_batch_queries(
        self, queries: list[str | ComparisonQuery], rag_names: list[str] | None = None
    ) -> list[ComparisonResult]:
        """Compare RAG systems on a batch of queries."""
        logger.info(f"Starting batch comparison of {len(queries)} queries")

        batch_results = []

        for i, query in enumerate(queries):
            logger.info(f"Processing query {i + 1}/{len(queries)}")

            try:
                result = self.compare_single_query(query, rag_names)
                batch_results.append(result)
            except Exception as e:
                logger.error(f"Failed to process query {i + 1}: {str(e)}")
                continue

        logger.info(f"Batch comparison complete: {len(batch_results)} successful comparisons")

        return batch_results

    def get_aggregate_analysis(self, rag_names: list[str] | None = None) -> dict[str, Any]:
        """Get aggregate analysis across all comparisons."""
        if rag_names is None:
            rag_names = list(self.rag_systems.keys())

        analysis = {}

        for rag_name in rag_names:
            if rag_name not in self.aggregate_metrics:
                continue

            metrics = self.aggregate_metrics[rag_name]

            # Calculate statistics
            rag_analysis = {}

            for metric_name, values in metrics.items():
                if values:
                    rag_analysis[metric_name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "median": np.median(values),
                        "count": len(values),
                    }
                else:
                    rag_analysis[metric_name] = {
                        "mean": 0.0,
                        "std": 0.0,
                        "min": 0.0,
                        "max": 0.0,
                        "median": 0.0,
                        "count": 0,
                    }

            # Calculate derived metrics
            if metrics["total_times"]:
                rag_analysis["queries_per_second"] = 1.0 / np.mean(metrics["total_times"])
            else:
                rag_analysis["queries_per_second"] = 0.0

            # Win rate
            wins = sum(1 for comp in self.comparison_history if comp.winner == rag_name)
            total_comparisons = len(self.comparison_history)
            rag_analysis["win_rate"] = wins / max(total_comparisons, 1)

            analysis[rag_name] = rag_analysis

        return analysis

    def generate_comparison_report(self, output_path: str | Path | None = None) -> dict[str, Any]:
        """Generate a comprehensive comparison report."""
        report = {
            "summary": {
                "total_comparisons": len(self.comparison_history),
                "rag_systems": list(self.rag_systems.keys()),
                "comparison_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "aggregate_analysis": self.get_aggregate_analysis(),
            "detailed_comparisons": [],
        }

        # Add detailed comparisons
        for comparison in self.comparison_history[-10:]:  # Last 10 comparisons
            comparison_data = {
                "query_id": comparison.query.id,
                "query_text": comparison.query.query,
                "winner": comparison.winner,
                "results_summary": {},
            }

            for rag_name, metrics in comparison.metrics.items():
                comparison_data["results_summary"][rag_name] = {
                    "total_time": metrics.total_time,
                    "confidence_score": metrics.confidence_score,
                    "chunks_retrieved": metrics.chunks_retrieved,
                    "response_length": metrics.response_length,
                    "error": metrics.error,
                }

            report["detailed_comparisons"].append(comparison_data)

        # Add performance rankings
        report["rankings"] = self._generate_rankings()

        # Save report if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Comparison report saved to {output_path}")

        return report

    def visualize_comparison(self, comparison_result: ComparisonResult) -> dict[str, Any]:
        """Create visualization data for a comparison result."""
        viz_data = {
            "query": comparison_result.query.query,
            "systems": [],
            "metrics_comparison": {},
            "response_comparison": {},
        }

        # System performance data
        for rag_name, metrics in comparison_result.metrics.items():
            system_data = {
                "name": rag_name,
                "total_time": metrics.total_time,
                "confidence": metrics.confidence_score,
                "chunks_retrieved": metrics.chunks_retrieved,
                "response_length": metrics.response_length,
                "is_winner": rag_name == comparison_result.winner,
                "error": metrics.error,
            }
            viz_data["systems"].append(system_data)

        # Metrics comparison
        metric_names = ["total_time", "confidence_score", "chunks_retrieved", "response_length"]
        for metric in metric_names:
            viz_data["metrics_comparison"][metric] = {
                rag_name: getattr(metrics, metric)
                for rag_name, metrics in comparison_result.metrics.items()
                if not metrics.error
            }

        # Response comparison
        for rag_name, result in comparison_result.results.items():
            viz_data["response_comparison"][rag_name] = {
                "response": result.response[:500] + "..." if len(result.response) > 500 else result.response,
                "confidence": result.confidence_score,
                "chunks_used": len(result.retrieved_chunks),
            }

        return viz_data

    def export_results(self, output_dir: str | Path, format: str = "json") -> None:
        """Export comparison results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if format == "json":
            # Export individual comparisons
            for i, comparison in enumerate(self.comparison_history):
                filename = f"comparison_{i + 1}_{comparison.query.id}.json"
                filepath = output_dir / filename

                # Convert to serializable format
                comparison_data = {
                    "query": asdict(comparison.query),
                    "metrics": {name: asdict(metrics) for name, metrics in comparison.metrics.items()},
                    "winner": comparison.winner,
                    "comparison_summary": comparison.comparison_summary,
                    "responses": {name: result.response for name, result in comparison.results.items()},
                }

                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(comparison_data, f, indent=2, default=str)

            # Export aggregate report
            report = self.generate_comparison_report()
            with open(output_dir / "aggregate_report.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)

        logger.info(f"Exported {len(self.comparison_history)} comparisons to {output_dir}")

    def _update_aggregate_metrics(self, rag_name: str, metrics: RAGPerformanceMetrics) -> None:
        """Update aggregate metrics for a RAG system."""
        if metrics.error:
            return  # Skip error cases

        agg_metrics = self.aggregate_metrics[rag_name]
        agg_metrics["retrieval_times"].append(metrics.retrieval_time)
        agg_metrics["generation_times"].append(metrics.generation_time)
        agg_metrics["total_times"].append(metrics.total_time)
        agg_metrics["confidence_scores"].append(metrics.confidence_score)
        agg_metrics["chunks_retrieved"].append(metrics.chunks_retrieved)
        agg_metrics["response_lengths"].append(metrics.response_length)

    def _analyze_comparison(
        self, results: dict[str, RAGResult], metrics: dict[str, RAGPerformanceMetrics]
    ) -> dict[str, Any]:
        """Analyze comparison results."""
        analysis = {"performance_summary": {}, "quality_summary": {}, "efficiency_summary": {}}

        # Performance analysis
        valid_metrics = {name: m for name, m in metrics.items() if not m.error}

        if valid_metrics:
            # Speed comparison
            fastest_system = min(valid_metrics.items(), key=lambda x: x[1].total_time)
            slowest_system = max(valid_metrics.items(), key=lambda x: x[1].total_time)

            analysis["performance_summary"] = {
                "fastest_system": fastest_system[0],
                "fastest_time": fastest_system[1].total_time,
                "slowest_system": slowest_system[0],
                "slowest_time": slowest_system[1].total_time,
                "speed_difference": slowest_system[1].total_time - fastest_system[1].total_time,
            }

            # Quality comparison
            highest_confidence = max(valid_metrics.items(), key=lambda x: x[1].confidence_score)
            lowest_confidence = min(valid_metrics.items(), key=lambda x: x[1].confidence_score)

            analysis["quality_summary"] = {
                "highest_confidence_system": highest_confidence[0],
                "highest_confidence_score": highest_confidence[1].confidence_score,
                "lowest_confidence_system": lowest_confidence[0],
                "lowest_confidence_score": lowest_confidence[1].confidence_score,
                "confidence_spread": highest_confidence[1].confidence_score - lowest_confidence[1].confidence_score,
            }

            # Efficiency analysis (chunks retrieved vs confidence)
            efficiency_scores = {}
            for name, m in valid_metrics.items():
                if m.chunks_retrieved > 0:
                    efficiency_scores[name] = m.confidence_score / m.chunks_retrieved
                else:
                    efficiency_scores[name] = 0.0

            if efficiency_scores:
                most_efficient = max(efficiency_scores.items(), key=lambda x: x[1])
                analysis["efficiency_summary"] = {
                    "most_efficient_system": most_efficient[0],
                    "efficiency_score": most_efficient[1],
                    "all_efficiency_scores": efficiency_scores,
                }

        return analysis

    def _determine_winner(self, metrics: dict[str, RAGPerformanceMetrics]) -> str | None:
        """Determine the winner based on multiple criteria with balanced scoring."""
        valid_metrics = {name: m for name, m in metrics.items() if not m.error}

        if not valid_metrics:
            return None

        # Improved scoring system: 60% confidence, 30% speed, 10% completeness
        # Speed scoring now rewards fast systems relative to the fastest, not penalized by slowest
        scores = {}

        # Get min/max values for normalization
        confidence_values = [m.confidence_score for m in valid_metrics.values()]
        time_values = [m.total_time for m in valid_metrics.values()]
        length_values = [m.response_length for m in valid_metrics.values()]

        max_confidence = max(confidence_values)
        min_time = min(time_values)  # Fastest time
        max_length = max(length_values)

        for name, m in valid_metrics.items():
            score = 0.0

            # Confidence score (60%) - primary quality metric
            if max_confidence > 0:
                score += 0.6 * (m.confidence_score / max_confidence)

            # Speed score (30%) - reward systems based on how close they are to the fastest
            # Fastest system gets full 0.3 points, others get proportionally less
            if min_time > 0:
                speed_score = 0.3 * (min_time / m.total_time)
                score += speed_score

            # Completeness score (10%) - reward comprehensive responses
            if max_length > 0:
                # Linear scaling - longer responses get higher scores (up to a point)
                completeness_score = min(1.0, m.response_length / max(1000, max_length * 0.7))
                score += 0.1 * completeness_score

            scores[name] = score

        # Return system with highest score
        return max(scores.items(), key=lambda x: x[1])[0] if scores else None

    def _generate_rankings(self) -> dict[str, Any]:
        """Generate performance rankings across all metrics."""
        analysis = self.get_aggregate_analysis()

        rankings = {"overall": {}, "by_metric": {}}

        # Overall ranking based on multiple factors
        overall_scores = {}

        for rag_name, rag_analysis in analysis.items():
            score = 0.0

            # Speed (30%)
            if rag_analysis["total_times"]["count"] > 0:
                avg_time = rag_analysis["total_times"]["mean"]
                # Lower time is better
                max_time = max(a["total_times"]["mean"] for a in analysis.values() if a["total_times"]["count"] > 0)
                if max_time > 0:
                    speed_score = 1.0 - (avg_time / max_time)
                    score += 0.3 * speed_score

            # Confidence (40%)
            if rag_analysis["confidence_scores"]["count"] > 0:
                avg_confidence = rag_analysis["confidence_scores"]["mean"]
                score += 0.4 * avg_confidence

            # Win rate (30%)
            score += 0.3 * rag_analysis["win_rate"]

            overall_scores[rag_name] = score

        # Sort by overall score
        sorted_overall = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        rankings["overall"] = {name: {"rank": i + 1, "score": score} for i, (name, score) in enumerate(sorted_overall)}

        # Rankings by individual metrics
        metric_names = ["total_times", "confidence_scores", "chunks_retrieved", "response_lengths"]

        for metric in metric_names:
            metric_values = {}
            for rag_name, rag_analysis in analysis.items():
                if rag_analysis[metric]["count"] > 0:
                    metric_values[rag_name] = rag_analysis[metric]["mean"]

            if metric_values:
                # For time metrics, lower is better
                reverse_sort = metric != "total_times"
                sorted_metric = sorted(metric_values.items(), key=lambda x: x[1], reverse=reverse_sort)
                rankings["by_metric"][metric] = {name: i + 1 for i, (name, _) in enumerate(sorted_metric)}

        return rankings

    def clear_history(self) -> None:
        """Clear comparison history and reset aggregate metrics."""
        self.comparison_history.clear()

        for rag_name in self.aggregate_metrics:
            self.aggregate_metrics[rag_name] = {
                "retrieval_times": [],
                "generation_times": [],
                "total_times": [],
                "confidence_scores": [],
                "chunks_retrieved": [],
                "response_lengths": [],
            }

        logger.info("Cleared comparison history and reset metrics")

    def get_system_stats(self) -> dict[str, Any]:
        """Get statistics about registered RAG systems."""
        stats = {
            "registered_systems": len(self.rag_systems),
            "total_comparisons": len(self.comparison_history),
            "systems": {},
        }

        for name, rag_system in self.rag_systems.items():
            try:
                system_stats = rag_system.get_stats()
                stats["systems"][name] = {"type": type(rag_system).__name__, "stats": system_stats}
            except Exception as e:
                stats["systems"][name] = {"type": type(rag_system).__name__, "error": str(e)}

        return stats
