"""Output formatter for architecture results"""
from typing import List
from core.result import ArchitectureResult


def format_output(results: List[ArchitectureResult]) -> str:
    """Format results in the required format"""
    output_lines = []
    
    for result in results:
        output_lines.append("=" * 80)
        output_lines.append(f"Architect name: {result.architect_name}")
        output_lines.append(f"Input query: {result.input_query}")
        output_lines.append(f"Output: {result.output}")
        output_lines.append(f"Latency to retrieve the data: {result.latency:.4f} seconds")
        output_lines.append(f"Accuracy: {result.accuracy:.4f}")
        output_lines.append(f"Confidence score: {result.confidence_score:.4f}")
        output_lines.append(f"Evidence score: {result.evidence_score:.4f}")
        output_lines.append(f"Average accuracy: {result.average_accuracy:.4f}")
        
        if result.error:
            output_lines.append(f"Error: {result.error}")
        
        output_lines.append("=" * 80)
        output_lines.append("")
    
    return "\n".join(output_lines)
