"""
Label logged interventions using the Activation Oracle.

This script implements Phase 2B (AO-Labeling Protocol) from docs/Impl.pdf:
- Loads successful interventions from logs/interventions/
- Uses MultiLayerActivationOracle to provide semantic labels
- Outputs labeled dataset D = {(δf_i, Label_i, Confidence_i)}

Usage:
    python src/label_interventions.py --log-file logs/interventions/policy_success.jsonl
    python src/label_interventions.py --all  # Label all intervention logs
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from models import MultiLayerActivationOracle
from rubrics import RubricDefinition


def load_intervention_logs(log_path: Path) -> List[Dict]:
    """Load intervention logs from JSONL file."""
    records = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def label_single_intervention(
    oracle: MultiLayerActivationOracle,
    delta_f: torch.Tensor,
    layer_idx: int,
    rubric: RubricDefinition,
    base_activation: Optional[torch.Tensor] = None,
    max_new_tokens: int = 64
) -> Tuple[str, float, str]:
    """
    Label a single intervention using the Activation Oracle.
    
    Implements the AO-Labelling Protocol from docs/Impl.pdf Section 2.2:
    <ACT>[fx,c serialized]</ACT> Identify the specific jailbreak mechanism 
    for rubric c. Output: [Short Label], [Confidence 0-1].
    
    Args:
        oracle: MultiLayerActivationOracle instance
        delta_f: Intervention direction [d_hidden]
        layer_idx: Layer index where intervention was applied
        rubric: RubricDefinition for context
        base_activation: Optional base activation f(x) to compute f'(x,c) = f(x) + δf
        max_new_tokens: Max tokens for AO response
        
    Returns:
        (label, confidence, mechanism): Semantic label, confidence score, and 
            description of the jailbreak mechanism
    """
    # If we have base activation, compute intervened activation
    # Otherwise, use delta_f directly (interpreting the intervention direction)
    if base_activation is not None:
        intervention_act = base_activation + delta_f
    else:
        # Use delta_f directly - this interprets the direction of change
        intervention_act = delta_f
    
    # Ensure correct shape: [1, d_hidden] or [T, d_hidden]
    if intervention_act.dim() == 1:
        intervention_act = intervention_act.unsqueeze(0)
    
    # Create activation dict for the layer
    intervention_activations = {layer_idx: intervention_act.cuda()}
    
    # Query AO using the PDF template
    label, confidence, mechanism = oracle.label_intervention(
        intervention_activations=intervention_activations,
        rubric=rubric,
        max_new_tokens=max_new_tokens
    )
    
    return label, confidence, mechanism


def label_all_interventions(
    log_path: Path,
    oracle: MultiLayerActivationOracle,
    output_path: Optional[Path] = None,
    max_new_tokens: int = 64,
    verbose: bool = True
) -> List[Dict]:
    """
    Label all interventions in a log file using the AO-Labelling Protocol.
    
    Implements Phase 2B from docs/Impl.pdf:
    For each successful intervention f'(x,c) = f(x) + δf, perform semantic enrichment
    using the AO prompt template to identify the jailbreak mechanism.
    
    Args:
        log_path: Path to intervention JSONL log
        oracle: MultiLayerActivationOracle instance
        output_path: Optional path to save labeled results
        max_new_tokens: Max tokens for AO responses
        verbose: Whether to print progress
        
    Returns:
        List of labeled intervention records with:
        - ao_label: Short label (e.g., "Roleplay-Jailbreak")
        - ao_confidence: Confidence score [0-1]
        - ao_mechanism: Description of the jailbreak mechanism
    """
    if verbose:
        print(f"\nLoading interventions from: {log_path}")
    
    records = load_intervention_logs(log_path)
    
    if verbose:
        print(f"Found {len(records)} intervention records")
    
    labeled_records = []
    
    for i, record in enumerate(records):
        if verbose:
            print(f"\n[{i+1}/{len(records)}] Labeling intervention...")
            print(f"  Rubric: {record['rubric_text'][:60]}...")
            print(f"  Score: {record['score']}")
        
        # Reconstruct delta_f tensor
        delta_f = torch.tensor(record['delta_f'], dtype=torch.float32)
        layer_idx = record['layer_idx']
        
        # Create rubric
        rubric = RubricDefinition(
            rubric_text=record['rubric_text'],
            target_answer=record['target_answer'],
            category=record.get('category', 'jailbreak')
        )
        
        # Label with AO using the PDF template
        try:
            label, confidence, mechanism = label_single_intervention(
                oracle=oracle,
                delta_f=delta_f,
                layer_idx=layer_idx,
                rubric=rubric,
                max_new_tokens=max_new_tokens
            )
            
            if verbose:
                print(f"  AO Label: {label}")
                print(f"  AO Confidence: {confidence:.2f}")
                print(f"  AO Mechanism: {mechanism[:80]}..." if len(mechanism) > 80 else f"  AO Mechanism: {mechanism}")
            
        except Exception as e:
            print(f"  Error labeling: {e}")
            label = "ERROR"
            confidence = 0.0
            mechanism = "Failed to identify mechanism"
        
        # Create labeled record following the dataset structure from PDF:
        # D = {(fi, Labeli, Scorei)} with mechanism for interpretability
        labeled_record = {
            **record,
            'ao_label': label,
            'ao_confidence': confidence,
            'ao_mechanism': mechanism,
            'labeled_at': datetime.utcnow().isoformat()
        }
        labeled_records.append(labeled_record)
    
    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as f:
            for rec in labeled_records:
                # Remove delta_f from output to keep file size reasonable
                output_rec = {k: v for k, v in rec.items() if k != 'delta_f'}
                f.write(json.dumps(output_rec) + '\n')
        
        if verbose:
            print(f"\n✓ Saved labeled interventions to: {output_path}")
    
    return labeled_records


def summarize_labels(labeled_records: List[Dict]) -> None:
    """
    Print summary of AO labels following the Expected Results Structure from 
    docs/Impl.pdf Table 1.
    """
    print("\n" + "="*70)
    print("AO LABELING SUMMARY (Phase 2B Complete)")
    print("="*70)
    
    # Count labels
    from collections import Counter
    label_counts = Counter(rec['ao_label'] for rec in labeled_records)
    
    print(f"\nTotal interventions labeled: {len(labeled_records)}")
    
    # Print table similar to PDF Table 1
    print("\n" + "-"*70)
    print(f"{'Cluster Label (AO)':<30} {'Size (n)':<10} {'Avg Conf':<10} {'Avg Score':<10}")
    print("-"*70)
    
    for label, count in label_counts.most_common():
        pct = count / len(labeled_records) * 100
        avg_conf = sum(
            rec['ao_confidence'] for rec in labeled_records 
            if rec['ao_label'] == label
        ) / count
        avg_score = sum(
            rec.get('score', 0) for rec in labeled_records 
            if rec['ao_label'] == label
        ) / count
        print(f"{label:<30} {count:<10} {avg_conf:<10.2f} {avg_score:<10.2f}")
    
    print("-"*70)
    
    # Average confidence
    avg_confidence = sum(rec['ao_confidence'] for rec in labeled_records) / len(labeled_records)
    print(f"\nOverall average confidence: {avg_confidence:.2f}")
    
    # Show mechanisms per label
    print("\n" + "="*70)
    print("JAILBREAK MECHANISMS IDENTIFIED")
    print("="*70)
    
    shown_labels = set()
    for rec in labeled_records:
        label = rec['ao_label']
        if label not in shown_labels:
            shown_labels.add(label)
            mechanism = rec.get('ao_mechanism', 'Unknown mechanism')
            print(f"\n[{label}]")
            print(f"  Mechanism: {mechanism}")
            print(f"  Confidence: {rec['ao_confidence']:.2f}")
            print(f"  Rubric: {rec['rubric_text'][:70]}...")
            if 'prompt' in rec:
                print(f"  Example prompt: {rec['prompt'][:70]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Label logged interventions using the Activation Oracle"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to specific intervention log file (JSONL)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Label all intervention logs in logs/interventions/"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/labeled",
        help="Output directory for labeled results"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Max tokens for AO responses (default: 64 to capture mechanism)"
    )
    parser.add_argument(
        "--ao-model",
        type=str,
        default="adamkarvonen/checkpoints_cls_latentqa_past_lens_gemma-3-1b-it",
        help="Activation Oracle model name"
    )
    
    args = parser.parse_args()
    
    # Determine log files to process
    if args.log_file:
        log_files = [Path(args.log_file)]
    elif args.all:
        log_dir = Path(__file__).parent.parent / "logs" / "interventions"
        log_files = list(log_dir.glob("*.jsonl"))
    else:
        # Default: process both success logs
        log_dir = Path(__file__).parent.parent / "logs" / "interventions"
        log_files = [
            log_dir / "exploration_success.jsonl",
            log_dir / "policy_success.jsonl"
        ]
        log_files = [f for f in log_files if f.exists()]
    
    if not log_files:
        print("No log files found to process!")
        print("Run 'python src/train.py' first to generate intervention logs.")
        return
    
    print("="*60)
    print("AO-LABELING PROTOCOL (Phase 2B)")
    print("="*60)
    print(f"Log files to process: {len(log_files)}")
    for f in log_files:
        print(f"  - {f}")
    
    # Load Activation Oracle
    print(f"\nLoading Activation Oracle: {args.ao_model}")
    print("(This may take a moment...)")
    
    try:
        oracle = MultiLayerActivationOracle(model_name=args.ao_model)
        print("✓ Activation Oracle loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load Activation Oracle: {e}")
        print("\nNote: The AO requires:")
        print("  - CUDA/GPU availability")
        print("  - Sufficient VRAM (~4GB for Gemma-3-1b)")
        print("  - Internet connection to download model")
        return
    
    # Process each log file
    all_labeled = []
    output_dir = Path(args.output_dir)
    
    for log_path in log_files:
        if not log_path.exists():
            print(f"\nSkipping (not found): {log_path}")
            continue
        
        output_path = output_dir / f"{log_path.stem}_labeled.jsonl"
        
        labeled = label_all_interventions(
            log_path=log_path,
            oracle=oracle,
            output_path=output_path,
            max_new_tokens=args.max_tokens,
            verbose=True
        )
        
        all_labeled.extend(labeled)
    
    # Print summary
    if all_labeled:
        summarize_labels(all_labeled)
    
    print("\n" + "="*60)
    print("LABELING COMPLETE")
    print("="*60)
    print(f"Total interventions labeled: {len(all_labeled)}")
    print(f"Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
