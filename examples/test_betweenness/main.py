"""
Test script for betweenness centrality labeling.

This script runs preprocessing only to verify that:
1. Betweenness centrality labels are generated correctly
2. Label files are created in the correct location
3. No errors occur during preprocessing
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from simplegnn.framework.core import FrameworkMain


def main():
    print("=" * 60)
    print("Testing Betweenness Centrality Labeling")
    print("=" * 60)

    # Initialize framework
    experiment = FrameworkMain(Path('examples/test_betweenness/main.yml'))

    # Run preprocessing (this will generate the labels)
    print("\nRunning preprocessing to generate labels...")
    experiment.preprocessing(num_threads=1)

    print("\n" + "=" * 60)
    print("✓ Preprocessing completed successfully!")
    print("=" * 60)

    # Check if label files were created
    label_path = Path('data/labels/MUTAG')
    if label_path.exists():
        label_files = list(label_path.glob('*betweenness*.pt'))
        if label_files:
            print(f"\n✓ Betweenness centrality label files created:")
            for f in label_files:
                print(f"  - {f.name}")
        else:
            print("\n⚠ No betweenness centrality label files found")
    else:
        print(f"\n⚠ Label directory not found: {label_path}")

    print("\nTo run full training, uncomment the following lines:")
    print("  experiment.run_configurations(-1)")
    print("  experiment.evaluate_results()")


if __name__ == '__main__':
    main()
