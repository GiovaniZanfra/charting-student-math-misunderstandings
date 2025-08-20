#!/usr/bin/env python3
"""
Test script to verify transformer setup works correctly.
"""
import os
import sys
import tempfile
import pandas as pd

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        import datasets
        print(f"✓ Datasets {datasets.__version__}")
    except ImportError as e:
        print(f"✗ Datasets import failed: {e}")
        return False
    
    try:
        import peft
        print(f"✓ PEFT {peft.__version__}")
    except ImportError as e:
        print(f"✗ PEFT import failed: {e}")
        return False
    
    try:
        from charting_student_math_misunderstandings.config import TRANSFORMER_CONFIG
        print("✓ Project config imported")
    except ImportError as e:
        print(f"✗ Project config import failed: {e}")
        return False
    
    return True


def test_data_creation():
    """Test creating sample data for transformer training."""
    print("\nTesting data creation...")
    
    # Create sample data
    sample_data = {
        'row_id': [1, 2, 3],
        'QuestionText': [
            'What is 2 + 2?',
            'Solve for x: 3x = 9',
            'What is the area of a circle with radius 5?'
        ],
        'MC_Answer': ['4', '3', '25π'],
        'StudentExplanation': [
            'I added 2 and 2 to get 4',
            'I divided both sides by 3 to get x = 3',
            'I used the formula A = πr² and got 25π'
        ],
        'Category': ['True_Correct', 'True_Correct', 'True_Correct'],
        'Misconception': ['NA', 'NA', 'NA']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Test prompt creation
    from charting_student_math_misunderstandings.features import format_input
    
    prompt = format_input(
        df.iloc[0]['QuestionText'],
        df.iloc[0]['MC_Answer'],
        df.iloc[0]['StudentExplanation']
    )
    
    print("✓ Sample prompt created:")
    print(f"  {prompt[:100]}...")
    
    return True


def test_model_setup():
    """Test model and tokenizer setup."""
    print("\nTesting model setup...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        # Use a small model for testing
        model_name = "distilbert-base-uncased"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=10
        )
        
        print(f"✓ Model and tokenizer loaded: {model_name}")
        
        # Test tokenization
        test_text = "This is a test sentence for tokenization."
        tokens = tokenizer(
            test_text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        print(f"✓ Tokenization test passed: {tokens['input_ids'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model setup failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing Transformer Setup for MAP Competition")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_creation,
        test_model_setup,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Transformer setup is ready.")
        print("\nNext steps:")
        print("1. Run: make features")
        print("2. Run: make train-transformer EXP=exp01")
        print("3. Run: make predict-transformer EXP=exp01")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
