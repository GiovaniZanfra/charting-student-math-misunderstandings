#!/usr/bin/env python3
"""
Test GPU setup for RTX 3050.
"""
import torch
import transformers

def test_gpu():
    print("Testing GPU Setup for RTX 3050")
    print("=" * 40)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.version.cuda}")
        print(f"✓ GPU device: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        print(f"✓ PyTorch CUDA: {torch.version.cuda}")
        
        # Test tensor operations
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.mm(x, y)
        print(f"✓ GPU tensor operations working")
        
        # Test model loading
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", 
            num_labels=10,
            torch_dtype=torch.float16
        ).cuda()
        print(f"✓ Model loaded on GPU with FP16")
        
        # Test inference
        inputs = tokenizer("Hello world", return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"✓ GPU inference working")
        
        return True
    else:
        print("✗ CUDA not available")
        return False

if __name__ == "__main__":
    test_gpu()
