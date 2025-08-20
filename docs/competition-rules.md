# MAP Competition Rules & Guidelines

## Competition Overview

**MAP Competition (Misconception Annotation Project)** - A Kaggle competition focused on building NLP/ML models to predict students' potential math misconceptions from open-ended explanations.

### Goal
Build an NLP/ML model that predicts students' potential math misconceptions from open-ended explanations. The model should output candidate misconceptions that help teachers identify and address misunderstandings.

### Why It Matters
- Students often misapply prior knowledge (e.g., thinking 0.355 > 0.8 because 355 > 8)
- Misconceptions are subtle, varied, and hard to label manually
- Automating this process improves diagnostic feedback at scale

## Task Details

### Input
- Student explanations in math (text data)

### Output
- Up to 3 predictions in the format `Category:Misconception` (space-delimited)
- Example: `True_Correct:NA False_Neither:NA False_Misconception:Incomplete`

### Evaluation Metric
- **Mean Average Precision @ 3 (MAP@3)**
- Only one correct label per row

## Submission Format

```csv
row_id,Category:Misconception
36696,True_Correct:NA False_Neither:NA False_Misconception:Incomplete
```

## Competition Details

### Hosts
- Vanderbilt University
- The Learning Agency
- Kaggle

### Timeline
- **Start Date**: July 10, 2025
- **Final Deadline**: October 15, 2025

### Prizes
Total prize pool: **$55,000**
- 1st Place: $20,000
- 2nd Place: $12,000
- 3rd Place: $8,000
- Additional prizes for other top performers

### Competition Rules
- **Type**: Code competition
- **Submission**: Must submit via Kaggle Notebook
- **Runtime Limit**: ≤ 9 hours
- **Internet Access**: No internet allowed during execution
- **External Data**: Public data allowed
- **Pre-trained Models**: Allowed

## Technical Requirements

### Model Constraints
- Must run within 9-hour time limit
- No internet access during execution
- Must be submitted as a Kaggle Notebook

### Data Handling
- Can use external public data
- Pre-trained models are permitted
- Must handle the specific output format requirements

### Output Format Validation
- Each prediction must follow `Category:Misconception` format
- Maximum 3 predictions per student explanation
- Space-delimited format for multiple predictions
- Must handle cases where fewer than 3 predictions are made

## Project Implementation Guidelines

### Data Processing
1. Load and preprocess student explanation text
2. Handle missing or malformed data
3. Implement text cleaning and normalization

### Model Development
1. Experiment with different NLP approaches
2. Consider pre-trained language models
3. Implement proper validation strategies
4. Optimize for MAP@3 metric

### Evaluation Strategy
1. Use cross-validation to estimate MAP@3
2. Implement proper train/validation/test splits
3. Monitor for overfitting
4. Test on diverse student explanation types

### Code Organization
- Follow the existing project structure
- Use the `charting_student_math_misunderstandings` package
- Implement modular code for easy experimentation
- Document all preprocessing and modeling steps

## Key Success Factors

1. **Text Understanding**: Deep understanding of mathematical misconceptions
2. **NLP Techniques**: Effective use of modern NLP methods
3. **Output Format**: Strict adherence to submission format
4. **Performance**: Optimization for MAP@3 metric
5. **Robustness**: Handling edge cases and diverse input types

## Resources & References

- Competition page on Kaggle
- Academic literature on math misconceptions
- NLP techniques for educational text analysis
- Previous similar competitions and approaches

---

**TL;DR**: Predict math misconceptions from student explanations using NLP. Output top-3 candidate `Category:Misconception` per student. Scored by MAP@3. Deadline Oct 15, 2025.
