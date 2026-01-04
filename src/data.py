"""
Module 1: Data & Tasks - Generate multi-layer jailbreak dataset
"""
def generate_jailbreak_dataset(n_examples=1000):
    """Generate L7:L12:L23 activation patterns → 'JAILBREAK' labels"""
    dataset = []
    for i in range(n_examples):
        layers = f"L7:[0.9,0.1,-0.8] L12:[0.2,0.95,0.3] L23:[-0.7,0.8,0.1]"
        dataset.append({
            "input": f"<ACT>{layers}</ACT> Analyze jailbreak",
            "target": f"DISTRIBUTED JAILBREAK v{i}: safety-inhibited + refusal-bypassed"
        })
    return dataset

if __name__ == "__main__":
    data = generate_jailbreak_dataset(10)
    print(f" Module 1 complete: {len(data)} examples ready")
