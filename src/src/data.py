def generate_jailbreak_dataset(n=100):
    dataset = []
    for i in range(n):
        layers = f"L7:[0.9,0.1,-0.8] L12:[0.2,0.95,0.3] L23:[-0.7,0.8,0.1]"
        dataset.append({
            "input": f"<ACT>{layers}</ACT> Analyze jailbreak",
            "target": f"DISTRIBUTED JAILBREAK v{i}"
        })
    return dataset
