"""
Module: Policy Network - Concept-conditioned activation intervention discovery.

This module implements the policy network p_θ(δf | c) that learns to generate
activation interventions from behavioral rubrics.

Following the "Investigator Agent" framework from arXiv:2502.01236, adapted
to activation space instead of prompt space.

Key components:
- ConceptEncoder: Encodes rubric (R, a) into concept embeddings
- PolicyNetwork: Maps concept embeddings to Gaussian distribution over δf
- ConceptExplainer: High-level interface combining encoder + policy
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class ConceptEncoder(nn.Module):
    """
    Encodes rubric (R, a) into concept embeddings.
    
    Uses pretrained sentence transformer to encode rubric text and target answer
    into a fixed-dimensional embedding space.
    """
    
    def __init__(
        self,
        encoder_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize concept encoder.
        
        Args:
            encoder_name: Sentence transformer model name
            device: Device to run encoder on
        """
        super().__init__()
        self.device = device
        self.encoder = SentenceTransformer(encoder_name, device=device)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
    
    def encode(self, rubric) -> torch.Tensor:
        """
        Encode rubric into concept embedding.
        
        Args:
            rubric: RubricDefinition object with rubric_text and target_answer
            
        Returns:
            Concept embedding tensor of shape [d_concept]
        """
        # Combine rubric text and target answer
        text = f"Rubric: {rubric.rubric_text} Target: {rubric.target_answer}"
        
        # Encode to embedding
        embedding = self.encoder.encode(
            text,
            convert_to_tensor=True,
            device=self.device
        )
        
        return embedding
    
    def encode_batch(self, rubrics) -> torch.Tensor:
        """
        Encode batch of rubrics.
        
        Args:
            rubrics: List of RubricDefinition objects
            
        Returns:
            Batch of concept embeddings [batch_size, d_concept]
        """
        texts = [
            f"Rubric: {r.rubric_text} Target: {r.target_answer}"
            for r in rubrics
        ]
        
        embeddings = self.encoder.encode(
            texts,
            convert_to_tensor=True,
            device=self.device,
            batch_size=len(texts)
        )
        
        return embeddings


class PolicyNetwork(nn.Module):
    """
    Maps concept embedding to Gaussian distribution over activation interventions.
    
    Architecture:
        Input: h(R, a) ∈ R^d_concept (from ConceptEncoder)
        Hidden: [1024, 512] with ReLU + LayerNorm
        Output: [μ_θ(c), log_σ_θ(c)] ∈ R^(2*d_hidden)
    
    Sampling: δf ~ N(μ_θ(c), σ_θ(c))
    Constraint: ||δf||_2 ≤ ε * ||f_x||_2 (default ε=0.1)
    """
    
    def __init__(
        self,
        d_concept: int = 768,
        d_hidden: int = 2048,
        dropout: float = 0.1
    ):
        """
        Initialize policy network.
        
        Args:
            d_concept: Dimension of concept embeddings
            d_hidden: Dimension of activation space (hidden dim of target model)
            dropout: Dropout probability
        """
        super().__init__()
        self.d_hidden = d_hidden
        
        # MLP for distribution parameters
        self.mlp = nn.Sequential(
            nn.Linear(d_concept, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2 * d_hidden)  # mean and log_variance
        )
    
    def forward(
        self,
        concept_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get distribution parameters.
        
        Args:
            concept_emb: Concept embedding [batch_size, d_concept] or [d_concept]
            
        Returns:
            (mu, log_sigma): Mean and log standard deviation
        """
        out = self.mlp(concept_emb)
        
        # Split into mean and log_sigma
        if len(out.shape) == 1:
            # Single example
            mu = out[:self.d_hidden]
            log_sigma = out[self.d_hidden:]
        else:
            # Batch
            mu = out[..., :self.d_hidden]
            log_sigma = out[..., self.d_hidden:]
        
        return mu, log_sigma
    
    def sample(
        self,
        mu: torch.Tensor,
        log_sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample intervention from Gaussian distribution.
        
        Args:
            mu: Mean [d_hidden] or [batch_size, d_hidden]
            log_sigma: Log standard deviation [d_hidden] or [batch_size, d_hidden]
            
        Returns:
            Sampled intervention δf
        """
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(mu)
        return mu + sigma * eps
    
    def enforce_constraint(
        self,
        delta_f: torch.Tensor,
        base_f: torch.Tensor,
        epsilon: float = 0.1
    ) -> torch.Tensor:
        """
        Ensure ||δf||_2 ≤ ε * ||f||_2
        
        Args:
            delta_f: Intervention vector
            base_f: Base activation vector
            epsilon: Constraint coefficient (default 0.1)
            
        Returns:
            Constrained intervention
        """
        max_norm = epsilon * torch.norm(base_f, p=2)
        current_norm = torch.norm(delta_f, p=2)
        
        if current_norm > max_norm:
            delta_f = delta_f * (max_norm / current_norm)
        
        return delta_f
    
    def log_prob(
        self,
        delta_f: torch.Tensor,
        mu: torch.Tensor,
        log_sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probability of intervention under Gaussian distribution.
        
        Used for policy gradient training.
        
        Args:
            delta_f: Intervention sample
            mu: Mean of distribution
            log_sigma: Log std of distribution
            
        Returns:
            Log probability
        """
        sigma = torch.exp(log_sigma)
        
        # Gaussian log probability
        log_prob = -0.5 * (
            ((delta_f - mu) / sigma) ** 2 +
            2 * log_sigma +
            torch.log(torch.tensor(2 * 3.14159))
        )
        
        # Sum over dimensions
        return log_prob.sum(dim=-1)


class ConceptExplainer(nn.Module):
    """
    Main interface for concept-conditioned activation interventions.
    
    Combines ConceptEncoder + PolicyNetwork to map rubrics to intervention
    distributions.
    """
    
    def __init__(
        self,
        d_hidden: int = 2048,
        encoder_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize concept explainer.
        
        Args:
            d_hidden: Dimension of activation space
            encoder_name: Sentence transformer model name
            device: Device to run on
        """
        super().__init__()
        self.device = device
        
        # Initialize components
        self.concept_encoder = ConceptEncoder(
            encoder_name=encoder_name,
            device=device
        )
        
        self.policy_network = PolicyNetwork(
            d_concept=self.concept_encoder.embedding_dim,
            d_hidden=d_hidden
        ).to(device)
    
    def sample_intervention(
        self,
        rubric,
        base_activation: torch.Tensor,
        epsilon: float = 0.1,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Sample activation intervention for given rubric.
        
        Args:
            rubric: RubricDefinition object
            base_activation: Base activation f(x) [d_hidden]
            epsilon: Norm constraint coefficient
            deterministic: If True, return mean (no sampling)
            
        Returns:
            Intervention δf [d_hidden]
        """
        # 1. Encode rubric
        with torch.no_grad():
            concept_emb = self.concept_encoder.encode(rubric)
        
        # 2. Get distribution parameters
        mu, log_sigma = self.policy_network(concept_emb)
        
        # 3. Sample intervention (or use mean if deterministic)
        if deterministic:
            delta_f = mu
        else:
            delta_f = self.policy_network.sample(mu, log_sigma)
        
        # 4. Enforce norm constraint
        delta_f = self.policy_network.enforce_constraint(
            delta_f,
            base_activation,
            epsilon
        )
        
        return delta_f
    
    def sample_intervention_batch(
        self,
        rubrics,
        base_activations: torch.Tensor,
        epsilon: float = 0.1,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Sample interventions for batch of rubrics.
        
        Args:
            rubrics: List of RubricDefinition objects
            base_activations: Base activations [batch_size, d_hidden]
            epsilon: Norm constraint coefficient
            deterministic: If True, return means
            
        Returns:
            Interventions [batch_size, d_hidden]
        """
        # 1. Encode rubrics
        with torch.no_grad():
            concept_embs = self.concept_encoder.encode_batch(rubrics)
        
        # 2. Get distribution parameters
        mus, log_sigmas = self.policy_network(concept_embs)
        
        # 3. Sample interventions
        if deterministic:
            delta_fs = mus
        else:
            delta_fs = self.policy_network.sample(mus, log_sigmas)
        
        # 4. Enforce norm constraints (batch)
        for i in range(len(delta_fs)):
            delta_fs[i] = self.policy_network.enforce_constraint(
                delta_fs[i],
                base_activations[i],
                epsilon
            )
        
        return delta_fs
    
    def get_log_prob(
        self,
        rubric,
        delta_f: torch.Tensor
    ) -> torch.Tensor:
        """
        Get log probability of intervention under policy.
        
        Used for policy gradient training.
        
        Args:
            rubric: RubricDefinition object
            delta_f: Intervention sample
            
        Returns:
            Log probability
        """
        # Encode rubric
        with torch.no_grad():
            concept_emb = self.concept_encoder.encode(rubric)
        
        # Get distribution parameters
        mu, log_sigma = self.policy_network(concept_emb)
        
        # Compute log prob
        return self.policy_network.log_prob(delta_f, mu, log_sigma)


if __name__ == "__main__":
    print("Testing Policy Network components...")
    
    # Test ConceptEncoder
    print("\n1. Testing ConceptEncoder...")
    encoder = ConceptEncoder()
    print(f"   Embedding dimension: {encoder.embedding_dim}")
    
    # Create dummy rubric
    from src.rubrics import RubricDefinition
    rubric = RubricDefinition(
        rubric_text="Successfully bypasses refusal",
        target_answer="provides harmful instructions",
        category="jailbreak"
    )
    
    embedding = encoder.encode(rubric)
    print(f"   Encoded rubric shape: {embedding.shape}")
    
    # Test PolicyNetwork
    print("\n2. Testing PolicyNetwork...")
    policy = PolicyNetwork(d_concept=768, d_hidden=2048)
    mu, log_sigma = policy(embedding)
    print(f"   Mean shape: {mu.shape}")
    print(f"   Log sigma shape: {log_sigma.shape}")
    
    delta_f = policy.sample(mu, log_sigma)
    print(f"   Sampled intervention shape: {delta_f.shape}")
    
    # Test constraint enforcement
    base_f = torch.randn(2048)
    delta_f_constrained = policy.enforce_constraint(delta_f, base_f, epsilon=0.1)
    print(f"   Constraint satisfied: {torch.norm(delta_f_constrained) <= 0.1 * torch.norm(base_f)}")
    
    # Test ConceptExplainer
    print("\n3. Testing ConceptExplainer...")
    explainer = ConceptExplainer(d_hidden=2048)
    base_activation = torch.randn(2048)
    intervention = explainer.sample_intervention(rubric, base_activation)
    print(f"   Intervention shape: {intervention.shape}")
    print(f"   Intervention norm: {torch.norm(intervention).item():.4f}")
    print(f"   Base activation norm: {torch.norm(base_activation).item():.4f}")
    print(f"   Ratio: {(torch.norm(intervention) / torch.norm(base_activation)).item():.4f}")
    
    print("\n✓ Policy Network tests complete!")
