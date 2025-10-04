"""
Weighted Random Sampler for Balanced Training

Implements PyTorch-style weighted sampling to achieve balanced bull/bear representation
during training without physical data duplication.
"""

import numpy as np
import torch
from torch.utils.data import Sampler
from typing import Iterator, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeightedPortfolioSampler(Sampler):
    """
    Weighted random sampler for balanced bull/bear training.

    Samples indices according to provided weights, ensuring balanced
    bull/bear representation over each epoch without duplicating data.

    Args:
        weights: Sample weights (higher weight = higher sampling probability)
        num_samples: Number of samples to draw per epoch (default: len(weights))
        replacement: Sample with replacement (default: True for balancing)
    """

    def __init__(
        self,
        weights: np.ndarray,
        num_samples: Optional[int] = None,
        replacement: bool = True
    ):
        if not isinstance(weights, torch.Tensor):
            weights = torch.as_tensor(weights, dtype=torch.float32)

        if torch.any(weights < 0):
            raise ValueError("Weights must be non-negative")

        self.weights = weights
        self.num_samples = num_samples if num_samples is not None else len(weights)
        self.replacement = replacement

        logger.info(f"WeightedPortfolioSampler initialized:")
        logger.info(f"  Total weights: {len(self.weights)}")
        logger.info(f"  Samples per epoch: {self.num_samples}")
        logger.info(f"  Sampling with replacement: {self.replacement}")

        # Calculate effective distribution
        total_weight = self.weights.sum()
        unique_weights = torch.unique(self.weights)
        logger.info(f"  Unique weight values: {unique_weights.tolist()}")
        logger.info(f"  Total weight: {total_weight:.1f}")

    def __iter__(self) -> Iterator[int]:
        """
        Generate sample indices according to weights.

        Note: torch.multinomial samples PROPORTIONALLY based on weights.
        If you have 10 samples with weight=35 and 90 samples with weight=1:
        - Total weight for bear = 10 * 35 = 350
        - Total weight for bull = 90 * 1 = 90
        - Bear probability = 350/(350+90) = 79.5%

        This achieves the desired balance when bear samples are rare.
        """
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=self.replacement
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


class BalancedBatchSampler(Sampler):
    """
    Batch sampler that ensures each batch has balanced bull/bear samples.

    More sophisticated than WeightedPortfolioSampler - guarantees exact
    50/50 split within each batch.

    Args:
        bear_indices: Indices of bear market samples
        bull_indices: Indices of bull market samples
        batch_size: Size of each batch
        drop_last: Drop incomplete batches (default: False)
    """

    def __init__(
        self,
        bear_indices: np.ndarray,
        bull_indices: np.ndarray,
        batch_size: int,
        drop_last: bool = False
    ):
        self.bear_indices = bear_indices
        self.bull_indices = bull_indices
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Each batch should have batch_size//2 bear and batch_size//2 bull
        self.bear_per_batch = batch_size // 2
        self.bull_per_batch = batch_size - self.bear_per_batch

        # Calculate number of batches
        self.num_batches = min(
            len(bear_indices) // self.bear_per_batch,
            len(bull_indices) // self.bull_per_batch
        )

        logger.info(f"BalancedBatchSampler initialized:")
        logger.info(f"  Bear samples: {len(bear_indices)}")
        logger.info(f"  Bull samples: {len(bull_indices)}")
        logger.info(f"  Batch size: {batch_size} ({self.bear_per_batch} bear + {self.bull_per_batch} bull)")
        logger.info(f"  Number of batches: {self.num_batches}")

    def __iter__(self) -> Iterator[list]:
        """Generate balanced batches."""
        # Shuffle indices
        bear_perm = np.random.permutation(self.bear_indices)
        bull_perm = np.random.permutation(self.bull_indices)

        for i in range(self.num_batches):
            # Get bear samples for this batch
            bear_batch = bear_perm[
                i * self.bear_per_batch : (i + 1) * self.bear_per_batch
            ]

            # Get bull samples for this batch
            bull_batch = bull_perm[
                i * self.bull_per_batch : (i + 1) * self.bull_per_batch
            ]

            # Combine and shuffle within batch
            batch = np.concatenate([bear_batch, bull_batch])
            np.random.shuffle(batch)

            yield batch.tolist()

    def __len__(self) -> int:
        return self.num_batches


def create_balanced_sampler(
    sample_weights: np.ndarray,
    sampler_type: str = 'weighted',
    batch_size: Optional[int] = None,
    bear_indices: Optional[np.ndarray] = None,
    bull_indices: Optional[np.ndarray] = None
) -> Sampler:
    """
    Factory function to create appropriate sampler.

    Args:
        sample_weights: Array of sample weights
        sampler_type: 'weighted' or 'balanced_batch'
        batch_size: Required if sampler_type='balanced_batch'
        bear_indices: Required if sampler_type='balanced_batch'
        bull_indices: Required if sampler_type='balanced_batch'

    Returns:
        Sampler instance
    """
    if sampler_type == 'weighted':
        return WeightedPortfolioSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

    elif sampler_type == 'balanced_batch':
        if batch_size is None or bear_indices is None or bull_indices is None:
            raise ValueError("balanced_batch sampler requires batch_size, bear_indices, and bull_indices")

        return BalancedBatchSampler(
            bear_indices=bear_indices,
            bull_indices=bull_indices,
            batch_size=batch_size,
            drop_last=False
        )

    else:
        raise ValueError(f"Unknown sampler_type: {sampler_type}. Use 'weighted' or 'balanced_batch'")


if __name__ == "__main__":
    # Test weighted sampler
    logger.info("Testing WeightedPortfolioSampler...")

    # Simulate: 100 samples, 10 bear (weight=35) and 90 bull (weight=1)
    weights = np.ones(100)
    weights[:10] = 35.0  # First 10 are bear

    sampler = WeightedPortfolioSampler(weights, num_samples=100)

    # Sample and count
    samples = list(sampler)
    bear_samples = sum(1 for s in samples if s < 10)
    bull_samples = sum(1 for s in samples if s >= 10)

    logger.info(f"Sampled {len(samples)} indices:")
    logger.info(f"  Bear samples (idx < 10): {bear_samples} ({bear_samples/len(samples):.1%})")
    logger.info(f"  Bull samples (idx >= 10): {bull_samples} ({bull_samples/len(samples):.1%})")
    logger.info(f"  Balance: {'YES ✅' if 0.45 <= bear_samples/len(samples) <= 0.55 else 'NO ❌'}")

    # Test balanced batch sampler
    logger.info("\nTesting BalancedBatchSampler...")

    bear_indices = np.arange(10)
    bull_indices = np.arange(10, 100)
    batch_sampler = BalancedBatchSampler(
        bear_indices=bear_indices,
        bull_indices=bull_indices,
        batch_size=32
    )

    # Check first batch
    batches = list(batch_sampler)
    first_batch = batches[0]
    bear_in_batch = sum(1 for idx in first_batch if idx < 10)
    bull_in_batch = sum(1 for idx in first_batch if idx >= 10)

    logger.info(f"First batch ({len(first_batch)} samples):")
    logger.info(f"  Bear: {bear_in_batch}")
    logger.info(f"  Bull: {bull_in_batch}")
    logger.info(f"  Ratio: {bear_in_batch/len(first_batch):.1%} bear")

    logger.info("\nTest complete!")
