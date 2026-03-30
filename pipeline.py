import numpy as np

from analysis import MispricingAnalyzer
from data_loader import DataBundle, DataLoader
from evaluation import Evaluator
from features import get_all_features, normalize
from features.temporal_overlap import TemporalOverlapFeature
from models import EvaluationResult, OptimizationResult, PipelineConfig
from optimizer import WeightOptimizer


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.loader = DataLoader(config)
        self.features = get_all_features(config)
        self.optimizer = WeightOptimizer(config)
        self.evaluator = Evaluator(config)
        self.analyzer = MispricingAnalyzer()

    def run(self) -> dict:
        """
        Full pipeline:
        1. Load data
        2. Split train/test
        3. Compute + normalize all 7 feature matrices on train
        4. Build ground truth and valid-pair mask
        5. Hill climb for optimal weights
        6. Repeat feature computation on test
        7. Evaluate on test
        8. Mispricing analysis
        9. Print results
        """
        # ── 1. Load ──────────────────────────────────────────────────────────
        print("=== Loading data ===")
        data = self.loader.load()

        # ── 2. Split ─────────────────────────────────────────────────────────
        print("\n=== Splitting train / test ===")
        train_idx, test_idx = self.evaluator.split_by_time(data.markets)
        print(f"Train markets: {len(train_idx)}  |  Test markets: {len(test_idx)}")

        if len(train_idx) < 10 or len(test_idx) < 5:
            raise ValueError(
                f"Too few markets after split "
                f"(train={len(train_idx)}, test={len(test_idx)}). "
                "Check train_cutoff in PipelineConfig."
            )

        train_data = data.subset(train_idx)
        test_data = data.subset(test_idx)

        # ── 3. Compute features on train ─────────────────────────────────────
        print("\n=== Computing features on train set ===")
        raw_train, normalized_train = self._compute_features(train_data)

        # ── 4. Ground truth + mask ────────────────────────────────────────────
        ground_truth_train = self.evaluator.build_ground_truth(data.markets, train_idx)
        overlap_days_train = TemporalOverlapFeature().compute_overlap_days(train_data)
        mask_train = self.evaluator.build_valid_pair_mask(overlap_days_train)

        valid_pairs = int(mask_train.sum())
        coresolved = int(ground_truth_train[mask_train].sum())
        print(f"Valid train pairs: {valid_pairs}  |  Co-resolved: {coresolved}")

        # ── 5. Optimize weights ───────────────────────────────────────────────
        print("\n=== Optimizing weights (hill climbing) ===")
        feature_names = [f.name for f in self.features]
        opt_result: OptimizationResult = self.optimizer.optimize(
            normalized_train, ground_truth_train, mask_train, feature_names
        )

        print(f"\nOptimal weights:")
        for name, w in zip(opt_result.feature_names, opt_result.weights):
            print(f"  {name:30s}  {w:.4f}")
        print(f"Train MSE: {opt_result.train_mse:.6f}")

        # ── 6. Compute features on test ───────────────────────────────────────
        print("\n=== Computing features on test set ===")
        raw_test, normalized_test = self._compute_features(test_data)

        ground_truth_test = self.evaluator.build_ground_truth(data.markets, test_idx)
        overlap_days_test = TemporalOverlapFeature().compute_overlap_days(test_data)
        mask_test = self.evaluator.build_valid_pair_mask(overlap_days_test)

        # ── 7. Evaluate ───────────────────────────────────────────────────────
        print("\n=== Evaluating on test set ===")
        weights = np.array(opt_result.weights)
        eval_result: EvaluationResult = self.evaluator.evaluate(
            normalized_test, ground_truth_test, mask_test, weights, feature_names
        )

        print(f"\nTest MSE:           {eval_result.test_mse:.6f}")
        print(f"Test Pearson r:     {eval_result.test_pearson_r:.4f}  (p={eval_result.test_pearson_p:.4f})")
        print(f"\nBaseline MSE comparison:")
        print(f"  Random weights:   {eval_result.baseline_mse_random:.6f}")
        print(f"  Equal weights:    {eval_result.baseline_mse_equal:.6f}")
        print(f"  Price-only:       {eval_result.baseline_mse_price_only:.6f}")
        print(f"  Linear regression:{eval_result.baseline_mse_linreg:.6f}")
        print(f"  Optimized:        {eval_result.test_mse:.6f}")

        # ── 8. Mispricing analysis ────────────────────────────────────────────
        print("\n=== Mispricing analysis ===")
        predictivity = sum(w * F for w, F in zip(weights, normalized_test))
        mispricing_df = self.analyzer.compute_mispricing_table(
            test_data, predictivity, test_idx, ground_truth_test, mask_test
        )
        stratified = self.analyzer.stratified_analysis(mispricing_df)

        if not stratified.empty:
            print("\nStratified mispricing (higher predicted score → larger gap?)")
            print(stratified.to_string(index=False))
        else:
            print("  (insufficient outcome price data for mispricing analysis)")

        return {
            "optimization": opt_result,
            "evaluation": eval_result,
            "mispricing_table": mispricing_df,
            "stratified_mispricing": stratified,
        }

    def _compute_features(
        self, data: DataBundle
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Run all 7 features and return (raw_matrices, normalized_matrices).
        """
        raw, normalized = [], []
        for feature in self.features:
            print(f"  Computing {feature.name}...")
            matrix = feature.compute(data)
            raw.append(matrix)
            normalized.append(normalize(matrix))
        return raw, normalized


if __name__ == "__main__":
    config = PipelineConfig()
    pipeline = Pipeline(config)
    pipeline.run()
