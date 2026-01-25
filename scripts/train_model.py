from typing import Optional, List
from src.core import StorageManager
from src.inference.factories import NeuralInferenceFactory


def train_model_from_simulations(
    input_files: List[str],
    prior_type: str,
    inference_type: str,
    output_name: Optional[str] = None
):
    storage = StorageManager()

    print(f"Loading simulations from {len(input_files)} files...")
    theta, x = storage.load_multiple_simulations(input_files)
    inference = NeuralInferenceFactory.get_inference(inference_type, prior_type)
    
    print(f"Training {inference_type} model with {prior_type} prior...")
    inference.append_simulations(theta, x)
    density_estimator = inference.train()
    storage.save_model(density_estimator, input_files, prior_type, inference_type, output_name)
    
    
if __name__ == "__main__":

    train_model_from_simulations(
        input_files=[
            "test_1_cov_binned.pt",
            "test_2_cov_binned.pt",
        ],
        prior_type="standard",
        inference_type="npse_regular",
        output_name="npse_regular_standard_test_50k_cov_binned.pth",
    )

    train_model_from_simulations(
        input_files=[
            "test_1_cov_binned.pt",
            "test_2_cov_binned.pt",
            "test_3_cov_binned.pt",
            "test_4_cov_binned.pt",
        ],
        prior_type="standard",
        inference_type="npse_regular",
        output_name="npse_regular_standard_test_100k_cov_binned.pth",
    )

    train_model_from_simulations(
        input_files=[
            "test_1_cov_binned.pt",
            "test_2_cov_binned.pt",
            "test_3_cov_binned.pt",
            "test_4_cov_binned.pt",
            "test_5_cov_binned.pt",
            "test_6_cov_binned.pt",
        ],
        prior_type="standard",
        inference_type="npse_regular",
        output_name="npse_regular_standard_test_150k_cov_binned.pth",
    )

    train_model_from_simulations(
        input_files=[
            "test_1_cov_binned.pt",
            "test_2_cov_binned.pt",
            "test_3_cov_binned.pt",
            "test_4_cov_binned.pt",
            "test_5_cov_binned.pt",
            "test_6_cov_binned.pt",
            "test_7_cov_binned.pt",
            "test_8_cov_binned.pt",
        ],
        prior_type="standard",
        inference_type="npse_regular",
        output_name="npse_regular_standard_test_200k_cov_binned.pth",
    )

    train_model_from_simulations(
        input_files=[
            "test_1_cov_binned.pt",
            "test_2_cov_binned.pt",
            "test_3_cov_binned.pt",
            "test_4_cov_binned.pt",
            "test_5_cov_binned.pt",
            "test_6_cov_binned.pt",
            "test_7_cov_binned.pt",
            "test_8_cov_binned.pt",
            "test_9_cov_binned.pt",
            "test_10_cov_binned.pt",
        ],
        prior_type="standard",
        inference_type="npse_regular",
        output_name="npse_regular_standard_test_250k_cov_binned.pth",
    )
