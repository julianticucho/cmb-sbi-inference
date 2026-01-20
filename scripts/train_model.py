from typing import Optional, List
from src.core import StorageManager
from src.inference.factories import InferenceFactory


def train_model_from_simulations(
    input_files: List[str],
    prior_type: str = "standard",
    inference_type: str = "snpe_c",
    output_name: Optional[str] = None
):
    storage = StorageManager()

    print(f"Loading simulations from {len(input_files)} files...")
    theta, x = storage.load_multiple_simulations(input_files)
    inference = InferenceFactory.get_inference(inference_type, prior_type)
    
    print(f"Training {inference_type} model with {prior_type} prior...")
    inference.append_simulations(theta, x)
    density_estimator = inference.train()
    storage.save_model(density_estimator, input_files, prior_type, inference_type, output_name)
    
    
if __name__ == "__main__":
    train_model_from_simulations(
        input_files=["test_1_cov_binned.pt"],
        prior_type="standard",
        inference_type="fmpe_default",
        output_name="fmpe_default_25k_cov_binned.pth",
    )
    


