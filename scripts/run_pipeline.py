from typing import Optional
from src.core import StorageManager
from src.preprocessing.factories import PipelineFactory
from typing import List


def run_pipeline_on_saved_simulations(
    input_file: List[str],
    pipeline_type: str = "planck_processing",
    output_name: Optional[str] = None
):
    storage = StorageManager()
    theta, x_clean = storage.load_multiple_simulations(input_file)
    pipeline = PipelineFactory.get_pipeline(pipeline_type)   
    x_processed = pipeline.run_batch(x_clean)
    if output_name:
        storage.save_simulations(theta, x_processed, output_name)
        print(f"Results saved as: {output_name}")
    
    print(f"Shape: {x_processed.shape}")
    return theta, x_processed


if __name__ == "__main__":
    run_pipeline_on_saved_simulations(
        input_file=[
            "calibration_standard_tt_1000_1.pt",
        ],
        pipeline_type="planck_processing",
        output_name="calibration_planck_processing_standard_tt_1000_1.pt"
    )
