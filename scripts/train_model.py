from src.inference.api import train_model

if __name__ == "__main__":
    train_model(
        input_files=[
            "tt_1sigma_unbinned_planck_processing_50000_1.pt",
        ],
        prior_type="1sigma",
        inference_type="snpe_c_default",
        output_name="snpe_c_default_tt_1sigma_unbinned_planck_processing_50000_1.pth"
    )

    