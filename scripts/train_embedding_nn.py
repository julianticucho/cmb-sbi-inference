from src.compression.api import train_embedding_nn

if __name__ == "__main__":
    train_embedding_nn(
        input_files=[
            "test_1_planck_binning_200.pt",
            "test_2_planck_binning_200.pt",
            "test_3_planck_binning_200.pt",
            "test_4_planck_binning_200.pt",
            "test_5_planck_binning_200.pt",
            "test_6_planck_binning_200.pt",
            "test_7_planck_binning_200.pt",
            "test_8_planck_binning_200.pt",
            "test_9_planck_binning_200.pt",
            "test_10_planck_binning_200.pt",
        ],
        dataloader_name="normalized",
        model_name="mlp_200",
        output_name="normalized_mlp_200_test_250k_cov_unbinned.pth"
    )