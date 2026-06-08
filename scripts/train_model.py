from src.inference.api import train_model

if __name__ == "__main__":

    train_model(
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
        prior_type="standard",
        inference_type="snpe_c_default",
        embedding_nn_filename="normalized_mlp_200_test_250k_cov_unbinned.pth",
        output_name="snpe_c_default_standard_test_250k_planck_binning_200_normalized_mlp_200_test_250k_cov_unbinned.pth"
    )

    # train_model(
    #     input_files=[
    #         "test_1_planck_binning_200.pt",
    #         "test_2_planck_binning_200.pt",
    #         "test_3_planck_binning_200.pt",
    #         "test_4_planck_binning_200.pt",
    #         "test_5_planck_binning_200.pt",
    #         "test_6_planck_binning_200.pt",
    #         "test_7_planck_binning_200.pt",
    #         "test_8_planck_binning_200.pt",
    #     ],
    #     prior_type="standard",
    #     inference_type="nle_default",
    #     output_name="nle_default_standard_test_200k_planck_binning_200.pth"
    # )
    # print("model saved")
    # train_model(
    #     input_files=[
    #         "test_1_planck_binning_200.pt",
    #         "test_2_planck_binning_200.pt",
    #         "test_3_planck_binning_200.pt",
    #         "test_4_planck_binning_200.pt",
    #         "test_5_planck_binning_200.pt",
    #         "test_6_planck_binning_200.pt",
    #         "test_7_planck_binning_200.pt",
    #         "test_8_planck_binning_200.pt",
    #         "test_9_planck_binning_200.pt",
    #         "test_10_planck_binning_200.pt",
    #     ],
    #     prior_type="standard",
    #     inference_type="nle_default",
    #     output_name="nle_default_standard_test_250k_planck_binning_200.pth"
    # )   
    # print("model saved")


        

    