from src.simulation.api import run_pipeline

if __name__ == "__main__":
    # run_pipeline(
    #     input_file=[
    #         "tt_1sigma_50000_1.pt",
    #     ],
    #     pipeline_type="unbinned_planck_processing",
    #     output_name="tt_1sigma_unbinned_planck_processing_50000_1.pt"
    # )

    run_pipeline(
        input_file=[
            "test_2.pt",
        ],
        pipeline_type="planck_binning_200",
        output_name="test_2_planck_binning_200.pt"
    )
    run_pipeline(
        input_file=[
            "test_3.pt",
        ],
        pipeline_type="planck_binning_200",
        output_name="test_3_planck_binning_200.pt"
    )
    run_pipeline(
        input_file=[
            "test_4.pt",
        ],
        pipeline_type="planck_binning_200",
        output_name="test_4_planck_binning_200.pt"
    )
    run_pipeline(
        input_file=[
            "test_5.pt",
        ],
        pipeline_type="planck_binning_200",
        output_name="test_5_planck_binning_200.pt"
    )
    run_pipeline(
        input_file=[
            "test_6.pt",
        ],
        pipeline_type="planck_binning_200",
        output_name="test_6_planck_binning_200.pt"
    )
    run_pipeline(
        input_file=[
            "test_7.pt",
        ],
        pipeline_type="planck_binning_200",
        output_name="test_7_planck_binning_200.pt"
    )
    run_pipeline(
        input_file=[
            "test_8.pt",
        ],
        pipeline_type="planck_binning_200",
        output_name="test_8_planck_binning_200.pt"
    )
    run_pipeline(
        input_file=[
            "test_9.pt",
        ],
        pipeline_type="planck_binning_200",
        output_name="test_9_planck_binning_200.pt"
    )
    run_pipeline(
        input_file=[
            "test_10.pt",
        ],
        pipeline_type="planck_binning_200",
        output_name="test_10_planck_binning_200.pt"
    )