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
            "tt_1sigma_50000_2.pt",
        ],
        pipeline_type="unbinned_planck_processing",
        output_name="tt_1sigma_unbinned_planck_processing_50000_2.pt"
    )

