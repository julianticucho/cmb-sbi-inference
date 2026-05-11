from src.inference.api import train_model

if __name__ == "__main__":

    train_model(
        input_files=[
            "auxiliary_observables_100000_0.pt"
        ],
        prior_type="standard",
        inference_type="snpe_c_default",
        output_name="snpe_c_default_auxiliary_observables_100000_0.pth"
    )
        

    