from src.visualization.api import plot_and_save_regression

if __name__ == "__main__":
    plot_and_save_regression(
        embedding_nn_filename="normalized_mlp_200_test_250k_cov_unbinned.pth",
        param_labels=[r'$\omega_b$', r'$\omega_c$', r'$100\theta_{MC}$', r'$\ln(10^{10}A_s)$', r'$n_s$'],
        limits=[
            [0.02212-0.00022*5, 0.02212+0.00022*5],
            [0.1206-0.0021*5, 0.1206+0.0021*5],
            [1.04077-0.00047*5, 1.04077+0.00047*5],
            [3.04-0.016*5, 3.04+0.016*5],
            [0.9626-0.0057*5, 0.9626+0.0057*5]
        ],
        output_name="normalized_mlp_200_test_250k_cov_unbinned.pdf"
    )