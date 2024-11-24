import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
from functools import partial
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, precision_recall_curve)
from settings import TEST_ANNOTATIONS_PATH, PREDICTIONS_PATH, FIGURES_PATH


class ECGStatistics:
    def __init__(self, model_id: int):
        # Constants
        self.cache = {}
        self.model_id = model_id
        self.score_fun = {"Precision": precision_score, "Recall": recall_score,
                          "Specificity": partial(recall_score, pos_label=0), "F1 score": f1_score}
        self.predictor_names = ["DNN", "cardio.", "emerg.", "stud."]
        self.diagnosis = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]

        # Cardiologists, residents and students performance
        self.y_gold = pd.read_csv(TEST_ANNOTATIONS_PATH / "gold_standard.csv").values
        self.y_student = pd.read_csv(TEST_ANNOTATIONS_PATH / "medical_students.csv").values
        self.y_emerg = pd.read_csv(TEST_ANNOTATIONS_PATH / "emergency_residents.csv").values
        self.y_cardio = pd.read_csv(TEST_ANNOTATIONS_PATH / "cardiology_residents.csv").values
        self.y_dnn = np.load(PREDICTIONS_PATH / f"prediction_{model_id}.npy")

        # Get neural network prediction
        self.threshold = self.get_optimal_precision_recall(self.y_gold, self.y_dnn)[2]
        self.y_neuralnet = np.zeros_like(self.y_dnn)
        self.y_neuralnet[self.y_dnn > self.threshold] = 1

    @property
    def _bootstrapped_scores(self):
        if "bootstrapped" not in self.cache:
            self._generate_bootstrapped_scores()
        return self.cache["bootstrapped"]

    @staticmethod
    def get_scores(y_true, y_nn, score_func):
        scores_arr = []
        for name, fun in score_func.items():
            scores_arr += [[fun(y_true[:, k], y_nn[:, k])
                            for k in range(np.shape(y_true)[1])]]
        return np.array(scores_arr).T

    @staticmethod
    def get_optimal_precision_recall(y_true, y_nn):
        """Find precision and recall values that maximize f1 score."""
        opt_recall = []
        opt_precision = []
        opt_threshold = []
        for k in range(np.shape(y_true)[1]):
            # Get precision-recall curve
            precision, recall, thresh = precision_recall_curve(y_true[:, k], y_nn[:, k])
            f1 = np.nan_to_num(2 * precision * recall / (precision + recall))

            # Select threshold that maximize f1 score
            f1_index = np.argmax(f1)

            opt_recall.append(recall[f1_index])
            opt_precision.append(precision[f1_index])

            t = thresh[f1_index - 1] if f1_index != 0 else thresh[0] - 1e-10
            opt_threshold.append(t)

        return np.array(opt_precision), np.array(opt_recall), np.array(opt_threshold)

    def set_model_id(self, m_id: int):
        self.model_id = m_id
        self.cache.pop("bootstrapped")
        self.y_dnn = np.load(PREDICTIONS_PATH / f"prediction_{m_id}.npy")

        self.threshold = self.get_optimal_precision_recall(self.y_gold, self.y_dnn)[2]
        self.y_neuralnet = np.zeros_like(self.y_dnn)
        self.y_neuralnet[self.y_dnn > self.threshold] = 1

    def generate_table_two(self):
        scores_list = []
        for y_pred in [self.y_neuralnet, self.y_cardio, self.y_emerg, self.y_student]:
            scores = self.get_scores(self.y_gold, y_pred, self.score_fun)
            scores_df = pd.DataFrame(scores, index=self.diagnosis,
                                     columns=np.array(self.score_fun.keys()))
            scores_list.append(scores_df)

        # Concatenate dataframes
        scores_all_df = pd.concat(scores_list, axis=1, keys=self.predictor_names)
        scores_all_df = scores_all_df.swaplevel(0, 1, axis=1)
        scores_all_df = scores_all_df.reindex(level=0, columns=self.score_fun.keys())

        # Save results
        scores_all_df.to_csv(FIGURES_PATH / f"scores_{self.model_id}.csv", float_format="%.3f")

    def _generate_bootstrapped_scores(self):
        bootstrap_nsamples = 1000
        percentiles = [2.5, 97.5]
        scores_resampled_list = []
        scores_percentiles_list = []
        for y_pred in [self.y_neuralnet, self.y_cardio, self.y_emerg, self.y_student]:
            # Compute bootstrapped samples
            np.random.seed(123)  # Never change this
            n, _ = np.shape(self.y_gold)
            samples = np.random.randint(n, size=n * bootstrap_nsamples)

            y_true_resampled = np.reshape(self.y_gold[samples, :],
                                          (bootstrap_nsamples, n, len(self.diagnosis)))
            y_doctors_resampled = np.reshape(y_pred[samples, :],
                                             (bootstrap_nsamples, n, len(self.diagnosis)))

            # Apply functions
            scores_resampled = np.array([self.get_scores(y_true_resampled[i, :, :],
                                                         y_doctors_resampled[i, :, :], self.score_fun)
                                         for i in range(bootstrap_nsamples)])

            # Sort and append scores
            scores_resampled.sort(axis=0)
            scores_resampled_list.append(scores_resampled)

            # Get percentiles
            i = [int(p / 100.0 * bootstrap_nsamples) for p in percentiles]
            scores_percentiles = scores_resampled[i, :, :]
            scores_percentiles_df = pd.concat([
                pd.DataFrame(x, index=self.diagnosis, columns=np.array(self.score_fun.keys()))
                for x in scores_percentiles], keys=["p1", "p2"], axis=1
            )
            scores_percentiles_df = scores_percentiles_df.swaplevel(0, 1, axis=1)
            scores_percentiles_df = (scores_percentiles_df.reindex(level=0, columns=self.score_fun.keys()))
            scores_percentiles_list.append(scores_percentiles_df)

        self.cache["bootstrapped"] = bootstrap_nsamples, scores_resampled_list, scores_percentiles_list

    def generate_supplementary_figure_one(self):
        scores_resampled_xr = xr.DataArray(
            np.array(self._bootstrapped_scores[1]),
            dims=["predictor", "n", "diagnosis", "score_fun"],
            coords={
                "predictor": self.predictor_names,
                "n": range(self._bootstrapped_scores[0]),
                "score_fun": list(self.score_fun.keys()),
                "diagnosis": self.diagnosis,
            }
        )

        for sf in self.score_fun:
            fig, ax = plt.subplots()
            f1_score_resampled_xr = scores_resampled_xr.sel(score_fun=sf)
            f1_score_resampled_df = f1_score_resampled_xr.to_dataframe(name=sf).reset_index(level=[0, 1, 2])

            sns.boxplot(x="diagnosis", y=sf, hue="predictor", data=f1_score_resampled_df, ax=ax)

            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

            plt.xlabel("")
            plt.ylabel("", fontsize=16)

            if sf == "F1 score":
                plt.legend(fontsize=17)
            else:
                ax.legend().remove()

            plt.tight_layout()
            plt.savefig(FIGURES_PATH / f"boxplot_bootstrap_{sf.lower()}_{self.model_id}.pdf")

        (scores_resampled_xr.to_dataframe(name="score")
         .to_csv(FIGURES_PATH / f"boxplot_bootstrap_data_{self.model_id}.txt"))
