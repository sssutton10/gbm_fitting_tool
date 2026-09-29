"""GBM Fitting: Polars-native GBM library for insurance modeling."""
__version__ = "0.1.0"

from ins_gbm.progress import ProgressEvent, ProgressCallback, PipelineCancelled
from ins_gbm.data.folds import CVConfig
from ins_gbm.data.loader import load_model_data
from ins_gbm.data.model_data import ModelData
from ins_gbm.data.schema import FeatureSchema
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.models.xgboost import XGBoostModel
from ins_gbm.models.catboost import CatBoostModel
from ins_gbm.models.random_forest import RandomForestModel
from ins_gbm.pipeline import ModelPipeline, ModelRecipe, FittedPipeline
from ins_gbm.preprocessing.encoder import OneHotEncoder
from ins_gbm.preprocessing.steps import PreprocessingStep
from ins_gbm.preprocessing.pca import PCAReducer
from ins_gbm.preprocessing.pls import PLSReducer
from ins_gbm.selection import ImportanceSelectionStage, StagedImportanceSelector
from ins_gbm.selection.boruta import BorutaSelector
from ins_gbm.selection.cv_importance import cv_feature_importance
from ins_gbm.tuning.tuner import HyperparameterTuner
from ins_gbm.persistence.io import load_model, load_pipeline, save_pipeline
from ins_gbm.persistence.cv_io import load_cv_result
from ins_gbm.evaluation.cv_report import CrossValidationReport, CVResult
from ins_gbm.evaluation.comparison import compare_cv_double_lift, compare_reports
from ins_gbm.ensemble.pipeline import EnsemblePipeline, EnsembleResult

__all__ = [
    "BorutaSelector", "CVConfig", "CVResult", "CatBoostModel",
    "CrossValidationReport", "EnsemblePipeline", "EnsembleResult",
    "FeatureSchema", "FittedPipeline",
    "HyperparameterTuner", "LightGBMModel", "ModelData", "ModelPipeline",
    "ImportanceSelectionStage", "ModelRecipe", "OneHotEncoder", "PCAReducer",
    "PLSReducer", "PipelineCancelled", "PreprocessingStep",
    "ProgressCallback", "ProgressEvent", "RandomForestModel", "XGBoostModel",
    "StagedImportanceSelector", "compare_cv_double_lift", "compare_reports",
    "cv_feature_importance",
    "load_cv_result", "load_model", "load_model_data", "load_pipeline",
    "save_pipeline",
]
