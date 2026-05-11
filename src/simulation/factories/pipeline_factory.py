from typing import Dict, Any
from ..pipelines import (
    IdentityPipeline,
    PlanckProcessingPipeline, 
    UnbinnedPlanckProcessingPipeline, 
    PlanckBinningPipeline
)


class PipelineFactory:

    @staticmethod
    def get_available_pipelines() -> Dict[str, Any]:
        return {
            "identity": PipelineFactory.create_identity,
            "planck_processing": PipelineFactory.create_planck_processing,
            "unbinned_planck_processing": PipelineFactory.create_unbinned_planck_processing,
            "planck_binning": PipelineFactory.create_planck_binning,
        }

    @staticmethod
    def get_pipeline(pipeline_type: str):
        pipelines = PipelineFactory.get_available_pipelines()
        if pipeline_type not in pipelines:
            raise ValueError(f"Pipeline {pipeline_type} not found, available pipelines: {list(pipelines.keys())}")
        return pipelines[pipeline_type]()
    
    def create_identity():
        return IdentityPipeline()
    
    @staticmethod
    def create_planck_processing():
        return PlanckProcessingPipeline()
    
    @staticmethod
    def create_unbinned_planck_processing():
        return UnbinnedPlanckProcessingPipeline()
    
    @staticmethod
    def create_planck_binning():
        return PlanckBinningPipeline()
    

