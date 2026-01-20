from typing import Dict, Any
from ..pipelines import PlanckProcessingPipeline, UnbinnedPlanckProcessingPipeline


class PipelineFactory:

    @staticmethod
    def get_available_pipelines() -> Dict[str, Any]:
        return {
            "planck_processing": PipelineFactory.create_planck_processing,
            "unbinned_planck_processing": PipelineFactory.create_unbinned_planck_processing,
        }

    @staticmethod
    def get_pipeline(pipeline_type: str):
        pipelines = PipelineFactory.get_available_pipelines()
        if pipeline_type not in pipelines:
            raise ValueError(f"Pipeline {pipeline_type} not found, available pipelines: {list(pipelines.keys())}")
        return pipelines[pipeline_type]()
    
    @staticmethod
    def create_planck_processing():
        return PlanckProcessingPipeline()
    
    @staticmethod
    def create_unbinned_planck_processing():
        return UnbinnedPlanckProcessingPipeline()
    

