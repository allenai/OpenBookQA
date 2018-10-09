from obqa.commands.evaluate_custom import EvaluateCustom
from allennlp.commands import main as main_allennlp

from obqa.commands.evaluate_predictions_qa_mc import EvaluatePredictionsQA_MC
from obqa.commands.evaluate_predictions_qa_mc_know_visualize import EvaluatePredictionsQA_MC_Knowledge_Visualize


def main(prog: str = None) -> None:
    subcommand_overrides = {
        "evaluate_custom": EvaluateCustom(),
        "evaluate_predictions_qa_mc": EvaluatePredictionsQA_MC(),
        "evaluate_predictions_qa_mc_know_visualize": EvaluatePredictionsQA_MC_Knowledge_Visualize()
    }
    main_allennlp(prog, subcommand_overrides=subcommand_overrides)
