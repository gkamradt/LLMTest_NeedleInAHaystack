from unittest.mock import patch, MagicMock, call, ANY

from needlehaystack.evaluators import OpenAIEvaluator

QUESTION_ASKED = "What is the color of the sky?"
QUESTION_ANSWER = "Sky is blue"
API_KEY = "abc"
SCORE = 123
TEMPERATURE = 0
MODEL = "gpt-3.5-turbo-0125"


@patch('needlehaystack.evaluators.openai.ChatOpenAI')
@patch('needlehaystack.evaluators.openai.load_evaluator')
def test_openai(mock_load_evaluator, mock_chat_open_ai, monkeypatch):
    monkeypatch.setenv('NIAH_EVALUATOR_API_KEY', API_KEY)

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate_strings.return_value = {'score': str(SCORE)}

    mock_load_evaluator.return_value = mock_evaluator

    evaluator = OpenAIEvaluator(question_asked=QUESTION_ASKED, true_answer=QUESTION_ANSWER)
    result = evaluator.evaluate_response("Something")

    assert mock_chat_open_ai.call_args == call(model=MODEL, temperature=TEMPERATURE, openai_api_key=API_KEY)
    assert mock_load_evaluator.call_args == call('labeled_score_string', criteria=OpenAIEvaluator.CRITERIA, llm=ANY)

    assert result == SCORE
