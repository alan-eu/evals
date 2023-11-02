import random
import re

import evals
import evals.metrics
from evals import CompletionFn


class Ecn(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_jsonl: str,
        num_few_shot: int = 0,
        few_shot_jsonl: str = None,
        **kwargs
    ):
        super().__init__(completion_fns, **kwargs)
        self.samples_jsonl = samples_jsonl
        self.num_few_shot = num_few_shot
        self.few_shot_jsonl = few_shot_jsonl

        if self.num_few_shot > 0:
            assert few_shot_jsonl is not None, "few shot requires few shot sample dataset"
            self.few_shot_jsonl = few_shot_jsonl
            self.few_shot = evals.get_jsonl(self.few_shot_jsonl)

    def run(self, recorder):
        """
        Called by the `oaieval` CLI to run the eval. The `eval_all_samples` method calls `eval_sample`.
        """

        samples = self.get_samples()
        self.eval_all_samples(recorder, samples)
        events = recorder.get_events("match")
        num_events = len(list(events))

        total_score = sum(map(lambda e: e.data["score"], events))
        average_score = total_score / num_events
        valid_format = f"{round(sum(map(lambda e: e.data['valid_format'], events)) / num_events * 100, 2)}%"

        return {
            "accuracy": evals.metrics.get_accuracy(events),
            "total_score": total_score,
            "average_score": average_score,
            "valid_format": valid_format,
        }

    def eval_sample(self, sample, rng: random.Random):
        """
        Called by the `eval_all_samples` method to evaluate a single sample.

        ARGS
        ====
        `test_sample`: a line from the JSONL test file
        `rng`: should be used for any randomness that is needed during evaluation

        This method does the following:
        1. Generate a prompt that contains the task statement, a few examples, and the test question.
        2. Generate a completion from the model.
        2. Check if the generated answer is correct.
        """

        prompt = sample["input"]
        expected = sample["ideal"].strip()
        if self.num_few_shot > 0:
            prompt = sample["input"][:-1]
            for s in self.few_shot[: self.num_few_shot]:
                prompt += s["sample"]
            prompt += sample["input"][-1:]

        result = self.completion_fn(
            prompt=prompt,
        )
        sampled = result.get_completions()[0].strip()

        valid_format = re.fullmatch(r'(^[A-E]$)|(^[A-E](,[A-E]){0,4}$)', sampled) is not None

        match = sampled == expected

        if valid_format:
            sampled_answers = set(sampled.split(','))
            try:
                expected_answers = set(expected.split(','))
            except Exception as e:
                print(f"{expected=}")
                raise e
            num_mistakes = len(sampled_answers - expected_answers) + len(expected_answers - sampled_answers)
            if num_mistakes == 0:
                score = 1.
            elif num_mistakes == 1:
                score = 0.5
            elif num_mistakes == 2:
                score = 0.2
            else:
                score = 0
        else:
            score = 0

        evals.record.record_match(
            match, expected=expected, sampled=sampled, score=score, valid_format=valid_format,
        )
