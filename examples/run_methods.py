"""Offline integration of paper mechanisms. Fixture scores are not LLM evidence."""
import hashlib
import re

from evoharness import (
    DistributionConfig,
    Evaluator,
    Experience,
    Genome,
    ModelSpec,
    PSOConfig,
    Response,
    Task,
    evaluate_frozen,
    jfk_scores,
    optimize_prompts,
    reinforce_graph,
    run_genome,
    search_roles,
    split_tasks,
)


class FixtureBackend:
    """Deterministic test fixture; reflection text is scripted, not learned."""
    def complete(self, prompt, model, sample=0):
        if prompt.startswith('Reflect on'):
            text = 'Check upstream answers against the task before choosing a final answer.'
        else:
            match = re.search(r'Task: q(\d+)\n', prompt)
            ref = 'a' + match[1] if match else '?'
            value = int(hashlib.sha256(f'{model.name}|{sample}|{prompt}'.encode()).hexdigest(), 16)
            text = 'Answer: ' + (ref if value % 10 < 7 else 'wrong')
        return Response(text, len(prompt) // 4, len(text) // 4, 0.001)


def main():
    tasks = [Task(f'q{i}', f'a{i}') for i in range(30)]
    split = split_tasks(tasks, seed=7)
    models = [ModelSpec('fixture-a', 0, 0), ModelSpec('fixture-b', 0, 0)]
    backend = FixtureBackend()
    train = Evaluator(split.train, backend, models)
    validation = Evaluator(split.validation, backend, models)
    test = Evaluator(split.test, backend, models)
    template = Genome([0, 1, 2], [0, 1, 0], {(0, 1), (1, 2)})

    pso = search_roles(template, train.utility, PSOConfig(particles=6, iterations=8, seed=7))
    policy = reinforce_graph(template, train.utility, DistributionConfig(iterations=12, batch_size=8, seed=7))
    candidates = [pso.genome, policy.best_genome]
    winner = max(candidates, key=lambda g: validation.score(g)[0]).copy()
    print('JFK scores:', jfk_scores(winner, len(models), train.utility, assignments=20, seed=7))

    # One completed historical window, containing observed full-graph I/O.
    history = []
    for task in split.train:
        result = run_genome(winner, task, backend, models, train.roles)
        reward = 1.0 if result.answer == task.reference else -1.0
        history.extend(Experience(i, task.prompt, output, reward) for i, output in result.outputs.items())
    update = optimize_prompts(winner, train.roles, history,
                              lambda g: train.attribute(g, deterministic=True),
                              backend, models[0], threshold=1.0)
    print('Contributions:', update.attribution.scores, 'updated node:', update.target)

    # Validate prompt changes before consulting the final test set.
    revised_validation = Evaluator(split.validation, backend, models, update.roles)
    if revised_validation.score(update.genome)[0] >= validation.score(winner)[0]:
        winner, test.roles = update.genome, update.roles
    report = evaluate_frozen([winner], test, search_tasks=split.train + split.validation, samples=3)[0]
    print('Frozen held-out accuracy:', report.accuracy, 'nominal cost:', report.mean_cost)
    print('Offline fixture only: these numbers do not measure real LLM performance.')


if __name__ == '__main__':
    main()
