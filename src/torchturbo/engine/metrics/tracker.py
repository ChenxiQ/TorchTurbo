from collections.abc import Callable

import torch


class MetricTracker:
    def __init__(self) -> None:
        self.totals: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = {}

    def record(self, metrics: dict[str, torch.Tensor]) -> None:
        for name, value in metrics.items():
            if name not in self.totals:
                self.totals[name] = torch.zeros(
                    (), device=value.device, dtype=torch.float32
                )
                self.counts[name] = 0
            self.totals[name].add_(value.detach().float())
            self.counts[name] += 1

    def collect(self, model: torch.nn.Module) -> None:
        for module in model.modules():
            pop_metrics = getattr(module, "pop_metrics", None)
            if pop_metrics is None:
                continue
            metrics = pop_metrics()
            self.record(metrics)

    def report(
        self,
        logs: dict,
        gather: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        for name, total in self.totals.items():
            count = self.counts[name]
            if count == 0:
                continue
            total_count = gather(total.new_tensor(count)).sum().item()
            logs[name] = round(gather(total).sum().item() / total_count, 8)
        self.clear()

    def clear(self) -> None:
        for total in self.totals.values():
            total.zero_()
        for name in self.counts:
            self.counts[name] = 0
