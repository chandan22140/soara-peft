from typing import Callable, Dict, List, Optional, Tuple, Union, Any
import math
import torch
import wandb
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import Trainer, Seq2SeqTrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.trainer import (
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
)
# CHANGED: Import SOARA layer instead of PEFT LoraLinear
# from peft.tuners.lora.layer import Linear as LoraLinear
from soara import SOARALinearLayer

# include_keywords = ["block.0", "block.4"]
include_keywords = ["encoder.block.2", "encoder.block.3", "encoder.block.4"]  # for T5
# include_keywords = ["layers.27", "layers.6"]  # for Llama
do_log = False


def get_forward_hook(name):
    def hook(module, input, output):
        wandb.log(
            {
                f"{name}/input_mean": input[0].mean().item(),
                f"{name}/input_std": input[0].std().item(),
                f"{name}/output_mean": output.mean().item(),
                f"{name}/output_std": output.std().item(),
            },
            commit=False,
        )

    return hook


class SOARAPhaseCallback(TrainerCallback):
    """Callback that advances V2 SOARA rotation phases during training.

    Different layers may have different ranks (e.g. 1024 vs 4096 in Llama-3-8B),
    yielding different ``phases_per_cycle`` (e.g. 10 vs 12 for butterfly).  This
    callback groups adapters by their phase count and maintains an independent
    ``steps_per_phase`` schedule for each group.

    At every group's ``steps_per_phase`` interval, the callback:
    1. Merges the current trained rotation angles into U/V matrices
    2. Advances to the next rotation phase (wrapping around per cycle)
    3. Resets angles to zero (identity) for the new phase

    Phases continue cycling beyond ``total_cycles`` to re-utilize remaining
    training steps for additional weight updates.
    """

    def __init__(self, adapter_groups):
        """
        Args:
            adapter_groups: list of dicts, each with keys:
                - 'adapters':        list of SOARALinearLayer
                - 'steps_per_phase': int
                - 'total_cycles':    int
                - 'phases_per_cycle': int
                - 'rank':            int (for logging)
                - 'count':           int (number of layers in group)
        """
        self.adapter_groups = adapter_groups

    def on_step_end(self, args, state, control, **kwargs):
        for group in self.adapter_groups:
            spp = group['steps_per_phase']
            if spp <= 0 or state.global_step <= 0:
                continue
            if state.global_step % spp != 0:
                continue

            for adapter in group['adapters']:
                adapter.step_phase()

            # Compute human-readable phase/cycle numbers
            phase_num = state.global_step // spp
            ppc = group['phases_per_cycle']
            cycle = (phase_num - 1) // ppc + 1
            phase_in_cycle = ((phase_num - 1) % ppc) + 1

            extra = " (bonus — reutilizing steps)" if cycle > group['total_cycles'] else ""
            print(
                f"\U0001f504 Phase transition at step {state.global_step} "
                f"[rank={group['rank']}, {group['count']} layers]: "
                f"cycle {cycle}/{group['total_cycles']}, "
                f"phase {phase_in_cycle}/{ppc}{extra}"
            )

            if wandb.run is not None:
                tag = f"r{group['rank']}"
                wandb.log({
                    f"soara/{tag}/phase_transition_step": state.global_step,
                    f"soara/{tag}/cycle": cycle,
                    f"soara/{tag}/phase_in_cycle": phase_in_cycle,
                }, commit=False)


class LogTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: Seq2SeqTrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        # CHANGED: Add soara_config for orthogonality regularization
        soara_config = None,
        # CHANGED: Add s_lr_multiplier for separate S learning rate
        s_lr_multiplier: float = 10.0,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        # CHANGED: Detect SOARA model instead of PEFT model
        self.is_peft = any(isinstance(m, SOARALinearLayer) for m in model.modules())
        
        # CHANGED: Store soara_config for orthogonality regularization
        self.soara_config = soara_config
        
        # CHANGED: Store s_lr_multiplier for separate S learning rate
        self.s_lr_multiplier = s_lr_multiplier
        
        # Collect all SOARA adapters for V2 phase transitions
        self._soara_adapters = []
        if self.is_peft:
            for name, module in model.named_modules():
                if isinstance(module, SOARALinearLayer):
                    self._soara_adapters.append(module)
            # Get scaling from first SOARA layer
            if self._soara_adapters:
                self.scaling = self._soara_adapters[0].scaling
        self.orig_A = None
        self.orig_B = None
        self.orig_W = None
        self.gradient_accumulation_counter = 0
        self._logged_step1_memory = False

    def train(self, resume_from_checkpoint=None, **kwargs):
        """Override to auto-compute V2 phase schedule and register phase callback.

        For V2 methods, groups adapters by ``phases_per_cycle`` (which varies
        by rank — e.g. rank-1024 layers have 10 butterfly phases while
        rank-4096 layers have 12).  Each group gets its own ``steps_per_phase``
        computed as ``total_steps // (phases_per_cycle × total_cycles)``.

        A single :class:`SOARAPhaseCallback` manages all groups, firing each
        group's phase transitions independently.  Phases continue beyond
        ``total_cycles`` so remaining steps are reutilized.
        """
        if (
            self.soara_config
            and self.soara_config.method == "V2"
            and self._soara_adapters
        ):
            # Compute total optimizer steps for the full training run
            train_dataloader = self.get_train_dataloader()
            num_update_steps_per_epoch = max(
                len(train_dataloader) // self.args.gradient_accumulation_steps, 1
            )
            total_steps = math.ceil(
                self.args.num_train_epochs * num_update_steps_per_epoch
            )
            total_cycles = self.soara_config.total_cycles

            # Group adapters by phases_per_cycle (varies by rank)
            from collections import defaultdict
            phase_groups = defaultdict(list)
            for adapter in self._soara_adapters:
                ppc = self._get_phases_per_cycle(adapter)
                phase_groups[ppc].append(adapter)

            print(f"\n\U0001f504 V2 Phase Schedule (auto-computed):")
            print(f"    Total training steps: {total_steps}")
            print(f"    Total cycles:         {total_cycles}")
            print(f"    Adapter groups ({len(phase_groups)} distinct phase counts):")

            adapter_groups = []
            min_spp = total_steps  # track smallest steps_per_phase for config
            for ppc, adapters in sorted(phase_groups.items()):
                total_phases = ppc * total_cycles
                spp = max(1, total_steps // total_phases)
                rank = adapters[0].r
                count = len(adapters)

                planned = total_phases * spp
                remaining = total_steps - planned
                extra = remaining // spp if spp > 0 else 0

                adapter_groups.append({
                    'adapters': adapters,
                    'steps_per_phase': spp,
                    'total_cycles': total_cycles,
                    'phases_per_cycle': ppc,
                    'rank': rank,
                    'count': count,
                })
                min_spp = min(min_spp, spp)

                print(f"      rank={rank} ({count} layers): "
                      f"{ppc} phases/cycle × {total_cycles} cycles = {total_phases} phases, "
                      f"steps_per_phase={spp}")
                if remaining > 0:
                    print(f"        └─ {remaining} remaining steps → {extra} bonus transitions")

            print(f"    Note: phases continue cycling beyond total_cycles to reutilize remaining steps\n")

            # Store the smallest steps_per_phase on config for external readers
            self.soara_config.steps_per_phase = min_spp

            # Register the phase transition callback
            self.add_callback(SOARAPhaseCallback(adapter_groups=adapter_groups))

        return super().train(
            resume_from_checkpoint=resume_from_checkpoint, **kwargs
        )

    @staticmethod
    def _get_phases_per_cycle(adapter):
        """Get the number of phases per cycle for a V2 SOARA adapter.

        Supports:
        - Persistent V2B (butterfly sequential): ``len(butterfly_k_values)``
        - Persistent V2A (Givens sequential):    ``num_phases`` from PhaseIndexedGivensOperator
        - Non-persistent V2A (base):             ``len(givens_pairings)``
        """
        # Persistent V2B (butterfly sequential)
        if hasattr(adapter, 'butterfly_k_values'):
            return len(adapter.butterfly_k_values)
        # Persistent V2A (Givens sequential)
        if hasattr(adapter, 'current_givens_u') and hasattr(adapter.current_givens_u, 'num_phases'):
            return adapter.current_givens_u.num_phases
        # Non-persistent V2A (base)
        if hasattr(adapter, 'givens_pairings'):
            return len(adapter.givens_pairings)
        return 1  # fallback

    def _log_step1_live_memory(self) -> None:
        """Log CUDA live memory once at global step 1."""
        if self._logged_step1_memory:
            return
        if not torch.cuda.is_available():
            return
        if int(self.state.global_step) != 1:
            return

        torch.cuda.synchronize()
        allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        max_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)

        print(
            f"[Memory step=1] allocated={allocated_mb:.2f}MB "
            f"max_allocated={max_allocated_mb:.2f}MB reserved={reserved_mb:.2f}MB"
        )

        if wandb.run is not None:
            wandb.log(
                {
                    "gpu/step1_memory_allocated_mb": allocated_mb,
                    "gpu/step1_max_memory_allocated_mb": max_allocated_mb,
                    "gpu/step1_memory_reserved_mb": reserved_mb,
                },
                commit=False,
            )

        self._logged_step1_memory = True

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        results = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        eval_loss = results.get(f"{metric_key_prefix}_loss")
        if isinstance(eval_loss, float) and math.isnan(eval_loss):
            try:
                current_eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
                eval_dataloader = self.get_eval_dataloader(current_eval_dataset)
                first_batch = next(iter(eval_dataloader))
                labels = first_batch.get("labels")
                if labels is not None:
                    valid_per_sample = (labels != -100).sum(dim=1)
                    total_samples = int(valid_per_sample.numel())
                    zero_label_samples = int((valid_per_sample == 0).sum().item())
                    min_valid = int(valid_per_sample.min().item()) if total_samples > 0 else 0
                    max_valid = int(valid_per_sample.max().item()) if total_samples > 0 else 0
                    print(
                        "[EvalNaNDebug] eval_loss is NaN. "
                        f"batch_samples={total_samples}, zero_label_samples={zero_label_samples}, "
                        f"min_valid_labels={min_valid}, max_valid_labels={max_valid}"
                    )
            except Exception as exc:
                print(f"[EvalNaNDebug] Failed to compute label diagnostics: {exc}")

        return results

    def create_optimizer(self):
        """Create optimizer with separate learning rate for S parameters.
        
        S (singular values) typically have values ~1-16, while R_U/R_V matrices
        have values ~0-1. With the same LR, S gets proportionally tiny updates.
        This method gives S a higher LR (s_lr_multiplier × base_lr).
        """
        if self.optimizer is not None:
            return self.optimizer
        
        # Separate S parameters from other trainable parameters
        s_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Match S parameters (e.g., "model.layers.0.self_attn.q_proj.S")
                if name.endswith('.S'):
                    s_params.append(param)
                else:
                    other_params.append(param)
        
        base_lr = self.args.learning_rate
        weight_decay = self.args.weight_decay
        
        # Create optimizer with separate param groups
        optimizer_grouped_parameters = []
        
        if other_params:
            optimizer_grouped_parameters.append({
                'params': other_params,
                'lr': base_lr,
                'weight_decay': weight_decay,
            })
        
        if s_params:
            optimizer_grouped_parameters.append({
                'params': s_params,
                'lr': base_lr * self.s_lr_multiplier,  # Higher LR for S
                'weight_decay': 0.0,  # No weight decay for singular values
            })
            print(f"[LogTrainer] S params: {len(s_params)}, LR: {base_lr * self.s_lr_multiplier:.2e} ({self.s_lr_multiplier}x)")
            print(f"[LogTrainer] Other params: {len(other_params)}, LR: {base_lr:.2e}")
        
        # Use the optimizer class from args
        # Using eps=1e-10 instead of default 1e-8 for better precision with small gradients
        # fused=True runs the entire update in a single CUDA kernel (5-15% faster optimizer step)
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-10, fused=True)
        
        return self.optimizer

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        if not do_log:
            # CHANGED: Add orthogonality regularization for SOARA
            if self.is_peft and self.soara_config is not None:
                return self._training_step_with_soara_reg(model, inputs, num_items_in_batch)
            
            try:
                loss = super().training_step(model, inputs, num_items_in_batch)
            except TypeError:
                # Fallback for older transformers versions that don't accept num_items_in_batch
                print("⚠️  Fallback for older transformers versions that don't accept num_items_in_batch")
                loss = super().training_step(model, inputs)

            self._log_step1_live_memory()
            return loss

        print("⚠️  Fallback for do_log=True")
        # Original logging code (unchanged when do_log=True)
        if self.is_peft:
            if self.orig_A is None:
                self.orig_A = {}
                self.orig_B = {}
                for name, param in model.named_parameters():
                    if param.requires_grad and any(
                        [kw in name for kw in include_keywords]
                    ):
                        # CHANGED: Adapt to SOARA parameter names
                        # For v1: R_U, R_V
                        # For v3/3: B_U, C_U, B_V, C_V
                        if "R_U" in name or "B_U" in name:
                            self.orig_A[name.split("R_U.")[0] if "R_U" in name else name.split("B_U.")[0]] = (
                                param.detach().clone()
                            )
                        elif "R_V" in name or "B_V" in name:
                            self.orig_B[name.split("R_V.")[0] if "R_V" in name else name.split("B_V.")[0]] = (
                                param.detach().clone()
                            )
                for name, module in model.named_modules():
                    if any([kw in name for kw in include_keywords]) and isinstance(
                        module, SOARALinearLayer
                    ):
                        breakpoint()
                        hook = get_forward_hook(name)
                        module.register_forward_hook(hook)
        else:
            if self.orig_W is None:
                self.orig_W = {}
                for name, param in model.named_parameters():
                    if param.requires_grad and any(
                        [kw in name for kw in include_keywords]
                    ):
                        self.orig_W[name] = param.detach().clone()

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss)
        with torch.no_grad():
            if (
                self.gradient_accumulation_counter
                % self.args.gradient_accumulation_steps
                == self.args.gradient_accumulation_steps - 1
            ):
                if self.is_peft:
                    # CHANGED: Log SOARA parameters instead of LoRA A/B
                    # This is complex and method-dependent, keeping simplified version
                    param_dict = {}
                    for name, param in model.named_parameters():
                        if param.requires_grad and any(
                            [kw in name for kw in include_keywords]
                        ):
                            param_dict[name] = param
                    
                    # Log parameter norms
                    for name, param in param_dict.items():
                        if param.grad is not None:
                            wandb.log(
                                {
                                    f"param_norm/{name}": torch.norm(param).item(),
                                    f"grad_norm/{name}": torch.norm(param.grad).item(),
                                    "train/global_step": self.state.global_step,
                                },
                                commit=False,
                            )
                else:
                    W_dict = {}
                    for name, param in model.named_parameters():
                        if (
                            param.requires_grad
                            and any([kw in name for kw in include_keywords])
                            and len(param.shape) == 2
                        ):
                            W_dict[name] = param
                    for key in W_dict.keys():
                        W = W_dict[key]
                        W_grad = W.grad
                        W_0 = self.orig_W[key]
                        W_diff = W - W_0
                        W_diff_norm = torch.norm(W_diff).item()
                        W_norm = torch.norm(W).item()
                        W_grad_norm = torch.norm(W_grad).item()
                        U, S, V = torch.svd(W_diff.float())
                        top_1_ratio = S[0] / S.sum()
                        top_4_ratio = S[:4].sum() / S.sum()
                        wandb.log(
                            {
                                f"W_norm/{key}": W_norm,
                                f"W_grad_norm/{key}": W_grad_norm,
                                f"W_diff_norm/{key}": W_diff_norm,
                                "train/global_step": self.state.global_step,
                                f"W_top_1_ratio/{key}": top_1_ratio.item(),
                                f"W_top_4_ratio/{key}": top_4_ratio.item(),
                            }
                        )
        self.gradient_accumulation_counter += 1

        return loss.detach() / self.args.gradient_accumulation_steps

    # CHANGED: New method for training step with SOARA regularization
    def _training_step_with_soara_reg(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """Training step with orthogonality regularization for SOARA."""
        if torch.cuda.is_available():
             torch.cuda.reset_peak_memory_stats()

        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            if num_items_in_batch is None:
                loss = self.compute_loss(model, inputs)
            else:
                # Use num_items_in_batch if compute_loss supports/needs it?
                # Actually compute_loss signature is (model, inputs, return_outputs=False, num_items_in_batch=None)
                # But let's check if we can pass it safely.
                # Inspect compute_loss signature to be safe.
                import inspect
                if "num_items_in_batch" in inspect.getfullargspec(self.compute_loss).args:
                     loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
                else:
                     loss = self.compute_loss(model, inputs)

            # DEBUG: Print raw loss values
            # print(f"[DEBUG] raw_loss={loss.item():.6f}, loss_dtype={loss.dtype}")
            
            # Add orthogonality regularization for v1
            if self.soara_config.method == "v1":
                ortho_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
                for module in model.modules():
                    if isinstance(module, SOARALinearLayer):
                        ortho_loss = ortho_loss + module.get_orthogonality_loss()
                
                # Combine losses
                total_loss = loss + ortho_loss
            else:
                ortho_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
                total_loss = loss
            

        # DEBUG: Print total loss before any normalization
        # print(f"[DEBUG] total_loss_before_norm={total_loss.item():.6f}, GA_steps={self.args.gradient_accumulation_steps}")

        if self.args.n_gpu > 1:
            total_loss = total_loss.mean()

        # FIX: Do backward with ORIGINAL loss, then divide for logging
        # This prevents bf16 precision loss from setting gradients to 0
        self.accelerator.backward(total_loss)

        # Log metrics (including VRAM) for ALL methods, not just v1
        # DO THIS AFTER BACKWARD so we capture the true peak memory!
        if self.state.global_step % self.args.logging_steps == 0:
            mem_metrics = {}
            if torch.cuda.is_available():
                mem_metrics = {
                    "gpu/peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
                    "gpu/current_memory_mb": torch.cuda.memory_allocated() / 1024**2,
                }
            
            if wandb.run is not None:
                wandb.log({
                    "train/ortho_loss": ortho_loss.item() if ortho_loss is not None else 0.0,
                    "train/task_loss": loss.item(),
                    "train/total_loss": total_loss.item(),
                    **mem_metrics,
                }, commit=False)

        # DEBUG: Check for NaN in parameters/gradients after backward
        # Guarded: this iterates all trainable params with 2 reductions each (~700 kernels/step)
        if self.soara_config and getattr(self.soara_config, 'debug_nan_check', False):
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    if torch.isnan(p.grad).any():
                        print(f"[NaN DETECTED] Gradient NaN in: {name}")
                    if torch.isnan(p).any():
                        print(f"[NaN DETECTED] Parameter NaN in: {name}")

        # Return scaled loss for logging (after backward)
        self._log_step1_live_memory()
        return total_loss.detach() / self.args.gradient_accumulation_steps