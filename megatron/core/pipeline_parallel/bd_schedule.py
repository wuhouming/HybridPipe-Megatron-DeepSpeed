import contextlib
import itertools
from typing import Iterator, List, Union

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron import core, get_args, get_num_microbatches, print_rank_0
from megatron.core import parallel_state
from megatron.core.utils import get_model_config, get_model_type
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
)
from megatron.core.pipeline_parallel.schedules import (
    recv_forward,
    send_forward,
    recv_backward,
    send_backward,
    deallocate_output_tensor,
    forward_step,
    backward_step,
    get_tensor_shapes,
)
from deepspeed.accelerator import get_accelerator
from megatron.timers import Timer
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import p2p_communication
from torch.nn.parallel.distributed import DistributedDataParallel as DDP


def forward_backward_pipelining_with_bidirectional(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
):
    assert isinstance(model, list), \
        "bidirectional pipeline parallelism expected model chunking"
    assert all(isinstance(chunk, torch.nn.Module) for chunk in model), \
        "invalid model chunking"
    assert isinstance(data_iterator, list), \
        "bidirectional pipeline parallelism expected each model chunk to have a data iterator"

    config = get_model_config(model[0])
    if config.overlap_p2p_comm and config.batch_p2p_comm:
        raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None and all(isinstance(chunk, torchDDP) for chunk in model):
        def multi_no_sync():
            stack = contextlib.ExitStack()
            for chunk in model:
                stack.enter_context(chunk.no_sync())
            return stack
        no_sync_func = multi_no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None
    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()
    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None
    disable_grad_sync()

    # Model chunk IDs with synchronized grads
    synchronized_model_chunks = set()

    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    forward_data_store = []
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]

    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()

    if num_microbatches % pipeline_parallel_size != 0:
        msg = f'number of microbatches ({num_microbatches}) is not divisible by '
        msg += f'pipeline-model-parallel-size ({pipeline_parallel_size}) '
        msg += 'when using interleaved schedule'
        raise RuntimeError(msg)

    model_type = get_model_type(model[0])
    if model_type == ModelType.encoder_and_decoder:
        raise RuntimeError("Bidirectional is not supported with an encoder and decoder model.")

    if decoder_seq_length is not None and decoder_seq_length != tensor_shape[0]:
        raise RuntimeError("Bidirectional is not supported with a different decoder sequence length.")

    tensor_shape = (seq_length, micro_batch_size, config.hidden_size)
    if config.sequence_parallel:
        tensor_shape[0] = tensor_shape[0] // parallel_state.get_tensor_model_parallel_world_size()

    # Compute number of warmup and remaining microbatches.
    num_model_chunks = len(model)
    total_num_microbatches = num_microbatches * num_model_chunks
    all_warmup_microbatches = False
    if forward_only:
        num_warmup_microbatches = total_num_microbatches
    else:
        # Run all forward passes and then all backward passes if number of
        # microbatches is just the number of pipeline stages.
        # Otherwise, perform (num_model_chunks-1)*pipeline_parallel_size on
        # all workers, followed by more microbatches after depending on
        # stage ID (more forward passes for earlier stages, later stages can
        # immediately start with 1F1B).
        if num_microbatches == pipeline_parallel_size:
            num_warmup_microbatches = total_num_microbatches // 2
        else:
            num_warmup_microbatches = total_num_microbatches-2 # 存疑

        num_warmup_microbatches += pipeline_parallel_rank
    num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    assert config.num_microbatches_with_partial_activation_checkpoints is None
    # max_outstanding_backprops = None
    # if config.num_microbatches_with_partial_activation_checkpoints is not None:
    #     max_outstanding_backprops = num_warmup_microbatches + 1

    # Synchronize params for first two model chunks
    if config.param_sync_func is not None:
        config.param_sync_func(model[0].parameters())
        config.param_sync_func(model[1].parameters())

    def get_model_chunk_id(microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        model_chunk_id =0
        if microbatch_id < pipeline_parallel_size -pipeline_parallel_rank:
            model_chunk_id =0
        elif microbatch_id >= total_num_microbatches-pipeline_parallel_size +pipeline_parallel_rank:
            model_chunk_id =1
        else:
            model_chunk_id =(microbatch_id-pipeline_parallel_size +pipeline_parallel_rank+1) %2

        return model_chunk_id if forward else num_model_chunks - model_chunk_id - 1

    def is_first_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Check if an iteration is the first for a model chunk."""
        model_chunk_id =get_model_chunk_id(microbatch_id, True)
        if model_chunk_id == 0:
            return microbatch_id == 0
        else:
            return microbatch_id == pipeline_parallel_size -pipeline_parallel_rank

    def is_last_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Check if an iteration is the last for a model chunk."""
        model_chunk_id =get_model_chunk_id(microbatch_id, True)
        if model_chunk_id == 0:
            return microbatch_id == pipeline_parallel_size +pipeline_parallel_rank-1
        else:
            return microbatch_id == total_num_microbatches -1

    def forward_step_helper(microbatch_id, checkpoint_activations_microbatch):
        """Helper method to run forward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        forward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # launch param synchronization for next model chunk
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        # if config.param_sync_func is not None:
        #     param_sync_microbatch_id = microbatch_id + pipeline_parallel_rank
        #     if param_sync_microbatch_id < num_microbatches and is_first_microbatch_for_model_chunk(param_sync_microbatch_id):
        #         param_sync_chunk_id = get_model_chunk_id(param_sync_microbatch_id, forward=True) + 1
        #         if 1 < param_sync_chunk_id < num_model_chunks:
        #             config.param_sync_func(model[param_sync_chunk_id].parameters())

        # forward step
        if parallel_state.is_pipeline_first_stage():
            if len(input_tensors[model_chunk_id]) == \
                    len(output_tensors[model_chunk_id]):
                input_tensors[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id][-1]
        # if not input_tensor==None:
        #     print(input_tensor.shape)
        output_tensor = forward_step(forward_step_func,
                                     data_iterator[model_chunk_id],
                                     model[model_chunk_id],
                                     num_microbatches,
                                     input_tensor,
                                     forward_data_store,
                                     config,
                                     collect_non_loss_data,
                                     checkpoint_activations_microbatch)
        output_tensors[model_chunk_id].append(output_tensor)

        # if forward-only, no need to save tensors for a backward pass
        if forward_only:
            input_tensors[model_chunk_id].pop()
            output_tensors[model_chunk_id].pop()

        return output_tensor

    def backward_step_helper(microbatch_id):
        """Helper method to run backward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        backward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # launch grad synchronization (default)
        if config.grad_sync_func is None and is_last_microbatch_for_model_chunk(microbatch_id):
            enable_grad_sync()
            synchronized_model_chunks.add(model_chunk_id)

        if parallel_state.is_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
        input_tensor_grad = \
            backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)

        # launch grad synchronization (custom grad sync)
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        # if config.grad_sync_func is not None:
        #     grad_sync_microbatch_id = microbatch_id - pipeline_parallel_rank
        #     if grad_sync_microbatch_id >= 0 and is_last_microbatch_for_model_chunk(grad_sync_microbatch_id):
        #         grad_sync_chunk_id = get_model_chunk_id(grad_sync_microbatch_id, forward=False)
        #         enable_grad_sync()
        #         config.grad_sync_func(model[grad_sync_chunk_id].parameters())
        #         synchronized_model_chunks.add(grad_sync_chunk_id)
        disable_grad_sync()

        return input_tensor_grad

    # Run warmup forward passes.
    parallel_state.set_virtual_pipeline_model_parallel_rank(0)
    input_tensors[0].append(
        p2p_communication.recv_forward(tensor_shape, config))

    fwd_wait_handles = None
    bwd_wait_handles = None

    # printrank =3
    for k in range(num_warmup_microbatches):

        if fwd_wait_handles is not None:
            for req in fwd_wait_handles:
                req.wait()

        # Decide to checkpoint all layers' activations of the current micro-batch
        # if max_outstanding_backprops is not None:
        #     checkpoint_activations_microbatch = k % max_outstanding_backprops >= \
        #         config.num_microbatches_with_partial_activation_checkpoints
        # else:
        #     checkpoint_activations_microbatch = None

        output_tensor = forward_step_helper(k, None)

        # Determine if tensor should be received from previous stage.
        forward_model_chunk_id =get_model_chunk_id(k, forward=True)
        # if pipeline_parallel_rank == printrank:
        #     print(f"f[{k}] {forward_model_chunk_id}")
        next_forward_model_chunk_id = get_model_chunk_id(k+1, forward=True)
        recv_prev = True # 均是以当前stage作为参照
        recv_next = True

        if k == (total_num_microbatches - 1):
            recv_prev = False

        # Don't send tensor downstream if on last stage.
        if parallel_state.is_pipeline_last_stage():
            output_tensor = None

        # Send and receive tensors as appropriate (send tensors computed
        # in this iteration; receive tensors for next iteration).
        if not config.overlap_p2p_comm:
            if k == (num_warmup_microbatches - 1) and not forward_only and \
                    not all_warmup_microbatches:
                assert forward_model_chunk_id==0 # 1F1B前的，chunk0
                input_tensor_grad = None
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True): #最后阶段，后续接chunk1
                    recv_prev = True  # 收上阶段backward
                    detached_output_tensor = output_tensor.detach()
                    detached_output_tensor.requires_grad_()
                    input_tensor=detached_output_tensor # chunk0最后阶段与chunk1第一阶段在同一个rank
                    # input_tensor = output_tensor.clone()
                    output_tensor_grad = p2p_communication.send_backward_recv_backward_bd(
                    input_tensor_grad, recv_prev=recv_prev,
                    tensor_shape=tensor_shape,
                    config=config)
                else:# 收下阶段forward，收上阶段backward，往下阶段发forward
                    if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                        recv_prev = False
                    input_tensor,output_tensor_grad = p2p_communication.send_forward_backward_recv_forward_backward_bd(
                        output_tensor,
                        input_tensor_grad,
                        recv_prev=recv_prev,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        config=config)

                output_tensor_grads[num_model_chunks-1].append(output_tensor_grad)
            else:# 分两类，连续则正常，否则交叉
                parallel_state.set_virtual_pipeline_model_parallel_rank(
                    forward_model_chunk_id
                )
                if forward_model_chunk_id ==next_forward_model_chunk_id:
                    if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                        if next_forward_model_chunk_id == 0:
                            recv_prev = False
                    input_tensor = \
                        p2p_communication.send_forward_recv_forward(
                            output_tensor, recv_prev=recv_prev,
                            tensor_shape=tensor_shape,
                            config=config)
                else:
                    if parallel_state.is_pipeline_last_stage(ignore_virtual=True) and next_forward_model_chunk_id == 1:                       
                        detached_output_tensor = output_tensor.detach()
                        detached_output_tensor.requires_grad_()
                        input_tensor=detached_output_tensor # chunk0最后阶段与chunk1第一阶段在同一个rank
                        # input_tensor = output_tensor.clone()
                    else:
                        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                            if next_forward_model_chunk_id == 0:
                                recv_next = False
                            if next_forward_model_chunk_id == 1:
                                recv_prev = False
                        input_tensor = \
                            p2p_communication.send_forward_recv_forward_bd0(
                                output_tensor, recv_next=recv_next,
                                tensor_shape=tensor_shape,
                                config=config)

            input_tensors[next_forward_model_chunk_id].append(input_tensor)

        deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # Run 1F1B in steady state.
    for k in range(num_microbatches_remaining):
        # Forward pass.
        forward_k = k + num_warmup_microbatches

        # Decide to checkpoint all layers' activations of the current micro-batch
        # if max_outstanding_backprops is not None:
        #     checkpoint_activations_microbatch = (
        #         forward_k % max_outstanding_backprops >= \
        #         config.num_microbatches_with_partial_activation_checkpoints
        #     )
        # else:
        #     checkpoint_activations_microbatch = None

        if not config.overlap_p2p_comm:
            output_tensor = forward_step_helper(forward_k, None)

            # Backward pass.
            backward_k = k
            input_tensor_grad = backward_step_helper(backward_k)

            # Send output_tensor and input_tensor_grad, receive input_tensor
            # and output_tensor_grad.

            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            # if pipeline_parallel_rank == printrank:
            #     print(f"f[{forward_k}] {forward_model_chunk_id}")
            parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            # if pipeline_parallel_rank == printrank:
            #     print(f"b[{backward_k}] {backward_model_chunk_id}")
            parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            # Determine if peers are sending, and where in data structure to put
            # received tensors.
            recv_prev = True
            next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1,forward=True)
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                recv_prev = False

            recv_next = True
            next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1,
                                                                  forward=False)
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                recv_next = False

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.

            # Communicate tensors.
            if k == (num_microbatches_remaining - 1):
                if parallel_state.is_pipeline_last_stage(True):
                    recv_prev = False
                    input_tensor= \
                        p2p_communication.send_forward_recv_forward(
                            output_tensor, recv_prev=recv_prev,
                            tensor_shape=tensor_shape,config=config)
                    output_tensor_grads[0].append(input_tensor_grad)
                else:
                    recv_prev = True
                    recv_next = False
                    input_tensor, output_tensor_grad = (
                        p2p_communication.send_forward_backward_recv_forward_backward_bd(
                            output_tensor,
                            input_tensor_grad,
                            recv_prev=recv_prev,
                            recv_next=recv_next,
                            tensor_shape=tensor_shape,
                            config=config,
                        )
                    )
                    output_tensor_grads[0].append(output_tensor_grad)
                # input_tensors[next_forward_model_chunk_id].append(input_tensor)
            else:
                input_tensor, output_tensor_grad = \
                        p2p_communication.send_forward_backward_recv_forward_backward(
                            output_tensor, input_tensor_grad,
                            recv_prev=recv_prev, recv_next=recv_next,
                            tensor_shape=tensor_shape, config=config)
                if recv_prev: 
                    input_tensors[next_forward_model_chunk_id].append(input_tensor)
                if recv_next:
                    output_tensor_grads[next_backward_model_chunk_id].append(
                        output_tensor_grad)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # Run cooldown backward passes (flush out pipeline).
    if not forward_only:
        if config.overlap_p2p_comm and bwd_wait_handles is not None:
            for wait_handle in bwd_wait_handles:
                wait_handle.wait()

        if all_warmup_microbatches:
            output_tensor_grads[num_model_chunks-1].append(
                p2p_communication.recv_backward(tensor_shape, config=config))
        for k in range(num_microbatches_remaining, total_num_microbatches):
            backward_model_chunk_id = get_model_chunk_id(k, forward=False)
            # if pipeline_parallel_rank == printrank:
            #     print(f"b[{k}] {backward_model_chunk_id}")
            input_tensor_grad = backward_step_helper(k)
            next_backward_model_chunk_id = get_model_chunk_id(k+1, forward=False)
            recv_next = True
            recv_prev = True
            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            if k == (total_num_microbatches - 1):
                recv_next = False
                output_tensor_grads[next_backward_model_chunk_id].append(
                    p2p_communication.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        config=config,
                    )
                )
            else:
                if backward_model_chunk_id == next_backward_model_chunk_id:
                    output_tensor_grads[next_backward_model_chunk_id].append(
                        p2p_communication.send_backward_recv_backward(
                            input_tensor_grad,
                            recv_next=recv_next,
                            tensor_shape=tensor_shape,
                            config=config,
                        )
                    )
                else:
                    if parallel_state.is_pipeline_last_stage(True) and next_backward_model_chunk_id == 0:
                        output_tensor_grads[next_backward_model_chunk_id].append(input_tensor_grad)
                    else:
                        output_tensor_grads[next_backward_model_chunk_id].append(
                            p2p_communication.send_backward_recv_backward_bd(
                                input_tensor_grad,
                                recv_prev=recv_prev,
                                tensor_shape=tensor_shape,
                                config=config,
                            )
                        )

    # Launch any remaining grad reductions
    enable_grad_sync()
    if config.grad_sync_func is not None:
        params = []
        for model_chunk_id in range(num_model_chunks):
            if model_chunk_id not in synchronized_model_chunks:
                params.extend(model[model_chunk_id].parameters())
                synchronized_model_chunks.add(model_chunk_id)
        if params:
            config.grad_sync_func(params)

    return forward_data_store
