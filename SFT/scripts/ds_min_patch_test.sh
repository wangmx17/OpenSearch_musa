#!/usr/bin/env bash
# Apply DeepSpeed patches one-by-one, run smoke train, find minimal working set.
set -euo pipefail

SFT_DIR="${SFT_DIR:-/home/jd/OpenSearch-VL-main/SFT}"
DS_ROOT="${DS_ROOT:-/home/DeepSpeed}"
RESULT_DIR="${RESULT_DIR:-${SFT_DIR}/logs/ds_min_patch_$(date +%Y%m%d_%H%M%S)}"
HOSTFILE="${HOSTFILE:-${SFT_DIR}/hostfile}"
YAML_CONFIG="${YAML_CONFIG:-${SFT_DIR}/examples/agentic_full/qwen3_vl_full_sft_8b_smoke.yaml}"
HANG_SECS="${HANG_SECS:-120}"   # no progress for this long after last loss/start => HANG
MAX_RUN_SECS="${MAX_RUN_SECS:-900}"

mkdir -p "${RESULT_DIR}"
RESULT_TSV="${RESULT_DIR}/results.tsv"
echo -e "step\tpatch\tstatus\tlog_dir\tnote" > "${RESULT_TSV}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

stop_train() {
  (cd "${SFT_DIR}" && bash stop_all.sh "${HOSTFILE}" >/dev/null 2>&1 || true)
  sleep 2
  pkill -f "llamafactory.cli train" 2>/dev/null || true
  pkill -f "llamafactory/launcher.py" 2>/dev/null || true
  sleep 1
}

reset_ds() {
  (cd "${DS_ROOT}" && git checkout -- \
    deepspeed/comm/torch.py \
    deepspeed/runtime/zero/mics.py \
    deepspeed/runtime/zero/partition_parameters.py \
    deepspeed/runtime/zero/partitioned_param_coordinator.py \
    deepspeed/runtime/zero/stage3.py)
}

# ---------- patches (each is one logical point) ----------
apply_p01_torch_sync_path() {
  python3 - <<'PY'
from pathlib import Path
p = Path("/home/DeepSpeed/deepspeed/comm/torch.py")
text = p.read_text()
old = '''    def all_gather_coalesced(self, output_tensors, input_tensors, group=None, async_op=False):
        """"""
        assert len(output_tensors) == len(input_tensors), ""
        if hasattr(torch.distributed.distributed_c10d, '_all_gather_base_coalesced'):'''
new = '''    def all_gather_coalesced(self, output_tensors, input_tensors, group=None, async_op=False):
        """"""
        assert len(output_tensors) == len(input_tensors), ""
        if not async_op:
            for output, input in zip(output_tensors, input_tensors):
                handle = torch.distributed.distributed_c10d.all_gather_into_tensor(output,
                                                                                   input,
                                                                                   group=group,
                                                                                   async_op=True)
                handle.wait()
            return
        if hasattr(torch.distributed.distributed_c10d, '_all_gather_base_coalesced'):'''
if old not in text:
    raise SystemExit("p01: pattern not found")
p.write_text(text.replace(old, new, 1))
print("applied p01")
PY
}

apply_p02_torch_drop_else_wait() {
  python3 - <<'PY'
from pathlib import Path
p = Path("/home/DeepSpeed/deepspeed/comm/torch.py")
text = p.read_text()
old = '''            if async_op:
                return reqs[-1]
            else:
                reqs[-1].wait()
'''
new = '''            if async_op:
                return reqs[-1]
'''
if old not in text:
    raise SystemExit("p02: pattern not found")
p.write_text(text.replace(old, new, 1))
print("applied p02")
PY
}

apply_p03_mics_wait1() {
  python3 - <<'PY'
from pathlib import Path
p = Path("/home/DeepSpeed/deepspeed/runtime/zero/mics.py")
text = p.read_text()
old = '''        all_gather_handle = dist.all_gather_coalesced(output_tensors,
                                                      input_tensors,
                                                      group=mics_comm_groups.param_shard_group,
                                                      async_op=True)

        for idx, param in enumerate(params):
'''
new = '''        all_gather_handle = dist.all_gather_coalesced(output_tensors,
                                                      input_tensors,
                                                      group=mics_comm_groups.param_shard_group,
                                                      async_op=True)
        all_gather_handle.wait()

        for idx, param in enumerate(params):
'''
if old not in text:
    raise SystemExit("p03: pattern not found")
p.write_text(text.replace(old, new, 1))
print("applied p03")
PY
}

apply_p04_mics_wait2() {
  python3 - <<'PY'
from pathlib import Path
p = Path("/home/DeepSpeed/deepspeed/runtime/zero/mics.py")
text = p.read_text()
old = '''        all_gather_handle = dist.all_gather_coalesced(intra_outputs,
                                                      intra_inputs,
                                                      group=intra_node_comm_group,
                                                      async_op=True)
        for i, param in enumerate(params):
'''
new = '''        all_gather_handle = dist.all_gather_coalesced(intra_outputs,
                                                      intra_inputs,
                                                      group=intra_node_comm_group,
                                                      async_op=True)
        all_gather_handle.wait()
        for i, param in enumerate(params):
'''
if old not in text:
    raise SystemExit("p04: pattern not found")
p.write_text(text.replace(old, new, 1))
print("applied p04")
PY
}

apply_p05_pp_wait_1253() {
  python3 - <<'PY'
from pathlib import Path
p = Path("/home/DeepSpeed/deepspeed/runtime/zero/partition_parameters.py")
text = p.read_text()
old = '''            handle = _dist_allgather_fn(partitions[rank_in_group], flat_tensor, ds_process_group)
            #Fix get_partition_dp_group(params[0]))
'''
new = '''            handle = _dist_allgather_fn(partitions[rank_in_group], flat_tensor, ds_process_group)
            handle.wait()
            #Fix get_partition_dp_group(params[0]))
'''
if old not in text:
    raise SystemExit("p05: pattern not found")
p.write_text(text.replace(old, new, 1))
print("applied p05")
PY
}

apply_p06_pp_wait_1338() {
  python3 - <<'PY'
from pathlib import Path
p = Path("/home/DeepSpeed/deepspeed/runtime/zero/partition_parameters.py")
text = p.read_text()
old = '''                    handles = _dist_allgather_fn(
                        param_ds_tensor.to(get_accelerator().current_device_name()).to(allgather_dtype),
                        param_buffer,
                        ds_process_group,
                    )
                    param.data = param_buffer.narrow(0, 0, param.ds_numel).view(param.ds_shape).to(param.device)
'''
new = '''                    handles = _dist_allgather_fn(
                        param_ds_tensor.to(get_accelerator().current_device_name()).to(allgather_dtype),
                        param_buffer,
                        ds_process_group,
                    )
                    handles.wait()
                    param.data = param_buffer.narrow(0, 0, param.ds_numel).view(param.ds_shape).to(param.device)
'''
if old not in text:
    raise SystemExit("p06: pattern not found")
p.write_text(text.replace(old, new, 1))
print("applied p06")
PY
}

apply_p07_pp_wait_1349() {
  python3 - <<'PY'
from pathlib import Path
p = Path("/home/DeepSpeed/deepspeed/runtime/zero/partition_parameters.py")
text = p.read_text()
old = '''                    handle = _dist_allgather_fn(quantized_param.to(get_accelerator().current_device_name()),
                                                param_buffer, ds_process_group)

                    quant_scale_buffer = torch.empty(
'''
new = '''                    handle = _dist_allgather_fn(quantized_param.to(get_accelerator().current_device_name()),
                                                param_buffer, ds_process_group)
                    handle.wait()

                    quant_scale_buffer = torch.empty(
'''
if old not in text:
    raise SystemExit("p07: pattern not found")
p.write_text(text.replace(old, new, 1))
print("applied p07")
PY
}

apply_p08_pp_wait_1359() {
  python3 - <<'PY'
from pathlib import Path
p = Path("/home/DeepSpeed/deepspeed/runtime/zero/partition_parameters.py")
text = p.read_text()
old = '''                    quant_handle = _dist_allgather_fn(scales.to(get_accelerator().current_device_name()),
                                                      quant_scale_buffer, ds_process_group)
                    quant_info = QuantizationInfo()
'''
new = '''                    quant_handle = _dist_allgather_fn(scales.to(get_accelerator().current_device_name()),
                                                      quant_scale_buffer, ds_process_group)
                    quant_handle.wait()
                    quant_info = QuantizationInfo()
'''
if old not in text:
    raise SystemExit("p08: pattern not found")
p.write_text(text.replace(old, new, 1))
print("applied p08")
PY
}

apply_p09_pp_wait_1386() {
  python3 - <<'PY'
from pathlib import Path
p = Path("/home/DeepSpeed/deepspeed/runtime/zero/partition_parameters.py")
text = p.read_text()
old = '''                    handle = dist.all_reduce(flat_tensor, group=ds_process_group, async_op=True)

                    return AllReduceCoalescedHandle(handle=handle, params=params)
'''
new = '''                    handle = dist.all_reduce(flat_tensor, group=ds_process_group, async_op=True)
                    handle.wait()

                    return AllReduceCoalescedHandle(handle=handle, params=params)
'''
if old not in text:
    raise SystemExit("p09: pattern not found")
p.write_text(text.replace(old, new, 1))
print("applied p09")
PY
}

apply_p10_pp_wait_1450() {
  python3 - <<'PY'
from pathlib import Path
p = Path("/home/DeepSpeed/deepspeed/runtime/zero/partition_parameters.py")
text = p.read_text()
old = '''                        handle = _dist_allgather_fn(quantized_param, flat_tensor, ds_process_group)
                        quant_handle = _dist_allgather_fn(scales, quant_scale_buffer, ds_process_group)
                        quant_info = QuantizationInfo()
'''
new = '''                        handle = _dist_allgather_fn(quantized_param, flat_tensor, ds_process_group)
                        handle.wait()
                        quant_handle = _dist_allgather_fn(scales, quant_scale_buffer, ds_process_group)
                        quant_handle.wait()
                        quant_info = QuantizationInfo()
'''
if old not in text:
    raise SystemExit("p10: pattern not found")
p.write_text(text.replace(old, new, 1))
print("applied p10")
PY
}

apply_p11_pp_launch_wait_each() {
  # Atomic: remove launch_handles batching + wait each collective (doc sections L1906-1953)
  python3 - <<'PY'
from pathlib import Path
p = Path("/home/DeepSpeed/deepspeed/runtime/zero/partition_parameters.py")
text = p.read_text()
old = '''        # launch
        launch_handles = []
        launch_quantize_handles = []
        for param_idx, param in enumerate(param_list):
            input_tensor = local_tensors[param_idx].view(-1)

            if self.use_all_gather_into_tensor:
                # try the _all_gather_base from Pytorch master
                h = dist.all_gather_into_tensor(allgather_params[param_idx],
                                                input_tensor,
                                                group=self.get_partition_dp_group(param),
                                                async_op=True)
                if quantize:
                    quantize_handle = dist.all_gather_into_tensor(allgather_quantize_scale[param_idx],
                                                                  quantize_scale_tensors[param_idx],
                                                                  group=self.get_partition_dp_group(param),
                                                                  async_op=True)
                    launch_quantize_handles.append(quantize_handle)
            else:
                output_list = []
                for i in range(self.num_partitions):
                    psize = partition_sizes[param_idx]
                    partition = allgather_params[param_idx].narrow(0, i * psize, psize)
                    output_list.append(partition)
                    if not get_accelerator().on_accelerator(partition):
                        logger.warning(
                            f'param {param_idx}, partition {i} is not on CUDA, partition shape {partition.size()}')

                # back to old all_gather function
                h = dist.all_gather(output_list, input_tensor, group=self.get_partition_dp_group(param), async_op=True)
                if quantize:
                    output_scale_list = []
                    for i in range(self.num_partitions):
                        psize = quantize_scale_sizes[param_idx]
                        partition = allgather_quantize_scale[param_idx].narrow(0, i * psize, psize)
                        output_scale_list.append(partition)
                    quant_handle = dist.all_gather(output_scale_list,
                                                   quantize_scale_tensors[param_idx],
                                                   group=self.get_partition_dp_group(param),
                                                   async_op=True)
                    launch_quantize_handles.append(quant_handle)
            launch_handles.append(h)

        # Wait ensures the operation is enqueued, but not necessarily complete.
        launch_handles[-1].wait()
        if quantize:
            for quant_handle in launch_quantize_handles:
                quant_handle.wait()
'''
new = '''        # launch and wait for each collective before issuing the next one
        for param_idx, param in enumerate(param_list):
            input_tensor = local_tensors[param_idx].view(-1)

            if self.use_all_gather_into_tensor:
                # try the _all_gather_base from Pytorch master
                h = dist.all_gather_into_tensor(allgather_params[param_idx],
                                                input_tensor,
                                                group=self.get_partition_dp_group(param),
                                                async_op=True)
                h.wait()
                if quantize:
                    quantize_handle = dist.all_gather_into_tensor(allgather_quantize_scale[param_idx],
                                                                  quantize_scale_tensors[param_idx],
                                                                  group=self.get_partition_dp_group(param),
                                                                  async_op=True)
                    quantize_handle.wait()
            else:
                output_list = []
                for i in range(self.num_partitions):
                    psize = partition_sizes[param_idx]
                    partition = allgather_params[param_idx].narrow(0, i * psize, psize)
                    output_list.append(partition)
                    if not get_accelerator().on_accelerator(partition):
                        logger.warning(
                            f'param {param_idx}, partition {i} is not on CUDA, partition shape {partition.size()}')

                # back to old all_gather function
                h = dist.all_gather(output_list, input_tensor, group=self.get_partition_dp_group(param), async_op=True)
                h.wait()
                if quantize:
                    output_scale_list = []
                    for i in range(self.num_partitions):
                        psize = quantize_scale_sizes[param_idx]
                        partition = allgather_quantize_scale[param_idx].narrow(0, i * psize, psize)
                        output_scale_list.append(partition)
                    quant_handle = dist.all_gather(output_scale_list,
                                                   quantize_scale_tensors[param_idx],
                                                   group=self.get_partition_dp_group(param),
                                                   async_op=True)
                    quant_handle.wait()
'''
if old not in text:
    raise SystemExit("p11: pattern not found")
# note: source uses allgather_params not allgather_buffers (doc name differs)
p.write_text(text.replace(old, new, 1))
print("applied p11")
PY
}

apply_p12_pp_reduce_scatter_seq() {
  python3 - <<'PY'
from pathlib import Path
p = Path("/home/DeepSpeed/deepspeed/runtime/zero/partition_parameters.py")
text = p.read_text()
old = '''        handles_and_reduced_partitions = []
        for param in param_list:
            assert param.grad.numel(
            ) == param.ds_numel, f"{param.grad.numel()} != {param.ds_numel} Cannot reduce scatter gradients whose size is not same as the params"

            handles_and_reduced_partitions.append(self._reduce_scatter_gradient(param))

        for param, (handle, reduced_partition) in zip(param_list, handles_and_reduced_partitions):
            if handle is not None:
                handle.wait()
'''
new = '''        for param in param_list:
            assert param.grad.numel(
            ) == param.ds_numel, f"{param.grad.numel()} != {param.ds_numel} Cannot reduce scatter gradients whose size is not same as the params"

            handle, reduced_partition = self._reduce_scatter_gradient(param)
            if handle is not None:
                handle.wait()
'''
if old not in text:
    raise SystemExit("p12: pattern not found")
p.write_text(text.replace(old, new, 1))
print("applied p12")
PY
}

apply_p13_pp_broadcast_seq() {
  python3 - <<'PY'
from pathlib import Path
p = Path("/home/DeepSpeed/deepspeed/runtime/zero/partition_parameters.py")
text = p.read_text()
old = '''        handles = [dist.broadcast(p.data, self.src_rank, group=p.ds_process_group, async_op=True) for p in self.params]
        for h in handles:
            h.wait()
        self.params[0].partition(param_list=self.params, has_been_updated=True)
'''
new = '''        for p in self.params:
            handle = dist.broadcast(p.data, self.src_rank, group=p.ds_process_group, async_op=True)
            handle.wait()
        self.params[0].partition(param_list=self.params, has_been_updated=True)
'''
if old not in text:
    raise SystemExit("p13: pattern not found")
p.write_text(text.replace(old, new, 1))
print("applied p13")
PY
}

apply_p14_coord_sync_wait() {
  python3 - <<'PY'
from pathlib import Path
p = Path("/home/DeepSpeed/deepspeed/runtime/zero/partitioned_param_coordinator.py")
text = p.read_text()
old = '''                if not param_group:
                    continue
                with get_accelerator().stream(self.__allgather_stream):
                    event_name = __class__.FORWARD_ALL_GATHER if forward else __class__.BACKWARD_ALL_GATHER
                    self.__profiler.start_event(event_name)
                    handle = param_group[0].all_gather_coalesced(param_group, quantize=quantize)
                    self.__profiler.stop_event(event_name, all_gather_numel)
                for param in param_group:
                    assert param.ds_status == ZeroParamStatus.INFLIGHT, param.ds_summary()
                    self.__inflight_param_registry[param] = handle
'''
new = '''                if not param_group:
                    continue
                if not get_accelerator().resolves_data_dependency():
                    self.__allgather_stream.wait_stream(get_accelerator().current_stream())
                with get_accelerator().stream(self.__allgather_stream):
                    event_name = __class__.FORWARD_ALL_GATHER if forward else __class__.BACKWARD_ALL_GATHER
                    self.__profiler.start_event(event_name)
                    handle = param_group[0].all_gather_coalesced(param_group, quantize=quantize)
                    handle.wait()
                    self.__profiler.stop_event(event_name, all_gather_numel)
                if not get_accelerator().resolves_data_dependency():
                    get_accelerator().current_stream().wait_stream(self.__allgather_stream)
                for param in param_group:
                    assert param.ds_status == ZeroParamStatus.AVAILABLE, param.ds_summary()
'''
if old not in text:
    raise SystemExit("p14: pattern not found")
p.write_text(text.replace(old, new, 1))
print("applied p14")
PY
}

apply_p15_stage3_wait_before() {
  python3 - <<'PY'
from pathlib import Path
p = Path("/home/DeepSpeed/deepspeed/runtime/zero/stage3.py")
text = p.read_text()
old = '''        if len(self.param_reduce_events) > self.max_param_reduce_events:
            self.param_reduce_events.popleft().synchronize()

        with get_accelerator().stream(self.reduce_and_partition_stream):
            if safe_mode:
'''
new = '''        if len(self.param_reduce_events) > self.max_param_reduce_events:
            self.param_reduce_events.popleft().synchronize()

        if not get_accelerator().resolves_data_dependency():
            self.reduce_and_partition_stream.wait_stream(get_accelerator().current_stream())
        with get_accelerator().stream(self.reduce_and_partition_stream):
            if safe_mode:
'''
if old not in text:
    raise SystemExit("p15: pattern not found")
p.write_text(text.replace(old, new, 1))
print("applied p15")
PY
}

apply_p16_stage3_wait_after() {
  python3 - <<'PY'
from pathlib import Path
p = Path("/home/DeepSpeed/deepspeed/runtime/zero/stage3.py")
text = p.read_text()
old = '''            if not get_accelerator().handles_memory_backpressure():
                event = get_accelerator().Event()
                event.record()
                self.param_reduce_events.append(event)

    @instrument_w_nvtx
    def __avg_scatter_contiguous_grads(self, buffer_to_reduce: Tensor,
'''
new = '''            if not get_accelerator().handles_memory_backpressure():
                event = get_accelerator().Event()
                event.record()
                self.param_reduce_events.append(event)
        if not get_accelerator().resolves_data_dependency():
            get_accelerator().current_stream().wait_stream(self.reduce_and_partition_stream)

    @instrument_w_nvtx
    def __avg_scatter_contiguous_grads(self, buffer_to_reduce: Tensor,
'''
if old not in text:
    raise SystemExit("p16: pattern not found")
p.write_text(text.replace(old, new, 1))
print("applied p16")
PY
}

run_one_trial() {
  local step="$1"
  local patch_name="$2"
  local note="$3"
  local stamp="step${step}_${patch_name}"
  local log_stamp="dsmin_${stamp}_$(date +%Y%m%d_%H%M%S)"

  stop_train
  log "=== RUN ${stamp}: ${note} ==="

  (
    cd "${SFT_DIR}"
    SKIP_INSTALL=1 \
    YAML_CONFIG="${YAML_CONFIG}" \
    LOG_TIMESTAMP="${log_stamp}" \
    bash launch_train_8b_no_blocking.sh "${HOSTFILE}"
  )

  # Prefer rank_0_* (trainer tee); wait up to 180s before falling back to rank0_*
  local log_dir="${SFT_DIR}/logs/${log_stamp}"
  local worker_log=""
  local wait_log_end=$(($(date +%s) + 180))
  while (( $(date +%s) < wait_log_end )); do
    for f in "${log_dir}"/rank_0_*.log; do
      if [[ -f "$f" ]]; then worker_log="$f"; break; fi
    done
    [[ -n "${worker_log}" ]] && break
    sleep 2
  done
  if [[ -z "${worker_log}" ]]; then
    for f in "${log_dir}"/rank0_*.log; do
      if [[ -f "$f" ]]; then worker_log="$f"; break; fi
    done
  fi
  if [[ -z "${worker_log}" ]]; then
    echo -e "${step}\t${patch_name}\tNO_LOG\t${log_dir}\t${note}" | tee -a "${RESULT_TSV}"
    return 1
  fi
  log "monitoring ${worker_log}"

  local start_ts end_ts
  start_ts=$(date +%s)
  end_ts=$((start_ts + MAX_RUN_SECS))
  local saw_running=0
  local last_progress_ts=0
  local last_loss_count=0
  local status="TIMEOUT"
  local detail=""

  while (( $(date +%s) < end_ts )); do
    if grep -Fq "Training completed" "${worker_log}" 2>/dev/null; then
      status="PASS"
      detail="Training completed"
      break
    fi
    if grep -qE "ChildFailedError|CalledProcessError" "${worker_log}" 2>/dev/null; then
      if ! pgrep -f "llamafactory/launcher.py" >/dev/null 2>&1; then
        status="FAIL"
        detail="error in log"
        break
      fi
    fi
    if grep -Fq "***** Running training *****" "${worker_log}" 2>/dev/null; then
      if (( saw_running == 0 )); then
        saw_running=1
        last_progress_ts=$(date +%s)
        log "saw Running training in ${worker_log}"
      fi
      local loss_count
      loss_count=$(grep -Fco "{'loss'" "${worker_log}" 2>/dev/null || true)
      loss_count=${loss_count:-0}
      if (( loss_count > last_loss_count )); then
        last_loss_count=${loss_count}
        last_progress_ts=$(date +%s)
        log "progress: loss_count=${loss_count}"
      fi
      if (( last_progress_ts > 0 )) && (( $(date +%s) - last_progress_ts > HANG_SECS )); then
        status="HANG"
        detail="no progress ${HANG_SECS}s (loss_count=${last_loss_count})"
        break
      fi
    fi
    if (( saw_running == 1 )) && ! pgrep -f "llamafactory/launcher.py" >/dev/null 2>&1 \
       && ! pgrep -f "train_8b_test_no_blocking.sh" >/dev/null 2>&1; then
      sleep 2
      if grep -Fq "Training completed" "${worker_log}" 2>/dev/null; then
        status="PASS"
        detail="Training completed"
      else
        status="FAIL"
        detail="process exited without Training completed"
      fi
      break
    fi
    sleep 5
  done

  if [[ "${status}" != "PASS" ]]; then
    stop_train
  else
    # wait a bit for clean exit
    local wait_end=$(($(date +%s) + 180))
    while pgrep -f "llamafactory/launcher.py" >/dev/null 2>&1 && (( $(date +%s) < wait_end )); do
      sleep 5
    done
    stop_train
  fi

  log "RESULT ${stamp}: ${status} (${detail})"
  echo -e "${step}\t${patch_name}\t${status}\t${log_dir}\t${note}" | tee -a "${RESULT_TSV}"
  [[ "${status}" == "PASS" ]]
}

# Dry-run: verify all patches apply cleanly then reset
verify_patches() {
  reset_ds
  apply_p01_torch_sync_path
  apply_p02_torch_drop_else_wait
  apply_p03_mics_wait1
  apply_p04_mics_wait2
  apply_p05_pp_wait_1253
  apply_p06_pp_wait_1338
  apply_p07_pp_wait_1349
  apply_p08_pp_wait_1359
  apply_p09_pp_wait_1386
  apply_p10_pp_wait_1450
  apply_p11_pp_launch_wait_each
  apply_p12_pp_reduce_scatter_seq
  apply_p13_pp_broadcast_seq
  apply_p14_coord_sync_wait
  apply_p15_stage3_wait_before
  apply_p16_stage3_wait_after
  log "all patches apply OK"
  (cd "${DS_ROOT}" && git diff --stat deepspeed/)
  reset_ds
}

MODE="${1:-run}"

case "${MODE}" in
  verify)
    verify_patches
    ;;
  run)
    verify_patches
    reset_ds
    stop_train

    declare -a NAMES=(
      p01_torch_sync_path
      p02_torch_drop_else_wait
      p03_mics_wait1
      p04_mics_wait2
      p05_pp_wait_1253
      p06_pp_wait_1338
      p07_pp_wait_1349
      p08_pp_wait_1359
      p09_pp_wait_1386
      p10_pp_wait_1450
      p11_pp_launch_wait_each
      p12_pp_reduce_scatter_seq
      p13_pp_broadcast_seq
      p14_coord_sync_wait
      p15_stage3_wait_before
      p16_stage3_wait_after
    )
    declare -a FUNCS=(
      apply_p01_torch_sync_path
      apply_p02_torch_drop_else_wait
      apply_p03_mics_wait1
      apply_p04_mics_wait2
      apply_p05_pp_wait_1253
      apply_p06_pp_wait_1338
      apply_p07_pp_wait_1349
      apply_p08_pp_wait_1359
      apply_p09_pp_wait_1386
      apply_p10_pp_wait_1450
      apply_p11_pp_launch_wait_each
      apply_p12_pp_reduce_scatter_seq
      apply_p13_pp_broadcast_seq
      apply_p14_coord_sync_wait
      apply_p15_stage3_wait_before
      apply_p16_stage3_wait_after
    )
    n=${#NAMES[@]}

    apply_range() {
      local start="$1" end="$2" k
      for k in $(seq "${start}" "${end}"); do
        ${FUNCS[$k]}
      done
    }

    # Phase 0: baseline (no patch) — expect FAIL/HANG
    reset_ds
    run_one_trial 0 "baseline" "no patches" || true

    # Phase 1: all recorded patches
    reset_ds
    apply_range 0 $((n - 1))
    if ! run_one_trial 1 "all_patches" "all recorded patches"; then
      log "ALL patches failed — fall back to forward cumulative one-by-one"
      reset_ds
      first_pass=""
      for i in $(seq 0 $((n - 1))); do
        ${FUNCS[$i]}
        if run_one_trial "fwd_${i}" "${NAMES[$i]}" "cumulative through ${NAMES[$i]}"; then
          first_pass="${NAMES[$i]}"
          echo "${first_pass}" > "${RESULT_DIR}/first_pass.txt"
          printf '%s\n' "${NAMES[@]:0:$((i + 1))}" > "${RESULT_DIR}/minimal.txt"
          break
        fi
      done
    else
      echo "all_patches" > "${RESULT_DIR}/first_pass.txt"
      log "ALL patches PASS — find minimal one point at a time"

      # Phase 2: try each patch alone (likely-first, then remaining doc order)
      solo_pass=""
      declare -a SOLO_ORDER=(13 14 15 11 10 4 0 1 2 3 5 6 7 8 9 12)  # p14,p15,p16,p12,p11,p05,p01,...
      for i in "${SOLO_ORDER[@]}"; do
        reset_ds
        ${FUNCS[$i]}
        if run_one_trial "solo_${i}" "${NAMES[$i]}_alone" "only ${NAMES[$i]}"; then
          solo_pass="${NAMES[$i]}"
          echo "${solo_pass}" > "${RESULT_DIR}/minimal.txt"
          log "MINIMAL = ${solo_pass} alone"
          break
        fi
      done

      # Phase 3: if no solo works, grow suffix from end (p16, p15+p16, ...)
      if [[ -z "${solo_pass}" ]]; then
        log "No single patch enough — grow suffix from end"
        for start in $(seq $((n - 1)) -1 0); do
          reset_ds
          apply_range "${start}" $((n - 1))
          applied_list=("${NAMES[@]:${start}}")
          if run_one_trial "suf_${start}" "suffix_${start}" "suffix ${NAMES[$start]}..${NAMES[$((n - 1))]}"; then
            printf '%s\n' "${applied_list[@]}" > "${RESULT_DIR}/minimal.txt"
            log "MINIMAL suffix = ${applied_list[*]}"
            break
          fi
        done
      fi
    fi

    log "REVERT all DeepSpeed changes"
    reset_ds
    stop_train
    log "Results: ${RESULT_TSV}"
    cat "${RESULT_TSV}"
    [[ -f "${RESULT_DIR}/minimal.txt" ]] && { log "Minimal set:"; cat "${RESULT_DIR}/minimal.txt"; }
    [[ -f "${RESULT_DIR}/first_pass.txt" ]] && { log "First pass:"; cat "${RESULT_DIR}/first_pass.txt"; }
    ;;
  *)
    echo "Usage: $0 [verify|run]"
    exit 1
    ;;
esac
