"""Read-only admission checks shared by the core and worker utility."""
from .receipt_schema import require, expected_components, same
from .layout import validate_block_ids
from .image_metadata import validate_images


def validate_config(config):
    p = config.parallel_config
    require(config.use_v2_model_runner is True, 'MRV2 required')
    require(config.scheduler_config.async_scheduling is True, 'Expected pinned async engine')
    require(config.speculative_config is None and config.num_speculative_tokens == 0, 'MTP0 required')
    require(p.tensor_parallel_size == p.pipeline_parallel_size == p.data_parallel_size == 1,
            'TP1/PP1/DP1 required')
    require(p.enable_dbo is False, 'DBO outside oracle scope')
    require(p.decode_context_parallel_size == 1, 'Context parallelism unsupported')
    require(config.cache_config.enable_prefix_caching is False, 'Prefix caching unsupported')
    require(config.cache_config.mamba_cache_mode == 'none', 'Only current recurrent state supported')
    require(config.kv_transfer_config is None and config.ec_transfer_config is None,
            'Transfer connector unsupported')
    require(config.model_config.runner_type == 'generate', 'Generation model required')


def validate_wire(wire):
    require(type(wire) is dict, 'Missing snapshot wire')
    require(type(wire['request_id']) is str and wire['request_id'], 'Invalid request ID')
    computed = wire['computed_tokens']
    expected_components(computed)
    ids = wire['accepted_token_ids']
    require(type(ids) is list and all(type(x) is int and 0 <= x < 2**31 for x in ids),
            'Malformed accepted token IDs')
    require(len(ids)-computed in (0, 1), 'Invalid accepted/computed frontier')
    require(same(wire['logical_positions'], list(range(computed))), 'Logical remapping unsupported')
    require(type(wire['original_prompt_tokens']) is int
            and 0 < wire['original_prompt_tokens'] <= computed, 'Invalid prompt frontier')
    validate_images(wire.get('image_layout',[]),wire['original_prompt_tokens'])
    validate_block_ids(wire['block_ids'])
    proof = wire['core_proof']
    require(type(proof) is dict and all(proof.get(k) is True for k in
        ('all_owners_paused', 'queues_empty', 'no_pending_transactions', 'exclusive_pages')),
        'Missing guarded core proof')
    require(proof['request_id'] == wire['request_id']
            and same(proof['block_ids'], wire['block_ids']), 'Core proof bound to another owner')
    return computed


def no_staged_writes(buffer):
    """A StagedWriteTensor must have nothing pending; a plain UvaBackedTensor (this
    runtime's block tables) has no staging lists and therefore nothing pending."""
    for key in ('_staged_write_indices', '_staged_write_starts',
                '_staged_write_contents', '_staged_write_cu_lens'):
        require(getattr(buffer, key, []) == [], 'Pending staged write: ' + key)
