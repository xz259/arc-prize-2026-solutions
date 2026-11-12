"""Early-gameplay T1 image admission; native context/MM limits remain authoritative."""
import os

FLAG = 'ARC3_PERSISTENT_ONE_LOOKAHEAD_IMAGE_DIAGNOSTIC'
IMAGE_TOKEN = 248056
VISION_START = 248053
VISION_END = 248054


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def enabled():
    value = os.environ.get(FLAG, '0')
    require(value in ('0', '1'), FLAG+' must be 0 or 1')
    return value == '1'


def image_layout(features, ids):
    """Inspect actual processed features; never infer a grid from placeholders."""
    require(type(ids) is list and all(type(x) is int and 0 <= x < 2**31 for x in ids),
            'Explicit integer token IDs required')
    result = []
    end = 0
    for feature in features:
        require(feature.modality == 'image', 'Only image modality is admitted')
        position = feature.mm_position
        offset, length = position.offset, position.length
        require(type(offset) is int and type(length) is int and length > 0
                and end <= offset and 0 < offset and offset+length < len(ids),
                'Invalid, overlapping or truncated image placement')
        require(position.is_embed is None, 'Embedding masks/pruning outside image diagnostic')
        require(ids[offset-1] == VISION_START and ids[offset+length] == VISION_END
                and ids[offset:offset+length] == [IMAGE_TOKEN]*length,
                'Actual visual placeholder IDs do not match the image span')
        require(type(feature.identifier) is str and bool(feature.identifier), 'Missing image identifier')
        require(feature.data is not None and 'image_grid_thw' in feature.data,
                'Processed image grid unavailable; cached payload not admitted without proof')
        grid_tensor = feature.data['image_grid_thw'].data
        require(str(grid_tensor.device) == 'cpu' and str(grid_tensor.dtype) == 'torch.int64',
                'Expected CPU int64 processed image grid')
        require(list(grid_tensor.shape) == [3], 'Expected one unbatched image grid')
        grid = grid_tensor.tolist()
        # Actual patch16 / temporal2 / spatial merge2 processor output. Cropped
        # Python boards can be rectangular; native processing owns resizing.
        require(all(type(x) is int and x > 0 for x in grid)
                and grid[0] == 1 and grid[1] % 2 == grid[2] % 2 == 0
                and length == (grid[1]//2)*(grid[2]//2),
                'Only actual unpruned T1 grids with exact merge2 token counts admitted')
        result.append(dict(identifier=feature.identifier, offset=offset, length=length, grid_thw=grid))
        end = offset+length
    return result


def validate_request_images(request, existing=None):
    """Called only for marked requests by the copied one-lookahead validator."""
    new = request.mm_features or []
    old = (existing.mm_features or []) if existing is not None else []
    if not new and not old:
        return
    require(enabled(), 'Images require the isolated image diagnostic flag')
    ids = list(request.prompt_token_ids or [])
    accepted = list(existing._all_token_ids) if existing is not None else []
    # Exact native initial/append guards enforce scheduler.max_model_len and
    # remaining output capacity; the native MM processor enforces prompt limits.
    # This removes only the two-image/one-page diagnostic restrictions.
    if existing is not None:
        state = getattr(existing, '_arc3_eviction_state', None)
        # Successful compaction republishes dense accepted IDs and remapped
        # resident feature offsets; original-position metadata remains owned
        # by the existing eviction state. Uncertain transactions stay closed.
        require(state is None or state.poisoned is False,
                'Image admission cannot follow an uncertain eviction transaction')
        image_layout(old, accepted)
    image_layout(new, ids)
