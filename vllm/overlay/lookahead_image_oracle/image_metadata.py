"""Bounded T=1 image layout and independent Qwen MRoPE arithmetic."""
import struct
from types import SimpleNamespace
from .strict_receipt import ContractFailure

MAX_IMAGES=2
MAX_VISUAL_TOKENS=512
SPATIAL_MERGE=2
ENCODER_HIDDEN=2560
MAX_ENCODER_BYTES=MAX_IMAGES*MAX_VISUAL_TOKENS*ENCODER_HIDDEN*2


def require(ok,message):
    if not ok:raise ContractFailure(message)


def require_image_flag(images):
    if images:
        import image_admission
        require(image_admission.enabled(),'Image state readout requires its isolated diagnostic flag')


def validate_images(images,prompt):
    require(type(images) is list and len(images)<=MAX_IMAGES,'Image inventory outside bounded scope')
    end=0
    for image in images:
        require(type(image) is dict and set(image)=={'identifier','offset','length','grid_thw'},'Unknown image layout fields')
        offset,length,grid=image['offset'],image['length'],image['grid_thw']
        require(type(image['identifier']) is str and bool(image['identifier']),'Missing image identifier')
        require(type(offset) is int and type(length) is int and end<=offset
                and 0<length<=MAX_VISUAL_TOKENS and offset+length<=prompt,'Image outside prompt or overlapping')
        require(type(grid) is list and len(grid)==3 and all(type(x) is int and x>0 for x in grid)
                and grid[0]==1 and grid[1]%SPATIAL_MERGE==grid[2]%SPATIAL_MERGE==0
                and (grid[1]//SPATIAL_MERGE)*(grid[2]//SPATIAL_MERGE)==length,
                'Image grid/placeholder disagrees or temporal/pruned image')
        end=offset+length
    return images


def feature_layout(features,prompt):
    require(type(features) is list,'Unexpected actual feature container')
    result=[]
    for feature in features:
        position=getattr(feature,'mm_position',None)
        require(getattr(feature,'modality',None)=='image' and position is not None
                and getattr(position,'is_embed',None) is None,
                'Only unpruned image placeholders admitted')
        require(feature.data is not None and 'image_grid_thw' in feature.data,'Actual image grid missing')
        grid=feature.data['image_grid_thw'].data.tolist()
        result.append(dict(identifier=feature.identifier,offset=position.offset,length=position.length,grid_thw=grid))
    return validate_images(result,prompt)


def position_rows(length,images):
    """Analytic counterpart of pinned Qwen3VL normal T=1, unpruned path."""
    require(type(length) is int and 0<length<=2048,'Position prefix outside oracle bound')
    validate_images(images,length)
    rows=[[],[],[]];cursor=0;next_pos=0
    for image in images:
        text=image['offset']-cursor
        for row in rows:row.extend(range(next_pos,next_pos+text))
        base=next_pos+text;h=image['grid_thw'][1]//SPATIAL_MERGE;w=image['grid_thw'][2]//SPATIAL_MERGE
        rows[0].extend([base]*(h*w))
        rows[1].extend(base+y for y in range(h) for _ in range(w))
        rows[2].extend(base+x for _ in range(h) for x in range(w))
        next_pos=base+max(h,w);cursor=image['offset']+image['length']
    for row in rows:row.extend(range(next_pos,next_pos+length-cursor))
    require(all(len(row)==length for row in rows),'Analytic image positions lost tokens')
    return rows,max(max(row) for row in rows)+1-length


def position_bytes(rows):
    require(type(rows) is list and len(rows)==3 and len({len(row) for row in rows})==1,'Expected equal three-axis rows')
    return b''.join(struct.pack('<i',p) for row in rows for p in row)


def computed_rows(prefill,computed,images):
    require(prefill<=computed<=2048,'Invalid computed MRoPE frontier')
    rows,delta=position_rows(prefill,images)
    for row in rows:row.extend(p+delta for p in range(prefill,computed))
    return rows,delta


class Grid:
    def __init__(self,values):self.values=values
    def tolist(self):return list(self.values)


def actual_position_rows(model,ids,images):
    config=model.config
    require(config.vision_config.spatial_merge_size==SPATIAL_MERGE,'Unexpected spatial merge')
    features=[SimpleNamespace(modality='image',mm_position=SimpleNamespace(offset=i['offset'],length=i['length']),
              data={'image_grid_thw':SimpleNamespace(data=Grid(i['grid_thw']))}) for i in images]
    positions,delta=model.get_mrope_input_positions(ids,features)
    actual=positions.tolist();expected,expected_delta=position_rows(len(ids),images)
    require(actual==expected and type(delta) is int and delta==expected_delta,'Actual model MRoPE disagrees with analytic expectation')
    return actual,delta


def encoder_descriptors(images):
    by_id={}
    for image in images:
        descriptor=dict(identifier=image['identifier'],dtype='bfloat16',shape=[image['length'],ENCODER_HIDDEN],
                        bytes=image['length']*ENCODER_HIDDEN*2)
        require(image['identifier'] not in by_id or by_id[image['identifier']]==descriptor,
                'Repeated image identifier has inconsistent geometry')
        by_id[image['identifier']]=descriptor
    return [by_id[key] for key in sorted(by_id)]
