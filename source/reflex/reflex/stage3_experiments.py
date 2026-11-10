"""Pure configuration contract for three mutually exclusive S3 experiments."""
import os

FLAGS={'A':'BOX_RETAIN_X10','B':'BOX_DYNAMIC_BUFFER','C':'BOX_FLEX_ARMS'}


def experiment(env=None):
    env=os.environ if env is None else env
    chosen=[]
    for group,key in FLAGS.items():
        value=env.get(key,'0')
        if value not in ('0','1'):raise ValueError(f'{key} must be 0 or 1')
        if value=='1':chosen.append(group)
    if len(chosen)>1:raise ValueError('S3 experiments are independent; enable only ONE of '+', '.join(FLAGS.values()))
    if chosen and env.get('BOX_BUFFER_STAGE','0')!='3':
        raise ValueError('S3 experiment flags require BOX_BUFFER_STAGE=3')
    return chosen[0] if chosen else 'S3'


def weights(env=None):
    env=os.environ if env is None else env
    group=experiment(env)
    intake=float(env.get('BOX_BUFFER_INTAKE_WEIGHT','.75'))
    retain=float(env.get('BOX_BUFFER_RETAIN_WEIGHT','.5'))
    if group!='S3' and (intake!=.75 or retain!=.5):
        raise ValueError('independent experiments require the matched S3 defaults intake=.75, retain=.5')
    return (6. if group=='B' else intake,5. if group=='A' else retain)


if __name__=='__main__':
    import sys
    group=experiment()
    if len(sys.argv)>1:
        expected=sys.argv[1]
        if group!=expected:raise ValueError(f'requested {expected}, inherited flags select {group}; unset experiment flags')
        if expected=='S3' and weights()!=(.75,.5):
            raise ValueError('matched S3 baseline requires intake=.75, retain=.5')
    print(group,weights())
