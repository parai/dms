import os

def RunCommand(cmd, e=True):
    if(os.name == 'nt'):
        cmd = cmd.replace('&&', '&')
    ret = os.system(cmd)
    if(0 != ret and e):
        raise Exception('FAIL of RunCommand "%s" = %s'%(cmd, ret))
    return ret
