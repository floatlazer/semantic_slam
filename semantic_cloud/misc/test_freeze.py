from ptsemseg.models import get_model

def freeze(m):
    if m.__class__.__name__ == 'BatchNorm2d':
        print m

psp = get_model('pspnet', 38, version = 'sunrgbd')
#print(psp.cbr_final.cbr_unit)
psp.apply(freeze)
