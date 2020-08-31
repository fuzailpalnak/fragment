# Fragment
![GitHub](https://img.shields.io/github/license/cypherics/ShapeMerge)
![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)


Splitting the Image in to Multiple fragments of smaller size and joining the fragment data back


## Installation

    pip install git+https://github.com/fuzailpalnak/fragment.git#egg=fragment


## Use

```python
import numpy as np
from fragment.fragment import ImageFragment

image = np.zeros((1024, 1024, 3))
new_image = np.zeros((1024, 1024, 3))

image_fragment = ImageFragment.get_image_fragment(fragment_size=(512, 512, 3), org_size=(1024, 1024, 3))
for fragment in image_fragment:
    # GET DATA THAT BELONGS TO THE FRAGMENT
    fragmented_image = fragment.get_fragment_data(image)
    
    # DO SOME OPERATION ON THE FRAGMENTED DATA
    operation_done_on_fragmented_data = np.rot(fragmented_image)
    
    # TRANSFER OPERATED IMAGE ON NEW IMAGE ON THE FRAGMENT POSITION
    new_image = fragment.transfer_fragment(transfer_from=operation_done_on_fragmented_data, transfer_to=new_image)
    
```