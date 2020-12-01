# Fragment
![GitHub](https://img.shields.io/github/license/fuzailpalnak/fragment)
![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
![Downloads](https://pepy.tech/badge/image-fragment)

Any extent is a set of smaller fragment extent , which can be as small as *1 x 1* to the *size* of the extent given.
This library will section the given extent in to specified smaller fragment extent


## Installation

    pip install image-fragment


## Image Fragments

![fragments](https://user-images.githubusercontent.com/24665570/91711219-2b7c8980-eba3-11ea-94d2-8239cf6713c4.gif)

An image extent can be sectioned into many fragments, these fragments holds positional information to where it is located 
to the original extent

Now this information can be used to extract selected fragments from the Image, perform operation, and transfer the new 
data to selected position.


```python
import numpy as np
from image_fragment.fragment import ImageFragment

image = np.zeros((1, 1024, 1024, 3))
new_image = np.zeros((1, 1024, 1024, 3))
image_fragment = ImageFragment.image_fragment_4d(fragment_size=(1, 512, 512, 3), org_size=(1, 1024, 1024, 3))
for fragment in image_fragment:
    # GET DATA THAT BELONGS TO THE FRAGMENT
    fragmented_image = fragment.get_fragment_data(image)
    
    # DO SOME OPERATION ON THE FRAGMENTED DATA
    operation_done_on_fragmented_data = np.rot(fragmented_image)
    
    # TRANSFER OPERATED IMAGE ON NEW IMAGE ON THE FRAGMENT POSITION
    new_image = fragment.transfer_fragment(transfer_from=operation_done_on_fragmented_data, transfer_to=new_image)
    
```

If image is 3 Dimensional then switch to `image_fragment = ImageFragment.image_fragment_3d(fragment_size=(512, 512, 3), org_size=(1024, 1024, 3))`, 
this will provide fragments for 3 dimensional image
