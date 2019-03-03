#  DeepLens
#  Copyright (c) 2019. Adam Dziedzic and Sanjay Krishnan
#  Licensed under The MIT License [see LICENSE for details]
#  Written by Adam Dziedzic

import matlab.engine

eng = matlab.engine.start_matlab()

eng.triarea(nargout=0)

eng.eval_sort_tracker_MOT16_train(nargout=0)
