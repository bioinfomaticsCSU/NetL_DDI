# NetL_DDI

License
=========

Copyright (C) 2014 Junwei Luo(luojunwei@hpu.edu.cn)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 3
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see <http://www.gnu.org/licenses/>.

NetL_DDI
=================

NetL_DDI, which is a novel DDI prediction and classification method based on network representation learning [1], can predict potential interactions between drugs and identify their hidden mechanisms.

1. Dataset.
```
(1) DrugIDs.txt: stores 2159 drugs with DrugBank ids;
(2) Redr.mat: 2159×86 drug-drug interaction matrix;
(3) group_191878.mat: 191,878×86 matrix, stores 191,878 drug-drug interactions and their DDI types;
(4) ddi_index_191878.npy: 191,878×2 matrix, stores 191,878 drug-drug interactions, each row corresponds to one drug pair which is indicated by indexes of drugs in file DrugIDs.txt;
* These files are constructed based on data from [2].
```
2. Code.
```
(1) train_DDI.py: perform 10-fold cross validation;
(2) load_data.py : construct train set, validation set and test set;
(3) eval_DDI.py: performance evaluation;
```

[1] Qiu, J. et al. (2018) Network embedding as matrix factorization: Unifying deepwalk, line, pte, and node2vec. In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining, 459-467.
[2] Ryu, J. Y. et al. (2018) Deep learning improves prediction of drug-drug and drug-food interactions. Proceedings of the National Academy of Sciences, 115(18), E4304-E4311.
