# SKA-DECONV
UNET network for deconvolving and denoising dirty images for SKA-LOW.  Very Very early stage, still under development.

Currently contains "deconv_ae.py" for training, and "load_model.py" for creating a 'cleaned' image using the trained model.
The out_src folder contains the polynomial fitting code used for EoR signal separation, which is modified from https://github.com/liweitianux/cdae-eor/tree/master/code/polyfit.py

## 方法设计

- [算子感知、nuisance-hardened 二维功率谱估计器详细测试计划](docs/nuisance_hardened_ps2d_test_plan.md)

Please contact zhenghao@shao.ac.cn if you want to know more or use the training set.
 
