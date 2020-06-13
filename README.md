## Masked Face Recognition(Capstone design 2020-1)

<p align="center">
    <img src="/Codes/Abstract.jpg" width="700px" height="300px" title="px(픽셀) 크기 설정" alt="RubberDuck"></img></div>
</p>
<br/>
### Overview
***
- Needs, Problems
    Because of covid-19, there is high misrecognition rate of masked face.
- Goals, Objectives (evaluation)
    NRMSE 5%, false positive 0%
    
### How To Run
***
1. git clone https://github.com/Hi-friends/SWCapstone.git
2. Download pretrained SHN model [here](https://drive.google.com/drive/folders/1AbTGhIBzUUINTH2GNL05tSWvOHnclRr4)
& Download your test image at "Codes/FINAL/image/test", file name should be maskx.jpg'
3. Locate that model at 'SWCapstone/Codes/SHN/models/model-hg2d3-cab/model/' dierectory
4. SWCapstone/Codes/SHN$python test.py
5. SWCapstone/Codes/FINAL$python align.py
6. $docker pull bamos/openface
7. Upload your testdata(which is in your SWCapstone/Codes/FINAL/transformed directory) to root/openface/aligned-images, then you can get labels.csv and reps.csv
8. SWCapstone/Codes/FINAL$python recog_test.py
9. Get Results


### Reference
***
- Paper   
[Multistage Model for Robust Face Alignment Using Deep Neural Networks(2020)](https://arxiv.org/pdf/2002.01075.pdf)   
[FaceID-GAN:Learning a Symmetry Three-Player GAN for Identity-Preserving Face Synthesis(2018)](http://openaccess.thecvf.com/content_cvpr_2018/papers/Shen_FaceID-GAN_Learning_a_CVPR_2018_paper.pdf)   
[ST-GAN: Spatial Transformer Generative Adversarial Networks for Image Compositing(2016)](https://arxiv.org/pdf/1506.02025.pdf)   
Robust face landmark estimation under occlusion(2013)   
A Deep Regression Architecture with Two-Stage Re-initialization for High Performance Facial Landmark Detection(2017)   
Occlusion Coherence: Detecting and Localizing Occluded Faces(2016)   
Robust Facial Landmark Detection via Aggregation on Geometrically Manipulated Faces(2020)   
STGAN: A Unified Selective Transfer Network for Arbitrary Image Attribute Editing(2019)   
[Robust Facial Landmark Detection via Occlusion-adaptive Deep Networks(2019)](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Robust_Facial_Landmark_Detection_via_Occlusion-Adaptive_Deep_Networks_CVPR_2019_paper.pdf)   

- GitHub   
[ST-GAN](https://github.com/chenhsuanlin/spatial-transformer-GAN)   
[InterFaceGAN](https://github.com/genforce/interfacegan)   
[STN](https://github.com/kevinzakka/spatial-transformer-network)   
[Face Frontalization GAN](https://github.com/scaleway/frontalization)   
[Hourglass Network](https://github.com/deepinx/deep-face-alignment)   
[Hourglass Network](https://github.com/viliusmat/SHN-based-2D-face-alignment)   
[Affine Transformation](https://github.com/cmusatyalab/openface/blob/master/openface/align_dlib.py)   
[Generate Embedding](https://gist.github.com/ageitgey/ddbae3b209b6344a458fa41a3cf75719)   
[SRGAN](https://github.com/dongheehand/SRGAN-PyTorch)   
[CCGAN](https://github.com/mafda/generative_adversarial_networks_101/blob/master/src/mnist/04_CCGAN_MNIST.ipynb)   
[Inpainting](https://github.com/JiahuiYu/generative_inpainting)   
[Generative-Inpainting](https://github.com/daa233/generative-inpainting-pytorch)   

- HomePage   
[Docker container to image](https://galid1.tistory.com/323)   


### Reports
***
+ Google Drive
+ Report : [Report](https://github.com/Hi-friends/SWCapstone/tree/master/Reports)
+ Final : [Report](https://github.com/Hi-friends/SWCapstone/tree/master/Reports), Demo
