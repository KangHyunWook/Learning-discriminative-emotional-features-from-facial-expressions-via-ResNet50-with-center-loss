
<h1>Learning discriminative emotional features from facial expressions via ResNet50 with center loss</h1>

Background: Although existing deep learning models are capable of learning inter-class variance using softmax function, they suffer from intra-class variance when they do label predictions as illustrated in the left figure below.

![alt text](https://github.com/KangHyunWook/Learning-discriminative-emotional-features-from-facial-expressions-via-ResNet50-with-center-loss/blob/main/Screenshot%20from%202023-04-29%2009-51-51.png)

To avoid intra-class variance and ensure both intra-class compactness and inter-class variance, the distance between the center and its corresponding deep features is penalized by the center loss, which has been first introduced by Wen et al.

This code trains the ResNet50 under the joint supervision of softmax and center loss. As a result, the deep features are effectively clustered to their respective centers as shown in the right figure above.

run as follows:
```
python main.py
```
add below options:
--center for center loss


<h2>Experimental results</h2>
Classification accuracy

<table>
  <tr align='center'><td>dataset</td><td>vanilla</td><td>w/ center loss</td></tr>
  <tr align='center'><td>RAF-DB</td><td>76.239</td><td><b>78.683</b></td></tr>
  <tr align='center'><td>FER2013</td><td>62.079</td><td><b>63.402</b></td></tr>  
</table>

RAF-DB dataset: http://www.whdeng.cn/raf/model1.html <br />
FER2013 datset: https://www.kaggle.com/datasets/msambare/fer2013

<b>References</b>:

Wen, Y., Zhang, K., Li, Z., & Qiao, Y. (2016). A discriminative feature learning approach for deep face recognition. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part VII 14 (pp. 499-515). Springer International Publishing.
