# KCI2023

<h1>Learning discriminative emotional features from facial expressions via ResNet50 with center loss</h1>

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
  <tr align='center'><td>RAF-DB</td><td>76.235</td><td><b>77.477</b></td></tr>
  <tr align='center'><td>FER2013</td><td>62.138</td><td><b>62.572</b></td></tr>  
</table>

FER2013 datset: https://www.kaggle.com/datasets/msambare/fer2013
