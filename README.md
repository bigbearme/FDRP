# End-to-End Flood Disaster Risk Prediction for the Poyang Lake Basin, China, Based on Informer
## Continuously updating!
Note: Still under review, this code repository is not yet fully complete.
## Task List
- [x] Release test data.
- [x] Release a simplified version of the code for testing purposes.
- [ ] Release the code for evaluating the results. In progress...
- [ ] Release the complete source code. 
- [ ] Release of the program for integrating end-to-end flood forecasting.
## 1.Setup
```
git clone https://github.com/bigbearme/FDRP.git
cd FDRP
conda create -n FDRP python=3.8
conda activate FDRP
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install rasterio
```
## 2.Data Preparation
The provided dataset structure is as follows. If using your own dataset, please organize it according to the structure below.

**Material Data**
```
Data/
├─ ─ NDVI
     ├─ ─ NDVI_06/01
     ├─ ─ NDVI_06/02
     ├─ ─ NDVI_06/03
     ├─ ─ ...
├─ ─ Elevation
├─ ─ GDP
├─ ─ ...
├─ ─ Daily rainfall
```
**Evaluation Data**
```
Risk/
├─ ─ risk_06/01
├─ ─ risk_06/02
├─ ─ risk_06/03
├─ ─ ...
```
## 3.Quick Start
## Get Positional Encodings
```
python get_postcode.py
```
## Training
```
python train.py
```
## Testing
```
python test.py
```
## 4.Results and Evaluation
# Contact:
```
Don't hesitate to contact me if you meet any problems when using this code.
Email: bigxiong23@gmail.com
```
