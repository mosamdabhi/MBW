
#!/bin/bash 

# Chimpanzee  Colobus_Monkey  Fish          Seahorse
# Clownfish	  Flamingo        Human36M      Tiger

dataset='Chimpanzee'

echo "**** Running unit tests for $dataset dataset ****"

./scripts/preprocess.sh $dataset
./scripts/infer_flow.sh $dataset
./scripts/train_mvnrsfm.sh $dataset
./scripts/train_detector.sh $dataset
