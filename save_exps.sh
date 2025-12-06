

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
model=$1
echo "Saving Model: $model"
mkdir -p "experiments/$model/$TIMESTAMP"
case $model in 
    "ct")
      cp -r results/cross_asset_transformer/ "experiments/$model/$TIMESTAMP/"
      cp configs/cross.yaml "experiments/$model/$TIMESTAMP/"
      echo "Saved experiment to experiments/$model/$TIMESTAMP/"
      ;;
    "mt")
      cp -r results/multi_asset_transformer_baseline/ "experiments/$model/$TIMESTAMP/"
      cp configs/multi.yaml "experiments/$model/$TIMESTAMP/"
      echo "Saved experiment to experiments/$model/$TIMESTAMP/"
      ;;
    "lstm")
      cp -r results/lstm_baseline/ "experiments/$model/$TIMESTAMP/"
      cp configs/lstm.yaml "experiments/$model/$TIMESTAMP/"
      echo "Saved experiment to experiments/$model/$TIMESTAMP/"
      ;;
    "bt")
      cp -r results/transformer_baseline/ "experiments/$model/$TIMESTAMP/"
      cp configs/base.yaml "experiments/$model/$TIMESTAMP/"
      echo "Saved experiment to experiments/$model/$TIMESTAMP/"
      ;;
    *)
      echo "Unknown model: $model. Options are: ct, mt, lstm, bt"
      ;;
esac