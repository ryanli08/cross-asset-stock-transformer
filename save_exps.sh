

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
model=$1
echo "Saving Model: $model"
mkdir -p "experiments/$model/$TIMESTAMP"
case $model in 
    "ct")
      cp -r results/cross_asset_transformer/ "../../exps/$model/$TIMESTAMP/"
      cp configs/cross.yaml "../../exps/$model/$TIMESTAMP/"
      echo "Saved experiment to ../../exps/$model/$TIMESTAMP/"
      ;;
    "mt")
      cp -r results/multi_asset_transformer_baseline/ "../../exps/$model/$TIMESTAMP/"
      cp configs/multi.yaml "../../exps/$model/$TIMESTAMP/"
      echo "Saved experiment to ../../exps/$model/$TIMESTAMP/"
      ;;
    "lstm")
      cp -r results/lstm_baseline/ "../../exps/$model/$TIMESTAMP/"
      cp configs/lstm.yaml "../../exps/$model/$TIMESTAMP/"
      echo "Saved experiment to ../../exps/$model/$TIMESTAMP/"
      ;;
    "bt")
      cp -r results/transformer_baseline/ "../../exps/$model/$TIMESTAMP/"
      cp configs/base.yaml "../../exps/$model/$TIMESTAMP/"
      echo "Saved experiment to ../../exps/$model/$TIMESTAMP/"
      ;;
    *)
      echo "Unknown model: $model. Options are: ct, mt, lstm, bt"
      ;;
esac