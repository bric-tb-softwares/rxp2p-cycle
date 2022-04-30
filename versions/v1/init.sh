
rm -rf rxcore || true
rm -rf rxrxpixp2pixcycle || true
rm -rf wandb  || true

export WANDB_API_KEY=$WANDB_API_KEY

git clone https://github.com/bric-tb-softwares/rxcore.git
git clone https://github.com/bric-tb-softwares/rxpixp2pixcycle.git

# setup into the python path
export PYTHONPATH=$PYTHONPATH:$PWD/rxcore
export PYTHONPATH=$PYTHONPATH:$PWD/rxwgan
export PATH=$PATH:$PWD/rxpixp2pixcycle
echo $PYTHONPATH
ls -lisah