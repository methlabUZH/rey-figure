
from config import MODEL_PATH
from watcher import Watcher
from restore_predict import restore

# adapt such that it works for all formats ('one-per-item', ...)
# restore commit in result.txt to ensure loading model parameters works

models = []

for i in range(10):
    models.append(restore(MODEL_PATH + "model_fold{}.ckpt".format(i)))

w = Watcher(models)
w.run()



