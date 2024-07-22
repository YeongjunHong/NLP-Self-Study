import sys
sys.path.append("..")
from trainer import Trainer
from optimizer import Adam
from simple_CBOW import SimpleCBOW
from utils import preprocess, create_context_target, convert_one_hot

window_size = 1
hidden_size =5
batch_size =3
max_epoch = 10000

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = len(word_to_id)
contexts, target = create_context_target(corpus, window_size)

target = convert_one_hot(target,vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

model = SimpleCBOW(vocab_size,hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()